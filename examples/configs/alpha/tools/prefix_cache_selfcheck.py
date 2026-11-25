#!/usr/bin/env python3
"""PC-S — same-server prefix-cache accuracy gate for the alpha serving engine (vLLM, `--mamba-cache-mode align`).

Why this shape: comparing a cache-off server with a cache-on server cannot resolve a cache error, because two
separately started servers already differ by more than the old threshold (GATES.md, 2026-11-25). Here one
cache-on server computes every prompt twice: once as a cache hit and once cold (a fresh `cache_salt` changes the
hash of block 0 and so of every later block). A single server repeats itself bit for bit, so a correct restore
must give bit-identical log-probs whenever the hit path and the cold path cut the prefill into the same chunks.

What is judged (all must hold; "identical" = token ids, token log-prob and the whole top-5 dict, 32 tokens, and —
when the engine returns them (`--routes`, router replay) — the routed-expert rows of the whole sequence, which
for a hit come from the cached blocks' slots):
  rep      the same prompt computed cold twice is identical            (otherwise the exact criterion is void)
  chain    cache hit vs cold where `Model` predicts the same chunk sequence; each hit is sent twice, so a hit
           request that damages the block it restored from is seen by the second one
  decode   a hit on a block boundary that was produced while *generating* continues the original generation
           identically (the first computed token is a single-token step, i.e. the decode kernel again); the
           generating request is cold in `decode` and itself starts from a hit in `decode_hit`
  marconi  a request that goes through the common-prefix admission cut (attention blocks hit further than the
           recurrent state) vs a plain hit at the same boundary, which lies off the chunk-budget grid and is
           followed by a multi-block chunk
  requests every request's cached-token count (engine counters) equals the predicted hit, cold requests report 0
Reported but not judged ("char"): hits whose chunk sequence differs from the cold one (they cannot be identical),
and a later hit on the state the admission cut saved (its attention blocks for the cut segment are the first
request's, computed in a longer chunk than the state: `Model` tracks how the state was cut, not the KV blocks).

Not shown by a PASS: that cache-on equals cache-off (a defect that changes every boundary state the same way in
the cold path too is invisible to a same-server comparison); a defect of a multi-block chunk that starts off the
budget grid whether it starts from a restore or not (both sides of `admit_vs_offgrid` share it; the one-block-chunk
server does not have such chunks); a generated boundary state read by a multi-token prefill chunk (only the
not-judged forced rows take that path); concurrent requests, preemption, eviction; the chat-template path; any
GDN prefill backend other than the one the server logs.

    prefix_cache_selfcheck.py run   --url http://127.0.0.1:8011/v1 --budget 8192 --set prod --out prod.json [--serve-log f]
                                    [--routes] [--backend Triton/FLA]     (training engine: budget 16384, routes, triton)
    prefix_cache_selfcheck.py judge --clean prod.json b1056.json --nc nc_scale.json nc_bf16.json [--json out.json]

Standard library only (`python3 -I`). Sequential single requests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
import uuid
from array import array
from pathlib import Path

BLOCK = 1056  # tokens per cache block with the fp32 recurrent state (engine log: "Setting attention block size to 1056")
N_GEN = 32  # tokens generated (and compared) per request
N_DECODE = 1200  # tokens generated in the decode scenarios: crosses two block boundaries
N_FORCED = 16  # forced positions in the not-judged part of the decode scenario
REPO = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------------------------- engine client
def _http(url: str, body: dict | None = None, timeout: int = 1800):
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_npy(b64: str | None) -> dict | None:
    """The engine sends routed experts as a base64 `.npy` array [tokens, MoE layers, top-k]. Parsed without numpy."""
    if not b64:
        return None
    raw = base64.b64decode(b64)
    if raw[:6] != b"\x93NUMPY":
        raise RuntimeError("routed_experts is not an .npy payload")
    off = 10 if raw[6] == 1 else 12
    hlen = int.from_bytes(raw[8:off], "little")
    header, data = raw[off : off + hlen].decode("latin1"), raw[off + hlen :]
    m = re.search(r"'shape':\s*\(([^)]*)\)", header)
    if not m or "'fortran_order': False" not in header:
        raise RuntimeError(f"unexpected .npy header: {header[:120]}")
    shape = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    rows = shape[0] if shape else 0
    if not rows or len(data) % rows:
        raise RuntimeError(f"routed_experts payload of {len(data)} bytes does not fit shape {shape}")
    return {"rows": rows, "row_bytes": len(data) // rows, "shape": shape, "data": data}


class Engine:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.base = self.url[: -len("/v1")] if self.url.endswith("/v1") else self.url
        self.run_id = uuid.uuid4().hex[:10]
        self.n_salt = 0

    def salt(self, tag: str) -> str:
        """A salt never used before on this server: the request it goes with can hit nothing."""
        self.n_salt += 1
        return f"pcs-{self.run_id}-{self.n_salt}-{tag}"

    def counters(self) -> tuple[float, float]:
        prompt = cached = None
        for line in _http(self.base + "/metrics", timeout=20).decode().splitlines():
            m = re.match(r"^vllm:(prompt_tokens|prompt_tokens_cached)_total(\{[^}]*\})?\s+([-+0-9.eE]+)\s*$", line)
            if m:
                if m.group(1) == "prompt_tokens":
                    prompt = (prompt or 0.0) + float(m.group(3))
                else:
                    cached = (cached or 0.0) + float(m.group(3))
        if prompt is None or cached is None:
            raise RuntimeError("engine /metrics has no vllm:prompt_tokens_total / vllm:prompt_tokens_cached_total")
        return prompt, cached

    def tokenize(self, text: str) -> list[int]:
        r = json.loads(_http(self.base + "/tokenize", {"model": "alpha", "prompt": text, "add_special_tokens": False}))
        return [int(t) for t in r["tokens"]]

    def gen(self, ids: list[int], salt: str, n: int) -> dict:
        """Greedy generation of n tokens with top-5 log-probs; the engine counters give the cached-token count."""
        p0, c0 = self.counters()
        t0 = time.time()
        r = json.loads(
            _http(
                self.url + "/completions",
                {
                    "model": "alpha", "prompt": ids, "max_tokens": n, "temperature": 0.0, "top_p": 1.0, "seed": 0,
                    "logprobs": 5, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                    "ignore_eos": True, "cache_salt": salt,
                },
            )
        )  # fmt: skip
        secs = time.time() - t0
        lp = r["choices"][0]["logprobs"]
        tid = lambda s: int(s.split(":", 1)[1])  # noqa: E731  ("token_id:123")
        out = {
            "tokens": [tid(t) for t in lp["tokens"]],
            "lps": [float(x) for x in lp["token_logprobs"]],
            "tops": [{tid(k): float(v) for k, v in d.items()} for d in lp["top_logprobs"]],
            "usage_prompt": int(r["usage"]["prompt_tokens"]),
            "secs": round(secs, 2),
            "routes": parse_npy(r["choices"][0].get("routed_experts")),
        }
        d_prompt = d_cached = float("nan")
        for _ in range(40):  # the counters are updated by the engine's stat loop, not by the response
            p1, c1 = self.counters()
            d_prompt, d_cached = p1 - p0, c1 - c0
            if d_prompt == len(ids):
                break
            time.sleep(0.25)
        out["d_prompt"], out["cached"] = d_prompt, d_cached
        return out


# -------------------------------------------------------------------------- model of the engine's chunk schedule
def prefix_keys(ids: list[int], block: int = BLOCK) -> dict[int, str]:
    """Hash of ids[:e] for every block boundary e."""
    h, out = hashlib.sha1(), {}
    for e in range(block, len(ids) + 1, block):
        h.update(array("q", ids[e - block : e]).tobytes())
        out[e] = h.hexdigest()
    return out


class Model:
    """What vLLM 0.25.1 does to one request at a time in align mode (read from the source, checked by the counters).

    Prefill is cut by `Scheduler._mamba_block_aligned_split`: chunks of the token budget rounded down to whole
    blocks counted from where the request starts computing, plus a forced cut at the last block boundary of the
    prompt. The recurrent state exists only at the end of a chunk and can be hit later only if that end is a block
    boundary; attention blocks are cached at every full block. A hit ends at the last state boundary inside the
    attention hit and leaves at least one token to compute. If the attention hit reaches further than the state
    hit by a block or more, the first chunk is cut at the end of the attention hit ("Marconi" admission), which
    saves a state there. While generating, the state moves to a new block at every boundary it crosses, which
    leaves the boundary state in the block it just filled. `prov` is the sequence of chunk ends that produced a
    state: two computations are bit-comparable only if their sequences are the same.
    """

    def __init__(self, budget: int, block: int = BLOCK):
        assert budget >= block, "align mode asserts max_num_batched_tokens >= block size"
        self.budget, self.block = budget, block
        self.state: dict[str, dict[int, tuple[str, tuple]]] = {}  # salt -> boundary -> (prefix key, provenance)
        self.attn: dict[str, set[tuple[int, str]]] = {}  # salt -> {(boundary, prefix key)}

    def ends(self, start: int, total: int, common: int = 0) -> list[int]:
        b, out, c, first = self.block, [], start, True
        last = total - total % b
        while c < total:
            n = min(self.budget, total - c)
            after = c + n
            if after < last:
                n = n // b * b
            elif c < last < after:
                n = last - c
            if first and common >= b and n > common:
                n = common // b * b
            first = False
            c += n
            out.append(c)
        return out

    def expect(self, salt: str, ids: list[int]) -> tuple[int, tuple, int]:
        """(predicted hit, provenance of the state at the end of the prompt, uncached common prefix)."""
        keys, b = prefix_keys(ids, self.block), self.block
        attn, state = self.attn.get(salt, set()), self.state.get(salt, {})
        attn_hit = next((e for e in range((len(ids) - 1) // b * b, 0, -b) if (e, keys[e]) in attn), 0)
        hit = next((e for e in range(attn_hit, 0, -b) if e in state and state[e][0] == keys[e]), 0)
        prov = state[hit][1] if hit else ()
        return hit, prov + tuple(self.ends(hit, len(ids), attn_hit - hit)), attn_hit - hit

    def commit(self, salt: str, ids: list[int], generated: list[int]) -> None:
        """Record what this request leaves in the cache."""
        hit, prov_end, common = self.expect(salt, ids)
        ends = self.ends(hit, len(ids), common)
        slot, n_before = self.state.setdefault(salt, {}), len(prov_end) - len(ends)
        seq = ids + generated
        keys = prefix_keys(seq, self.block)
        for i, e in enumerate(ends):
            if e % self.block == 0:
                slot.setdefault(e, (keys[e], prov_end[: n_before + i + 1]))
        computed = len(seq) - 1  # the last generated token is never fed back
        for e in range((len(ids) // self.block + 1) * self.block, computed + 1, self.block):
            slot.setdefault(e, (keys[e], prov_end + (("decode", e),)))
        self.attn.setdefault(salt, set()).update((e, keys[e]) for e in range(self.block, computed + 1, self.block))


class Run:
    """Every request goes through `ask`: predicted, sent, logged and committed to the model."""

    def __init__(self, eng: Engine, model: Model):
        self.eng, self.model, self.log = eng, model, []

    def ask(self, salt: str, ids: list[int], n: int, what: str) -> tuple[dict, int, tuple, int]:
        hit, prov, common = self.model.expect(salt, ids)
        r = self.eng.gen(ids, salt, n)
        self.model.commit(salt, ids, r["tokens"])
        ok = r["cached"] == hit and r["d_prompt"] == len(ids) == r["usage_prompt"] and len(r["tokens"]) == n
        # routed experts: one row per forwarded token — every prompt token and every generated token but the last.
        # A short or long array is the "missing routes" defect of prefix caching with chunked prefill (NeMo-RL utils.py).
        rows, want_rows = (r["routes"]["rows"] if r["routes"] else None), len(ids) + n - 1
        ok = ok and (rows is None or rows == want_rows)
        self.log.append({"what": what, "len": len(ids), "n": n, "got": len(r["tokens"]), "expect": hit, "cached": r["cached"],
                         "d_prompt": r["d_prompt"], "usage_prompt": r["usage_prompt"], "common": common, "ok": bool(ok),
                         "route_rows": rows, "route_rows_expected": want_rows if rows is not None else None, "secs": r["secs"]})  # fmt: skip
        r["ok"], r["why"] = ok, "" if ok else (
            f"{what}: engine reports {r['cached']} cached tokens (predicted {hit}), prompt tokens counted "
            f"{r['d_prompt']} / usage {r['usage_prompt']} (sent {len(ids)}), {len(r['tokens'])} tokens returned (asked {n})"
            + (f", routed-expert rows {rows} (expected {want_rows})" if rows is not None else "")
        )  # fmt: skip
        return r, hit, prov, common


# ----------------------------------------------------------------------------------------------- comparisons
def compare(a: dict, b: dict) -> dict:
    """Position-wise comparison of two generations. Stops at the first differing token (inputs differ after it)."""
    n = min(len(a["tokens"]), len(b["tokens"]))
    compared = equal = 0
    dmax, exps, div = 0.0, [], None
    for i in range(n):
        compared += 1
        same_tok = a["tokens"][i] == b["tokens"][i]
        if same_tok:
            d = abs(a["lps"][i] - b["lps"][i])
            equal += a["lps"][i] == b["lps"][i] and a["tops"][i] == b["tops"][i]
        else:
            other = b["tops"][i].get(a["tokens"][i])
            d = abs(a["lps"][i] - other) if other is not None else float("nan")
        if d == d:
            dmax = max(dmax, d)
            exps.append(math.exp(d))
        if not same_tok:
            div = i
            break
    full = len(a["tokens"]) == len(b["tokens"]) == compared and compared > 0
    lp_identical = bool(full and equal == compared and div is None)
    # routed experts: rows are per forwarded token from position 0, so the common leading rows are comparable
    ra, rb = a.get("routes"), b.get("routes")
    route_rows = 0
    if ra is None and rb is None:
        routes = "absent"
    elif ra is None or rb is None:
        routes = "missing on one side"
    else:
        route_rows = min(ra["rows"], rb["rows"])
        k = route_rows * ra["row_bytes"]
        routes = "identical" if (ra["row_bytes"] == rb["row_bytes"] and route_rows > 0 and ra["data"][:k] == rb["data"][:k]) else "different"
    return {
        "compared": compared,
        "equal": equal,
        "lp_identical": lp_identical,
        "routes": routes,
        "route_rows": route_rows,
        "identical": bool(lp_identical and routes in ("absent", "identical")),
        "first_token_divergence": div,
        "max_abs_dlogprob": dmax,
        "mean_exp_abs_dlogprob": (sum(exps) / len(exps)) if exps else float("nan"),
    }


def _row(rid: str, kind: str, exact: bool, a: dict, b: dict, reqs: list[dict], extra_why: list[str] | None = None, **more) -> dict:
    why = [r["why"] for r in reqs if not r["ok"]] + (extra_why or [])
    return {"id": rid, "kind": kind, "expect_exact": bool(exact), **compare(a, b), "valid": not why, "why_invalid": why, **more}


# -------------------------------------------------------------------------------------------------- scenarios
def scenario_table(set_name: str, budget: int, block: int = BLOCK) -> list[dict]:
    """The scenarios of a set. `conv` picks one of three different documents; lengths are prompt lengths in tokens."""
    b, c = block, budget // block * block
    rep21 = {"name": "rep_21k", "kind": "rep", "conv": 0, "length": 20 * b + 100}
    dec = {"name": "decode", "kind": "decode", "conv": 2, "length": 19 * b + b - 16, "pre": 0, "forced": True}
    dec_hit = {"name": "decode_hit", "kind": "decode", "conv": 1, "length": 19 * b + b - 16, "pre": 12 * b + 300, "forced": False}
    # same salt: A[:shared+6b+300], then A[:shared] + other text, with shared = k*c + 6b: the attention hit (shared)
    # passes the last state boundary (k*c) by 6 blocks. For c = 7b this is 26b+300, 20b, 30b+300 (k = 2).
    shared = (20 * b // c) * c + 6 * b if c > b else 20 * b
    marconi = {"name": "marconi", "kind": "marconi", "conv": 0, "conv_tail": 1, "shared": shared, "first": shared + 6 * b + 300,
               "length": shared + 10 * b + 300, "probe": shared + 50, "pre": shared + 100}  # fmt: skip
    ch = lambda name, exact, conv, *lengths, rehit=0: {"name": name, "kind": "chain", "exact": exact, "conv": conv,  # noqa: E731
                                                      "lengths": list(lengths), "rehit": rehit}  # fmt: skip
    # `rehit`: a last prompt under the same salt that hits the first turn's boundary again, after the later turns
    # restored from it and computed several blocks (its chunk sequence equals the cold one)
    if set_name == "prod":  # the fleet's chunk budget: bit-identity is expected only at aligned hit positions
        assert c in (7 * b, 15 * b), "the prod set is built for a chunk budget of 7 blocks (8192) or 15 blocks (16384)"
        return [
            rep21,
            {"name": "rep_60k", "kind": "rep", "conv": 2, "length": 57 * b + 500},
            # hit boundary = forced cut at the last block of the earlier prompt; the later prompt ends in the same block
            ch("sameblock_21k", True, 0, 20 * b + 100, 20 * b + 700),
            ch("sameblock_43k", True, 1, 41 * b + 17, 41 * b + 1040),
            ch("sameblock_60k", True, 2, 57 * b + 500, 57 * b + 900),
            # hit boundary = a multiple of the chunk budget; the later prompt is many blocks longer
            ch("budget_30k", True, 1, 4 * c + 300, 4 * c + 9300, rehit=4 * c + 6 * b + 10),
            ch("budget_52k", True, 0, 7 * c + 50, 7 * c + 8750),
            # three turns: the third hits a state that was saved by a request that itself started from a hit
            ch("turns3_budget", True, 2, 2 * c + 200, 5 * c + 900, 8 * c + 400),
            # the typical production case: the hit path cuts the prefill differently from the cold path (not judged)
            ch("offgrid_28k", False, 0, 20 * b + 100, 26 * b + 300),
            ch("offgrid_60k", False, 1, 41 * b + 17, 57 * b + 300),
            dec,
            dec_hit,
            marconi,
        ]
    if set_name == "b1056":  # chunk budget = one block: every block boundary is a chunk end, any hit is comparable
        assert c == b, "the b1056 set needs --budget equal to the block size"
        return [
            rep21,
            ch("any_28k", True, 0, 20 * b + 100, 26 * b + 300, rehit=20 * b + 500),
            ch("any_60k", True, 1, 41 * b + 17, 57 * b + 300),
            ch("turns3_any", True, 2, 13 * b + 77, 24 * b + 555, 40 * b + 1000),
            ch("sameblock_60k", True, 2, 57 * b + 500, 57 * b + 900),
            dec,
            dec_hit,
        ]
    if set_name == "nc":  # negative control (fault injected on restore): the judged comparisons must all differ
        assert c in (7 * b, 15 * b)
        return [
            rep21,
            ch("sameblock_21k", True, 0, 20 * b + 100, 20 * b + 700),
            ch("budget_30k", True, 1, 4 * c + 300, 4 * c + 9300, rehit=4 * c + 6 * b + 10),
            ch("turns3_budget", True, 2, 2 * c + 200, 5 * c + 900, 8 * c + 400),
            # not judged: how large the injected fault is where bit-identity is not available (compare with the clean run)
            ch("offgrid_28k", False, 0, 20 * b + 100, 26 * b + 300),
            dec,
            dec_hit,
            marconi,
        ]
    raise SystemExit(f"unknown scenario set {set_name!r}")


def decode_bounds(length: int, block: int) -> list[int]:
    """Block boundaries e with length < e <= length + N_DECODE - 1 (the last generated token is never fed back)."""
    return list(range((length // block + 1) * block, length + N_DECODE, block))


def expected_rows(set_name: str, budget: int, block: int = BLOCK) -> dict[str, bool]:
    """Row ids a complete run must contain, with whether each is judged for bit-identity.

    The judge fails a run that lacks a row or whose row disagrees with this intent, so a wrong prediction by
    `Model` cannot quietly move a comparison out of the judged set.
    """
    out: dict[str, bool] = {}
    for s in scenario_table(set_name, budget, block):
        name = s["name"]
        if s["kind"] == "rep":
            out[f"{name}:cold_vs_cold"] = True
        elif s["kind"] == "chain":
            for t in range(1, len(s["lengths"])):
                out[f"{name}:turn{t + 1}:hit_vs_cold"] = s["exact"]
                out[f"{name}:turn{t + 1}:hit2_vs_cold"] = s["exact"]
            if s["rehit"]:
                out[f"{name}:rehit_vs_cold"] = True
        elif s["kind"] == "decode":
            bounds = decode_bounds(s["length"], block)
            for e in bounds:
                out[f"{name}:continue_at_{e}"] = True
                out[f"{name}:continue_again_at_{e}"] = True
            if s["forced"]:
                out[f"{name}:forced_after_{bounds[-1]}"] = False
        else:
            out[f"{name}:probe_hit_vs_cold"] = False  # the hit position still proves that the cut saved a state
            out[f"{name}:admit_vs_offgrid"] = True
            out[f"{name}:admit_vs_cold"] = False
            out[f"{name}:offgrid_vs_cold"] = False
    return out


def run_rep(run: Run, s: dict, seqs: list[list[int]]) -> list[dict]:
    ids = seqs[s["conv"]][: s["length"]]
    a, *_ = run.ask(run.eng.salt(s["name"] + "-a"), ids, N_GEN, s["name"] + " cold a")
    b, *_ = run.ask(run.eng.salt(s["name"] + "-b"), ids, N_GEN, s["name"] + " cold b")
    return [_row(f"{s['name']}:cold_vs_cold", "rep", True, a, b, [a, b], length=len(ids))]


def run_chain(run: Run, s: dict, seqs: list[list[int]]) -> list[dict]:
    salt, rows, seq = run.eng.salt(s["name"]), [], seqs[s["conv"]]
    first: dict = {}
    for t, length in enumerate(s["lengths"]):
        ids = seq[:length]
        a, hit_at, prov, _ = run.ask(salt, ids, N_GEN, f"{s['name']} turn{t + 1} hit")
        if t == 0:  # the first turn only fills the cache; it must itself be cold (checked through its `ok`)
            first = a
            continue
        b, _, prov_cold, _ = run.ask(run.eng.salt(s["name"] + "-cold"), ids, N_GEN, f"{s['name']} turn{t + 1} cold")
        a2, hit2_at, prov2, _ = run.ask(salt, ids, N_GEN, f"{s['name']} turn{t + 1} hit again")
        for tag, x, h, p in (("hit", a, hit_at, prov), ("hit2", a2, hit2_at, prov2)):
            why = [] if h else ["the model predicts no hit: nothing is tested"]
            rows.append(_row(f"{s['name']}:turn{t + 1}:{tag}_vs_cold", "chain", p == prov_cold, x, b, [first, x, b], why,
                             length=length, hit=h, chunks_hit=[e for e in p if isinstance(e, int)], chunks_cold=list(prov_cold)))  # fmt: skip
    if s["rehit"]:
        ids = seq[: s["rehit"]]
        want = s["lengths"][0] // run.model.block * run.model.block  # the boundary the second turn restored from
        a, hit_at, prov, _ = run.ask(salt, ids, N_GEN, f"{s['name']} re-hit of the first boundary")
        b, _, prov_cold, _ = run.ask(run.eng.salt(s["name"] + "-cold"), ids, N_GEN, f"{s['name']} re-hit cold")
        why = [] if hit_at == want else [f"the model predicts a hit at {hit_at}, the scenario needs {want}"]
        rows.append(_row(f"{s['name']}:rehit_vs_cold", "chain", prov == prov_cold, a, b, [first, a, b], why, length=len(ids),
                         hit=hit_at, chunks_hit=[e for e in prov if isinstance(e, int)], chunks_cold=list(prov_cold)))  # fmt: skip
    return rows


def run_decode(run: Run, s: dict, seqs: list[list[int]]) -> list[dict]:
    """Hits on boundary states that were produced while generating."""
    block, salt, rows, name = run.model.block, run.eng.salt(s["name"]), [], s["name"]
    ids = seqs[s["conv"]][: s["length"]]
    setup, why0 = [], []
    if s["pre"]:  # make the generating request itself start from a cache hit
        pre, *_ = run.ask(salt, ids[: s["pre"]], N_GEN, f"{name} request before the generation")
        setup.append(pre)
    first, gen_hit, _, _ = run.ask(salt, ids, N_DECODE, f"{name} generation")
    setup.append(first)
    if len(first["tokens"]) != N_DECODE:
        raise RuntimeError(f"{name}: the generation returned {len(first['tokens'])} tokens, asked {N_DECODE}")
    if bool(gen_hit) != bool(s["pre"]):
        why0.append(f"the generation was predicted to hit at {gen_hit}; the scenario needs {'a hit' if s['pre'] else 'no hit'}")
    g = first["tokens"]
    bounds = decode_bounds(len(ids), block)
    assert len(bounds) == 2 and bounds[-1] + 1 - len(ids) + N_GEN <= N_DECODE, "decode scenario lengths are off"

    def continue_rows(tag: str) -> None:  # judged: continue the original generation from one token past the boundary
        for e in bounds:
            m = e + 1 - len(ids)
            a, hit_at, _, _ = run.ask(salt, ids + g[:m], N_GEN, f"{name} {tag} {e}")
            ref = {"tokens": g[m : m + N_GEN], "lps": first["lps"][m : m + N_GEN], "tops": first["tops"][m : m + N_GEN],
                   "routes": first["routes"]}  # the original generation's rows cover the continuation's whole sequence
            why = why0 + ([] if hit_at == e else [f"the model predicts a hit at {hit_at}, the scenario needs {e}"])
            rows.append(_row(f"{name}:{tag}_{e}", "decode", True, a, ref, setup + [a], why, length=len(ids) + m, hit=hit_at,
                             generation_started_from_hit=gen_hit))  # fmt: skip

    continue_rows("continue_at")
    if s["forced"]:
        # not judged: the next-token log-prob after a hit on the last generated boundary, against a cold prefill
        # and against the original generation, at the same positions
        e = bounds[-1]
        lo, hi = e + 1 - len(ids) + 8, N_DECODE - 1
        js = sorted({lo + round(i * (hi - lo) / (N_FORCED - 1)) for i in range(N_FORCED)})
        tri, why, missing = [], list(why0), 0
        for j in js:
            p = ids + g[:j]
            a, hit_at, _, _ = run.ask(salt, p, 1, f"{name} forced {j} hit")
            b, *_ = run.ask(run.eng.salt(name + "-cold"), p, 1, f"{name} forced {j} cold")
            why += [r["why"] for r in (a, b) if not r["ok"]]
            if hit_at != e:
                why.append(f"forced position {j}: predicted hit {hit_at}, wanted {e}")
            lh, lc = a["tops"][0].get(g[j]), b["tops"][0].get(g[j])
            if lh is None or lc is None:
                missing += 1
                continue
            tri.append({"j": j, "hit": lh, "cold": lc, "decode": first["lps"][j], "top1_hit": a["tokens"][0], "top1_cold": b["tokens"][0]})
        mean_exp = lambda x, y: sum(math.exp(abs(r[x] - r[y])) for r in tri) / len(tri) if tri else float("nan")  # noqa: E731
        rows.append(
            {
                "id": f"{name}:forced_after_{e}", "kind": "decode_forced", "expect_exact": False, "length": len(ids), "hit": e,
                "positions": len(js), "reference_token_missing_in_top5": missing,
                "mean_exp_hit_vs_cold": mean_exp("hit", "cold"), "mean_exp_decode_vs_cold": mean_exp("decode", "cold"),
                "mean_exp_hit_vs_decode": mean_exp("hit", "decode"),
                "identical_hit_vs_cold": sum(r["hit"] == r["cold"] for r in tri),
                "top1_agree_hit_vs_cold": sum(r["top1_hit"] == r["top1_cold"] for r in tri),
                "identical": bool(tri) and all(r["hit"] == r["cold"] for r in tri),
                "valid": not why, "why_invalid": why, "rows": tri,
            }  # fmt: skip
        )
    continue_rows("continue_again_at")  # the same boundary blocks, after they have been hit before
    return rows


def run_marconi(run: Run, s: dict, seqs: list[list[int]]) -> list[dict]:
    """The common-prefix admission cut, and a hit at a boundary off the chunk-budget grid followed by a long chunk."""
    b, name, eng = run.model.block, s["name"], run.eng
    r0_ids = seqs[s["conv"]][: s["first"]]
    r1_ids = seqs[s["conv"]][: s["shared"]] + seqs[s["conv_tail"]][: s["length"] - s["shared"]]
    x, y = eng.salt(name + "-x"), eng.salt(name + "-y")
    r0, *_ = run.ask(x, r0_ids, N_GEN, f"{name} first prompt")
    adm, adm_hit, adm_prov, adm_common = run.ask(x, r1_ids, N_GEN, f"{name} shared prefix, new tail (admission cut)")
    cold, _, cold_prov, _ = run.ask(eng.salt(name + "-cold"), r1_ids, N_GEN, f"{name} new prompt cold")
    probe_ids = r1_ids[: s["probe"]]
    probe, probe_hit, probe_prov, _ = run.ask(x, probe_ids, N_GEN, f"{name} probe of the state saved by the cut")
    probe_cold, _, probe_cold_prov, _ = run.ask(eng.salt(name + "-pcold"), probe_ids, N_GEN, f"{name} probe cold")
    pre, *_ = run.ask(y, r1_ids[: s["pre"]], N_GEN, f"{name} second salt, prefix only")
    off, off_hit, off_prov, off_common = run.ask(y, r1_ids, N_GEN, f"{name} second salt, hit off the budget grid")
    why = []
    if not (adm_hit and adm_common >= b):
        why.append(f"the model predicts hit {adm_hit} / uncached common prefix {adm_common}: no admission cut to test")
    if probe_hit != s["shared"]:
        why.append(f"the probe is predicted to hit at {probe_hit}, the cut should have saved a state at {s['shared']}")
    if off_hit != s["shared"] or off_common:
        why.append(f"the second salt is predicted to hit at {off_hit} with common prefix {off_common}, wanted {s['shared']} / 0")
    ints = lambda p: [e for e in p if isinstance(e, int)]  # noqa: E731
    return [
        _row(f"{name}:probe_hit_vs_cold", "marconi", False, probe, probe_cold, [r0, adm, probe, probe_cold],
             why, length=len(probe_ids), hit=probe_hit, chunks_hit=ints(probe_prov), chunks_cold=ints(probe_cold_prov)),
        _row(f"{name}:admit_vs_offgrid", "marconi", adm_prov == off_prov, adm, off, [r0, adm, pre, off], why,
             length=len(r1_ids), hit=adm_hit, hit_other=off_hit, chunks_hit=ints(adm_prov), chunks_cold=ints(off_prov)),
        _row(f"{name}:admit_vs_cold", "marconi", adm_prov == cold_prov, adm, cold, [r0, adm, cold], why,
             length=len(r1_ids), hit=adm_hit, chunks_hit=ints(adm_prov), chunks_cold=ints(cold_prov)),
        _row(f"{name}:offgrid_vs_cold", "marconi", off_prov == cold_prov, off, cold, [pre, off, cold], why,
             length=len(r1_ids), hit=off_hit, chunks_hit=ints(off_prov), chunks_cold=ints(cold_prov)),
    ]  # fmt: skip


def build_documents(eng: Engine, need: int) -> tuple[list[list[int]], dict]:
    """Three different token sequences of at least `need` tokens, cut from this repository's own source files."""
    files = sorted(p for p in (REPO / "nemo_rl").rglob("*.py") if p.is_file())
    if len(files) < 60:
        raise RuntimeError(f"too few source files under {REPO / 'nemo_rl'} to build the documents")
    seqs, info = [], []
    for conv in range(3):
        order = files[conv::3] + files[(conv + 1) % 3 :: 3] + files[(conv + 2) % 3 :: 3]
        text = "<|im_start|>user\nContinue the following source files.\n"
        for p in order:
            text += f"\n\n### FILE {p.relative_to(REPO)}\n" + p.read_text(errors="ignore")
            if len(text) > need * 6:  # at least 6 characters per token would be needed to fall short
                break
        ids = eng.tokenize(text)
        if len(ids) < need:
            raise RuntimeError(f"document {conv} has {len(ids)} tokens, {need} needed")
        seqs.append(ids)
        info.append({"conv": conv, "tokens": len(ids), "sha1_text": hashlib.sha1(text.encode()).hexdigest()})
    if len({tuple(s[:BLOCK]) for s in seqs}) != 3:
        raise RuntimeError("the three documents do not differ within their first block")
    return seqs, {"documents": info}


def read_serve_log(path: str | None) -> dict:
    """Engine settings and fault-injection markers from the server's log."""
    if not path:
        return {}
    txt = Path(path).read_text(errors="ignore")
    grab = lambda pat: sorted(set(re.findall(pat, txt)))  # noqa: E731
    fired = re.findall(r"\[ALPHA_GATE_FAULT pid=\d+\] FIRED (?:routes_)?restore_\w+ [^\n]*computed=(\d+)[^\n]*tensors=[1-9]", txt)
    return {
        "serve_log": path,
        "enable_prefix_caching": grab(r"enable_prefix_caching=(\w+)"),
        "mamba_cache_mode": grab(r"(?:mamba_cache_mode\W+|Mamba cache mode is set to ')(\w+)"),
        "mamba_ssm_cache_dtype": grab(r"mamba_ssm_cache_dtype\W+(\w+)"),
        "max_num_batched_tokens": grab(r"max_num_batched_tokens['\"]?[=:]\s*(\d+)"),
        "attention_block_size": grab(r"Setting attention block size to (\d+) tokens"),
        "gdn_prefill_backend": grab(r"([\w/]+) GDN prefill kernel"),
        "enable_return_routed_experts": grab(r"enable_return_routed_experts['\"]?[=:]\s*(\w+)"),
        "fault_patched": grab(r"\[ALPHA_GATE_FAULT pid=\d+\] patched ([^\n]+)"),
        "fault_fired_at": sorted(int(x) for x in fired),
        "fault_lines": len(re.findall(r"\[ALPHA_GATE_FAULT", txt)),
    }


def cmd_run(a: argparse.Namespace) -> int:
    run = Run(Engine(a.url), Model(a.budget, a.block))
    table = scenario_table(a.set, a.budget, a.block)
    need = max(max(s.get("lengths", [0]) + [s.get("length", 0), s.get("first", 0)]) for s in table) + N_DECODE + 64
    seqs, meta = build_documents(run.eng, need)
    meta.update({"url": a.url, "set": a.set, "budget": a.budget, "block": a.block, "n_gen": N_GEN, "routes_expected": bool(a.routes),
                 "backend_expected": a.backend, "started": time.strftime("%F %T")})  # fmt: skip
    # one discarded request, so that anything the engine does only on its first request is not part of a comparison
    run.ask(run.eng.salt("warmup"), seqs[1][: 3 * a.block + 7], 4, "warm-up (discarded)")
    rows = []
    for s in table:
        t0 = time.time()
        new = {"rep": run_rep, "chain": run_chain, "decode": run_decode, "marconi": run_marconi}[s["kind"]](run, s, seqs)
        rows += new
        for r in new:
            tag = "identical" if r["identical"] else "DIFFERENT"
            extra = f"compared {r['compared']} equal {r['equal']} max|d| {r['max_abs_dlogprob']:.3g} routes {r['routes']} ({r['route_rows']} rows)" if "compared" in r else \
                f"hit~cold {r['mean_exp_hit_vs_cold']:.4f} decode~cold {r['mean_exp_decode_vs_cold']:.4f}"
            print(f"[pcs] {r['id']:<36} {'judged' if r['expect_exact'] else 'char  '} {tag:<9} hit@{r.get('hit')} "
                  f"{extra} {'' if r['valid'] else 'INVALID ' + '; '.join(r['why_invalid'][:2])} ({time.time() - t0:.0f}s)",
                  flush=True)  # fmt: skip
    meta.update({"finished": time.strftime("%F %T"), "requests": run.log, **read_serve_log(a.serve_log)})
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------ judge
def judge_run(d: dict, role: str) -> dict:
    meta, rows = d["meta"], {r["id"]: r for r in d["rows"]}
    fails, notes = [], []
    want = expected_rows(meta["set"], meta["budget"], meta["block"])
    for rid, exact in want.items():
        if rid not in rows:
            fails.append(f"row missing: {rid}")
        elif rows[rid]["expect_exact"] != exact:
            fails.append(f"{rid}: the chunk model predicts {'the same' if rows[rid]['expect_exact'] else 'a different'} "
                         f"chunk sequence, the scenario was built for the opposite")  # fmt: skip
    if role == "clean" and meta["set"] not in ("prod", "b1056") or role == "nc" and meta["set"] != "nc":
        fails.append(f"scenario set {meta['set']!r} is not a {role} set")
    # engine settings (from the server log; a run without them is not accepted)
    for name, ok in (
        ("enable_prefix_caching=True", meta.get("enable_prefix_caching") == ["True"]),
        ("mamba_cache_mode align", meta.get("mamba_cache_mode") == ["align"]),
        ("recurrent state float32", meta.get("mamba_ssm_cache_dtype") == ["float32"]),
        (f"max_num_batched_tokens={meta['budget']}", meta.get("max_num_batched_tokens") == [str(meta["budget"])]),
        (f"block size {meta['block']}", meta.get("attention_block_size") == [str(meta["block"])]),
        (f"GDN prefill backend {meta.get('backend_expected')}", not meta.get("backend_expected") or meta.get("gdn_prefill_backend") == [meta["backend_expected"]]),
        ("enable_return_routed_experts=True", not meta.get("routes_expected") or meta.get("enable_return_routed_experts") == ["True"]),
    ):
        if not ok:
            fails.append(f"engine setting not confirmed in the server log: {name}")
    # every request, including the ones that only set a scenario up
    reqs = meta.get("requests") or []
    if not reqs:
        fails.append("the run has no request log")
    for q in reqs:
        if not q["ok"]:
            fails.append(f"request invalid — {q['what']}: cached {q['cached']} (predicted {q['expect']}), prompt tokens "
                         f"{q['d_prompt']}/{q['usage_prompt']} (sent {q['len']}), tokens {q['got']}/{q['n']}"
                         + (f", routed-expert rows {q['route_rows']}/{q['route_rows_expected']}" if q.get("route_rows") is not None else ""))  # fmt: skip
    hits = sorted(int(q["expect"]) for q in reqs if q["expect"] > 0)
    present = [rows[r] for r in want if r in rows]
    for r in present:
        if not r["valid"]:
            fails.append(f"{r['id']}: invalid — {'; '.join(r['why_invalid'][:3])}")
    judged = [r for r in present if r["expect_exact"]]
    if meta.get("routes_expected"):
        for r in judged:
            if r.get("routes") not in ("identical", "different"):
                fails.append(f"{r['id']}: routed experts {r.get('routes')} — the engine was expected to return them")
    reps = [r for r in judged if r["kind"] == "rep"]
    tests = [r for r in judged if r["kind"] != "rep"]
    if not reps or not tests:
        fails.append("no judged rows")
    for r in reps:
        if not r["identical"]:
            fails.append(f"{r['id']}: two cold computations differ (max|d| {r['max_abs_dlogprob']:.3g}) — the exact criterion is void")
    if role == "clean":
        if meta.get("fault_lines", 0) != 0:
            fails.append("fault-injection markers in a clean server's log")
        for r in tests:
            if not r["identical"]:
                fails.append(f"{r['id']}: hit differs from the reference ({r['equal']}/{r['compared']} positions equal, "
                             f"max|d| {r['max_abs_dlogprob']:.3g}, first token divergence {r['first_token_divergence']}, "
                             f"routes {r.get('routes')})")  # fmt: skip
    else:
        if not meta.get("fault_patched"):
            fails.append("negative control: the server log has no 'patched' marker")
        if meta.get("fault_fired_at") != hits:
            fails.append(f"negative control: the fault fired on {len(meta.get('fault_fired_at') or [])} restores, {len(hits)} hit "
                         f"requests were sent, or their positions differ (fired at {(meta.get('fault_fired_at') or [])[:6]}…, "
                         f"hits at {hits[:6]}…)")  # fmt: skip
        for r in tests:
            if r["identical"]:
                fails.append(f"{r['id']}: the injected fault was NOT detected (hit identical to the reference)")
    for r in present:
        if not r["expect_exact"]:
            if r["kind"] == "decode_forced":
                notes.append(f"{r['id']}: mean exp|d| hit~cold {r['mean_exp_hit_vs_cold']:.4f} · decode~cold "
                             f"{r['mean_exp_decode_vs_cold']:.4f} · hit~decode {r['mean_exp_hit_vs_decode']:.4f} "
                             f"({r['positions']} positions, top-1 {r['top1_agree_hit_vs_cold']}/{len(r['rows'])})")  # fmt: skip
            else:
                notes.append(f"{r['id']}: {r['equal']}/{r['compared']} positions equal · mean exp|d| "
                             f"{r['mean_exp_abs_dlogprob']:.4f} · max|d| {r['max_abs_dlogprob']:.3g}")  # fmt: skip
    return {
        "set": meta["set"], "role": role, "url": meta["url"], "pass": not fails, "fails": fails, "not_judged": notes,
        "judged_identical": f"{sum(r['identical'] for r in tests)}/{len(tests)}",
        "judged_routes_identical": f"{sum(r.get('routes') == 'identical' for r in tests)}/{len(tests)}",
        "judged_logprobs_identical": f"{sum(bool(r.get('lp_identical')) for r in tests)}/{len(tests)}",
        "route_rows": sum(r.get("route_rows", 0) for r in tests),
        "judged_positions": sum(r.get("compared", 0) for r in tests),
        "cold_repeat_identical": f"{sum(r['identical'] for r in reps)}/{len(reps)}",
        "requests": len(reqs), "hit_requests": len(hits), "fault_fired_restores": len(meta.get("fault_fired_at") or []),
        "fault": meta.get("fault_patched"), "gdn_prefill_backend": meta.get("gdn_prefill_backend"),
    }  # fmt: skip


def cmd_judge(a: argparse.Namespace) -> int:
    out = {"clean": [], "nc": [], "fails": []}
    for role, paths in (("clean", a.clean), ("nc", a.nc or [])):
        for p in paths:
            try:
                out[role].append({"file": p, **judge_run(json.loads(Path(p).read_text()), role)})
            except (OSError, ValueError, KeyError, TypeError) as e:
                out[role].append({"file": p, "pass": False, "fails": [f"unreadable result: {type(e).__name__}: {e}"]})
    sets = sorted(r.get("set", "?") for r in out["clean"])
    if sets != ["b1056", "prod"]:
        out["fails"].append(f"clean runs must be exactly one 'prod' and one 'b1056' set, got {sets}")
    if not a.clean_only:
        kinds = {json.dumps(r.get("fault")) for r in out["nc"] if r.get("fault")}
        if len(out["nc"]) < 2 or len(kinds) < 2:
            out["fails"].append(f"{len(out['nc'])} negative controls with {len(kinds)} different faults, at least 2 different ones required")
    out["pass"] = not out["fails"] and all(r["pass"] for r in out["clean"] + out["nc"])
    for r in out["clean"] + out["nc"]:
        print(f"[{r.get('role', '?'):5}] {r.get('set', '?'):6} {'PASS' if r['pass'] else 'FAIL'}  judged identical "
              f"{r.get('judged_identical')} ({r.get('judged_positions')} positions; log-probs {r.get('judged_logprobs_identical')} · routes "
              f"{r.get('judged_routes_identical')} over {r.get('route_rows')} rows) · cold repeat {r.get('cold_repeat_identical')} "
              f"· requests {r.get('requests')} (hits {r.get('hit_requests')}) · fault fired {r.get('fault_fired_restores')} "
              f"· backend {r.get('gdn_prefill_backend')}  {r['file']}")  # fmt: skip
        for f in r["fails"]:
            print(f"        FAIL  {f}")
        for n in r.get("not_judged", []):
            print(f"        note  {n}")
    for f in out["fails"]:
        print(f"FAIL  {f}")
    print(f"PC-S verdict{' (clean runs only — not a gate verdict)' if a.clean_only else ''}: {'PASS' if out['pass'] else 'FAIL'}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--budget", type=int, required=True, help="the server's --max-num-batched-tokens")
    r.add_argument("--block", type=int, default=BLOCK)
    r.add_argument("--set", required=True, choices=["prod", "b1056", "nc"])
    r.add_argument("--out", required=True)
    r.add_argument("--serve-log")
    r.add_argument("--routes", action="store_true", help="the engine returns routed experts (router replay): compare them and require them")
    r.add_argument("--backend", help="GDN prefill backend the server log must name, e.g. Triton/FLA or FlashInfer")
    j = sub.add_parser("judge")
    j.add_argument("--clean", nargs="+", required=True)
    j.add_argument("--nc", nargs="*")
    j.add_argument("--clean-only", action="store_true", help="judge the clean runs alone (to decide whether the negative controls are worth running)")
    j.add_argument("--json")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_judge(a)


if __name__ == "__main__":
    sys.exit(main())
