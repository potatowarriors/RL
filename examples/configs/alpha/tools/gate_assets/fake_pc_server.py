#!/usr/bin/env python3
"""CPU stand-in for the serving engine, used only to test `prefix_cache_selfcheck.py` without a GPU. NOT production code.

It simulates what the gate relies on: attention blocks and recurrent-state boundaries cached per salt and token
prefix, block-aligned chunked prefill with the common-prefix admission cut, boundary states left behind while
generating, the two token counters, and "numerics" that depend on how the prefill was cut (two different chunk
sequences give different values, the same sequence gives the same bits). It encodes the same reading of vLLM as
the gate tool, so it tests the tool's logic, not the engine.

    fake_pc_server.py <port> <budget> [fault] [routes]   (stdout carries the lines the gate reads from a server log;
                                                           `routes` makes it return routed experts like a router-replay engine)

Faults, each a way the real engine could be broken or the gate could be fooled:
    restore_scale   the restored state is multiplied by 1.0001
    restore_zero    the restored state is replaced by zero
    inplace_src     a hit is right, but it damages the cached state it restored from (the next hit on it is wrong)
    inplace_multi   the same, but only when the hit request goes on to compute more than one block
    decode_off1     boundary states made while generating are saved one token late
    hit_decode      boundary states made while generating are wrong when the generating request started from a hit
    admit_cut       the chunk that ends at the common-prefix admission cut leaves a wrong state
    offgrid_multi   a restore at a boundary off the chunk-budget grid is wrong when a multi-block chunk follows
    nondet          every request gets its own noise (two cold computations differ)
    no_salt         `cache_salt` is ignored (a "cold" reference would silently be a hit)
    short_hit       the newest state boundary is never hit
    routes_drop_tail       a hit returns three routed-expert rows too few (the "missing routes" defect)
    routes_restore_shift   every hit damages the stored routed experts of the cached block that ends at its hit
                           position (this hit and later ones return them wrong; log-probs are right)
`restore_*` and `routes_restore_*` print the marker line the real fault injector prints, so the judge can count them.
"""

import base64
import hashlib
import json
import math
import random
import sys
import zlib
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, BUDGET = int(sys.argv[1]), int(sys.argv[2])
FAULT = sys.argv[3] if len(sys.argv) > 3 else ""
ROUTES = len(sys.argv) > 4 and sys.argv[4] == "routes"
BLOCK = 1056
STATE: dict[tuple, float] = {}  # (salt, boundary, prefix key) -> recurrent state
ATTN: set[tuple] = set()  # (salt, boundary, prefix key): attention blocks up to the boundary are cached
SHIFT: dict[tuple, int] = {}  # (salt, boundary, prefix key) -> how often the routes of the block ending there were damaged
COUNT = {"prompt": 0, "cached": 0}


def prefix_keys(ids):
    h, out = hashlib.sha1(), {}
    for e in range(BLOCK, len(ids) + 1, BLOCK):
        h.update(array("q", ids[e - BLOCK : e]).tobytes())
        out[e] = h.hexdigest()
    return out


def chunk(s, ids, a, b):  # prefill kernel over [a, b): the result depends on where the chunk starts and ends
    h = int.from_bytes(hashlib.sha1(array("q", ids[a:b]).tobytes()).digest()[:6], "big") / 2**48
    return math.fmod(s * 1.0000001 + h * math.sqrt(b - a) + 0.37, 97.0)


def step(s, tok):  # decode kernel; a prefill of exactly one token takes this path too
    return math.fmod(s * 1.0000001 + (tok % 9973) * 1e-4 + 0.11, 97.0)


def dist(s, last):
    return [((last * 31 + 7 * k + 1) % 50000, -(0.05 + 0.3 * k + 0.01 * abs(math.sin(s * (k + 1))))) for k in range(5)]


def npy_b64(rows: list[bytes]) -> str:
    """A [tokens, 2, 1] uint8 array as a base64 .npy payload (what vLLM puts in `routed_experts`)."""
    header = f"{{'descr': '|u1', 'fortran_order': False, 'shape': ({len(rows)}, 2, 1), }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    return base64.b64encode(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode() + b"".join(rows)).decode()


def run(ids, salt, n):
    if FAULT == "no_salt":
        salt = ""
    total, keys = len(ids), prefix_keys(ids)
    attn_hit = next((e for e in range((total - 1) // BLOCK * BLOCK, 0, -BLOCK) if (salt, e, keys[e]) in ATTN), 0)
    bounds = [e for e in range(attn_hit, 0, -BLOCK) if (salt, e, keys[e]) in STATE]
    if FAULT == "short_hit":
        bounds = bounds[1:]
    hit, s = (bounds[0], STATE[(salt, bounds[0], keys[bounds[0]])]) if bounds else (0, 0.0)
    common = attn_hit - hit
    if hit:
        if FAULT in ("restore_scale", "restore_zero"):
            s = s * 1.0001 if FAULT == "restore_scale" else 0.0
            print(f"[ALPHA_GATE_FAULT pid=1] FIRED {FAULT} computed={hit} tensors=1", flush=True)
        if FAULT == "routes_restore_shift":
            SHIFT[(salt, hit, keys[hit])] = SHIFT.get((salt, hit, keys[hit]), 0) + 1
            print(f"[ALPHA_GATE_FAULT pid=1] FIRED {FAULT} computed={hit} tensors=1", flush=True)
        if FAULT == "inplace_src" or FAULT == "inplace_multi" and total - hit > BLOCK:
            STATE[(salt, hit, keys[hit])] = s * 1.0001
    if FAULT == "nondet":
        s += random.random() * 1e-9
    c, last, first = hit, total - total % BLOCK, True
    while c < total:
        m = min(BUDGET, total - c)
        after = c + m
        if after < last:
            m = m // BLOCK * BLOCK
        elif c < last < after:
            m = last - c
        admitted = first and common >= BLOCK and m > common
        if admitted:
            m = common // BLOCK * BLOCK
        if FAULT == "offgrid_multi" and first and hit and hit % (BUDGET // BLOCK * BLOCK) and m > BLOCK:
            s += 1e-7
        s = step(s, ids[c]) if m == 1 else chunk(s, ids, c, c + m)
        c += m
        if FAULT == "admit_cut" and admitted:
            s *= 1.0001
        if c % BLOCK == 0:
            STATE.setdefault((salt, c, keys[c]), s)
        first = False
    seq, toks, lps, tops = list(ids), [], [], []
    for i in range(n):
        d = dist(s, seq[-1])
        toks.append(d[0][0])
        lps.append(d[0][1])
        tops.append({f"token_id:{t}": lp for t, lp in d})
        if i == n - 1:
            break
        before = s
        seq.append(d[0][0])
        s = step(s, d[0][0])
        if len(seq) % BLOCK == 0:
            saved = before if FAULT == "decode_off1" else s * 1.0001 if FAULT == "hit_decode" and hit else s
            STATE.setdefault((salt, len(seq), prefix_keys(seq)[len(seq)]), saved)
    for e, k in prefix_keys(seq).items():
        ATTN.add((salt, e, k))
    COUNT["prompt"] += total
    COUNT["cached"] += hit
    choice = {"logprobs": {"tokens": [f"token_id:{t}" for t in toks], "token_logprobs": lps, "top_logprobs": tops}}
    if ROUTES:  # one row per forwarded token (every token but the last sampled one)
        rows = [bytes([(tok * 7 + pos) % 192, (tok * 13 + 3 * pos) % 192]) for pos, tok in enumerate(seq[: len(seq)])]
        for e, k in prefix_keys(seq).items():  # damaged blocks stay damaged for every request that reads them
            d = SHIFT.get((salt, e, k), 0)
            if d:
                rows[e - BLOCK : e] = [bytes([(r[0] + d) % 192, r[1]]) for r in rows[e - BLOCK : e]]
        if FAULT == "routes_drop_tail" and hit:
            rows = rows[:-3]
        choice["routed_experts"] = npy_b64(rows)
    return {"choices": [choice], "usage": {"prompt_tokens": total}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/metrics":
            self._send(
                f'vllm:prompt_tokens_total{{model_name="alpha"}} {COUNT["prompt"]}.0\n'
                f'vllm:prompt_tokens_cached_total{{model_name="alpha"}} {COUNT["cached"]}.0\n'.encode(),
                "text/plain",
            )
        else:
            self._send(json.dumps({"data": [{"id": "alpha"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            t = body["prompt"]
            self._send(json.dumps({"tokens": [zlib.crc32(t[i : i + 3].encode()) % 50000 for i in range(0, len(t), 3)]}).encode())
        else:
            self._send(json.dumps(run(body["prompt"], body.get("cache_salt", ""), body["max_tokens"])).encode())


if __name__ == "__main__":
    print(f"non-default args: enable_prefix_caching=True, mamba_cache_mode='align', mamba_ssm_cache_dtype='float32', "
          f"max_num_batched_tokens={BUDGET}\nSetting attention block size to {BLOCK} tokens to ensure ...\n"
          f"Using FakeCPU GDN prefill kernel (requested=auto)" + ("\nnon-default args: {'enable_return_routed_experts': True}" if ROUTES else ""),
          flush=True)  # fmt: skip
    if FAULT.startswith(("restore_", "routes_restore_")):
        print(f"[ALPHA_GATE_FAULT pid=1] patched fake restore ({FAULT})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
