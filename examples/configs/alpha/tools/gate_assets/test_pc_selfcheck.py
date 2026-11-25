#!/usr/bin/env python3
"""CPU test of the PC-S gate tool against `fake_pc_server.py`: the gate must pass on a correct engine and fail on
each broken one. Run before any GPU run of the gate:    python3 -I tools/gate_assets/test_pc_selfcheck.py

Every case states the verdict it expects and, for failures, a phrase that must appear among the reasons, so a
failure for the wrong reason does not count.
"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "prefix_cache_selfcheck.py"
FAKE = HERE / "fake_pc_server.py"
PY = [sys.executable, "-I"]
TMP = Path(tempfile.mkdtemp(prefix="pcs_test_"))
PROCS = []


def serve(name: str, port: int, budget: int, fault: str = "") -> dict:
    log = TMP / f"{name}.serve.log"
    PROCS.append(subprocess.Popen(PY + [str(FAKE), str(port), str(budget), fault], stdout=log.open("w"), stderr=subprocess.STDOUT))
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).read()
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError(f"fake server {name} did not start")
    return {"name": name, "url": f"http://127.0.0.1:{port}/v1", "log": str(log), "budget": budget}


def run(srv: dict, set_name: str, budget: int | None = None) -> str:
    out = TMP / f"{srv['name']}.{set_name}.json"
    r = subprocess.run(
        PY + [str(TOOL), "run", "--url", srv["url"], "--budget", str(budget or srv["budget"]), "--set", set_name,
              "--out", str(out), "--serve-log", srv["log"]],
        capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode != 0:
        return f"RUN-ERROR: {(r.stderr or r.stdout).strip().splitlines()[-1]}"
    return str(out)


def judge(clean: list[str], nc: list[str]) -> tuple[bool, str]:
    r = subprocess.run(PY + [str(TOOL), "judge", "--clean", *clean, "--nc", *nc], capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def main() -> int:
    port = iter(range(18300, 18360))
    good = serve("good", next(port), 8192)
    good1056 = serve("good1056", next(port), 1056)
    nc_scale = serve("nc_scale", next(port), 8192, "restore_scale")
    nc_zero = serve("nc_zero", next(port), 8192, "restore_zero")
    prod, b1056 = run(good, "prod"), run(good1056, "b1056")
    ncs = [run(nc_scale, "nc"), run(nc_zero, "nc")]
    cases = [("correct engine, two detected negative controls", [prod, b1056], ncs, True, "")]
    # fault, phrase the failure must mention on the prod set, same for the b1056 set (None = that set cannot see the fault)
    for fault, phrase, phrase1056 in (
        ("restore_scale", "hit differs from the reference", "hit differs from the reference"),
        ("restore_zero", "hit differs from the reference", "hit differs from the reference"),
        ("inplace_src", "hit2_vs_cold: hit differs", "hit2_vs_cold: hit differs"),
        ("decode_off1", "decode:continue_at_", "decode:continue_at_"),
        ("hit_decode", "decode_hit:continue_at_", "decode_hit:continue_at_"),
        ("admit_state", "marconi:probe_hit_vs_cold: hit differs", None),
        ("offgrid_multi", "marconi:admit_vs_offgrid: hit differs", None),
        ("nondet", "two cold computations differ", "two cold computations differ"),
        ("no_salt", "invalid", "invalid"),
        ("short_hit", "invalid", "invalid"),
    ):
        bad = serve(f"bad_{fault}", next(port), 8192, fault)
        cases.append((f"prod engine broken: {fault}", [run(bad, "prod"), b1056], ncs, False, phrase))
        if phrase1056:
            bad1056 = serve(f"bad1056_{fault}", next(port), 1056, fault)
            cases.append((f"b1056 engine broken: {fault}", [prod, run(bad1056, "b1056")], ncs, False, phrase1056))
    cases.append(("negative control on a server without the fault", [prod, b1056], [ncs[0], run(good, "nc")], False, "NOT detected"))
    cases.append(("only one negative control", [prod, b1056], ncs[:1], False, "negative controls"))
    nc_scale2 = serve("nc_scale2", next(port), 8192, "restore_scale")
    cases.append(("two negative controls with the same fault", [prod, b1056], [ncs[0], run(nc_scale2, "nc")], False, "different faults"))
    lost = json.loads(Path(ncs[0]).read_text())
    lost["meta"]["fault_fired_at"] = lost["meta"]["fault_fired_at"][1:]
    (TMP / "lost.json").write_text(json.dumps(lost))
    cases.append(("negative control whose fault missed one restore", [prod, b1056], [str(TMP / "lost.json"), ncs[1]], False, "the fault fired on"))
    setup = json.loads(Path(prod).read_text())
    for q in setup["meta"]["requests"]:
        if q["what"] == "decode_hit generation":
            q["cached"], q["ok"] = 0, False
    (TMP / "setup.json").write_text(json.dumps(setup))
    cases.append(("a set-up request that did not hit as predicted", [str(TMP / "setup.json"), b1056], ncs, False, "request invalid"))
    cases.append(("b1056 run missing", [prod], ncs, False, "exactly one 'prod' and one 'b1056'"))
    other = serve("budget4224", next(port), 4224)
    cases.append(("server chunk budget differs from the one the tool was told", [run(other, "prod", 8192), b1056], ncs, False, "max_num_batched_tokens"))
    cut = json.loads(Path(prod).read_text())
    cut["rows"] = [r for r in cut["rows"] if not r["id"].startswith("turns3")]
    (TMP / "cut.json").write_text(json.dumps(cut))
    cases.append(("a judged row is missing", [str(TMP / "cut.json"), b1056], ncs, False, "row missing"))
    relabel = json.loads(Path(prod).read_text())
    for r in relabel["rows"]:
        if r["id"].startswith("budget_30k"):
            r["expect_exact"] = False
    (TMP / "relabel.json").write_text(json.dumps(relabel))
    cases.append(("a judged row relabelled as not judged", [str(TMP / "relabel.json"), b1056], ncs, False, "built for the opposite"))

    bad = 0
    for name, clean, nc, want, phrase in cases:
        errs = [p for p in clean + nc if p.startswith("RUN-ERROR")]
        ok, text = (False, " ".join(errs)) if errs else judge(clean, nc)
        good_case = ok == want and (want or phrase in text)
        bad += not good_case
        print(f"{'ok  ' if good_case else 'BAD '} {name}: verdict {'PASS' if ok else 'FAIL'} (expected {'PASS' if want else 'FAIL'}"
              f"{'' if want else ', reason must mention ' + repr(phrase)})")  # fmt: skip
        if not good_case:
            print("      " + "\n      ".join(text.strip().splitlines()[-12:]))
    print(f"{len(cases) - bad}/{len(cases)} cases as expected · results in {TMP}")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        for p in PROCS:
            p.kill()
    sys.exit(rc)
