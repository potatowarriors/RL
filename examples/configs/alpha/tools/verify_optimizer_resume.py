"""옵티마이저 상태가 재개를 건너 이어지는지 본다 (CPU, G7 2026-10-07). 연속 저장 구간과 재개를 건넌 구간의 상태 변화를 비교한다.

Megatron torch_dist 체크포인트(`<ckpt_dir>/step_N/policy/weights/iter_*`)에서 대표 층(GDN 층 0·attention 층 3·임베딩·출력·최종 norm)의
옵티마이저 상태만 읽는다. 체크포인트 순서는 `--ckpts S_0 S_1 ... S_k` 이고, 이웃한 두 체크포인트 사이가 한 구간이다.
`--resume-after i` 는 S_i 에서 재개한 런이 S_{i+1} 을 저장했다는 뜻이다 (그 구간이 재개 구간, 나머지는 연속 구간).
  - Muon `momentum_buffer`·Adam `exp_avg`: m' = β·m + g 라 cos(m, m') 가 연속 구간과 비슷해야 한다. 재개에서 초기화되면 ~0 이다.
  - Adam `exp_avg_sq`: |v'|/|v| 가 연속 구간과 비슷해야 한다. 초기화되면 크게 작아진다.
  - fp32 master(`fp32_param`): master 는 bf16 가중치에서 출발해 bf16 해상도 아래의 갱신을 쌓는다. 그 누적량
    R = |p − bf16(p)|/|p| 는 스텝마다 커져야 한다 (bf16 ULP 의 절반에 닿을 때까지). 재개가 bf16 가중치에서 master 를 다시 만들면
    R 이 한 스텝 갱신 크기로 떨어진다 — lr 1e-6 의 작은 갱신이 재개마다 지워진다. 판정은 재개 구간의 R 비율 ≥ 1 이다.
    R 이 fp32 해상도에 가까운 텐서(R < 5e-7, 주로 norm)는 판정하지 않는다.

판정은 실패 양상별 신호로 한다 (갱신식에서 나온 값, G7 2026-10-07). 상태는 0 에서 시작해 lerp 로 갱신된다
(Muon `momentum_buffer.lerp_(g, 1-β)`, Adam exp_avg·exp_avg_sq 도 같은 꼴) — 초기화는 "0 에서 한 스텝" 이다.
  - Adam exp_avg_sq: Σv' = β2·Σv + (1−β2)·Σg² ≥ β2·Σv 라 정상이면 합 비율 ≥ β2 (하한이 정확하다). 초기화면 (1−β2)Σg²/Σv ≈ 0.35.
    β2 는 체크포인트의 config.yaml 에서 읽는다. 판정: 비율 < β2 − 0.01.
  - Muon momentum: 정상이면 norm 비율 ≈ 1, cos 는 새 gradient 크기에 따라 0.85~0.99. 초기화면 m' = (1−β)g 라 norm 비율 ≈ 0.4~0.6,
    다른 상태(다른 런·다른 파라미터에 매핑)면 cos ≈ 0. 판정: cos < 0.5 또는 norm 비율 < 0.7.
  - master 를 bf16 에서 재생성: 위 R 비율 < 1.
  - Adam exp_avg 는 참고만 한다 — 정상·초기화의 norm 비율 범위가 겹친다 (0.9m + 0.1g).
  - 참고(판정 아님): 연속 구간과 다른 변동(cos 차 > 0.2, 비율이 연속의 1/1.5~1.5 배 밖)은 UNUSUAL 로 표시한다.
    그 스텝의 gradient 가 크거나 방향이 바뀐 것이고, 상태 손실과는 반대 방향일 수 있다 (예: exp_avg_sq 가 늘어남).
  음성 대조: 다른 런의 체크포인트를 사슬에 넣으면 FAIL 이어야 한다 (`--ckpts runA/step_2 runA/step_3 runB/step_2`).

  python examples/configs/alpha/tools/verify_optimizer_resume.py --ckpts <ckpt>/step_2 <ckpt>/step_3 <ckpt2>/step_4 \
      --resume-after 1 [--out report.json]
종료 코드: 재개 구간에 초기화·재생성·다른 상태 신호가 하나라도 있으면 1.
"""

import argparse
import glob
import json
import sys

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

PARAM_PATTERNS = ("decoder.layers.0.", "decoder.layers.3.", "embedding.", "output_layer.", "decoder.final_layernorm.")
KINDS = ("momentum_buffer", "exp_avg", "exp_avg_sq", "fp32_param")


def weights_dir(step_dir):
    d = glob.glob(f"{step_dir}/policy/weights/iter_*")
    assert len(d) == 1, f"{step_dir}: policy/weights/iter_* 가 1개가 아님 {d}"
    return d[0]


def load_keys(path, keys):
    """선택한 키만 읽는다. Megatron torch_dist 메타데이터는 planner_data 가 없어 `_load_state_dict_from_keys` 를 못 쓴다
    → 메타데이터의 shape·dtype 으로 빈 텐서를 만들어 기본 planner 로 채운다 (프로세스 그룹 없이 단일 프로세스)."""
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    sd = {}
    for k in keys:
        m = md[k]
        assert isinstance(m, TensorStorageMetadata), f"{k}: 텐서 메타데이터가 아님 ({type(m).__name__})"
        sd[k] = torch.empty(m.size, dtype=m.properties.dtype)
    dcp_load(sd, storage_reader=FileSystemReader(path))
    return sd


def select_keys(path):
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    out = []
    for k in md:
        parts = k.split(".")
        if len(parts) > 3 and parts[0] == "optimizer" and parts[1] == "state" and parts[2] in KINDS:
            param = ".".join(parts[3:])
            if param.startswith(PARAM_PATTERNS):
                out.append(k)
    return sorted(out)


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def transition(kind, a, b):
    a, b = a.float(), b.float()
    if kind in ("momentum_buffer", "exp_avg"):
        return {"cos": cos(a, b), "norm_ratio": float(b.norm() / a.norm().clamp_min(1e-30))}
    if kind == "exp_avg_sq":
        return {"sum_ratio": float(b.double().sum() / a.double().sum().clamp_min(1e-30))}
    return {"residual_ratio": residual(b) / max(residual(a), 1e-30), "rel_delta": float((b - a).norm() / a.norm().clamp_min(1e-30))}


def residual(p):
    """bf16 해상도 아래에 쌓인 master 갱신량 |p − bf16(p)|/|p|."""
    p = p.float()
    return float((p - p.bfloat16().float()).norm() / p.norm().clamp_min(1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--resume-after", type=int, required=True, help="재개 구간의 시작 체크포인트 인덱스 (0-based)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = [weights_dir(s) for s in args.ckpts]
    import yaml

    beta2 = yaml.safe_load(open(f"{args.ckpts[0]}/config.yaml"))["policy"]["megatron_cfg"]["optimizer"]["adam_beta2"]
    print(f"adam_beta2={beta2} (exp_avg_sq 합 비율 하한)")
    keys = select_keys(paths[0])
    for p in paths[1:]:
        missing = set(keys) - set(select_keys(p))
        assert not missing, f"{p}: 키 누락 {sorted(missing)[:5]}"
    print(f"optimizer state keys: {len(keys)} (layers 0·3, embedding, output, final norm)")
    states = [load_keys(p, keys) for p in paths]
    rows, fails = [], []
    for k in keys:
        kind, param = k.split(".")[2], ".".join(k.split(".")[3:])
        tr = [transition(kind, states[i][k], states[i + 1][k]) for i in range(len(paths) - 1)]
        row = {"key": k, "kind": kind, "param": param, "transitions": tr}
        if kind == "fp32_param":
            row["residual"] = [residual(states[i][k]) for i in range(len(paths))]
        rows.append(row)
        cont = [t for i, t in enumerate(tr) if i != args.resume_after]
        res = tr[args.resume_after]
        metric = next(iter(res))
        if kind == "fp32_param":
            bad = None if row["residual"][args.resume_after] < 5e-7 else res["residual_ratio"] < 1.0
        elif kind == "momentum_buffer":
            bad = res["cos"] < 0.5 or res["norm_ratio"] < 0.7
        elif kind == "exp_avg":
            bad = None
        else:
            bad = res["sum_ratio"] < beta2 - 0.01
        row["ok"] = None if bad is None else not bad
        if bad:
            fails.append(k)
        if cont and kind != "fp32_param":
            ref = sum(t[metric] for t in cont) / len(cont)
            row["unusual"] = abs(res[metric] - ref) > 0.2 if metric == "cos" else not (ref / 1.5 <= res[metric] <= ref * 1.5)
        else:
            row["unusual"] = False
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    labels = [f"{i}->{i + 1}{'(resume)' if i == args.resume_after else ''}" for i in range(len(paths) - 1)]
    for kind, rs in by_kind.items():
        metric = next(iter(rs[0]["transitions"][0]))
        print(f"== {kind} [{metric}] {' | '.join(labels)}" + (" | R per ckpt" if kind == "fp32_param" else ""))
        for r in rs:
            vals = " | ".join(f"{t[metric]:.4g}" for t in r["transitions"])
            extra = " | " + " ".join(f"{x:.2e}" for x in r["residual"]) if kind == "fp32_param" else ""
            if kind in ("momentum_buffer", "exp_avg"):
                extra = "  | norm " + " | ".join(f"{t['norm_ratio']:.3f}" for t in r["transitions"])
            tag = "n/a" if r["ok"] is None else ("OK " if r["ok"] else "BAD")
            print(f"   {tag}{'*' if r['unusual'] else ' '} {r['param'][:58]:58s} {vals}{extra}")
    print(f"keys={len(rows)} judged={sum(r['ok'] is not None for r in rows)} bad={len(fails)} "
          f"unusual(*, 판정 아님)={sum(r['unusual'] for r in rows)}")
    print("PASS" if not fails else "FAIL")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpts": args.ckpts, "resume_after": args.resume_after, "rows": rows, "bad": fails}, f, indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
