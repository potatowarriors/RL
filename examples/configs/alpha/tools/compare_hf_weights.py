"""RL 반출 HF 가중치를 시작점 HF 와 대조한다 (CPU, G6). 텐서 이름·shape 일치, dtype 변경의 무손실 여부, 동결 텐서 비트 동일, 학습 텐서 변화량.

동결 기대값 (alpha RL 레시피): 라우터 가중치(`mlp.gate.weight`, `freeze_moe_router: true`) · expert bias(`e_score_correction_bias`,
`moe_router_bias_update_rate: 0.0`). 이 둘이 바뀌었으면 동결이 깨졌거나 반출이 잘못된 것이다.
dtype 변경은 시작점 값이 새 dtype 으로 정확히 표현될 때만 허용한다 (예: Pai 반출은 bf16 로 학습한 `A_log` 를 fp32 로 저장하고
NeMo-RL 반출은 bf16 로 저장한다 — 값은 같다, G6 2026-10-07).

  python examples/configs/alpha/tools/compare_hf_weights.py --base <hfmodel_00NNNNN> --new <exported_hf> [--out report.json]
종료 코드: 이름·shape 불일치, 손실 있는 dtype 변경, 동결 텐서 변화 중 하나라도 있으면 1.
"""

import argparse
import json
import sys

import torch
from safetensors import safe_open

FROZEN_PATTERNS = ("mlp.gate.weight", "e_score_correction_bias")


def weight_map(path):
    with open(f"{path}/model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def group_of(name):
    for p in FROZEN_PATTERNS:
        if p in name:
            return p
    parts = name.split(".")
    return parts[-2] if parts[-1] in ("weight", "bias") and len(parts) > 1 else parts[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bi, ni = weight_map(args.base), weight_map(args.new)
    errors, dtype_changes, frozen_changed = [], {}, []
    if set(bi) != set(ni):
        errors.append(f"tensor names differ: only_base={sorted(set(bi) - set(ni))[:5]} only_new={sorted(set(ni) - set(bi))[:5]}")
    handles = {}

    def get(path, wm, name):
        key = (path, wm[name])
        if key not in handles:
            handles[key] = safe_open(f"{path}/{wm[name]}", framework="pt")
        return handles[key].get_tensor(name)

    groups = {}
    n = 0
    for name in sorted(set(bi) & set(ni)):
        a, b = get(args.base, bi, name), get(args.new, ni, name)
        n += 1
        if a.shape != b.shape:
            errors.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if a.dtype != b.dtype:
            lossless = torch.equal(a.to(b.dtype).to(a.dtype), a)
            k = f"{group_of(name)}: {a.dtype} -> {b.dtype}"
            st = dtype_changes.setdefault(k, {"n": 0, "lossless": 0})
            st["n"] += 1
            st["lossless"] += lossless
            if not lossless:
                errors.append(f"{name}: lossy dtype change {a.dtype} -> {b.dtype}")
        af, bf = a.float(), b.float()
        d = (af - bf).abs()
        same = torch.equal(af, bf)
        st = groups.setdefault(group_of(name), {"n": 0, "bit_identical": 0, "max_abs": 0.0, "max_rel": 0.0})
        st["n"] += 1
        st["bit_identical"] += same
        st["max_abs"] = max(st["max_abs"], float(d.max()))
        st["max_rel"] = max(st["max_rel"], float(d.norm() / af.norm().clamp_min(1e-12)))
        if any(p in name for p in FROZEN_PATTERNS) and not same:
            frozen_changed.append(name)
    rep = {"tensors": n, "errors": errors, "dtype_changes": dtype_changes, "n_frozen_changed": len(frozen_changed),
           "frozen_changed": frozen_changed[:20], "groups": groups}
    for g, st in sorted(groups.items()):
        print(f"{g:32s} n={st['n']:5d} bit_identical={st['bit_identical']:5d} max_abs={st['max_abs']:.3e} max_rel={st['max_rel']:.3e}")
    for k, st in sorted(dtype_changes.items()):
        print(f"dtype change {k}: {st['lossless']}/{st['n']} lossless")
    print(f"tensors={n} errors={len(errors)} frozen_changed={len(frozen_changed)}")
    for e in errors[:10]:
        print("ERROR", e)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if errors or frozen_changed else 0


if __name__ == "__main__":
    sys.exit(main())
