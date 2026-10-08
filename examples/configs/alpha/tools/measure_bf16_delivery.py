"""학습 체크포인트의 갱신이 bf16 가중치에 얼마나 반영됐는지 잰다 (CPU, Megatron 워커 venv).

lr 1e-6 teacher_tool 런에서 fp32 master 는 움직였지만 bf16 가중치(학습 forward·vLLM refit 이 쓰는 값)에는
50스텝 누적 갱신의 약 12% 만 반영됐다 (docs/KNOWN_ISSUES.md 2026-10-08 "bf16 가중치에 갱신이 반영되지 않는다").
시작 가중치가 bf16 HF 체크포인트라 master 가 bf16 격자점에서 출발하고, 누적 갱신이 반올림 경계(ulp/2)를 넘어야 bf16 값이 바뀐다.

출력 (시작점 HF 대비, 텐서별):
  flip%       bf16 값이 바뀐 원소 비율
  dW rms      fp32 master 누적 갱신의 RMS
  delivered   bf16 변화량을 fp32 갱신 방향에 투영한 계수 (1 이면 갱신이 그대로 반영)
  >ulp/2 %    누적 갱신이 반올림 경계를 넘은 원소 비율 (flip% 와 같으면 반올림 dead zone 이 원인)
  router 는 동결 대조군이다 — 시작점과 다른 원소가 0 이어야 한다.

사용:
  $NRL_ROOT/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python -I \
    examples/configs/alpha/tools/measure_bf16_delivery.py <ckpt>/step_N <hfmodel_dir>
"""

import json
import sys

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

# mcore 이름 → HF 이름 (1:1 매핑만). 전문가 텐서는 192개를 쌓는다.
TENSORS = {
    "decoder.layers.0.self_attention.out_proj.weight": "model.layers.0.linear_attn.out_proj.weight",
    "decoder.layers.12.self_attention.out_proj.weight": "model.layers.12.linear_attn.out_proj.weight",
    "decoder.layers.3.self_attention.linear_proj.weight": "model.layers.3.self_attn.o_proj.weight",
    "decoder.layers.12.mlp.experts.experts.linear_fc2.weight": "experts:12",
    "output_layer.weight": "lm_head.weight",
    "decoder.layers.0.mlp.router.weight": "model.layers.0.mlp.gate.weight",
}


def load_ckpt(step_dir):
    path = f"{step_dir}/policy/weights/iter_0000000"
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    sd = {}
    for name in TENSORS:
        for key in (name, f"optimizer.state.fp32_param.{name}"):
            if key in md:
                sd[key] = torch.empty(md[key].size, dtype=md[key].properties.dtype)
    dcp.load(sd, storage_reader=dcp.FileSystemReader(path), no_dist=True)
    return sd


def load_hf(hf_dir, spec):
    weight_map = json.load(open(f"{hf_dir}/model.safetensors.index.json"))["weight_map"]

    def get(name):
        with safe_open(f"{hf_dir}/{weight_map[name]}", "pt") as f:
            return f.get_tensor(name)

    if spec.startswith("experts:"):
        layer = spec.split(":")[1]
        return torch.stack([get(f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight") for e in range(192)])
    return get(spec)


def main():
    step_dir, hf_dir = sys.argv[1], sys.argv[2]
    ck = load_ckpt(step_dir)
    print(f"{'tensor':56s} {'flip%':>6s} {'dW rms':>9s} {'delivered':>9s} {'>ulp/2 %':>8s}")
    for name, spec in TENSORS.items():
        b0 = load_hf(hf_dir, spec)
        b = ck[name]
        if b0.shape != b.shape:
            b0 = b0[: b.shape[0]]
        master = ck.get(f"optimizer.state.fp32_param.{name}")
        if master is None:
            print(f"{name:56s} frozen — 시작점과 다른 원소 {(b != b0).sum().item()} (0 이어야 한다)")
            continue
        dw = master - b0.float()
        db = b.float() - b0.float()
        delivered = ((db * dw).sum() / dw.pow(2).sum()).item()
        # 시작값의 bf16 ulp: 절댓값의 비트 패턴에 1 을 더한 값과의 차이
        ulp = (b0.abs().view(torch.int16) + 1).view(torch.bfloat16).float() - b0.float().abs()
        print(
            f"{name:56s} {100 * (b != b0).float().mean().item():6.2f} {dw.pow(2).mean().sqrt().item():9.2e} "
            f"{delivered:9.3f} {100 * (dw.abs() > ulp / 2).float().mean().item():8.2f}"
        )


if __name__ == "__main__":
    main()
