"""G1 — sequence packing 상태 누출 판별: packed 묶음 안의 시퀀스가 앞 시퀀스의 GDN 재귀·conv 상태나 attention 을 보는가.

같은 길이의 서로 다른 앞 시퀀스 A·A' 뒤에 같은 B 를 묶어 B 의 토큰 logprob 을 비교한다.
정확한 수학에서 B 의 logprob 은 A 내용과 무관하다. 위치·패딩·커널이 같으므로 차이는 누출만 남는다.
누출이 있으면 차이가 B 의 앞부분(conv 폭 4 토큰, 재귀 상태는 감쇠하며 더 길게)에 몰린다.

비교 (B 의 logprob, 첫 토큰 제외):
  noise : 같은 묶음 [A,B] 를 두 번 — 실행 간 잡음 바닥
  leak  : [A,B] vs [A',B] — 판정 대상. 기준: max|Δ| 가 noise 수준이고 앞 64 토큰에 몰리지 않는다
  path  : 단독 packed [B] vs unpacked [B] — THD 커널 경로 차이 (참고)
  offset: [A,B] 안의 B vs 단독 packed [B] — 묶음 내 오프셋 차이 (참고)

모델 forward 는 NeMo-RL 학습 경로와 같은 MegatronPolicyWorkerImpl.get_logprobs 다 (R3 끔, CP1, EP8).
EP 전 랭크가 같은 마이크로배치 수를 돌도록 모든 랭크에 같은 데이터를 준다.

실행 (NeMo-RL 루트, 8 GPU):
  $NRL_ROOT/clean_run.sh /usr/bin/env [ALPHA_GDN_BACKEND=flashqla] $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
      examples/configs/alpha/tools/verify_packing_isolation.py --config examples/configs/alpha/grpo_alpha_smoke.yaml \
      --blend /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/rlvr1_alpha.jsonl --out g1.json </dev/null
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# (len A, len B). 64 의 배수가 아닌 길이·아주 짧은 A 를 섞는다 (패딩이 묶음 안 구간 경계에 들어가게).
DEFAULT_PAIRS = [(4097, 2048), (16, 777), (8191, 3001), (1000, 1000)]


def load_text_tokens(blend: str, tok, need: int, skip: int) -> list[int]:
    """RL 블렌드의 user 프롬프트 텍스트를 이어 붙여 실제 분포의 토큰열을 만든다."""
    ids: list[int] = []
    seen = 0
    with open(blend) as f:
        for line in f:
            seen += 1
            if seen <= skip:
                continue
            row = json.loads(line)
            for m in row["responses_create_params"]["input"]:
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                if m.get("role") == "user" and isinstance(c, str) and c:
                    ids.extend(tok.encode(c, add_special_tokens=False))
            if len(ids) >= need:
                return ids[:need]
    raise RuntimeError(f"블렌드에서 토큰 {need} 개를 못 모았다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--pad-multiple", type=int, default=16, help="make_sequence_length_divisible_by (128K CP8 레시피 = 16)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ray

    local_rank = int(os.environ["LOCAL_RANK"])
    ray.get_gpu_ids = lambda: [local_rank]

    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)
    world = int(os.environ["WORLD_SIZE"])
    pol = cfg["policy"]
    max_pair = max(a + b for a, b in DEFAULT_PAIRS)
    L = 16384
    assert max_pair + 2 * args.pad_multiple <= L
    pol["max_total_sequence_length"] = L
    pol["logprob_batch_size"] = 1
    pol["make_sequence_length_divisible_by"] = args.pad_multiple
    pol["megatron_cfg"]["context_parallel_size"] = 1
    pol["sequence_packing"]["enabled"] = True
    pol["sequence_packing"]["train_mb_tokens"] = L
    pol["sequence_packing"]["logprob_mb_tokens"] = L
    pol["router_replay"] = {"enabled": False}
    pol["generation"]["colocated"]["enabled"] = False
    pol["megatron_cfg"].setdefault("train_iters", 10)

    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    sharding = NamedSharding(layout=np.arange(world).reshape(1, -1, 1, 1),
                             names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"])
    tok = get_tokenizer(pol["tokenizer"])
    worker = MegatronPolicyWorkerImpl(pol, tok, init_optimizer=False, init_reference_model=False,
                                      worker_sharding_annotations=sharding)
    rank = torch.distributed.get_rank()
    gdn_backend = os.environ.get("ALPHA_GDN_BACKEND", "fla")

    def batch(seqs: list[list[int]]) -> BatchedDataDict:
        S = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), S), tok.pad_token_id or 0, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return BatchedDataDict({"input_ids": ids, "input_lengths": torch.tensor([len(s) for s in seqs])})

    def logprobs_of(seqs: list[list[int]], packed: bool, target: list[int]) -> torch.Tensor:
        """seqs 를 한 묶음(packed) 또는 1개씩(unpacked)으로 돌려 target 시퀀스의 logprob[1:len] 을 돌려준다."""
        data = batch(seqs)
        worker.cfg["sequence_packing"]["enabled"] = packed
        if packed:
            # concatenative: 입력 순서를 지켜 [A, B] 에서 B 가 A 뒤에 온다 (FFD 는 길이 내림차순으로 재배열한다)
            sp = {"algorithm": "concatenative", "input_key": "input_ids",
                  "input_lengths_key": "input_lengths", "sequence_length_pad_multiple": args.pad_multiple,
                  "max_tokens_per_microbatch": L}
            shards, _ = data.shard_by_batch_size(1, batch_size=data.size, sequence_packing_args=sp)
            data = shards[0]
            assert len(data.micro_batch_indices[0]) == 1, f"한 묶음이어야 한다: {data.micro_batch_indices}"
        out = worker.get_logprobs(data=data)["logprobs"].float().cpu()
        t = torch.tensor(target)
        rows = [i for i in range(data.size) if int(data["input_lengths"][i]) == len(target)
                and torch.equal(data["input_ids"][i, : len(target)], t)]
        assert len(rows) == 1, f"target 행을 하나로 특정하지 못했다: {rows}"
        assert rows[0] == len(seqs) - 1, f"target 이 묶음의 마지막이 아니다 (row {rows[0]}) — 누출 판별이 무의미"
        return out[rows[0], 1 : len(target)]

    def diff(x: torch.Tensor, y: torch.Tensor) -> dict:
        d = (x - y).abs()
        head = d[:64]
        return {"max": float(d.max()), "mean": float(d.mean()), "head64_mean": float(head.mean()),
                "tail_mean": float(d[64:].mean()) if d.numel() > 64 else None, "argmax_pos": int(d.argmax()) + 1,
                "bitwise_equal": bool(torch.equal(x, y))}

    results = []
    skip = 0
    for la, lb in DEFAULT_PAIRS:
        A = load_text_tokens(args.blend, tok, la, skip); skip += 50
        A2 = load_text_tokens(args.blend, tok, la, skip); skip += 50
        Bseq = load_text_tokens(args.blend, tok, lb, skip); skip += 50
        assert A != A2
        ab1 = logprobs_of([A, Bseq], True, Bseq)
        ab2 = logprobs_of([A, Bseq], True, Bseq)
        a2b = logprobs_of([A2, Bseq], True, Bseq)
        b_packed = logprobs_of([Bseq], True, Bseq)
        b_unpacked = logprobs_of([Bseq], False, Bseq)
        rec = {"len_a": la, "len_b": lb, "noise": diff(ab1, ab2), "leak": diff(ab1, a2b),
               "path": diff(b_packed, b_unpacked), "offset": diff(ab1, b_packed),
               "b_mean_logprob": float(ab1.mean())}
        results.append(rec)
        if rank == 0:
            print(f"G1 PAIR {json.dumps(rec)}", flush=True)

    worker.cfg["sequence_packing"]["enabled"] = True
    if rank == 0:
        # 판정: leak 의 max 가 noise·path 기준선의 max 를 넘지 않고, 앞 64 토큰 평균이 뒤쪽 평균의 3배를 넘지 않는다
        verdicts = []
        for r in results:
            base = max(r["noise"]["max"], r["path"]["max"], 1e-6)
            head_ratio = (r["leak"]["head64_mean"] / max(r["leak"]["tail_mean"] or 0.0, 1e-9)
                          if r["leak"]["tail_mean"] else None)
            ok = r["leak"]["bitwise_equal"] or (r["leak"]["max"] <= base and (head_ratio is None or head_ratio <= 3.0))
            verdicts.append({"len_a": r["len_a"], "len_b": r["len_b"], "leak_max": r["leak"]["max"], "baseline_max": base,
                             "head_ratio": head_ratio, "pass": bool(ok)})
        summary = {"gdn_backend": gdn_backend, "pad_multiple": args.pad_multiple, "model": pol["model_name"],
                   "pairs": results, "verdicts": verdicts, "pass": all(v["pass"] for v in verdicts)}
        print("G1 RESULT " + json.dumps({"gdn_backend": gdn_backend, "pass": summary["pass"], "verdicts": verdicts}),
              flush=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
