"""MOPD 학습 덤프(train_data_step*.jsonl)의 teacher 슬롯별 추이 (CPU, 2026-11-10 `mopd1/s2`·`s3` 진단에 쓴 것을 정리).

슬롯 = 블렌드 agent → teacher 경로 (--routing, `build_mopd1_blend.py` 의 `<out>.routing.yaml`). 덤프 한 파일 = 한 step.
step 마다 슬롯별로 낸다:
  rows · 평균 생성 길이 (token_loss_mask 합) · 시퀀스 평균 차이 (행마다 mean(A) 의 평균) · 토큰 평균 차이 (ΣA / Σmask) · 평균 |A| ·
  잘림 비율 (마지막 생성 구간 길이 ≥ --cap) · 환경 응답 없이 assistant 턴이 이어진 행 수 (구간 사이에 'user' 턴이 없음, --tokenizer 를 줄 때만)
A = advantages = clip(log π_teacher − log π_student, ±clip) 라 "차이" 는 teacher − 학생 logprob 이다 (0 에 가까울수록 teacher 에 가깝다).
전체 줄에는 손실 집계 확인용 기대값도 낸다: 토큰 평균 −ΣA/Σmask · 시퀀스 평균 −mean_i(mean_t A) (force_on_policy_ratio · KL 0 가정) —
TensorBoard `train/loss` 와 비교해 `loss_fn.token_level_loss` 가 실제로 적용됐는지 본다 (결정 57 확인 때 0.2762 대 0.2745 / 0.2647).

  python analyze_mopd_slots.py --routing <routing.yaml> <dump>/train_data_step1.jsonl <dump>/train_data_step60.jsonl ... \
      [--cap 32768] [--tokenizer <HF 경로>] [--json-out <file>]
덤프는 필드마다 배치 차원 1 로 감싼다 (`analyze_mopd_smoke.py` 머리말). 파일 하나가 1~2 GB 라 줄 단위로 읽는다.
"""

import argparse
import collections
import json
import re
import sys


def un(v):
    return v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) else v


def slot_name(path):
    m = re.search(r"/alpha/([^/]+)/hf/", path)
    return m.group(1) if m else path


def segments(mask):
    spans, i = [], 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        spans.append((mask[i], i, j))
        i = j
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--routing", required=True)
    ap.add_argument("--cap", type=int, default=32768, help="턴당 생성 상한 (max_new_tokens)")
    ap.add_argument("--tokenizer", help="주면 환경 응답 없는 assistant 연속 턴을 센다")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    import yaml

    routing = yaml.safe_load(open(a.routing))
    user_ids = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        user_ids = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True).encode("user", add_special_tokens=False)
    out = {}
    for path in a.dumps:
        st = collections.defaultdict(collections.Counter)
        tok_num = tok_den = 0.0
        seq_means = []
        for line in open(path):
            d = json.loads(line)
            slm = d.get("sample_loss_mask", 1.0)
            slm = slm[0] if isinstance(slm, list) else slm
            if not slm:
                continue
            ref = d.get("agent_ref")
            ref = ref[0] if isinstance(ref, list) else ref
            slot = slot_name(routing.get((ref or {}).get("name"), "?"))
            m, adv = un(d["token_loss_mask"]), un(d["advantages"])
            n = sum(m)
            if not n:
                continue
            s = sum(x * y for x, y in zip(adv, m))
            spans = segments(m)
            gens = [(b - e_) for v, e_, b in spans if v]
            c = st[slot]
            c["rows"] += 1
            c["tokens"] += n
            c["sum"] += s
            c["abs"] += sum(abs(x) * y for x, y in zip(adv, m))
            c["seq_mean_sum"] += s / n
            c["truncated"] += gens[-1] >= a.cap
            tok_num += s
            tok_den += n
            seq_means.append(s / n)
            if user_ids is not None:
                ids = un(d["token_ids"])
                for k in range(1, len(spans) - 1):
                    v, lo, hi = spans[k]
                    if v == 0 and spans[k - 1][0] and spans[k + 1][0]:
                        gap = ids[lo:hi]
                        if not any(gap[x:x + len(user_ids)] == user_ids for x in range(len(gap))):
                            c["asst_asst_rows"] += 1
                            break
        step = re.search(r"step(\d+)", path)
        step = int(step.group(1)) if step else path
        rec = {"expected_loss_token_level": -tok_num / tok_den if tok_den else None,
               "expected_loss_sequence_level": -sum(seq_means) / len(seq_means) if seq_means else None, "slots": {}}
        for slot, c in sorted(st.items()):
            r = c["rows"]
            rec["slots"][slot] = {"rows": r, "mean_gen_tokens": c["tokens"] / r, "gap_seq": c["seq_mean_sum"] / r,
                                  "gap_tok": c["sum"] / c["tokens"], "abs_gap": c["abs"] / c["tokens"], "truncated": c["truncated"] / r}
            if user_ids is not None:
                rec["slots"][slot]["asst_asst_rows"] = c["asst_asst_rows"]
            x = rec["slots"][slot]
            print(f"step {step!s:>4} {slot:28s} rows={r:4d} len={x['mean_gen_tokens']:7.0f} gap_seq={x['gap_seq']:+.3f} "
                  f"gap_tok={x['gap_tok']:+.3f} |gap|={x['abs_gap']:.3f} trunc={x['truncated']:.1%}"
                  + (f" asst→asst={x['asst_asst_rows']}" if user_ids is not None else ""))
        print(f"step {step!s:>4} expected loss — token-level {rec['expected_loss_token_level']:.4f} · sequence-level {rec['expected_loss_sequence_level']:.4f}")
        out[str(step)] = rec
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
