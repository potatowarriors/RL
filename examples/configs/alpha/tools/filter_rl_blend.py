"""alpha RL 블렌드를 agent 허용 목록으로 걸러 새 블렌드를 만든다 (행 순서·내용 그대로, 결정론적).

용도: 결정 D1 (사용자 2026-10-07) — 첫 RLVR 런은 judge·sandbox 가 필요 없는 환경만 쓴다.
출력 옆에 `<out>.stats.json`(입력·출력 행 수, agent 별 행 수, 제외 사유별 행 수)을 남긴다. 이어서 `verify_rl_blend.py`(D1 게이트)로 구조를 검사한다.

  python examples/configs/alpha/tools/filter_rl_blend.py --preset judge_free \
      --in /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/rlvr1_alpha.jsonl \
      --out /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/rlvr1_alpha_judgefree.jsonl
"""

import argparse
import collections
import json
import os
import sys

# docs/RLVR_READINESS.md H4 — judge·sandbox 불필요, 라이선스 확인 완료 환경 (Gym 설정 파일은 student_rlvr1_alpha.yaml config_paths)
PRESETS = {
    "judge_free": {
        "single_step_tool_use_with_argument_comparison_agent",
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
        "instruction_following_simple_agent",
        "code_gen_simple_agent",
        "math_with_judge_simple_agent",  # should_use_judge: false 로 math-verify 만 쓴다
        "mcqa_simple_agent",
        "reasoning_gym_simple_agent",
        "structured_outputs_simple_agent",
        "calendar_simple_agent",
    },
}
# 제외 사유 (stats 기록용)
EXCLUDE_REASON = {
    "genrm_simple_agent": "GenRM", "genrm_simple_agent_reasoning_off": "GenRM", "abstention_simple_agent": "GenRM",
    "multichallenge_simple_agent": "LLM judge", "ns_tools_simple_agent": "sandbox",
    "math_formal_lean_refinement_agent": "sandbox", "nvarc_inductive_simple_agent": "license",
    "nvarc_transductive_simple_agent": "license",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"출력이 이미 있다: {args.out} — 지우고 다시 실행한다", file=sys.stderr)
        return 1
    allow = PRESETS[args.preset]
    kept = collections.Counter()
    dropped = collections.Counter()
    n_in = 0
    tmp = args.out + ".tmp"
    with open(args.inp) as fi, open(tmp, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            n_in += 1
            name = json.loads(line)["agent_ref"]["name"]
            if name in allow:
                fo.write(line if line.endswith("\n") else line + "\n")
                kept[name] += 1
            else:
                dropped[EXCLUDE_REASON.get(name, "jailbreak judge" if name.startswith("jailbreak") else "other")] += 1
    os.replace(tmp, args.out)  # 마지막에 생기는 파일로 완료를 판정한다 (Pai 09-23 교훈)
    stats = {"input": args.inp, "output": args.out, "preset": args.preset, "rows_in": n_in,
             "rows_out": sum(kept.values()), "kept_by_agent": dict(kept.most_common()),
             "dropped_by_reason": dict(dropped.most_common())}
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "kept_by_agent"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
