#!/usr/bin/env bash
# RL 체크포인트(step_N) → Pai 호환 HF 디렉토리 (G6, 2026-10-07). 두 스택의 유일한 인터페이스가 HF 체크포인트라 반출 형식을 고정한다.
#  1) NeMo-RL 변환기(examples/converters/convert_megatron_to_hf.py)로 가중치를 반출한다. 변환은 CPU(gloo)에서 돈다.
#     Megatron 워커 venv 의 python 을 직접 쓴다 — `uv run --extra mcore` 는 driver venv 의 패키지를 바꿔 실행 중인 런과 충돌한다.
#  2) 가중치가 아닌 파일은 시작점 hfmodel(`policy.model_name`)에서 그대로 복사한다. 변환기는 transformers 5.8 로 메타데이터를 다시 쓴다:
#     tokenizer_config.json 의 `tokenizer_class: TokenizersBackend` 는 Pai(transformers 4.57)가 모르는 클래스라 토크나이저 로드가 실패하고,
#     chat template 이 chat_template.jinja 로 빠지며 special_tokens_map.json·tokenizer_metadata.json 이 없다 (config.json 은 해석 동일).
#     RL 은 모델 메타데이터를 바꾸지 않으므로 시작점 것을 쓴다.
#  3) compare_hf_weights.py 로 시작점과 대조한다 (이름·shape, 동결 텐서 비트 동일, dtype 변경은 무손실만 허용).
#
# 사용: export_rl_hf.sh <checkpoint_dir>/step_N <out_hf_dir>
#   결과: <out_hf_dir> (+ rl_export.json 출처 기록), <out_hf_dir>.compare.json. 종료 코드 0 = 반출·메타데이터·가중치 대조 모두 통과.
set -euo pipefail
STEP=$(realpath "${1:?step dir (…/ckpt/step_N)}")
OUT=${2:?output HF dir}
NRL_ROOT=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PY=$NRL_ROOT/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CFG=$STEP/config.yaml
log() { echo "[export_rl_hf $(date '+%T')] $*"; }

[ -f "$CFG" ] || { log "config.yaml 없음: $CFG"; exit 1; }
[ -e "$OUT" ] && { log "출력이 이미 있음: $OUT"; exit 1; }
mapfile -t WEIGHTS < <(ls -d "$STEP"/policy/weights/iter_* 2>/dev/null)
[ "${#WEIGHTS[@]}" -eq 1 ] || { log "policy/weights/iter_* 가 1개가 아님: ${WEIGHTS[*]:-없음}"; exit 1; }
read -r SRC TOK < <("$PY" -I -c 'import sys, yaml; p = yaml.safe_load(open(sys.argv[1]))["policy"]; print(p["model_name"], p["tokenizer"]["name"])' "$CFG")
[ -f "$SRC/config.json" ] || { log "시작점 hfmodel 없음: $SRC"; exit 1; }
[ "$TOK" = "$SRC" ] || log "WARN tokenizer.name($TOK) ≠ model_name — 메타데이터는 model_name 에서 복사한다"

log "1) 가중치 반출 ${WEIGHTS[0]} → $OUT"
cd "$REPO"
"$NRL_ROOT/clean_run.sh" "$PY" examples/converters/convert_megatron_to_hf.py \
    --config "$CFG" --megatron-ckpt-path "${WEIGHTS[0]}" --hf-ckpt-path "$OUT" </dev/null

log "2) 메타데이터를 시작점에서 복사 ($SRC)"
is_weight() { case "$1" in *.safetensors | model.safetensors.index.json) return 0 ;; *) return 1 ;; esac; }
for f in "$OUT"/*; do  # 시작점에 없는 비가중치 파일(예: chat_template.jinja)은 지운다
    b=$(basename "$f")
    if [ -f "$f" ] && ! is_weight "$b" && [ ! -e "$SRC/$b" ]; then log "  삭제 $b (시작점에 없음)"; rm -f "$f"; fi
done
for f in "$SRC"/*; do  # 하위 디렉토리(eval_results_* 등 벤치 산출물)는 모델이 아니다
    b=$(basename "$f")
    if [ -f "$f" ] && ! is_weight "$b"; then cp -p "$f" "$OUT/$b"; fi
done
bad=0
for f in "$SRC"/*; do
    b=$(basename "$f")
    if [ -f "$f" ] && ! is_weight "$b"; then cmp -s "$f" "$OUT/$b" || { log "  불일치 $b"; bad=1; }; fi
done
[ "$bad" = 0 ] || exit 1
"$PY" -I - "$OUT/rl_export.json" "$SRC" "$STEP" "${WEIGHTS[0]}" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import json, sys, time
out, src, step, weights, commit = sys.argv[1:]
info = json.load(open(f"{step}/training_info.json"))
json.dump({"source_hf": src, "step_dir": step, "weights": weights, "training_info": info, "repo_commit": commit,
           "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out, "w"), indent=1)
PY

log "3) 가중치 대조 → ${OUT%/}.compare.json"
"$NRL_ROOT/clean_run.sh" "$PY" -I "$REPO/examples/configs/alpha/tools/compare_hf_weights.py" \
    --base "$SRC" --new "$OUT" --out "${OUT%/}.compare.json" </dev/null
log "완료: $OUT"
