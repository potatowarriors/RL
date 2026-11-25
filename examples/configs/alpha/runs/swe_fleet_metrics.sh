#!/bin/bash
# fleet 백엔드 8 개의 /metrics 를 60 초마다 적는다 — 학생 롤아웃 측정의 캐시 적중률 · 엔진 정지 감지 (2026-11-25, 적대적 검토 반영).
#   swe_fleet_metrics.sh <결과 디렉토리>     → <디렉토리>/fleet_metrics.tsv (시각 포트 지표{라벨} 값). <디렉토리>/metrics.stop 이 생기면 끝난다.
# 적중률은 Δvllm:prompt_tokens_cached_total / Δvllm:prompt_tokens_total 로 낸다 (요청당 한 번만 세는 누계).
#   로그의 "Prefix cache hit rate" 와 prefix_cache_hits/queries 는 대기 중 재조회가 섞인 이동창이라 쓰지 않는다 (참고로만 적는다).
D=${1:?result dir}
PAT='^vllm:(prompt_tokens_total|prompt_tokens_cached_total|prefix_cache_hits_total|prefix_cache_queries_total|generation_tokens_total|request_success_total|num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|e2e_request_latency_seconds_(sum|count)|request_prefill_time_seconds_(sum|count)|request_queue_time_seconds_(sum|count)|request_decode_time_seconds_(sum|count)|time_to_first_token_seconds_(sum|count))(\{| )'
while :; do
  ts=$(date +%s)
  for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
    curl -s -m 5 http://127.0.0.1:$p/metrics 2>/dev/null | grep -E "$PAT" \
      | awk -v ts=$ts -v p=$p '{ name=$1; sub(/\{.*finished_reason="/, "{", name); sub(/",?[^}]*\}/, "}", name); sub(/\{engine=.*/, "", name); print ts "\t" p "\t" name "\t" $NF }' >> $D/fleet_metrics.tsv
  done
  [ -e $D/metrics.stop ] && break
  sleep 60
done
