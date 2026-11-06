"""레시피의 Gym 설정을 Gym 과 같은 순서로 합쳐 judge 연결을 단언한다 (CPU) — JG1 적대적 검토 #9 (2026-11-06).

Gym global_config 는 config_paths 파일들을 합친 뒤 레시피의 env.nemo_gym 인라인 dict 를 마지막에 합친다 (global_config.py 1068~1070행).
이 순서를 OmegaConf 로 재현해, judge_model_server 를 가진 모든 resources 서버가 어디를 가리키는지 출력하고
  - 정책 서버(policy_model · policy_model_reasoning_off) 를 judge 로 가리키면 ERROR
  - 가리키는 모델 서버가 합친 설정에 없거나 서버 종류가 2개 이상이면 ERROR
  - --expect 서버=모델서버 를 주면 그 값과 다르면 ERROR

  $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/configs/alpha/tools/check_gym_judge_routing.py \
      examples/configs/alpha/teacher_general_alpha.yaml --expect multichallenge=judge_model ... </dev/null
"""
import argparse
import os
import sys

from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ROOTS = [os.path.join(REPO, "examples/configs/alpha/gym_plugins"), os.path.join(REPO, "3rdparty/Gym-workspace/Gym")]
POLICY = {"policy_model", "policy_model_reasoning_off"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--expect", action="append", default=[], help="resources 서버 이름=judge 모델 서버 이름")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    register_omegaconf_resolvers()
    cfg = load_config(a.recipe)
    ng = OmegaConf.to_container(cfg.env.nemo_gym, resolve=False)
    files = []
    for rel in ng.get("config_paths") or []:
        hit = next((os.path.join(r, rel) for r in ROOTS if os.path.exists(os.path.join(r, rel))), None)
        if hit is None:
            print(f"ERROR config_paths 항목을 찾지 못함: {rel}")
            return 1
        files.append(OmegaConf.load(hit))
    inline = OmegaConf.create({k: v for k, v in ng.items() if k != "config_paths"})
    merged = OmegaConf.to_container(OmegaConf.merge(*files, inline), resolve=False)
    errors = 0
    models = {}
    for inst, body in merged.items():
        if isinstance(body, dict) and isinstance(body.get("responses_api_models"), dict):
            models[inst] = list(body["responses_api_models"])
    routes = {}
    for inst, body in merged.items():
        if not (isinstance(body, dict) and isinstance(body.get("resources_servers"), dict)):
            continue
        for stype, scfg in body["resources_servers"].items():
            if not isinstance(scfg, dict):
                continue
            for key in ("judge_model_server", "genrm_model_server"):
                ref = scfg.get(key)
                if isinstance(ref, dict) and ref.get("name"):
                    routes[stype] = ref["name"]
                    tgt = ref["name"]
                    kinds = models.get(tgt)
                    bad = []
                    if tgt in POLICY:
                        bad.append("정책 서버를 judge 로 씀")
                    if kinds is None:
                        bad.append("모델 서버 정의 없음")
                    elif len(kinds) != 1:
                        bad.append(f"서버 종류 {len(kinds)}개 {kinds}")
                    print(f"{'ERROR' if bad else 'OK   '} {inst}.{stype}.{key} → {tgt} {kinds} {'; '.join(bad)}")
                    errors += bool(bad)
    for e in a.expect:
        srv, tgt = e.split("=")
        if routes.get(srv) != tgt:
            print(f"ERROR 기대 {srv} → {tgt}, 실제 {routes.get(srv)}")
            errors += 1
    print(f"check_gym_judge_routing: ERROR {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
