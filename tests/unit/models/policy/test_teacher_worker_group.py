# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_teacher_resource_config_defaults():
    from nemo_rl.algorithms.opd import TeacherResourceConfig

    res = TeacherResourceConfig(tensor_model_parallel_size=4)
    assert res.tensor_model_parallel_size == 4
    assert res.pipeline_model_parallel_size == 1
    assert res.gpus_per_node == 8
    assert res.precision == "bf16"


def test_create_teacher_configs_homogeneous():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4}
            },
        }
    )
    assert len(configs) == 2
    assert all(c.tensor_model_parallel_size == 4 for c in configs)


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4},
                "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
            },
        }
    )
    code_cfg = [c for c in configs if c.alias == "code"][0]
    assert code_cfg.tensor_model_parallel_size == 8


def test_create_teacher_configs_deduplicates():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {
                "math": "/shared",
                "code": "/shared",
                "rlhf": "/rlhf",
            },
            "deduplicate_shared_teacher_checkpoints": True,
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 2}
            },
        }
    )
    assert len(configs) == 2


def _build_teacher_with_fake_workers(monkeypatch, policy_config):
    """Construct a TeacherWorkerGroup without Ray; return (teacher, cfg passed to the worker builder)."""
    from unittest.mock import MagicMock

    import nemo_rl.distributed.worker_groups as worker_groups
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    captured = {}

    class FakeWorkerBuilder:
        def __init__(self, worker_path, cfg, **kwargs):
            del worker_path, kwargs
            captured["cfg"] = cfg

    class FakeWorkerGroup:
        def __init__(self, cluster, worker_builder, **kwargs):
            del cluster, worker_builder, kwargs

    monkeypatch.setattr(worker_groups, "RayWorkerBuilder", FakeWorkerBuilder)
    monkeypatch.setattr(worker_groups, "RayWorkerGroup", FakeWorkerGroup)
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    teacher_config = TeacherConfig(
        alias="teacher",
        model_name="/ckpt/teacher",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_nodes=1,
        gpus_per_node=1,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides={},
    )
    teacher = TeacherWorkerGroup(teacher_config, cluster, policy_config, MagicMock())
    return teacher, captured["cfg"]


# Ported from upstream 790952514 (router replay) and 93b1c1d03 (pretrained_checkpoint).
def test_teacher_worker_group_disables_student_router_replay(monkeypatch):
    """Frozen teachers do not require rollout-to-training route consistency."""
    policy_config = {
        "model_name": "/ckpt/student",
        "megatron_cfg": {"enabled": True},
        "dtensor_cfg": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "router_replay": {"enabled": True},
    }
    teacher, worker_cfg = _build_teacher_with_fake_workers(monkeypatch, policy_config)

    assert worker_cfg["router_replay"]["enabled"] is False
    assert teacher.cfg["router_replay"]["enabled"] is False
    assert policy_config["router_replay"]["enabled"] is True


def test_teacher_worker_group_drops_the_student_pretrained_checkpoint(monkeypatch):
    """Left in, the teacher would load the student's weights and distill itself."""
    policy_config = {
        "model_name": "/ckpt/student",
        "pretrained_checkpoint": {"format": "megatron_bridge", "path": "/ckpt/sft"},
        "megatron_cfg": {"enabled": True},
        "dtensor_cfg": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }
    teacher, worker_cfg = _build_teacher_with_fake_workers(monkeypatch, policy_config)

    assert "pretrained_checkpoint" not in worker_cfg
    assert "pretrained_checkpoint" not in teacher.cfg
    assert worker_cfg["model_name"] == "/ckpt/teacher"
    assert policy_config["pretrained_checkpoint"]["path"] == "/ckpt/sft"
