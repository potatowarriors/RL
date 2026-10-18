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
"""``grpo.async_grpo.rollout_group_log``: one jsonl record per buffered prompt group.

The collector remembers each NeMo-Gym row's identity when it stamps the task
index and writes a record (row key, weight versions, per-sample rewards,
generated tokens, truncation) once the group is buffered. No Ray.
"""

import json
import threading
import unittest.mock as mock

import torch

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
from nemo_rl.algorithms.async_utils.trajectory_collector import (
    _row_log_meta,
    rollout_group_record,
)
from nemo_rl.algorithms.grpo import AsyncGRPOConfig, GRPOConfig, MasterConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import RolloutGroupResult


def _group_batch(rewards, gen_lengths, truncated):
    message_logs = [
        [
            {"role": "user", "content": "q", "token_ids": torch.arange(5)},
            {"role": "assistant", "content": "a", "token_ids": torch.arange(n)},
        ]
        for n in gen_lengths
    ]
    return BatchedDataDict(
        {
            "message_log": message_logs,
            "total_reward": torch.tensor(rewards),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
        }
    )


def _local_collector(log_path):
    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    master_config = MasterConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=2,
            max_rollout_turns=1,
            async_grpo=AsyncGRPOConfig.model_construct(
                max_trajectory_age_steps=2,
                max_generation_failures=0,
                rollout_overprovision=0,
                rollout_group_log=log_path,
            ),
        ),
        policy={
            "max_total_sequence_length": 512,
            "make_sequence_length_divisible_by": 1,
            "generation": {"backend": "vllm", "vllm_cfg": {"async_engine": False}},
        },
        env={"should_use_nemo_gym": True},
        logger={
            "wandb_enabled": False,
            "wandb": {"log_nemo_gym_full_result_tables": False},
        },
    )
    collector = collector_cls(
        policy_generation=mock.MagicMock(),
        tokenizer=mock.MagicMock(),
        task_to_env={},
        master_config=master_config,
        replay_buffer=mock.MagicMock(),
        start_step=0,
        next_nemo_gym_task_index=7,
    )
    for attr in (
        "_manual_pause_cleared",
        "_refit_pause_cleared",
        "_generation_limit_cleared",
    ):
        ev = threading.Event()
        ev.set()
        setattr(collector, attr, ev)
    return collector


def _gym_rows():
    return [
        {
            "pool_key": "k1",
            "dataset": "taco",
            "agent_ref": {"name": "code_gen_simple_agent"},
        },
        {
            "pool_key": "k2",
            "dataset": "dapo",
            "agent_ref": {"name": "math_with_judge_simple_agent"},
        },
    ]


def test_row_log_meta_reads_gym_row_identity():
    assert _row_log_meta(_gym_rows()[0]) == {
        "pool_key": "k1",
        "dataset": "taco",
        "agent": "code_gen_simple_agent",
    }
    assert _row_log_meta({}) == {"pool_key": None, "dataset": None, "agent": None}


def test_rollout_group_record_fields():
    batch = _group_batch([1.0, 0.0], [12, 30], [False, True])
    rec = rollout_group_record(
        batch,
        task_index=9,
        row_meta={"pool_key": "k1", "dataset": "taco", "agent": "code"},
        generation_weight_version=4,
        target_weight_version=6,
    )
    assert rec["task_index"] == 9
    assert rec["pool_key"] == "k1" and rec["agent"] == "code"
    assert rec["generation_weight_version"] == 4 and rec["target_weight_version"] == 6
    assert rec["rewards"] == [1.0, 0.0]
    assert rec["gen_tokens"] == [12, 30]
    assert rec["truncated"] == [False, True]
    json.dumps(rec)  # serializable


def test_collector_logs_buffered_group_with_row_key(tmp_path):
    log_path = str(tmp_path / "groups.jsonl")
    collector = _local_collector(log_path)
    batch = BatchedDataDict({"extra_env_info": _gym_rows()})
    collector._stamp_nemo_gym_task_indices(batch)
    assert set(collector._task_row_meta) == {7, 8}

    result = RolloutGroupResult(
        group_index=1,
        final_batch=_group_batch([0.0, 1.0], [5, 6], [False, False]),
        rollout_metrics={},
        task_index=8,
    )
    collector._write_rollout_group_log(result, 3, 4)

    lines = open(log_path).read().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["pool_key"] == "k2" and rec["agent"] == "math_with_judge_simple_agent"
    assert rec["task_index"] == 8 and rec["rewards"] == [0.0, 1.0]
    assert 8 not in collector._task_row_meta  # popped once written


def test_collector_log_disabled_is_noop(tmp_path):
    collector = _local_collector(None)
    batch = BatchedDataDict({"extra_env_info": _gym_rows()})
    collector._stamp_nemo_gym_task_indices(batch)
    assert collector._task_row_meta == {}
    result = RolloutGroupResult(
        group_index=0,
        final_batch=_group_batch([1.0, 1.0], [3, 3], [False, False]),
        rollout_metrics={},
        task_index=7,
    )
    collector._write_rollout_group_log(result, 0, 1)
    assert list(tmp_path.iterdir()) == []


def test_collector_log_write_error_does_not_raise(tmp_path):
    collector = _local_collector(str(tmp_path / "missing_dir" / "groups.jsonl"))
    result = RolloutGroupResult(
        group_index=0,
        final_batch=_group_batch([1.0], [3], [False]),
        rollout_metrics={},
        task_index=None,
    )
    collector._write_rollout_group_log(result, 0, 1)  # must not raise
