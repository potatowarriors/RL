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
"""DAPO dynamic sampling for async GRPO (``grpo.async_grpo.drop_zero_std_groups``).

Prompt groups whose rewards are all equal are dropped before the replay buffer
(and still logged); groups with reward variance are buffered as before. No Ray.
"""

import asyncio
import json
import threading
import unittest.mock as mock

import torch

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
from nemo_rl.algorithms.async_utils.trajectory_collector import zero_std_group_kind
from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    GRPOConfig,
    MasterConfig,
    dynamic_sampling_step_metrics,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import RolloutGroupResult


def _group(rewards):
    return BatchedDataDict(
        {
            "message_log": [
                [
                    {"role": "user", "content": "q", "token_ids": torch.arange(3)},
                    {"role": "assistant", "content": "a", "token_ids": torch.arange(4)},
                ]
                for _ in rewards
            ],
            "total_reward": torch.tensor(rewards),
            "truncated": torch.zeros(len(rewards), dtype=torch.bool),
        }
    )


def _collector(drop: bool, log_path=None):
    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    master_config = MasterConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=2,
            max_rollout_turns=1,
            debug_payload_metrics=False,
            async_grpo=AsyncGRPOConfig.model_construct(
                max_trajectory_age_steps=2,
                max_generation_failures=0,
                rollout_overprovision=0,
                rollout_group_log=log_path,
                drop_zero_std_groups=drop,
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
    replay_buffer = mock.MagicMock()
    replay_buffer.add.remote = mock.AsyncMock(return_value="success")
    collector = collector_cls(
        policy_generation=mock.MagicMock(),
        tokenizer=mock.MagicMock(),
        task_to_env={},
        master_config=master_config,
        replay_buffer=replay_buffer,
        start_step=0,
        next_nemo_gym_task_index=0,
    )
    for attr in (
        "_manual_pause_cleared",
        "_refit_pause_cleared",
        "_generation_limit_cleared",
    ):
        ev = threading.Event()
        ev.set()
        setattr(collector, attr, ev)
    collector.running = True
    return collector, replay_buffer


def _enqueue(collector, rewards, group_index=0, buffered=None):
    buffered = set() if buffered is None else buffered
    result = RolloutGroupResult(
        group_index=group_index,
        final_batch=_group(rewards),
        rollout_metrics={},
        task_index=None,
    )
    asyncio.run(
        collector._enqueue_rollout_group(
            rollout_result=result,
            generation_weight_version=3,
            target_weight_version=4,
            expected_prompt_groups=2,
            buffered_group_indices=buffered,
            collection_started_at=0.0,
        )
    )
    return buffered


def test_zero_std_group_kind():
    assert zero_std_group_kind(torch.tensor([0.0, 0.0, 0.0])) == "all_wrong"
    assert zero_std_group_kind([1.0, 1.0]) == "all_correct"
    assert zero_std_group_kind([0.0, 1.0, 0.0]) is None
    assert zero_std_group_kind([1.0]) is None  # a single sample is never dropped


def test_dynamic_sampling_step_metrics():
    m = dynamic_sampling_step_metrics(
        {"kept": 10, "all_wrong": 4, "all_correct": 1},
        {"kept": 16, "all_wrong": 7, "all_correct": 2},
    )
    assert m["dynamic_sampling/groups"] == 10.0
    assert m["dynamic_sampling/kept_fraction"] == 0.6
    assert m["dynamic_sampling/all_wrong_fraction"] == 0.3
    assert m["dynamic_sampling/all_correct_fraction"] == 0.1
    assert dynamic_sampling_step_metrics({}, {})["dynamic_sampling/groups"] == 0.0


def test_zero_variance_group_is_dropped_resolved_and_logged(tmp_path):
    log_path = str(tmp_path / "groups.jsonl")
    collector, replay_buffer = _collector(drop=True, log_path=log_path)
    buffered = _enqueue(collector, [0.0, 0.0], group_index=1)
    assert buffered == {1}  # resolved so the batch does not retry it
    replay_buffer.add.remote.assert_not_called()
    rec = json.loads(open(log_path).read().splitlines()[0])
    assert rec["dropped"] == "all_wrong" and rec["rewards"] == [0.0, 0.0]
    assert collector.get_dynamic_sampling_counts() == {"all_wrong": 1}


def test_group_with_variance_is_buffered(tmp_path):
    log_path = str(tmp_path / "groups.jsonl")
    collector, replay_buffer = _collector(drop=True, log_path=log_path)
    buffered = _enqueue(collector, [1.0, 0.0], group_index=0)
    assert buffered == {0}
    replay_buffer.add.remote.assert_awaited_once()
    rec = json.loads(open(log_path).read().splitlines()[0])
    assert rec["dropped"] is None
    assert collector.get_dynamic_sampling_counts() == {"kept": 1}


def test_disabled_keeps_zero_variance_groups():
    collector, replay_buffer = _collector(drop=False)
    buffered = _enqueue(collector, [1.0, 1.0], group_index=0)
    assert buffered == {0}
    replay_buffer.add.remote.assert_awaited_once()
    assert collector.get_dynamic_sampling_counts() == {}
