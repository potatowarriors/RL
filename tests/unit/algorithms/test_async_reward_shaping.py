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
"""``grpo.reward_shaping`` in the async GRPO training step.

The async loop applies DAPO overlong shaping to the batch sampled from the
replay buffer (``apply_async_reward_shaping``), at the same place the sync loop
calls ``apply_reward_shaping``. Pure-function tests, no Ray.
"""

import torch

from nemo_rl.algorithms.grpo import (
    _max_assistant_turn_lengths,
    apply_async_reward_shaping,
)
from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _msg(role: str, num_tokens: int, as_list: bool = False) -> dict:
    token_ids = list(range(num_tokens)) if as_list else torch.arange(num_tokens)
    return {"role": role, "content": "x", "token_ids": token_ids}


def _batch(turn_lengths: list[list[int]], rewards: list[float]) -> BatchedDataDict:
    """One sample per entry; each entry lists its assistant turn lengths."""
    message_logs = []
    for lengths in turn_lengths:
        log = [_msg("user", 4)]
        for n in lengths:
            log.append(_msg("assistant", n))
            log.append(_msg("user", 2))
        message_logs.append(log)
    return BatchedDataDict(
        {"message_log": message_logs, "total_reward": torch.tensor(rewards)}
    )


def _cfg(enabled: bool = True) -> RewardShapingConfig:
    # expected_response_length = 25 - 5 = 20
    return RewardShapingConfig(
        enabled=enabled,
        overlong_buffer_length=5,
        overlong_buffer_penalty=1.0,
        max_response_length=25,
    )


def test_max_assistant_turn_lengths_uses_longest_turn():
    logs = [
        [_msg("user", 3), _msg("assistant", 5), _msg("user", 2), _msg("assistant", 9)],
        [_msg("user", 4)],
        [_msg("user", 1), _msg("assistant", 7, as_list=True)],
    ]
    assert _max_assistant_turn_lengths(logs).tolist() == [9, 0, 7]


def test_disabled_returns_batch_unchanged_and_no_metrics():
    batch = _batch([[10], [30]], [1.0, 0.0])
    original = batch["total_reward"].clone()
    out, metrics = apply_async_reward_shaping(batch, _cfg(enabled=False))
    assert out is batch
    assert metrics == {}
    assert torch.equal(out["total_reward"], original)
    assert "unshaped_total_reward" not in out
    assert "response_token_lengths" not in out


def test_enabled_shapes_rewards_and_reports_metrics():
    # per-turn max: 10, 22, 25 (two turns, the longer counts), 30
    batch = _batch([[10], [22], [12, 25], [30]], [1.0, 0.0, 1.0, 0.0])
    out, metrics = apply_async_reward_shaping(batch, _cfg())
    # exceed = len - 20 → penalty min(-exceed/5, 0): 0, -0.4, -1.0, -2.0
    assert torch.allclose(
        out["total_reward"], torch.tensor([1.0, -0.4, 0.0, -2.0]), atol=1e-6
    )
    assert torch.equal(out["unshaped_total_reward"], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert "response_token_lengths" not in out
    assert metrics["reward_shaping/penalized_rate"] == 0.75
    assert abs(metrics["reward_shaping/mean_penalty"] - (-3.4 / 4)) < 1e-6


def test_existing_response_token_lengths_are_honored():
    """Data-plane slices carry response_token_lengths and no message_log."""
    batch = BatchedDataDict(
        {
            "total_reward": torch.tensor([1.0, 1.0]),
            "response_token_lengths": torch.tensor([20, 25]),
        }
    )
    out, metrics = apply_async_reward_shaping(batch, _cfg())
    assert torch.allclose(out["total_reward"], torch.tensor([1.0, 0.0]))
    assert out["response_token_lengths"].tolist() == [20, 25]
    assert metrics["reward_shaping/penalized_rate"] == 0.5
