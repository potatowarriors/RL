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
"""``grpo.async_grpo.shuffle_train_batch``: seeded row shuffle of the async
training batch before it is split into global batches. No Ray."""

import torch

from nemo_rl.algorithms.grpo import AsyncGRPOConfig, shuffle_train_batch_rows
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _batches(n=12):
    train = BatchedDataDict(
        {
            "input_ids": torch.arange(n).unsqueeze(1).repeat(1, 3),
            "advantages": torch.arange(n, dtype=torch.float32),
            "sample_mask": torch.ones(n),
        }
    )
    repeated = BatchedDataDict(
        {"message_log": [[{"row": i}] for i in range(n)], "idx": list(range(n))}
    )
    return train, repeated


def test_default_off():
    assert AsyncGRPOConfig.model_construct().shuffle_train_batch is False


def test_shuffle_keeps_rows_aligned_and_is_seeded():
    train, repeated = _batches()
    perm = shuffle_train_batch_rows(train, repeated, seed=7)
    assert sorted(perm) == list(range(12))
    rows = train["input_ids"][:, 0].tolist()
    assert rows != list(range(12))  # actually reordered
    assert sorted(rows) == list(range(12))
    assert train["advantages"].tolist() == [float(r) for r in rows]
    assert [m[0]["row"] for m in repeated["message_log"]] == rows
    assert repeated["idx"] == rows

    train2, repeated2 = _batches()
    shuffle_train_batch_rows(train2, repeated2, seed=7)
    assert train2["input_ids"][:, 0].tolist() == rows  # same seed, same order
    train3, repeated3 = _batches()
    shuffle_train_batch_rows(train3, repeated3, seed=8)
    assert train3["input_ids"][:, 0].tolist() != rows


def test_size_mismatch_raises():
    train, _ = _batches(12)
    _, repeated = _batches(8)
    try:
        shuffle_train_batch_rows(train, repeated, seed=0)
    except AssertionError:
        return
    raise AssertionError("expected size mismatch to raise")
