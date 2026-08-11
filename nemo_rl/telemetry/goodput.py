# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Option-B goodput tagging for NeMo-RL telemetry.

Frameworks keep dialect phase / span-group names. Each *bucketed* phase is
tagged with a shared goodput bucket so offline monitors (e.g. wandb-monitor)
can SUM GPU-time by bucket without requiring identical leaf names across
Cosmos / NeMo / etc.

Contract
--------
* Shared bucket tokens: ``productive`` | ``overhead`` | ``idle`` | ``wasted``.
* Umbrella groups (``job``, ``step``, ``rollout``, …) are timed but **not**
  tagged — monitors must exclude them from the goodput denominator.
* Apps do **not** emit rolled-up ``rl.bucket.*`` / ``rl.goodput``; the monitor
  derives those from tagged phase durations / span walls.

Span attribute key: ``rl.bucket``.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Optional

from nemo_rl.telemetry.span_groups import RLSpanGroup

# OTel / OneLogger-shared attribute key (flat sinks encode this in the name).
RL_BUCKET_ATTR = "rl.bucket"


class Bucket(str, Enum):
    """Shared goodput buckets (lingua franca across RL frameworks)."""

    PRODUCTIVE = "productive"
    OVERHEAD = "overhead"
    IDLE = "idle"
    WASTED = "wasted"


# Span groups that are umbrellas / lifecycle only — no rl.bucket tag.
UMBRELLA_GROUPS: frozenset[str] = frozenset(
    {
        RLSpanGroup.JOB,
        RLSpanGroup.STEP,
        RLSpanGroup.ROLLOUT,  # collect_rollouts umbrella (like Cosmos generate)
        RLSpanGroup.MODEL_INIT,
        RLSpanGroup.EVALUATE,  # eval pass; treat as umbrella unless timed as idle
    }
)

# Default classification for RLSpanGroup members that are leaf work.
# logprob / advantage / reference_policy count as overhead (prep), not the
# productive policy gradient update itself.
_DEFAULT_GROUP_BUCKET: Mapping[str, Bucket] = {
    RLSpanGroup.GENERATION: Bucket.PRODUCTIVE,
    RLSpanGroup.REWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.POLICY_UPDATE: Bucket.PRODUCTIVE,
    RLSpanGroup.FORWARD_BACKWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.OPTIMIZER: Bucket.PRODUCTIVE,
    RLSpanGroup.DATA_PROCESSING: Bucket.OVERHEAD,
    RLSpanGroup.CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOAD_CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOGPROB: Bucket.OVERHEAD,
    RLSpanGroup.ADVANTAGE: Bucket.OVERHEAD,
    RLSpanGroup.REFERENCE_POLICY: Bucket.OVERHEAD,
}

# Async efficiency category labels → bucket (when emitted as phase metrics).
# These are not RLSpanGroup members; listed for monitor / future metric tee.
EFFICIENCY_CATEGORY_BUCKET: Mapping[str, Bucket] = {
    "init/total": Bucket.OVERHEAD,
    "idle/buffer_starvation": Bucket.IDLE,
    "idle/refit_bubble": Bucket.IDLE,
    "idle/validation": Bucket.IDLE,
    "idle/buffer_full_backoff": Bucket.IDLE,
    "idle/generation_limit_pause": Bucket.IDLE,
    "idle/refit_event_wait": Bucket.IDLE,
    "wasted/failed_trajectory": Bucket.WASTED,
}


def bucket_for_span_group(group: str) -> Optional[Bucket]:
    """Return the goodput bucket for a span group, or None if umbrella / unknown.

    Unknown non-umbrella groups default to ``overhead`` so new leaves are not
    silently dropped from the denominator.
    """
    if group in UMBRELLA_GROUPS:
        return None
    if group in _DEFAULT_GROUP_BUCKET:
        return _DEFAULT_GROUP_BUCKET[group]
    return Bucket.OVERHEAD


def bucket_for_efficiency_category(category: str) -> Optional[Bucket]:
    """Return the bucket for an async efficiency category label, if known."""
    return EFFICIENCY_CATEGORY_BUCKET.get(category)


def goodput_span_attributes(group: str) -> dict[str, str]:
    """Attributes to merge into ``managed_span`` for *group*.

    Empty when the group is an umbrella (no ``rl.bucket``).
    """
    bucket = bucket_for_span_group(group)
    if bucket is None:
        return {}
    return {RL_BUCKET_ATTR: bucket.value}
