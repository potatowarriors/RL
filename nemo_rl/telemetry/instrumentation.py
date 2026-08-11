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

"""Instrumentation helpers that attach option-B goodput tags.

Algorithms should import ``managed_span`` / ``trace_fn`` from here (not raw
nemo-lens) so every leaf span gets ``rl.bucket`` when applicable.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any

from nemo_rl.telemetry._fallbacks import (
    is_span_group_enabled,
    managed_span as _managed_span,
    safe_set_span_attributes,
    span_cm,
)
from nemo_rl.telemetry.goodput import RL_BUCKET_ATTR, goodput_span_attributes

__all__ = [
    "managed_span",
    "trace_fn",
    "span_cm",
    "is_span_group_enabled",
    "safe_set_span_attributes",
]


@contextmanager
def managed_span(group: str, name: str, tracer=None, **attributes: Any):
    """Like lens ``managed_span``, but injects ``rl.bucket`` for leaf groups.

    Callers may override by passing ``rl.bucket=...`` explicitly. Umbrella
    groups (job / step / rollout / …) receive no bucket attribute.
    """
    attrs = dict(attributes)
    if RL_BUCKET_ATTR not in attrs:
        attrs.update(goodput_span_attributes(group))
    with _managed_span(group, name, tracer=tracer, **attrs) as span:
        yield span


def trace_fn(group: str, name: str, tracer=None):
    """Decorator that wraps a function in a bucket-tagged ``managed_span``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with managed_span(group, name, tracer=tracer):
                return func(*args, **kwargs)

        return wrapper

    return decorator
