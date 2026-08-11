# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for option-B goodput bucket tagging."""

from nemo_rl.telemetry.goodput import (
    EFFICIENCY_CATEGORY_BUCKET,
    RL_BUCKET_ATTR,
    UMBRELLA_GROUPS,
    Bucket,
    bucket_for_efficiency_category,
    bucket_for_span_group,
    goodput_span_attributes,
)
from nemo_rl.telemetry.span_groups import RLSpanGroup


def test_shared_bucket_tokens():
    assert {b.value for b in Bucket} == {
        "productive",
        "overhead",
        "idle",
        "wasted",
    }


def test_umbrellas_have_no_bucket():
    for group in (
        RLSpanGroup.JOB,
        RLSpanGroup.STEP,
        RLSpanGroup.ROLLOUT,
        RLSpanGroup.MODEL_INIT,
        RLSpanGroup.EVALUATE,
    ):
        assert group in UMBRELLA_GROUPS
        assert bucket_for_span_group(group) is None
        assert goodput_span_attributes(group) == {}


def test_leaf_groups_map_to_expected_buckets():
    assert bucket_for_span_group(RLSpanGroup.GENERATION) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.REWARD) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.POLICY_UPDATE) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.DATA_PROCESSING) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.CHECKPOINT) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.LOGPROB) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.ADVANTAGE) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.REFERENCE_POLICY) is Bucket.OVERHEAD


def test_goodput_span_attributes_shape():
    attrs = goodput_span_attributes(RLSpanGroup.GENERATION)
    assert attrs == {RL_BUCKET_ATTR: "productive"}


def test_unknown_non_umbrella_defaults_to_overhead():
    assert bucket_for_span_group("brand_new_leaf") is Bucket.OVERHEAD
    assert goodput_span_attributes("brand_new_leaf")[RL_BUCKET_ATTR] == "overhead"


def test_efficiency_categories_mapped():
    assert bucket_for_efficiency_category("idle/buffer_starvation") is Bucket.IDLE
    assert (
        bucket_for_efficiency_category("wasted/failed_trajectory") is Bucket.WASTED
    )
    assert bucket_for_efficiency_category("init/total") is Bucket.OVERHEAD
    assert set(EFFICIENCY_CATEGORY_BUCKET)  # non-empty


def test_every_rl_span_group_is_classified():
    """Every known RLSpanGroup is either umbrella or has an explicit/default bucket."""
    for group in RLSpanGroup.ALL_GROUPS:
        bucket = bucket_for_span_group(group)
        if group in UMBRELLA_GROUPS:
            assert bucket is None, group
        else:
            assert bucket in Bucket, group
