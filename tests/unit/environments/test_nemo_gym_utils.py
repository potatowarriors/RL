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
"""Pure-Python (vllm-free) unit tests for NeMo-Gym helpers.

These run in the default L0 suite. Keep this module free of heavy imports
(e.g. vllm) so the fast detector tests are not gated behind the nemo_gym extra.
"""

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _compile_tool_output_patterns,
    _detect_invalid_tool_call_and_malformed_thinking,
    _is_rejected_tool_call_output,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
)


@pytest.mark.parametrize(
    ("output_item_dict", "expected_invalid_tool_call", "expected_malformed_thinking"),
    [
        (
            {"content": [{"text": "use <tool_call>{}</tool_call>"}]},
            True,
            False,
        ),
        (
            {"content": [{"text": "final answer leaked <think>reasoning</think>"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think>"}]},
            False,
            False,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think><think>b"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "bad <function_call>{}"}]},
            True,
            False,
        ),
    ],
)
def test_detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict,
    expected_invalid_tool_call,
    expected_malformed_thinking,
):
    assert _detect_invalid_tool_call_and_malformed_thinking(output_item_dict) == (
        expected_invalid_tool_call,
        expected_malformed_thinking,
    )


def test_get_nemo_gym_venv_dir_returns_env_value(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    assert get_nemo_gym_venv_dir() == "/opt/gym_venvs"


def test_get_nemo_gym_venv_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)
    assert get_nemo_gym_venv_dir() is None


def test_get_nemo_gym_uv_cache_dir_none_outside_container(monkeypatch):
    # Outside a container the caller should omit the arg; uv must not be invoked.
    monkeypatch.delenv("NRL_CONTAINER", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("uv should not be invoked outside a container")

    monkeypatch.setattr(nemo_gym_mod.subprocess, "check_output", _fail)
    assert get_nemo_gym_uv_cache_dir() is None


def test_get_nemo_gym_uv_cache_dir_uses_uv_inside_container(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setattr(
        nemo_gym_mod.subprocess,
        "check_output",
        lambda *args, **kwargs: b"  /root/.cache/uv\n",
    )
    assert get_nemo_gym_uv_cache_dir() == "/root/.cache/uv"


# Error messages an agent harness (nv-OpenHands) returns for a tool call it rejects.
_REJECTION_PATTERNS = [
    r"Validation failure for (?P<name>[^:\n]+): ",
    r"(?s)Tool (?P<name>.+?) is not registered\. \(arguments: ",
]
_VALIDATION_ERROR = (
    'Validation failure for execute_bash: Missing required argument "command" '
    "in tool call execute_bash"
)
_UNKNOWN_TOOL_ERROR = (
    "Tool shellcommand is not registered. (arguments: {'command': 'ls'}). "
    "Please check the tool name and retry with an existing tool."
)


@pytest.mark.parametrize(
    ("output_item_dict", "called_tool_name", "expected"),
    [
        # Rejections of the called tool.
        (
            {"type": "function_call_output", "output": _VALIDATION_ERROR},
            "execute_bash",
            True,
        ),
        (
            {"type": "function_call_output", "output": _UNKNOWN_TOOL_ERROR},
            "shellcommand",
            True,
        ),
        # List-form output content.
        (
            {
                "type": "function_call_output",
                "output": [{"type": "input_text", "text": _VALIDATION_ERROR}],
            },
            "execute_bash",
            True,
        ),
        # Ordinary tool output.
        (
            {"type": "function_call_output", "output": "a.py\nb.py"},
            "execute_bash",
            False,
        ),
        # The words appear, but not at the start of the output.
        (
            {
                "type": "function_call_output",
                "output": "$ cat log\n" + _VALIDATION_ERROR,
            },
            "execute_bash",
            False,
        ),
        # Starts like a rejection but names another tool than the one called.
        (
            {"type": "function_call_output", "output": _VALIDATION_ERROR},
            "str_replace_editor",
            False,
        ),
        # The answered call is unknown.
        ({"type": "function_call_output", "output": _VALIDATION_ERROR}, None, False),
        # Not a tool output item.
        (
            {"type": "message", "content": [{"text": _VALIDATION_ERROR}]},
            "execute_bash",
            False,
        ),
        ({"type": "function_call_output", "output": None}, "execute_bash", False),
    ],
)
def test_is_rejected_tool_call_output(output_item_dict, called_tool_name, expected):
    assert (
        _is_rejected_tool_call_output(
            output_item_dict,
            rejection_patterns=_compile_tool_output_patterns(_REJECTION_PATTERNS),
            called_tool_name=called_tool_name,
        )
        is expected
    )


def test_is_rejected_tool_call_output_pattern_without_name_group():
    patterns = _compile_tool_output_patterns([r"Validation failure for "])
    item = {"type": "function_call_output", "output": _VALIDATION_ERROR}
    assert _is_rejected_tool_call_output(
        item, rejection_patterns=patterns, called_tool_name="another_tool"
    )


def test_compile_tool_output_patterns_unset_is_empty():
    assert _compile_tool_output_patterns(None) == []
    assert _compile_tool_output_patterns([]) == []


@pytest.mark.parametrize("patterns", [["(unclosed"], [3]])
def test_compile_tool_output_patterns_rejects_bad_pattern(patterns):
    with pytest.raises(ValueError, match="invalid_tool_call_output_patterns"):
        _compile_tool_output_patterns(patterns)


class _CountingTokenizer:
    def batch_decode(self, batch):
        return ["decoded"] * len(batch)


def _tool_call(call_id, name, *, prompt_token_ids=None, generation_token_ids=None):
    item = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }
    if generation_token_ids is not None:
        item["prompt_token_ids"] = prompt_token_ids
        item["generation_token_ids"] = generation_token_ids
        item["generation_log_probs"] = [-0.1] * len(generation_token_ids)
    return item


def _tool_output(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _postprocess_flags(output_items, patterns):
    class _MockSelf:
        cfg = {"invalid_tool_call_output_patterns": patterns}
        _processor = None

    nemo_gym_result = {
        "response": {"output": output_items},
        "responses_create_params": {"input": []},
    }
    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _CountingTokenizer()
        )
    )
    flags = [
        message["is_invalid_tool_call"]
        for message in result["message_log"]
        if message["role"] == "assistant"
    ]
    # The per-rollout count is reported only when tool outputs are inspected, and
    # every flag in these cases comes from a rejection.
    if patterns:
        assert result["full_result"]["num_harness_rejected_tool_calls"] == sum(flags)
    else:
        assert "num_harness_rejected_tool_calls" not in result["full_result"]
    return flags


def _three_model_calls(second_output, *, second_name="execute_bash"):
    """Three model calls; the second one's tool output is ``second_output``."""
    return [
        _tool_call(
            "c1", "execute_bash", prompt_token_ids=[1, 2], generation_token_ids=[3]
        ),
        _tool_output("c1", "a.py"),
        _tool_call(
            "c2",
            second_name,
            prompt_token_ids=[1, 2, 3, 4],
            generation_token_ids=[5, 6],
        ),
        _tool_output("c2", second_output),
        _tool_call(
            "c3",
            "execute_bash",
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
            generation_token_ids=[8],
        ),
        _tool_output("c3", "done"),
    ]


def test_postprocess_flags_only_the_rejected_model_call():
    flags = _postprocess_flags(
        _three_model_calls(_VALIDATION_ERROR), _REJECTION_PATTERNS
    )
    assert flags == [False, True, False]


def test_postprocess_flags_unknown_tool_call():
    flags = _postprocess_flags(
        _three_model_calls(_UNKNOWN_TOOL_ERROR, second_name="shellcommand"),
        _REJECTION_PATTERNS,
    )
    assert flags == [False, True, False]


@pytest.mark.parametrize("patterns", [None, []])
def test_postprocess_does_not_inspect_tool_outputs_when_patterns_unset(patterns):
    assert _postprocess_flags(_three_model_calls(_VALIDATION_ERROR), patterns) == [
        False,
        False,
        False,
    ]


def test_postprocess_does_not_flag_ordinary_tool_output_with_the_same_words():
    # A command that prints the harness's words is a valid call.
    output = "$ cat log.txt\n" + _VALIDATION_ERROR
    assert _postprocess_flags(_three_model_calls(output), _REJECTION_PATTERNS) == [
        False,
        False,
        False,
    ]


def test_postprocess_does_not_flag_a_later_message_for_an_earlier_call():
    # The rejection answers c1, but arrives after the second model call: it must not
    # mark the second assistant message, whose own call (c2) ran normally.
    items = [
        _tool_call(
            "c1", "execute_bash", prompt_token_ids=[1, 2], generation_token_ids=[3]
        ),
        _tool_call(
            "c2",
            "execute_bash",
            prompt_token_ids=[1, 2, 3, 4],
            generation_token_ids=[5],
        ),
        _tool_output("c1", _VALIDATION_ERROR),
        _tool_output("c2", "ok"),
    ]
    assert _postprocess_flags(items, _REJECTION_PATTERNS) == [False, False]


def test_postprocess_tolerates_tool_call_items_without_name_or_call_id():
    # A tool call whose name is unknown cannot be tied to a named rejection.
    items = [
        {
            "type": "function_call",
            "call_id": "c1",
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
            "generation_log_probs": [-0.1],
        },
        _tool_output("c1", _VALIDATION_ERROR),
        {
            "type": "function_call",
            "prompt_token_ids": [1, 2, 3, 4],
            "generation_token_ids": [5],
            "generation_log_probs": [-0.1],
        },
        {"type": "function_call_output", "output": _VALIDATION_ERROR},
    ]
    assert _postprocess_flags(items, _REJECTION_PATTERNS) == [False, False]


def test_postprocess_flags_model_call_with_parallel_tool_calls_when_one_is_rejected():
    # Only the last output item of a model call carries the token ids.
    items = [
        _tool_call("c1", "execute_bash"),
        _tool_call(
            "c2", "execute_bash", prompt_token_ids=[1, 2], generation_token_ids=[3, 4]
        ),
        _tool_output("c1", "a.py"),
        _tool_output("c2", _VALIDATION_ERROR),
        _tool_call(
            "c3",
            "execute_bash",
            prompt_token_ids=[1, 2, 3, 4, 5],
            generation_token_ids=[6],
        ),
        _tool_output("c3", "done"),
    ]
    assert _postprocess_flags(items, _REJECTION_PATTERNS) == [True, False]


def test_postprocess_counts_rejections_apart_from_text_flags():
    # A leftover textual tool call (the text detector's case) is flagged but is not a
    # harness rejection; a call rejected by two outputs is counted once.
    items = [
        {
            "type": "message",
            "content": [{"text": "<tool_call> oops"}],
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
            "generation_log_probs": [-0.1],
        },
        _tool_call("c1", "execute_bash"),
        _tool_call(
            "c2",
            "execute_bash",
            prompt_token_ids=[1, 2, 3, 4],
            generation_token_ids=[5],
        ),
        _tool_output("c1", _VALIDATION_ERROR),
        _tool_output("c2", _VALIDATION_ERROR),
    ]

    class _MockSelf:
        cfg = {"invalid_tool_call_output_patterns": _REJECTION_PATTERNS}
        _processor = None

    nemo_gym_result = {
        "response": {"output": items},
        "responses_create_params": {"input": []},
    }
    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _CountingTokenizer()
        )
    )
    flags = [
        m["is_invalid_tool_call"]
        for m in result["message_log"]
        if m["role"] == "assistant"
    ]
    assert flags == [True, True]
    assert result["full_result"]["num_harness_rejected_tool_calls"] == 1
