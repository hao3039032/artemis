# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashRunner loop wiring for the stagnation advisor (mount-point test).

Verifies the production mount point — after ``_commit_turn``, before
``_build_tail`` — end to end: two identical successful action turns with
unchanged screens produce exactly one advisory notice riding the next
observation tail (once per action+target signature).
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from artemis.agents.flash.runner import FlashRunner
from artemis.context import ArtemisContext
from artemis.llm.decision import FakeDecisionClient


def _noise(seed: int, size=(200, 100)) -> bytes:
    import io
    import random

    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(size[0] * size[1])
        ]
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class _LoopHarness:
    """Drives FlashRunner.run() with a scripted model and executor."""

    def __init__(self, turns_script: list[list[dict]]):
        # turns_script: one entry per turn, each a list of tool-call dicts.
        self.turns_script = turns_script
        self.turn_index = 0
        self.tails: list[HumanMessage] = []
        self.post_images = [
            _noise(1),
            _noise(1),  # identical screen both turns
            _noise(1),
            _noise(1),
        ]
        self.executed: list[tuple[str, dict]] = []

    async def observe(self, *args, **kwargs):
        return MagicMock(ok=True, elements_text="ui list", screenshot_path=None)

    def next_response(self, *args, **kwargs):
        if self.turn_index >= len(self.turns_script):
            return AIMessage(
                content="done",
                tool_calls=[
                    {
                        "name": "report_task_status",
                        "args": {"status": "completed", "explanation": "ok"},
                        "id": "fin",
                    }
                ],
            )
        calls = self.turns_script[self.turn_index]
        self.turn_index += 1
        return AIMessage(
            content="acting",
            tool_calls=[
                {"name": c["name"], "args": c["args"], "id": f"tc{self.turn_index}"} for c in calls
            ],
        )


@pytest.mark.asyncio
async def test_stagnation_notice_reaches_the_tail():
    harness = _LoopHarness(
        turns_script=[
            [
                {
                    "name": "click",
                    "args": {"target": [500, 900], "target_description": "Login button"},
                }
            ],
            [
                {
                    "name": "click",
                    "args": {"target": [500, 900], "target_description": "Login button"},
                }
            ],
        ]
    )
    ctx = Mock(spec=ArtemisContext)
    ctx.llm_config = None
    ctx.data_engine = None
    ctx.step_memory = None
    ctx.transcript_ledger = None
    ctx.execution_setup = None
    ctx.actuator = None

    decision = FakeDecisionClient(
        scripted={
            "stagnation_detection": {
                "materially_changed": 0.03,
                "situation": "true_stagnation",
            }
        }
    )

    runner = None
    with patch("artemis.controllers.unified_controller.get_driver"):
        runner = FlashRunner(ctx, goal="g", max_turns=6)
    runner.summarizer = None
    runner._prepare_conversation = AsyncMock(return_value=(MagicMock(), _noise(1), "ui list"))

    async def fake_execute(name, args, tc_id, state, index_elements=None):
        harness.executed.append((name, args))
        result = MagicMock()
        result.status = "success"
        result.text_summary = f"{name} ok"
        result.ui_elements_text = None
        result.screenshot_bytes = harness.post_images[min(len(harness.executed) - 1, 3)]
        result.screenshot_path = None
        result.raw_result = None
        result.metadata = {
            "target_coordinates": args.get("target"),
            "target_semantics": {"target_description": args.get("target_description")},
        }
        return result

    runner.executor = MagicMock()
    runner.executor.action_tool_names = ["click"]
    runner.executor.execute = fake_execute

    original_build_tail = runner._build_tail

    def spy_build_tail(ledger, turns, img_bytes, xml_list, **kwargs):
        tail = original_build_tail(ledger, turns, img_bytes, xml_list, **kwargs)
        harness.tails.append(tail)
        return tail

    runner._build_tail = spy_build_tail

    async def fake_invoke(llm, tools, messages):
        return harness.next_response()

    runner._invoke_model = fake_invoke

    with patch("artemis.agents.flash.runner.get_decision_client", return_value=decision):
        report = await runner.run(MagicMock())

    assert report["status"] == "completed"
    # Exactly one stagnation notice was queued, and none repeated afterwards.
    notices = [
        block["text"]
        for tail in harness.tails
        if isinstance(tail.content, list)
        for block in tail.content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and "Stagnation advisor" in block.get("text", "")
    ]
    assert len(notices) == 1
    assert "Login button" in notices[0]
    assert len(harness.executed) == 2


@pytest.mark.asyncio
async def test_no_decision_client_leaves_loop_untouched():
    """Without a configured client the loop never consults the advisor."""
    harness = _LoopHarness(
        turns_script=[
            [{"name": "click", "args": {"target": [500, 900]}}],
            [{"name": "click", "args": {"target": [500, 900]}}],
        ]
    )
    ctx = Mock(spec=ArtemisContext)
    ctx.llm_config = None
    ctx.data_engine = None
    ctx.step_memory = None
    ctx.transcript_ledger = None
    ctx.execution_setup = None
    ctx.actuator = None

    with patch("artemis.controllers.unified_controller.get_driver"):
        runner = FlashRunner(ctx, goal="g", max_turns=4)
    runner.summarizer = None
    runner._prepare_conversation = AsyncMock(return_value=(MagicMock(), _noise(1), "ui"))

    async def fake_execute(name, args, tc_id, state, index_elements=None):
        harness.executed.append((name, args))
        result = MagicMock()
        result.status = "success"
        result.text_summary = "ok"
        result.ui_elements_text = None
        result.screenshot_bytes = harness.post_images[0]
        result.screenshot_path = None
        result.raw_result = None
        result.metadata = {}
        return result

    runner.executor = MagicMock()
    runner.executor.action_tool_names = ["click"]
    runner.executor.execute = fake_execute

    original_build_tail = runner._build_tail

    def spy_build_tail(ledger, turns, img_bytes, xml_list, **kwargs):
        tail = original_build_tail(ledger, turns, img_bytes, xml_list, **kwargs)
        harness.tails.append(tail)
        return tail

    runner._build_tail = spy_build_tail
    runner._invoke_model = lambda llm, tools, messages: _async_return(harness.next_response())

    with patch("artemis.agents.flash.runner.get_decision_client", return_value=None):
        report = await runner.run(MagicMock())

    assert report["status"] == "completed"
    assert all(
        "Stagnation advisor" not in str(block)
        for tail in harness.tails
        for block in (tail.content if isinstance(tail.content, list) else [tail.content])
    )


async def _async_return(value):
    return value
