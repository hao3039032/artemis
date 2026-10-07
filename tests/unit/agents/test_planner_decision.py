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

"""Decision-model (Clef) path of the async planner validation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.agents.planner import planner
from artemis.agents.planner.planner import run_async_planner_validation
from artemis.llm.decision import DecisionError


class DummyDataEngine:
    def get_agent_friendly_steps(self):
        return []


class DummyCtx:
    data_engine = DummyDataEngine()


class _FakeDecision:
    def __init__(self, p_approved=0.95, concern="none", concern_p=0.9, error=None):
        self.p_approved = p_approved
        self.concern = concern
        self.concern_p = concern_p
        self.error = error
        self.calls = []

    def use_enabled(self, name):
        return True

    async def decide(self, state, questions, images=None, *, decision_point=None):
        self.calls.append({"state": state, "questions": questions, "point": decision_point})
        if self.error is not None:
            raise self.error
        result = MagicMock()
        result.noul = lambda qid, default=None: self.p_approved
        concern_answer = MagicMock()
        concern_answer.chosen = self.concern
        concern_answer.confidence = self.concern_p
        result.choice = lambda qid: concern_answer
        return result


_BEFORE = "- [/] Step one\n- [ ] Step two"
_AFTER = "- [/] Step one\n- [ ] Step two\n- [ ] New unrelated subgoal"


async def _run(client, vlm_result=None):
    llm = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=vlm_result or planner.ValidationResult(is_approved=True, feedback="")
    )
    llm.with_structured_output.return_value = structured
    with patch.object(planner, "get_llm", return_value=llm) as get_llm:
        res = await run_async_planner_validation(
            ctx=DummyCtx(),
            initial_goal="Do the task",
            content_before=_BEFORE,
            content_after=_AFTER,
            operator_raw_thinking="adding a subgoal",
            operator_native_thinking="hmm",
        )
    return res, get_llm


@pytest.mark.asyncio
async def test_approved_decision_short_circuits_llm():
    client = _FakeDecision(p_approved=0.93)
    with patch.object(planner, "get_decision_client", return_value=client):
        res, get_llm = await _run(client)
    assert res == {"status": "success", "feedback": ""}
    assert get_llm.called is False  # LLM judge never invoked
    call = client.calls[0]
    assert call["point"] == "planner_validation"
    assert "Do the task" in call["state"]
    assert set(call["questions"]) == {"is_approved", "concern"}
    assert "goal_drift" in call["questions"]["concern"]["options"]


@pytest.mark.asyncio
async def test_rejected_decision_renders_templated_feedback():
    client = _FakeDecision(p_approved=0.19, concern="goal_drift", concern_p=0.81)
    with patch.object(planner, "get_decision_client", return_value=client):
        res, get_llm = await _run(client)
    assert res["status"] == "failed"
    assert "goal_drift" in res["feedback"]
    assert "(p=0.81)" in res["feedback"]
    assert "advisory" in res["feedback"].lower()
    assert get_llm.called is False


@pytest.mark.asyncio
async def test_boundary_at_half_approves():
    client = _FakeDecision(p_approved=0.50)
    with patch.object(planner, "get_decision_client", return_value=client):
        res, _ = await _run(client)
    assert res["status"] == "success"
    client = _FakeDecision(p_approved=0.49, concern="weakened_checks", concern_p=0.7)
    with patch.object(planner, "get_decision_client", return_value=client):
        res, _ = await _run(client)
    assert res["status"] == "failed"
    assert "weakened_checks" in res["feedback"]


@pytest.mark.asyncio
async def test_decision_error_falls_back_to_llm_judge():
    client = _FakeDecision(error=DecisionError("timeout", kind="timeout"))
    with patch.object(planner, "get_decision_client", return_value=client):
        res, get_llm = await _run(
            client,
            vlm_result=planner.ValidationResult(is_approved=False, feedback="VLM says goal drift."),
        )
    assert res["status"] == "failed"
    assert res["feedback"] == "VLM says goal drift."
    assert get_llm.called is True


@pytest.mark.asyncio
async def test_point_disabled_keeps_llm_path():
    client = _FakeDecision()
    client.use_enabled = lambda name: False
    with patch.object(planner, "get_decision_client", return_value=client):
        res, get_llm = await _run(client)
    assert res["status"] == "success"
    assert get_llm.called is True


@pytest.mark.asyncio
async def test_no_client_keeps_legacy_path():
    with patch.object(planner, "get_decision_client", return_value=None):
        res, get_llm = await _run(None)
    assert res["status"] == "success"
    assert get_llm.called is True
