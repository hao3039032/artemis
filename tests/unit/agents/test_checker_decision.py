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

"""Decision-model (Clef) verdict path of the Checker."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from artemis.agents.checker import checker
from artemis.agents.checker.checker import (
    CheckReport,
    CheckVerdict,
    _decision_check_report,
    _final_report,
)
from artemis.llm.decision import DecisionAnswer, DecisionResult, DecisionError


def _items():
    return [
        SimpleNamespace(kind="verify", when="on_complete", text="Alarm named Wake Up exists"),
        SimpleNamespace(kind="assert", when="final", text="Battery above 20 percent"),
    ]


def _messages():
    return [
        SystemMessage(content="You are the Checker."),
        HumanMessage(content="audit these items"),
        AIMessage(content="Checking the alarm."),
        ToolMessage(
            tool_call_id="t1",
            name="probe_device",
            content="Wake Up alarm scheduled at 07:00 next Tuesday",
        ),
        AIMessage(content="Checking battery."),
        ToolMessage(tool_call_id="t2", name="probe_device", content="level: 45"),
    ]


class _FakeDecision:
    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error
        self.calls = []

    def use_enabled(self, name):
        return True

    async def decide(self, state, questions, images=None, *, decision_point=None):
        self.calls.append({"state": state, "questions": questions, "point": decision_point})
        if self.error is not None:
            raise self.error
        return DecisionResult(answers=self.answers, model="clef-flash", latency_ms=30.0)


def _result(statuses, probabilities=None):
    answers = {}
    for i, status in enumerate(statuses):
        p = (probabilities or {}).get(i, 0.9)
        answers[f"item_{i}"] = DecisionAnswer(
            question_id=f"item_{i}",
            kind="choice",
            chosen=status,
            distribution={status: p},
        )
    return DecisionResult(answers=answers, model="clef-flash")


@pytest.mark.asyncio
async def test_final_report_uses_decision_model_with_carried_evidence():
    client = _FakeDecision(answers=_result(["passed", "failed"]).answers)
    llm = MagicMock()
    report = await _final_report(llm, _messages(), _items(), client)
    call = client.calls[0]
    assert call["point"] == "checker_verdict"
    assert "Alarm named Wake Up exists" in call["state"]
    assert "Wake Up alarm scheduled" in call["state"]  # probe output carried over
    assert set(call["questions"]) == {"item_0", "item_1"}
    assert call["questions"]["item_0"]["options"] == ["passed", "failed", "inconclusive"]
    assert [v.status for v in report.verdicts] == ["passed", "failed"]
    assert report.verdicts[0].kind == "verify"
    assert report.verdicts[1].kind == "assert"
    # Evidence carried from the conversation, not generated.
    assert "07:00" in report.verdicts[0].evidence
    assert "45" in report.verdicts[1].evidence


@pytest.mark.asyncio
async def test_low_confidence_verdict_degrades_to_inconclusive():
    answers = _result(["failed"], probabilities={0: 0.59}).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items()[:1], _messages())
    assert report.verdicts[0].status == "inconclusive"


@pytest.mark.asyncio
async def test_missing_answer_degrades_to_inconclusive():
    report = _decision_check_report(DecisionResult(answers={}), _items()[:1], _messages())
    assert report.verdicts[0].status == "inconclusive"


@pytest.mark.asyncio
async def test_decision_error_falls_back_to_structured_report():
    client = _FakeDecision(error=DecisionError("timeout", kind="timeout"))
    llm = MagicMock()
    fallback = CheckReport(
        verdicts=[CheckVerdict(item_text="x", kind="verify", status="passed", evidence="e")]
    )
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=fallback)
    llm.with_structured_output.return_value = structured
    report = await _final_report(llm, _messages(), _items()[:1], client)
    assert report.verdicts[0].status == "passed"
    assert structured.ainvoke.called


@pytest.mark.asyncio
async def test_no_client_uses_structured_report():
    llm = MagicMock()
    fallback = CheckReport(verdicts=[])
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=fallback)
    llm.with_structured_output.return_value = structured
    report = await _final_report(llm, _messages(), _items(), None)
    assert report.verdicts == []
    assert structured.ainvoke.called


@pytest.mark.asyncio
async def test_verdicts_allow_release_unchanged_by_decision_report():
    # verify passed + assert failed: assert failures never block release.
    answers = _result(["passed", "failed"]).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items(), _messages())
    assert checker.verdicts_allow_release(report) is True
    # verify failed: release blocked.
    answers = _result(["failed", "passed"]).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items(), _messages())
    assert checker.verdicts_allow_release(report) is False


@pytest.mark.asyncio
async def test_check_loop_routes_final_verdict_through_decision_model(monkeypatch):
    """The loop's report step consults the decision client resolved from ctx."""
    client = _FakeDecision(answers=_result(["passed", "passed"]).answers)
    ctx = MagicMock()
    ctx.execution_setup = None
    monkeypatch.setattr(checker, "get_llm", lambda **kwargs: MagicMock())
    monkeypatch.setattr(checker, "get_decision_client", lambda c: client)

    async def fake_invoke(coro, **kw):
        return AIMessage(content="done", tool_calls=[])

    monkeypatch.setattr(checker, "invoke_llm_with_timeout_message", fake_invoke)
    monkeypatch.setattr(checker, "acomplete", lambda llm, messages: _noop_coro())

    report = await checker._run_check_loop(ctx, _messages()[:2], [], _items())
    assert [v.status for v in report.verdicts] == ["passed", "passed"]
    assert client.calls[0]["point"] == "checker_verdict"


def _noop_coro():
    import asyncio

    async def _c():
        return AIMessage(content="ok", tool_calls=[])

    return asyncio.ensure_future(_c())


@pytest.mark.asyncio
async def test_state_is_tail_bounded():
    huge = HumanMessage(content="x" * 60000)
    text = checker._conversation_evidence([huge])
    assert len(text) <= checker._DECISION_STATE_MAX_CHARS + 100
    assert text.startswith("... [conversation truncated]")


# --- Final-check subgoal questions (release gate) --------------------------------------


def _subgoal_result(item_statuses, subgoal_ps):
    """A decision result with item_<i> choices plus subgoal_<i> noul answers."""
    answers = {}
    for i, status in enumerate(item_statuses):
        answers[f"item_{i}"] = DecisionAnswer(
            question_id=f"item_{i}",
            kind="choice",
            chosen=status,
            distribution={status: 0.9},
        )
    for i, p in enumerate(subgoal_ps):
        answers[f"subgoal_{i}"] = DecisionAnswer(
            question_id=f"subgoal_{i}", kind="noul", probability=p
        )
    return DecisionResult(answers=answers, model="clef-flash")


_SUBGOALS = ["Set the alarm for 7:00", "Verify battery saver is off"]


@pytest.mark.asyncio
async def test_final_subgoal_questions_reach_the_client():
    client = _FakeDecision(answers=_subgoal_result(["passed"], [0.05]).answers)
    llm = MagicMock()
    report = await _final_report(llm, _messages()[:2], _items()[:1], client, _SUBGOALS)
    call = client.calls[0]
    assert set(call["questions"]) == {"item_0", "subgoal_0", "subgoal_1"}
    assert call["questions"]["subgoal_0"]["type"] == "noul"
    assert "Set the alarm for 7:00" in call["questions"]["subgoal_0"]["question"]
    assert "# Completed Plan Subgoals" in call["state"]
    # Low P(unmet): no unmet subgoals, release gate stays open.
    assert report.unmet_subgoals == []


@pytest.mark.asyncio
async def test_high_probability_unmet_subgoal_closes_the_release_gate():
    from artemis.agents.checker.checker import verdicts_allow_release

    answers = _subgoal_result(["passed"], [0.87, 0.05]).answers
    report = _decision_check_report(
        DecisionResult(answers=answers), _items()[:1], _messages(), _SUBGOALS
    )
    assert report.unmet_subgoals == ["Set the alarm for 7:00"]
    # graph.py gate: passed = verdicts_allow_release(report) and not report.unmet_subgoals
    assert (verdicts_allow_release(report) and not report.unmet_subgoals) is False

    answers_ok = _subgoal_result(["passed"], [0.59, 0.05]).answers
    report_ok = _decision_check_report(
        DecisionResult(answers=answers_ok), _items()[:1], _messages(), _SUBGOALS
    )
    assert report_ok.unmet_subgoals == []
    assert (verdicts_allow_release(report_ok) and not report_ok.unmet_subgoals) is True


def test_subgoal_threshold_is_inclusive_at_point_six():
    # P(unmet) == 0.60 exactly -> counted as unmet (conservative boundary).
    answers = _subgoal_result(["passed"], [0.60]).answers
    report = _decision_check_report(
        DecisionResult(answers=answers), _items()[:1], _messages(), _SUBGOALS[:1]
    )
    assert report.unmet_subgoals == _SUBGOALS[:1]


def test_checkpoint_entry_passes_no_subgoals_so_unmet_stays_empty():
    answers = _subgoal_result(["passed"], []).answers
    report = _decision_check_report(
        DecisionResult(answers=answers), _items()[:1], _messages(), None
    )
    assert report.unmet_subgoals == []


@pytest.mark.asyncio
async def test_too_many_questions_fall_back_to_structured_report():
    from artemis.llm.decision import MAX_QUESTIONS

    client = _FakeDecision(answers=_subgoal_result(["passed"], [0.9]).answers)
    llm = MagicMock()
    fallback = CheckReport(verdicts=[])
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=fallback)
    llm.with_structured_output.return_value = structured
    many_items = [
        SimpleNamespace(kind="verify", when="final", text=f"item {i}") for i in range(MAX_QUESTIONS)
    ]
    report = await _final_report(llm, _messages()[:2], many_items, client, ["extra subgoal"])
    assert report is fallback
    assert client.calls == []  # never called: guard fired before decide()


@pytest.mark.asyncio
async def test_placeholder_evidence_failed_verdict_is_downgraded():
    # A conversation with no tool output at all: evidence is the placeholder,
    # so a decision "failed" verdict must downgrade to inconclusive.
    bare = [
        SystemMessage(content="You are the Checker."),
        HumanMessage(content="audit"),
        AIMessage(content="no probes needed"),
    ]
    answers = _result(["failed"]).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items()[:1], bare)
    assert report.verdicts[0].status == "inconclusive"
    assert "downgraded" in report.verdicts[0].evidence


@pytest.mark.asyncio
async def test_real_evidence_failed_verdict_is_kept():
    answers = _result(["failed"]).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items()[:1], _messages())
    assert report.verdicts[0].status == "failed"
    assert "07:00" in report.verdicts[0].evidence


@pytest.mark.asyncio
async def test_unexpected_client_exception_falls_back_to_structured_report():
    # Defense-in-depth: a non-DecisionError from the client must not skip the
    # structured-report fallback.
    client = _FakeDecision()
    client.decide = AsyncMock(side_effect=RuntimeError("unexpected client bug"))
    llm = MagicMock()
    fallback = CheckReport(verdicts=[])
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=fallback)
    llm.with_structured_output.return_value = structured
    report = await _final_report(llm, _messages(), _items()[:1], client)
    assert report is fallback
    assert structured.ainvoke.called


@pytest.mark.asyncio
async def test_verdict_probability_boundary_keeps_verdict_at_point_six():
    # The item-verdict downgrade threshold is inclusive: p=0.60 keeps "failed".
    answers = _result(["failed"], probabilities={0: 0.60}).answers
    report = _decision_check_report(DecisionResult(answers=answers), _items()[:1], _messages())
    assert report.verdicts[0].status == "failed"
