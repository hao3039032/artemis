# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS BASIS" WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decision-model (Clef) path of the pixel safety net."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.agents.validator import precondition_pixel as pp
from artemis.agents.validator.categories import ValidationErrorCategory
from artemis.llm.decision import DecisionError


class _FakeDecision:
    """Minimal decision-client double for the pixel safety net."""

    def __init__(self, p_present=0.95, situation="unchanged", error=None):
        self.p_present = p_present
        self.situation = situation
        self.error = error
        self.calls = []

    def use_enabled(self, name):
        return True

    async def decide(self, state, questions, images=None, *, decision_point=None):
        self.calls.append(
            {"state": state, "questions": questions, "images": images, "point": decision_point}
        )
        if self.error is not None:
            raise self.error
        return MagicMock(
            noul=lambda qid, default=None: self.p_present,
            choice=lambda qid: MagicMock(chosen=self.situation),
        )


def _item():
    return {
        "action": "tap",
        "coordinates": [500, 900],
        "target_text": "Login",
    }


async def _run_with_client(client, vlm_json=None):
    """Runs one attempt with a decision client; VLM fallback optional."""
    session = MagicMock()
    screenshot = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    session.screenshot_b64 = AsyncMock(return_value=screenshot)
    llm = MagicMock()
    if vlm_json is not None:
        llm.acomplete = None
    ctx = MagicMock()
    ctx.llm_config = None

    vlm_calls = {"n": 0}

    async def fake_acomplete_structured(llm_, messages_):
        vlm_calls["n"] += 1
        if isinstance(vlm_json, Exception):
            raise vlm_json
        return vlm_json or {}

    with (
        patch.object(pp, "acomplete_structured", side_effect=fake_acomplete_structured),
        patch.object(pp.visualization, "crop_and_annotate_target", return_value=b"crop"),
        patch.object(pp, "_init_llm_and_prompt", return_value=(llm, "RULES")),
    ):
        verdict = await pp._run_attempt(
            session,
            llm,
            "RULES",
            b"orig",
            [500, 900],
            _item(),
            None,
            attempt=1,
            decision_client=client,
        )
    return verdict, vlm_calls


@pytest.mark.asyncio
async def test_probability_mapping_boundaries():
    # P >= 0.5 -> present -> pass.
    v, _ = await _run_with_client(_FakeDecision(p_present=0.50))
    assert v[0] is True and v[1] is ValidationErrorCategory.NONE
    # P < 0.5 with verdict-confidence (1-P) below 0.7 -> bypass.
    v, _ = await _run_with_client(_FakeDecision(p_present=0.49))
    assert v[0] is True and v[1] is ValidationErrorCategory.PIXEL_BYPASSED
    # 1-P == 0.70 exactly -> confident absence -> reject.
    v, _ = await _run_with_client(_FakeDecision(p_present=0.30))
    assert v[0] is False and v[1] is ValidationErrorCategory.PIXEL_TARGET_DISAPPEARED
    assert "decision model" in v[2]
    # P well below -> reject with situation in the reason.
    v, _ = await _run_with_client(_FakeDecision(p_present=0.13, situation="shifted"))
    assert v[0] is False and v[1] is ValidationErrorCategory.PIXEL_TARGET_DISAPPEARED
    assert "situation=shifted" in v[2]
    assert "P(present)=0.13" in v[2]


@pytest.mark.asyncio
async def test_decision_state_carries_target_rules_and_images():
    client = _FakeDecision(p_present=0.9)
    await _run_with_client(client)
    call = client.calls[0]
    assert call["point"] == "pixel_safety_net"
    assert "[Target]" in call["state"]
    assert "Kind: specific UI control" in call["state"]
    assert "[Judgment Rules]" in call["state"]
    assert call["images"] == [b"orig", b"crop"]  # reference first, live second
    assert set(call["questions"]) == {"is_present", "situation"}
    assert call["questions"]["is_present"]["type"] == "noul"
    assert "popup_blocked" in call["questions"]["situation"]["options"]


@pytest.mark.asyncio
async def test_decision_error_falls_back_to_vlm_within_attempt():
    client = _FakeDecision(error=DecisionError("boom", kind="timeout"))
    v, vlm = await _run_with_client(
        client, vlm_json={"is_present": True, "confidence": 0.9, "reasoning": "vlm"}
    )
    assert v[0] is True and v[1] is ValidationErrorCategory.NONE
    assert vlm["n"] == 1  # same attempt, VLM took over


@pytest.mark.asyncio
async def test_decision_error_then_vlm_reject_preserves_branches():
    client = _FakeDecision(error=DecisionError("5xx", kind="http"))
    v, _ = await _run_with_client(
        client, vlm_json={"is_present": False, "confidence": 0.9, "reasoning": "gone"}
    )
    assert v[0] is False and v[1] is ValidationErrorCategory.PIXEL_TARGET_DISAPPEARED


@pytest.mark.asyncio
async def test_no_client_runs_vlm_exactly_as_before():
    v, vlm = await _run_with_client(
        None, vlm_json={"is_present": False, "confidence": 0.5, "reasoning": "unsure"}
    )
    assert v[0] is True and v[1] is ValidationErrorCategory.PIXEL_BYPASSED
    assert vlm["n"] == 1


@pytest.mark.asyncio
async def test_entry_point_resolves_disabled_client_to_none():
    """validate_action_precondition_pixel with no decision config stays VLM-only."""
    session = MagicMock()
    session.screenshot_b64 = AsyncMock(
        return_value="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    ctx = MagicMock()
    get_llm_calls = {"n": 0}

    def get_llm_fn(ctx, name=None):
        get_llm_calls["n"] += 1
        return MagicMock()

    vlm_json = {"is_present": True, "confidence": 1.0, "reasoning": ""}

    async def fake_acomplete_structured(llm_, messages_):
        return vlm_json

    with (
        patch.object(pp, "acomplete_structured", side_effect=fake_acomplete_structured),
        patch.object(pp.visualization, "crop_and_annotate_target", return_value=b"crop"),
        patch.object(pp, "_init_llm_and_prompt", return_value=(MagicMock(), "RULES")),
        patch.object(pp, "get_decision_client", return_value=None),
    ):
        verdict = await pp.validate_action_precondition_pixel(
            ctx, session, _item(), "aW1n", get_llm_fn=get_llm_fn
        )
    assert verdict[0] is True and verdict[1] is ValidationErrorCategory.NONE
