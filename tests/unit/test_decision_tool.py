# Copyright 2026 Google LLC
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

"""The ``ask_decision`` agent tool: validation, fail-open, formatting, wiring.

Covers the tool contract end to end: the question normalization
(:func:`artemis.llm.decision.ask_decision_questions`), the fail-open behavior
of :func:`artemis.tools.decision_tool.run_ask_decision` (unavailable client,
HTTP failure, malformed arguments), the single-screenshot image path against a
mocked transport, the answer formatting, the ``decision_call`` telemetry, and
the Flash / Pro-Operator wiring gates (tool declaration, wrapper availability,
prompt teaching).
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from jinja2 import Template

from artemis.agents.flash.runner import FlashRunner
from artemis.agents.operator.prompts import (
    OPERATOR_PROMPT_TOOLSET,
    apply_operator_prompt_contract,
    load_operator_prompts,
    resolve_operator_prompt_tools,
)
from artemis.config import DecisionModelConfig, DecisionModelUseConfig
from artemis.context import ArtemisContext
from artemis.core.tool_failure import ToolFailure, is_tool_failure
from artemis.llm.decision import (
    DecisionClient,
    DecisionError,
    FakeDecisionClient,
    ask_decision_questions,
)
from artemis.mcp.action_executor import AGENT_TOOL_NAMES, McpActionExecutor
from artemis.mcp.action_manifest import BACKEND_INDEPENDENT_TOOLS
from artemis.tools.decision_tool import (
    AskDecisionArgs,
    ask_decision_available,
    ask_decision_wrapper,
    get_ask_decision_tool,
    run_ask_decision,
    state_screenshot_bytes,
)
from third_party.mobile_use.context import DeviceContext


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tiny_jpeg(color=(200, 60, 60)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 160), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _ctx_with_decision(client) -> ArtemisContext:
    """A real context carrying a pre-resolved decision client."""
    ctx = ArtemisContext(device=DeviceContext())
    ctx.decision_client = client
    return ctx


def _ctx_with_config(agent_tool: bool = True) -> ArtemisContext:
    """A context whose agent config enables the decision layer."""
    cfg = DecisionModelConfig(
        enabled=True,
        provider="custom",
        base_url="http://decide.test/decide",
        use=DecisionModelUseConfig(agent_tool=agent_tool),
    )
    ctx = ArtemisContext(device=DeviceContext())
    ctx.agent_config = SimpleNamespace(decision_model=cfg)
    return ctx


_QUESTIONS_ARG = [
    {"id": "is_settlement", "type": "bool", "question": "Is this the settlement screen?"},
    {
        "id": "popup_kind",
        "type": "choice",
        "question": "What kind of popup is this?",
        "options": ["revive_reward", "ad", "system_dialog"],
    },
    {"id": "retry_value", "type": "score", "question": "Retry value?", "low": 1, "high": 5},
]


# --------------------------------------------------------------------------- #
# Question normalization
# --------------------------------------------------------------------------- #


def test_ask_decision_questions_normalizes_every_type():
    questions = ask_decision_questions(_QUESTIONS_ARG)
    assert set(questions) == {"is_settlement", "popup_kind", "retry_value"}
    assert questions["is_settlement"] == {
        "type": "noul",
        "question": "Is this the settlement screen?",
    }
    assert questions["popup_kind"]["type"] == "choice"
    assert questions["popup_kind"]["options"] == ["revive_reward", "ad", "system_dialog"]
    assert questions["retry_value"] == {
        "type": "score",
        "question": "Retry value?",
        "low": 1,
        "high": 5,
    }


def test_ask_decision_questions_accepts_eight_and_defaults_score_bounds():
    eight = [{"id": f"q{i}", "type": "bool", "question": f"Q{i}?"} for i in range(8)]
    assert len(ask_decision_questions(eight)) == 8
    score = ask_decision_questions([{"id": "s", "type": "score", "question": "How much?"}])
    assert score["s"]["low"] == 1 and score["s"]["high"] == 5


@pytest.mark.parametrize(
    "raw,match",
    [
        ([], "1-8"),
        (
            [{"id": f"q{i}", "type": "bool", "question": "Q?"} for i in range(9)],
            "1-8",
        ),
        (["not a dict"], "object"),
        ([{"type": "bool", "question": "Q?"}], "id"),
        ([{"id": "bad id!", "type": "bool", "question": "Q?"}], "id"),
        ([{"id": "x" * 65, "type": "bool", "question": "Q?"}], "id"),
        (
            [
                {"id": "q", "type": "bool", "question": "Q?"},
                {"id": "q", "type": "bool", "question": "Q?"},
            ],
            "duplicate",
        ),
        ([{"id": "q", "type": "bool"}], "question"),
        ([{"id": "q", "type": "maybe", "question": "Q?"}], "type"),
        ([{"id": "q", "type": "choice", "question": "Q?", "options": ["only"]}], "options"),
        (
            [
                {
                    "id": "q",
                    "type": "choice",
                    "question": "Q?",
                    "options": [str(i) for i in range(7)],
                }
            ],
            "options",
        ),
        ([{"id": "q", "type": "score", "question": "Q?", "low": 5, "high": 5}], "low"),
        ([{"id": "q", "type": "score", "question": "Q?", "low": 2, "high": 1}], "low"),
        ([{"id": "q", "type": "score", "question": "Q?", "high": 21}], "high"),
        ([{"id": "q", "type": "score", "question": "Q?", "low": "1", "high": "5"}], "bounds"),
        ("not-a-list", "list"),
    ],
)
def test_ask_decision_questions_rejects_invalid_input(raw, match):
    with pytest.raises(DecisionError) as excinfo:
        ask_decision_questions(raw)
    assert excinfo.value.kind == "protocol"
    assert match in str(excinfo.value)


def test_score_bounds_accept_full_legal_range():
    ok = ask_decision_questions(
        [{"id": "s", "type": "score", "question": "Q?", "low": 0, "high": 20}]
    )
    assert ok["s"]["low"] == 0 and ok["s"]["high"] == 20


# --------------------------------------------------------------------------- #
# run_ask_decision: fail-open paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unavailable_client_returns_guidance_not_an_error():
    ctx = ArtemisContext(device=DeviceContext())  # no decision config at all
    out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None)
    assert is_tool_failure(out) is False
    assert "unavailable" in out
    assert "judge the situation yourself" in out


@pytest.mark.asyncio
async def test_agent_tool_switch_off_returns_disabled_text():
    client = SimpleNamespace(
        decide=AsyncMockDecide(),
        use_enabled=lambda point: point != "agent_tool",
    )
    ctx = _ctx_with_decision(client)
    out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None)
    assert "disabled" in out and "agent_tool" in out


@pytest.mark.asyncio
async def test_malformed_args_return_tool_failure():
    ctx = _ctx_with_decision(FakeDecisionClient())
    out = await run_ask_decision(ctx, {"questions": _QUESTIONS_ARG[:1]}, None)
    assert isinstance(out, ToolFailure)
    assert "argument error" in out


@pytest.mark.asyncio
async def test_rejected_questions_return_tool_failure():
    ctx = _ctx_with_decision(FakeDecisionClient())
    args = {"context": "s", "questions": [{"id": "q", "type": "maybe", "question": "Q?"}]}
    out = await run_ask_decision(ctx, args, None)
    assert isinstance(out, ToolFailure)
    assert "rejected the questions" in out
    assert "use 'bool'" in out  # actionable: names the legal types


@pytest.mark.asyncio
async def test_http_failure_returns_proceed_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="payload too large")

    client = DecisionClient(
        DecisionModelConfig(enabled=True, provider="custom", base_url="http://decide.test/decide")
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = _ctx_with_decision(client)
    try:
        out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None)
    finally:
        await client._http.aclose()
    assert is_tool_failure(out) is False  # fail-open: the turn continues
    assert "ask_decision failed" in out
    assert "proceed without it" in out


@pytest.mark.asyncio
async def test_unexpected_exception_is_caught():
    class Boom:
        def use_enabled(self, point):
            return True

        async def decide(self, *a, **kw):
            raise RuntimeError("boom")

    ctx = _ctx_with_decision(Boom())
    out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None)
    assert "ask_decision failed" in out and "proceed without it" in out


class AsyncMockDecide:
    """decide() double that records the call; never reached in the off-switch test."""

    async def __call__(self, *args, **kwargs):
        raise AssertionError("decide() must not be called")


# --------------------------------------------------------------------------- #
# run_ask_decision: request shape & formatting
# --------------------------------------------------------------------------- #


def _ok_handler():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        answers = {
            "is_settlement": {"p": 0.97},
            "popup_kind": {
                "probabilities": {"revive_reward": 0.91, "ad": 0.06, "system_dialog": 0.03}
            },
            "retry_value": {"score": 3.4},
        }
        return httpx.Response(200, json={"answers": answers})

    return handler, requests


def _cloudflare_client(monkeypatch) -> DecisionClient:
    """A cloudflare-provider client with settings-shaped credentials stubbed."""
    from pydantic import SecretStr

    from artemis.config import settings as artemis_settings

    monkeypatch.setattr(artemis_settings, "CLOUDFLARE_ACCOUNT_ID", "acct123", raising=False)
    monkeypatch.setattr(artemis_settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    return DecisionClient(DecisionModelConfig(enabled=True, model="clef-flash"))


@pytest.mark.asyncio
async def test_single_screenshot_rides_the_request_as_one_data_uri(monkeypatch):
    handler, requests = _ok_handler()
    client = _cloudflare_client(monkeypatch)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = _ctx_with_decision(client)
    try:
        out = await run_ask_decision(
            ctx, {"context": "scene", "questions": _QUESTIONS_ARG}, _tiny_jpeg()
        )
    finally:
        await client._http.aclose()

    body = json.loads(requests[0].read())
    images = body.get("images") or []
    # Exactly one image, embedded as a base64 data URI (the 413/422 lesson:
    # Workers AI rejects bare base64, and two images overflow the context).
    assert len(images) == 1
    assert images[0].startswith("data:image/jpeg;base64,")
    assert "- is_settlement: YES p=0.97" in out
    assert "- retry_value: 3.4 (1-5)" in out


@pytest.mark.asyncio
async def test_include_screenshot_false_sends_a_text_only_request():
    handler, requests = _ok_handler()
    client = DecisionClient(
        DecisionModelConfig(enabled=True, provider="custom", base_url="http://decide.test/decide")
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = _ctx_with_decision(client)
    args = {
        "context": "scene",
        "questions": _QUESTIONS_ARG,
        "include_screenshot": False,
    }
    try:
        await run_ask_decision(ctx, args, _tiny_jpeg())
    finally:
        await client._http.aclose()
    assert "images" not in json.loads(requests[0].read())


@pytest.mark.asyncio
async def test_missing_screenshot_is_answered_text_only_with_a_note():
    client = FakeDecisionClient(scripted={"agent_tool": {"is_settlement": 0.9}})
    ctx = _ctx_with_decision(client)
    out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None)
    assert "- is_settlement: YES p=0.90" in out
    assert "no screenshot was available" in out


@pytest.mark.asyncio
async def test_answers_format_bool_choice_and_score():
    client = FakeDecisionClient(
        scripted={
            "agent_tool": {
                "is_settlement": 0.97,
                "popup_kind": {"revive_reward": 0.91, "ad": 0.06, "system_dialog": 0.03},
                "retry_value": 3.4,
            }
        }
    )
    ctx = _ctx_with_decision(client)
    out = await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG}, None)

    assert "- is_settlement: YES p=0.97" in out
    # choice: chosen option, its probability, and the full distribution.
    assert "- popup_kind: revive_reward p=0.91" in out
    assert "{revive_reward: 0.91, ad: 0.06, system_dialog: 0.03}" in out
    assert "- retry_value: 3.4 (1-5)" in out
    # The calibration legend rides along.
    assert "p>=0.75 high confidence" in out

    # A bool below 0.5 reads NO.
    client2 = FakeDecisionClient(scripted={"agent_tool": {"is_settlement": 0.02}})
    out2 = await run_ask_decision(
        _ctx_with_decision(client2), {"context": "s", "questions": _QUESTIONS_ARG[:1]}, None
    )
    assert "- is_settlement: NO p=0.02" in out2


@pytest.mark.asyncio
async def test_integer_score_renders_without_decimal():
    client = FakeDecisionClient(scripted={"agent_tool": {"retry_value": 4}})
    ctx = _ctx_with_decision(client)
    out = await run_ask_decision(ctx, {"context": "s", "questions": [_QUESTIONS_ARG[2]]}, None)
    assert "- retry_value: 4 (1-5)" in out


@pytest.mark.asyncio
async def test_decision_call_telemetry_labels_the_agent_tool_point(monkeypatch):
    from artemis.services import llm as llm_service

    events: list[tuple] = []
    monkeypatch.setattr(
        llm_service,
        "_record_llm_event",
        lambda name, payload, status=None: events.append((name, payload, status)),
        raising=False,
    )
    handler, _ = _ok_handler()
    client = DecisionClient(
        DecisionModelConfig(enabled=True, provider="custom", base_url="http://decide.test/decide")
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = _ctx_with_decision(client)
    try:
        await run_ask_decision(ctx, {"context": "s", "questions": _QUESTIONS_ARG}, None)
    finally:
        await client._http.aclose()

    decision_events = [e for e in events if e[0] == "decision_call"]
    assert decision_events, "decision_call telemetry must be recorded"
    name, payload, status = decision_events[0]
    assert payload["decision_point"] == "agent_tool"
    assert status == "success"
    assert payload["answers"]["is_settlement"] == 0.97


# --------------------------------------------------------------------------- #
# Screenshot plumbing
# --------------------------------------------------------------------------- #


def test_state_screenshot_bytes_reads_the_latest_screenshot(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(_tiny_jpeg())
    state = SimpleNamespace(latest_screenshot=str(shot))
    assert state_screenshot_bytes(state) == shot.read_bytes()

    assert state_screenshot_bytes(SimpleNamespace(latest_screenshot=None)) is None
    assert state_screenshot_bytes(SimpleNamespace(latest_screenshot="")) is None
    missing = SimpleNamespace(latest_screenshot=str(tmp_path / "gone.jpg"))
    assert state_screenshot_bytes(missing) is None


# --------------------------------------------------------------------------- #
# Wiring: availability gates (Flash declaration, Operator wrapper, prompt)
# --------------------------------------------------------------------------- #


def test_availability_follows_the_decision_layer():
    assert ask_decision_available(None) is False
    assert ask_decision_available(ArtemisContext(device=DeviceContext())) is False
    assert ask_decision_available(_ctx_with_decision(FakeDecisionClient())) is True
    # use.agent_tool=false alone turns the tool off (master switch stays on).
    assert ask_decision_available(_ctx_with_config(agent_tool=False)) is False


def test_agent_tool_defaults_on_in_the_use_config():
    from artemis.config import DecisionModelUseConfig

    assert DecisionModelUseConfig().agent_tool is True
    assert "agent_tool" in DecisionModelUseConfig.model_fields


def test_flash_declares_ask_decision_only_with_the_decision_layer():
    def make_runner(ctx):
        with patch("artemis.controllers.unified_controller.get_driver"):
            return FlashRunner(ctx, goal="g")

    enabled = make_runner(_ctx_with_config())
    names = [t.name for t in enabled._get_tools()]
    assert "ask_decision" in names
    assert names[-1] == "report_task_status"  # sentinel stays last-bound

    disabled = make_runner(ArtemisContext(device=DeviceContext()))
    names_off = [t.name for t in disabled._get_tools()]
    assert "ask_decision" not in names_off

    # The prompt teaching follows the declaration (same gate).
    assert "ask_decision" in enabled._render_system_prompt(enabled._get_tools())
    assert "ask_decision" not in disabled._render_system_prompt(disabled._get_tools())


def test_flash_declaration_schema_matches_the_contract():
    from artemis.agents.validator.tool_declarations import ASK_DECISION_TOOL

    assert ASK_DECISION_TOOL.name == "ask_decision"
    params = ASK_DECISION_TOOL.parameters
    assert set(params["required"]) == {"context", "questions"}
    items = params["properties"]["questions"]["items"]
    assert items["properties"]["type"]["enum"] == ["bool", "choice", "score"]
    assert params["properties"]["questions"]["maxItems"] == 8
    assert params["properties"]["include_screenshot"]["type"] == "boolean"


def test_ask_decision_is_an_agent_side_backend_independent_tool():
    assert "ask_decision" in AGENT_TOOL_NAMES
    assert "ask_decision" in BACKEND_INDEPENDENT_TOOLS


def test_operator_wrapper_gates_on_availability():
    from artemis.tools.index import get_tools_from_wrappers

    ctx_on = _ctx_with_decision(FakeDecisionClient())
    tools_on = get_tools_from_wrappers(ctx_on, [ask_decision_wrapper])
    assert [t.name for t in tools_on] == ["ask_decision"]

    ctx_off = ArtemisContext(device=DeviceContext())
    assert get_tools_from_wrappers(ctx_off, [ask_decision_wrapper]) == []


def test_operator_prompt_advertises_ask_decision_with_the_decision_layer():
    ctx_on = _ctx_with_decision(FakeDecisionClient())
    assert "ask_decision" in resolve_operator_prompt_tools(ctx_on)

    ctx_off = ArtemisContext(device=DeviceContext())
    assert "ask_decision" not in resolve_operator_prompt_tools(ctx_off)

    template = load_operator_prompts()["main_template"]
    with_tool = apply_operator_prompt_contract(
        template, available_tools=resolve_operator_prompt_tools(ctx_on)
    )
    without_tool = apply_operator_prompt_contract(
        template, available_tools=resolve_operator_prompt_tools(ctx_off)
    )
    assert "Calibrated Second Opinion" in with_tool
    assert "ask_decision" not in without_tool
    # The full-set render (every known tool present) keeps the teaching.
    assert "Calibrated Second Opinion" in apply_operator_prompt_contract(template)
    assert "ask_decision" in OPERATOR_PROMPT_TOOLSET


def test_flash_prompt_off_render_keeps_the_prompt_bytes_stable():
    """Decision layer off: the Flash prompt keeps its pre-tool bytes.

    The ``ask_decision`` teaching sits between the "subsequent turns." line
    and the ``---`` section rule; with the tool absent the render must be
    byte-for-byte what it was before the tool existed (one blank line, not
    two). This pins the template's ``{% endif %}`` spacing contract.
    """
    template = (
        Path(__file__).resolve().parents[2] / "artemis" / "agents" / "flash" / "flash_runner.md"
    )
    rendered = Template(template.read_text(encoding="utf-8")).render(
        goal="g", available_tools=frozenset()
    )
    # Scoped to the "subsequent turns." line: section 3 legitimately renders
    # wider blank gaps from its own gated lines, on HEAD alike.
    assert "subsequent turns.\n\n---" in rendered
    assert "subsequent turns.\n\n\n" not in rendered


def test_flash_prompt_on_render_keeps_one_blank_line_before_the_rule():
    """Decision layer on: the teaching is present and still reads as prose."""
    template = (
        Path(__file__).resolve().parents[2] / "artemis" / "agents" / "flash" / "flash_runner.md"
    )
    rendered = Template(template.read_text(encoding="utf-8")).render(
        goal="g", available_tools=frozenset({"ask_decision"})
    )
    assert "Semantic Uncertainty (`ask_decision`)" in rendered
    assert "two screens.\n\n---" in rendered
    assert "two screens.\n\n\n" not in rendered


def test_ask_decision_defers_like_the_other_ask_tools():
    """A mixed [ask_decision, click] turn must not run click before the answer."""
    from artemis.agents.operator.operator import DEFERRING_TOOLS

    assert "ask_decision" in DEFERRING_TOOLS
    assert "ask_explorer" in DEFERRING_TOOLS


@pytest.mark.asyncio
async def test_graph_mounts_ask_decision_only_with_the_decision_layer():
    """The Operator node binds ask_decision iff the decision layer is on."""
    from artemis.context import DevicePlatform, ExecutionSetup
    from artemis.graph.graph import get_graph

    device = DeviceContext(
        host_platform="LINUX",
        mobile_platform=DevicePlatform.ANDROID,
        device_id="dummy",
        device_width=1080,
        device_height=2400,
    )
    ctx_on = ArtemisContext(
        device=device,
        execution_setup=ExecutionSetup(
            decision_model=DecisionModelConfig(
                enabled=True, provider="custom", base_url="http://decide.test/decide"
            )
        ),
    )
    graph_on = await get_graph(ctx_on)
    names_on = [t.name for t in graph_on.nodes["operator"].bound.afunc.tools]
    assert "ask_decision" in names_on

    ctx_off = ArtemisContext(
        device=device,
        execution_setup=ExecutionSetup(decision_model=None),
    )
    graph_off = await get_graph(ctx_off)
    names_off = [t.name for t in graph_off.nodes["operator"].bound.afunc.tools]
    assert "ask_decision" not in names_off


@pytest.mark.asyncio
async def test_executor_routes_ask_decision_with_the_current_screenshot(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(_tiny_jpeg())
    fake = FakeDecisionClient(scripted={"agent_tool": {"is_settlement": 0.97}})
    ctx = _ctx_with_decision(fake)
    executor = McpActionExecutor(ctx, actuator=Mock(), agent_name="flash")
    state = Mock()
    state.latest_screenshot = str(shot)

    result = await executor.execute(
        "ask_decision",
        {"context": "post-battle screen", "questions": [_QUESTIONS_ARG[0]]},
        "tc1",
        state,
    )

    assert result.status == "success"
    assert "- is_settlement: YES p=0.97" in result.text_summary
    call = fake.calls[0]
    assert call["decision_point"] == "agent_tool"
    assert call["images"] == [shot.read_bytes()]  # exactly one, this turn's screen

    # A rejected question set surfaces as an error result the agent can fix.
    bad = await executor.execute(
        "ask_decision",
        {"context": "x", "questions": [{"id": "q", "type": "maybe", "question": "Q?"}]},
        "tc2",
        state,
    )
    assert bad.status == "error"
    assert "rejected the questions" in bad.text_summary


@pytest.mark.asyncio
async def test_executor_skips_screenshot_when_disabled(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(_tiny_jpeg())
    fake = FakeDecisionClient(scripted={"agent_tool": {"is_settlement": 0.9}})
    ctx = _ctx_with_decision(fake)
    executor = McpActionExecutor(ctx, actuator=Mock(), agent_name="flash")
    state = Mock()
    state.latest_screenshot = str(shot)

    await executor.execute(
        "ask_decision",
        {
            "context": "x",
            "questions": [_QUESTIONS_ARG[0]],
            "include_screenshot": False,
        },
        "tc1",
        state,
    )
    assert fake.calls[0]["images"] == []


@pytest.mark.asyncio
async def test_operator_langchain_tool_injects_state_screenshot(tmp_path):
    from artemis.tools.tool_wrapper import invoke_tool_with_injection

    shot = tmp_path / "shot.jpg"
    shot.write_bytes(_tiny_jpeg())
    fake = FakeDecisionClient(scripted={"agent_tool": {"is_settlement": 0.9}})
    ctx = _ctx_with_decision(fake)
    tool = get_ask_decision_tool(ctx)

    state: Any = SimpleNamespace(latest_screenshot=str(shot))
    result = await invoke_tool_with_injection(
        tool=tool,
        args={"context": "scene", "questions": [_QUESTIONS_ARG[0]]},
        tool_call_id="tc1",
        state=state,
    )
    assert "- is_settlement: YES p=0.90" in result
    assert fake.calls[0]["images"] == [shot.read_bytes()]
    # The model-supplied 'context' argument survives the state injection.
    assert fake.calls[0]["state"] == "scene"
