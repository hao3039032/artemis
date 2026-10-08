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

"""``ask_decision``: calibrated second-opinion questions for the acting agent.

Exposes the Clef decision layer as one agent tool, bound to both the Flash
runner (ToolDeclaration via ``McpActionExecutor``) and the Pro Operator
(LangChain tool via :data:`ask_decision_wrapper`). The agent describes the
situation in ``context`` and asks 1-8 typed questions (``bool`` / ``choice``
/ ``score``); the current screenshot is attached automatically (exactly one
image — two 1568px screenshots cost ~132k tokens and overflow the endpoint,
the 413 lesson from the shadow evaluation) and the answer comes back as
calibrated probabilities the agent weighs as a second opinion.

Fail-open on every layer: without a usable decision layer the tool is not
declared at all (:func:`ask_decision_available` gates the declaration, the
wrapper and the prompt teaching alike); a failed call returns guidance text
so the turn continues on the agent's own judgment; low probabilities are the
agent's to ignore.

The public argument is ``context`` — not the wire field ``state`` — because
the Pro Operator's LangChain binding reserves the parameter name ``state``
for the injected graph state: a model-supplied ``state`` argument would be
silently overwritten by the injection in
:func:`artemis.tools.tool_wrapper.invoke_tool_with_injection`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from artemis.context import ArtemisContext
from artemis.core.tool_failure import ToolFailure
from artemis.drivers.base import BaseDeviceDriver
from artemis.graph.state import State
from artemis.llm.decision import (
    DecisionError,
    DecisionResult,
    ask_decision_questions,
)
from artemis.services.decision import get_decision_client
from artemis.tools.base import ArtemisTool
from artemis.tools.tool_wrapper import ToolWrapper
from artemis.utils.cython_compat import CyFunctionDetector
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "ASK_DECISION_DESCRIPTION",
    "ASK_DECISION_TOOL_NAME",
    "AskDecisionArgs",
    "AskDecisionTool",
    "ask_decision_available",
    "ask_decision_wrapper",
    "format_decision_answers",
    "get_ask_decision_tool",
    "run_ask_decision",
    "state_screenshot_bytes",
]


ASK_DECISION_TOOL_NAME = "ask_decision"

ASK_DECISION_DESCRIPTION = (
    "[SECOND OPINION] Ask a calibrated decision model about semantic"
    " uncertainty on the current screen: which situation this is, whether a"
    " popup is X or Y, whether a strategy is still worth pursuing. Provide"
    " 'context' (the scene, what you just did, what you are deciding) and"
    " 1-8 typed questions: bool (yes/no probability), choice (one of 2-6"
    " options), score (a value on a low..high integer scale). The current"
    " screenshot is attached automatically (include_screenshot=false to"
    " disable). Returns calibrated probabilities in about a second:"
    " p>=0.75 high confidence, 0.40-0.75 uncertain, <0.40 no signal. NOT for"
    " locating elements (use ask_explorer), reading precise on-screen text,"
    " or comparing two screens."
)

#: Calibration bands every answer carries, so the agent needs no memory of them.
_CALIBRATION_LEGEND = "calibration: p>=0.75 high confidence; 0.40-0.75 uncertain; <0.40 no signal"


class AskDecisionArgs(BaseModel):
    """Arguments of ``ask_decision`` as seen by every calling agent."""

    model_config = {"ignored_types": (CyFunctionDetector,)}

    context: str = Field(
        ...,
        description=(
            "Background for the decision: the scene on screen, what you just"
            " did, and what you are trying to decide."
        ),
    )
    questions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "1-8 typed questions. Each object: id (stable answer key,"
            " [A-Za-z0-9_.-] up to 64 chars), type ('bool' | 'choice' |"
            " 'score'), question (plain language); 'choice' adds options"
            " (2-6 strings); 'score' adds low/high integer bounds"
            " (defaults 1 and 5, high at most 20)."
        ),
    )
    include_screenshot: bool = Field(
        default=True,
        description="Attach the current screenshot to the decision (default true).",
    )


# --------------------------------------------------------------------------- #
# Availability & screenshot plumbing
# --------------------------------------------------------------------------- #


def ask_decision_available(ctx: ArtemisContext | None) -> bool:
    """Whether the decision layer is on with the ``agent_tool`` use switch.

    Gates the tool's declaration (Flash), the Operator wrapper
    (:data:`ask_decision_wrapper`) and the prompt teaching alike, so an
    installation without the decision layer never sees the tool at all.
    """
    if ctx is None:
        return False
    try:
        client = get_decision_client(ctx)
        return client is not None and client.use_enabled("agent_tool")
    except Exception as e:  # availability must never raise on the hot paths
        logger.debug(f"ask_decision availability check failed: {e}")
        return False


def state_screenshot_bytes(state: Any) -> bytes | None:
    """The current screenshot bytes off the graph state, or ``None``.

    ``state.latest_screenshot`` is the same source ``ask_explorer`` resolves
    against — the pre-action screenshot of the current turn, refreshed by
    every device action.
    """
    path = getattr(state, "latest_screenshot", None)
    if not path:
        return None
    try:
        return Path(str(path)).read_bytes()
    except OSError as e:
        logger.debug(f"ask_decision could not read the current screenshot: {e}")
        return None


def _parse_args(raw: Any) -> AskDecisionArgs | ToolFailure:
    """Validates the model-supplied arguments, or a ``ToolFailure`` to show."""
    if isinstance(raw, AskDecisionArgs):
        return raw
    try:
        return AskDecisionArgs.model_validate(raw or {})
    except ValidationError as e:
        return ToolFailure(f"ask_decision argument error: {e}")


# --------------------------------------------------------------------------- #
# Answer formatting
# --------------------------------------------------------------------------- #


def format_decision_answers(questions: dict[str, dict[str, Any]], result: DecisionResult) -> str:
    """Renders one answer line per question plus the calibration legend.

    Compact by design — this is tool-result text the agent reads inline:

    - ``bool``:    ``- is_settlement: YES p=0.97``
    - ``choice``:  ``- popup_kind: ad p=0.91 {ad: 0.91, reward: 0.06, ...}``
    - ``score``:   ``- retry_value: 3.4 (1-5)``
    """
    lines: list[str] = []
    for qid, spec in questions.items():
        answer = result.answers.get(qid)
        if answer is None:  # pragma: no cover - decide() fails on missing answers
            continue
        if answer.kind == "choice" and answer.chosen is not None:
            distribution = ", ".join(
                f"{option}: {p:.2f}"
                for option, p in sorted(
                    answer.distribution.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            lines.append(f"- {qid}: {answer.chosen} p={answer.confidence:.2f} {{{distribution}}}")
        elif answer.kind == "score" and answer.score is not None:
            low, high = spec.get("low", 1), spec.get("high", 10)
            value = f"{answer.score:.1f}".rstrip("0").rstrip(".")
            lines.append(f"- {qid}: {value} ({low}-{high})")
        elif answer.probability is not None:
            verdict = "YES" if answer.probability >= 0.5 else "NO"
            lines.append(f"- {qid}: {verdict} p={answer.probability:.2f}")
        else:  # pragma: no cover - defensive: every kind carries a value
            lines.append(f"- {qid}: (no answer)")
    footer = _CALIBRATION_LEGEND
    if result.latency_ms is not None:
        footer = f"answered in {(result.latency_ms or 0.0) / 1000.0:.1f}s; {footer}"
    lines.append(f"({footer})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_ask_decision(
    ctx: ArtemisContext | None,
    args: AskDecisionArgs | dict[str, Any],
    screenshot_bytes: bytes | None = None,
) -> str:
    """Runs one agent-authored decision request; never raises.

    ``screenshot_bytes`` is the single optional screenshot the caller resolved
    (the current turn's pre-action screen); the decision request carries it
    only while the call asks for it — never more than one image.

    Returns :class:`ToolFailure` only for malformed arguments and rejected
    questions (the agent can fix and retry); every runtime failure — client
    unavailable, endpoint error — returns guidance text so the turn continues
    on the agent's own judgment (fail-open).
    """
    parsed = _parse_args(args)
    if isinstance(parsed, ToolFailure):
        return parsed
    try:
        questions = ask_decision_questions(parsed.questions)
    except DecisionError as e:
        return ToolFailure(f"ask_decision rejected the questions: {e}")

    client = None
    agent_tool_enabled = False
    try:
        client = get_decision_client(ctx) if ctx is not None else None
        agent_tool_enabled = client is not None and client.use_enabled("agent_tool")
    except Exception as e:
        logger.warning(f"ask_decision client resolution failed: {e}")
    if client is None:
        return (
            "ask_decision unavailable: the decision model is not enabled;"
            " judge the situation yourself."
        )
    if not agent_tool_enabled:
        return "ask_decision is disabled for this installation (use.agent_tool=false)."

    images = [screenshot_bytes] if (parsed.include_screenshot and screenshot_bytes) else None
    try:
        result = await client.decide(parsed.context, questions, images, decision_point="agent_tool")
    except DecisionError as e:
        return f"ask_decision failed ({e.kind}): {e}; proceed without it."
    except Exception as e:  # defensive: a helper tool must never break the loop
        logger.warning(f"ask_decision unexpected failure: {e}")
        return f"ask_decision failed: {e}; proceed without it."

    text = format_decision_answers(questions, result)
    if parsed.include_screenshot and not screenshot_bytes:
        text += "\n(no screenshot was available; answered from 'context' alone)"
    return text


# --------------------------------------------------------------------------- #
# Tool bindings (Pro Operator: LangChain; Flash: validator declaration)
# --------------------------------------------------------------------------- #


class AskDecisionTool(ArtemisTool):
    """``ask_decision`` as an :class:`ArtemisTool` (LangChain / MCP export)."""

    def __init__(self):
        super().__init__(
            name=ASK_DECISION_TOOL_NAME,
            description=ASK_DECISION_DESCRIPTION,
            args_schema=AskDecisionArgs,
            category="custom",
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def execute(
        self,
        driver: BaseDeviceDriver | None = None,  # pylint: disable=unused-argument
        ctx: ArtemisContext | None = None,
        state: State | None = None,
        **kwargs: Any,
    ) -> str:
        raw = {k: v for k, v in kwargs.items() if k in AskDecisionArgs.model_fields}
        parsed = _parse_args(raw)
        if isinstance(parsed, ToolFailure):
            return parsed
        screenshot = state_screenshot_bytes(state) if parsed.include_screenshot else None
        return await run_ask_decision(ctx, parsed, screenshot)


ask_decision = AskDecisionTool()


def get_ask_decision_tool(ctx: ArtemisContext) -> BaseTool:
    """Exports ``ask_decision`` as a LangChain tool bound to ``ctx``."""
    return AskDecisionTool().to_langchain_tool(ctx)


ask_decision_wrapper = ToolWrapper(
    tool_fn_getter=get_ask_decision_tool,
    on_success_fn=lambda output: output,
    on_failure_fn=lambda error: f"ask_decision failed: {error}",
    is_available_fn=ask_decision_available,
)
