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

"""Flash stagnation advisor (dHash gate + decision-model review, advisory).

The Flash runner historically had no code-level detection for a stuck loop:
the prompt asks the model to notice unchanging screens, but nothing verifies
it. This module adds a conservative advisory layer:

1. **Heuristic gate (zero cost)** — the last two executed turns ran the same
   action on the same target AND their post-action screenshots' 64-bit dHash
   Hamming distance is <= :data:`GATE_MAX_DHASH_DISTANCE` (the same
   perceptual-hash util the DataEngine stamps per step).
2. **Decision-model review** — the gated pair goes to the configured Clef
   endpoint with both screenshots ("did the screen materially change?", "what
   situation is this?"). Only ``materially_changed=no`` combined with
   ``situation ∈ {true_stagnation, overlay_blocking}`` produces a notice.
3. **Advisory only** — the notice is appended to the runner's pending tail
   notices so the model sees it on the next turn; nothing is stopped or
   forced. Every failure mode (client unavailable, call error, parse error)
   is silent: a missed hint is acceptable, a false alarm is not.

Each (action+target) signature is reported at most once per session.
"""

from __future__ import annotations

from dataclasses import dataclass

from artemis.llm.decision import DecisionError, choice_question, noul_question
from artemis.utils.image_hash import dhash_hex, hamming_distance_hex
from artemis.utils.logger import get_logger

logger = get_logger(__name__)

#: Screens whose post-action dHash Hamming distance is at most this are
#: considered "no material change" by the zero-cost gate. Generous on purpose
#: (the decision model still confirms): it only needs to catch same-screen
#: re-captures, which cluster at distance <=4 on-device.
GATE_MAX_DHASH_DISTANCE = 8


@dataclass
class StagnationWindowTurn:
    """One executed turn as the gate sees it."""

    action_signature: str | None
    """Canonical "action(s)+target(s)" string; None when the turn ran no
    (fully successful) device action, which breaks the repetition chain."""

    action_label: str = ""
    """Human-readable "action on target" for the notice text."""

    post_dhash: str | None = None
    post_img_bytes: bytes | None = None


def action_signature(action_dict: dict) -> str:
    """Canonical per-action signature: action verb + target identity + point.

    Target identity prefers the observed element label / the operator's own
    description, then the resource id, then the raw coordinates — whichever
    provenance the recorded action carries.
    """
    action = action_dict.get("action") or "?"
    target = (
        action_dict.get("target_text")
        or action_dict.get("target_description")
        or action_dict.get("target_resource_id")
        or ""
    )
    coords = action_dict.get("coordinates")
    return f"{action}:{target}:{coords}"


def action_label(action_dict: dict) -> str:
    """Human-readable action description for the advisory notice."""
    action = action_dict.get("action") or "action"
    target = (
        action_dict.get("target_text")
        or action_dict.get("target_description")
        or action_dict.get("target_resource_id")
        or action_dict.get("coordinates")
        or "its target"
    )
    return f"{action} on {target}"


def _decision_questions() -> dict:
    return {
        "materially_changed": noul_question(
            "Did the screen materially change between Image 1 (before the"
            " first action) and Image 2 (after the repeated action)? Ignore"
            " negligible rendering noise, blinking cursors and clock ticks."
        ),
        "situation": choice_question(
            "Which situation best describes the pair of screenshots?",
            [
                "transition_in_progress",
                "waiting_appropriate",
                "true_stagnation",
                "overlay_blocking",
            ],
        ),
    }


def _decision_state(action_desc: str) -> str:
    return (
        "An autonomous mobile agent repeated the same action twice"
        f" ({action_desc}) and the screen appears unchanged. Decide whether it"
        " is stuck.\n\n[Images]\nImage 1: screen before the first action.\n"
        "Image 2: screen after the repeated action."
    )


class StagnationAdvisor:
    """Tracks the last executed turns and issues at most one notice per signature."""

    def __init__(self):
        self._previous: StagnationWindowTurn | None = None
        self._last: StagnationWindowTurn | None = None
        self._notified_signatures: set[str] = set()

    def record_turn(self, turn: StagnationWindowTurn | None) -> None:
        """Feeds one executed turn into the sliding window.

        A ``None`` turn (no successful device action this turn) resets the
        repetition chain — the gate compares two consecutive action turns.
        """
        if turn is None or not turn.action_signature:
            self._previous = None
            self._last = None
            return
        self._previous = self._last
        self._last = turn

    def gate_passes(self) -> StagnationWindowTurn | None:
        """The matched turn pair when the zero-cost gate fires, else ``None``."""
        prev, last = self._previous, self._last
        if prev is None or last is None:
            return None
        if not prev.action_signature or prev.action_signature != last.action_signature:
            return None
        distance = hamming_distance_hex(prev.post_dhash, last.post_dhash)
        if distance is None or distance > GATE_MAX_DHASH_DISTANCE:
            return None
        if prev.action_signature in self._notified_signatures:
            return None
        return last

    async def review_and_notify(self, decision_client) -> str | None:
        """Runs the decision review when the gate fires; returns the notice.

        ``None`` means "no notice" for every reason: gate not passed, client
        unusable, call failed, or the model does not consider it stagnation.
        Failures are silent by design (prefer missing a hint over a false one).
        """
        if decision_client is None:
            return None
        triggered = self.gate_passes()
        if triggered is None or self._previous is None:
            return None
        prev, last = self._previous, triggered
        try:
            result = await decision_client.decide(
                state=_decision_state(last.action_label or "the same action"),
                questions=_decision_questions(),
                images=[
                    prev.post_img_bytes,
                    last.post_img_bytes,
                ],
                decision_point="stagnation_detection",
            )
        except DecisionError as e:
            logger.debug(f"Stagnation decision review failed (silent): {e}")
            return None
        except Exception as e:  # defensive: advisory only, never raise
            logger.debug(f"Stagnation decision review error (silent): {e}")
            return None

        p_changed = result.noul("materially_changed", default=1.0)
        situation = result.choice("situation")
        situation_label = situation.chosen if situation and situation.chosen else ""
        if p_changed >= 0.5:
            return None
        if situation_label not in ("true_stagnation", "overlay_blocking"):
            return None

        # Report this signature at most once per session.
        self._notified_signatures.add(last.action_signature or "")
        p_stuck = 1.0 - p_changed
        action_desc = last.action_label or "the same action"
        if situation_label == "overlay_blocking":
            hint = "Consider an alternative path or dismiss the overlay."
        else:
            hint = "Consider an alternative path."
        notice = (
            f"Stagnation advisor: repeated {action_desc} produced no screen"
            f" change (P={p_stuck:.2f}, situation={situation_label}). {hint}"
        )
        logger.info(f"Flash stagnation notice queued: {notice}")
        return notice


def post_turn_snapshot(
    action_records: list[tuple[dict | None, str]],
    post_img_bytes: bytes | None,
) -> StagnationWindowTurn | None:
    """Builds the window turn for one executed turn.

    Takes the turn's ``(action_dict, status)`` pairs. The signature only
    covers fully successful turns: a turn with any failed (or unrecorded)
    action resets the chain — an unchanged screen after a failed action is
    expected, not stagnation.
    """
    if not action_records:
        return None
    action_dicts: list[dict] = []
    for action_dict, status in action_records:
        if status != "success" or not isinstance(action_dict, dict):
            return None
        action_dicts.append(action_dict)
    signature = "|".join(action_signature(d) for d in action_dicts)
    label = ", ".join(action_label(d) for d in action_dicts)
    return StagnationWindowTurn(
        action_signature=signature,
        action_label=label,
        post_dhash=dhash_hex(post_img_bytes),
        post_img_bytes=post_img_bytes,
    )
