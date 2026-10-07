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

"""Flash stagnation advisor: heuristic gate, decision review, dedupe."""

import io

import pytest

from artemis.agents.flash.stagnation import (
    GATE_MAX_DHASH_DISTANCE,
    StagnationAdvisor,
    StagnationWindowTurn,
    post_turn_snapshot,
)
from artemis.utils.image_hash import dhash_hex


def _action(action="click", target="Login button", coords=(500, 900)):
    return {
        "action": action,
        "target_text": target,
        "coordinates": list(coords),
    }


def _noise_bytes(seed: int, size=(200, 100)) -> bytes:
    """Deterministic noise image: hashes differ across seeds."""
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


def _turn(target="Login button", seed=1, action="click"):
    return post_turn_snapshot([(_action(action, target), "success")], _noise_bytes(seed))


def _hashes(turn):
    return turn.post_dhash


class _FakeDecision:
    def __init__(self, p_changed=0.05, situation="true_stagnation", error=None):
        self.p_changed = p_changed
        self.situation = situation
        self.error = error
        self.calls = []

    def use_enabled(self, name):
        return True

    async def decide(self, state, questions, images=None, *, decision_point=None):
        self.calls.append({"state": state, "questions": questions, "images": images})
        if self.error is not None:
            raise self.error

        class _A:
            def __init__(self, chosen, p):
                self.chosen = chosen
                self.confidence = p

        return type(
            "R",
            (),
            {
                "noul": lambda self, qid, default=None: self.p_changed,  # noqa: ARG005
                "choice": lambda self, qid: _A(self.situation, 0.9),  # noqa: ARG005
                "p_changed": self.p_changed,
                "situation": self.situation,
            },
        )()


@pytest.mark.asyncio
async def test_gate_requires_same_action_and_unchanged_screen():
    advisor = StagnationAdvisor()
    client = _FakeDecision()

    # Same action, same screen -> gate passes, decision consulted.
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    notice = await advisor.review_and_notify(client)
    assert notice is not None
    assert "Login button" in notice
    assert "(P=0.95" in notice
    assert client.calls[0]["images"]  # two screenshots attached

    # Different target -> gate never passes.
    advisor2 = StagnationAdvisor()
    advisor2.record_turn(_turn(target="Menu", seed=1))
    advisor2.record_turn(_turn(target="Search", seed=1))
    assert await advisor2.review_and_notify(client) is None
    assert len(client.calls) == 1  # no new decision call


@pytest.mark.asyncio
async def test_gate_blocks_when_screen_materially_changed():
    advisor = StagnationAdvisor()
    client = _FakeDecision()
    t1 = _turn(seed=1)
    t2 = _turn(seed=99)
    distance = bin(int(t1.post_dhash, 16) ^ int(t2.post_dhash, 16)).count("1")
    assert distance > GATE_MAX_DHASH_DISTANCE  # fixture check: truly different
    advisor.record_turn(t1)
    advisor.record_turn(t2)
    assert await advisor.review_and_notify(client) is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_gate_boundary_distance():
    # Build a pair of images with dHash distance <= GATE_MAX_DHASH_DISTANCE.
    advisor = StagnationAdvisor()
    client = _FakeDecision()
    t1 = _turn(seed=7)
    # A near-identical screenshot: the same noise with a small overlay patch.
    from PIL import Image

    buf = io.BytesIO()
    img = Image.open(io.BytesIO(_noise_bytes(7))).convert("RGB")
    for y in range(0, 6):
        for x in range(0, 6):
            img.putpixel((x, y), (250, 250, 250))
    img.save(buf, format="JPEG", quality=85)
    t2 = post_turn_snapshot([(_action(), "success")], buf.getvalue())
    distance = bin(int(t1.post_dhash, 16) ^ int(t2.post_dhash, 16)).count("1")
    assert distance <= GATE_MAX_DHASH_DISTANCE, f"fixture too far: {distance}"
    advisor.record_turn(t1)
    advisor.record_turn(t2)
    assert await advisor.review_and_notify(client) is not None


@pytest.mark.asyncio
async def test_dedupe_once_per_signature():
    advisor = StagnationAdvisor()
    client = _FakeDecision()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    first = await advisor.review_and_notify(client)
    assert first is not None
    # A third identical turn: same signature, already notified -> silent.
    advisor.record_turn(_turn())
    assert await advisor.review_and_notify(client) is None


@pytest.mark.asyncio
async def test_failed_action_resets_chain():
    advisor = StagnationAdvisor()
    client = _FakeDecision()
    advisor.record_turn(_turn())
    # A turn whose action failed -> snapshot is None -> chain reset.
    advisor.record_turn(post_turn_snapshot([(_action(), "error")], _noise_bytes(1)))
    advisor.record_turn(_turn())
    assert await advisor.review_and_notify(client) is None


@pytest.mark.asyncio
async def test_non_stagnation_situations_do_not_notify():
    client_transition = _FakeDecision(p_changed=0.05, situation="transition_in_progress")
    advisor = StagnationAdvisor()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    assert await advisor.review_and_notify(client_transition) is None

    client_waiting = _FakeDecision(p_changed=0.05, situation="waiting_appropriate")
    advisor2 = StagnationAdvisor()
    advisor2.record_turn(_turn())
    advisor2.record_turn(_turn())
    assert await advisor2.review_and_notify(client_waiting) is None


@pytest.mark.asyncio
async def test_changed_screen_answer_suppresses_notice():
    client = _FakeDecision(p_changed=0.9, situation="true_stagnation")
    advisor = StagnationAdvisor()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    assert await advisor.review_and_notify(client) is None


@pytest.mark.asyncio
async def test_overlay_blocking_hint_mentions_overlay():
    client = _FakeDecision(p_changed=0.05, situation="overlay_blocking")
    advisor = StagnationAdvisor()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    notice = await advisor.review_and_notify(client)
    assert notice is not None
    assert "overlay" in notice


@pytest.mark.asyncio
async def test_decision_failure_is_silent():
    from artemis.llm.decision import DecisionError

    client = _FakeDecision(error=DecisionError("timeout", kind="timeout"))
    advisor = StagnationAdvisor()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    assert await advisor.review_and_notify(client) is None


@pytest.mark.asyncio
async def test_none_client_is_free():
    advisor = StagnationAdvisor()
    advisor.record_turn(_turn(seed=1))
    advisor.record_turn(_turn(seed=1))
    assert await advisor.review_and_notify(None) is None


def test_post_turn_snapshot_uses_recorded_action_shape():
    turn = post_turn_snapshot(
        [
            (
                {
                    "action": "click",
                    "coordinates": [500, 900],
                    "target_description": "the play button",
                },
                "success",
            )
        ],
        _noise_bytes(1),
    )
    assert turn.action_signature == "click:the play button:[500, 900]"
    assert "click on the play button" in turn.action_label
    assert turn.post_dhash and len(turn.post_dhash) == 16


def test_gate_constant_is_pinned():
    """The gate threshold is part of the review contract; pin it so a silent
    loosening/tightening cannot slip through (fixtures below then bracket
    both sides of exactly this value)."""
    assert GATE_MAX_DHASH_DISTANCE == 8


def _hash_pair_at_distance(target: int) -> tuple[str, str]:
    """Two 16-hex dHashes at exactly `target` Hamming distance, by construction.

    Derived from a real noise-image hash so the bit pattern is realistic;
    flipping exactly `target` low bits gives a deterministic pair (no search).
    """
    base = dhash_hex(_noise_bytes(1, size=(200, 120)))
    assert base
    n = int(base, 16)
    flipped = n ^ ((1 << target) - 1)
    other = f"{flipped:0{len(base)}x}"
    assert bin(n ^ flipped).count("1") == target
    return base, other


def test_gate_fires_at_exactly_the_threshold_distance():
    a, b = _hash_pair_at_distance(GATE_MAX_DHASH_DISTANCE)
    advisor = StagnationAdvisor()

    class _Img:
        pass

    # Build window turns directly from the matched hashes.
    t1 = StagnationWindowTurn(
        action_signature="click:X:[1, 2]",
        action_label="click on X",
        post_dhash=a,
        post_img_bytes=b"img1",
    )
    t2 = StagnationWindowTurn(
        action_signature="click:X:[1, 2]",
        action_label="click on X",
        post_dhash=b,
        post_img_bytes=b"img2",
    )
    advisor.record_turn(t1)
    advisor.record_turn(t2)
    assert advisor.gate_passes() is t2


def test_gate_blocks_at_threshold_plus_one():
    a, b = _hash_pair_at_distance(GATE_MAX_DHASH_DISTANCE + 1)
    advisor = StagnationAdvisor()
    t1 = StagnationWindowTurn(
        action_signature="click:X:[1, 2]", action_label="click on X", post_dhash=a
    )
    t2 = StagnationWindowTurn(
        action_signature="click:X:[1, 2]", action_label="click on X", post_dhash=b
    )
    advisor.record_turn(t1)
    advisor.record_turn(t2)
    assert advisor.gate_passes() is None


def test_signature_includes_multi_action_sequences():
    a = post_turn_snapshot(
        [(_action(), "success"), (_action("swipe", "list"), "success")], _noise_bytes(1)
    )
    b = post_turn_snapshot([(_action(), "success")], _noise_bytes(1))
    advisor = StagnationAdvisor()
    advisor.record_turn(a)
    advisor.record_turn(b)
    assert advisor.gate_passes() is None  # different sequences never match
