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

"""Unit tests for the Clef/Jev decision-model client."""

import base64
import io

import httpx
import pytest

from artemis.config import DecisionModelConfig
from artemis.llm.decision import (
    DecisionClient,
    _is_bounded_jpeg,
    _to_cloudflare_question,
    DecisionError,
    FakeDecisionClient,
    MAX_IMAGES,
    MAX_QUESTIONS,
    choice_question,
    normalize_image_bytes,
    normalize_images,
    noul_question,
    parse_decision_answers,
    score_question,
)


def _tiny_jpeg(width: int = 200, height: int = 100, color=(120, 30, 30)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _custom_client(**overrides) -> DecisionClient:
    cfg = DecisionModelConfig(
        enabled=True, provider="custom", base_url="http://decide.test/decide", **overrides
    )
    return DecisionClient(cfg)


def _handler(responses):
    """Returns (handler, requests) where responses is a list of Response or Exception."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = responses.pop(0) if responses else httpx.Response(200, json={"answers": {}})
        if isinstance(item, Exception):
            raise item
        return item

    return handler, requests


_QUESTIONS = {
    "is_present": noul_question("Is it present?"),
    "situation": choice_question("What changed?", ["unchanged", "shifted", "disappeared"]),
}


@pytest.mark.asyncio
async def test_request_body_is_jev_shaped():
    responses = [
        httpx.Response(
            200,
            json={"answers": {"is_present": {"p": 0.9}, "situation": {"choice": "unchanged"}}},
        )
    ]
    handler, requests = _handler(responses)
    client = _custom_client(model="clef")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("the state", _QUESTIONS, decision_point="test")
    finally:
        await client._http.aclose()

    body = requests[0].read()
    import json

    payload = json.loads(body)
    assert payload["model"] == "clef"
    assert payload["state"] == "the state"
    assert payload["questions"]["is_present"]["type"] == "noul"
    assert payload["questions"]["situation"]["options"] == ["unchanged", "shifted", "disappeared"]
    assert "images" not in payload
    assert result.noul("is_present") == pytest.approx(0.9)
    assert result.choice("situation").chosen == "unchanged"
    assert result.model == "clef"
    assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_images_are_normalized_and_embedded():
    big = _tiny_jpeg(width=3000, height=2000, color=(10, 200, 10))
    responses = [
        httpx.Response(
            200,
            json={
                "result": {
                    "answers": {"is_present": {"p": 0.1}, "situation": {"p": {"shifted": 0.9}}}
                }
            },
        )
    ]
    handler, requests = _handler(responses)
    client = _custom_client()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS, images=[big, big], decision_point="img")
    finally:
        await client._http.aclose()

    import json

    payload = json.loads(requests[0].read())
    assert len(payload["images"]) == 2
    decoded = base64.b64decode(payload["images"][0])
    from PIL import Image

    with Image.open(io.BytesIO(decoded)) as img:
        assert max(img.size) <= 1568
    assert result.noul("is_present") == pytest.approx(0.1)
    assert result.choice("situation").chosen == "shifted"
    assert result.choice("situation").confidence == pytest.approx(0.9)


def test_normalize_image_bytes_rejects_garbage():
    with pytest.raises(DecisionError) as exc:
        normalize_image_bytes(b"not-an-image")
    assert exc.value.kind == "protocol"


def test_normalize_images_caps_request_size():
    # A bounded JPEG passes through; a huge non-JPEG raises a protocol error.
    bounded = _tiny_jpeg()
    assert normalize_images([bounded]) == [base64.b64encode(bounded).decode()]
    with pytest.raises(DecisionError):
        normalize_images([b"\x00" * (13 * 1024 * 1024)])


def test_normalize_images_rejects_request_over_the_whole_request_cap():
    """Four individually-bounded images whose base64 total exceeds the 13 MiB
    request cap raise from the request-size branch (not the per-image cap)."""
    import io as _io
    import random

    from PIL import Image

    # Incompressible noise below the 4 MiB per-image cap but large enough
    # that four of them exceed the 13 MiB whole-request cap after base64.
    rng = random.Random(42)
    size = (1568, 1568)
    img = Image.new("RGB", size)
    img.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(size[0] * size[1])
        ]
    )
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    noise = buf.getvalue()
    assert len(noise) <= 4 * 1024 * 1024, f"fixture too large: {len(noise)}"
    assert _is_bounded_jpeg(noise), "fixture must pass the per-image cap untouched"
    with pytest.raises(DecisionError) as exc:
        normalize_images([noise] * 4)
    # The per-image cap did not fire (each image is bounded): this is the
    # whole-request budget branch.
    assert "exceed" in str(exc.value)


@pytest.mark.asyncio
async def test_timeout_retries_once_then_succeeds():
    """A timeout retries exactly once; the second attempt's answer wins."""
    client = _custom_client(timeout=0.2)
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(
            200, json={"answers": {"is_present": {"p": 0.5}, "situation": {"choice": "shifted"}}}
        )

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert calls["n"] == 2
    assert result.noul("is_present") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_timeout_both_attempts_raises_decision_error():
    client = _custom_client(timeout=0.2)
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        raise httpx.ConnectTimeout("down")

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DecisionError) as exc:
            await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert exc.value.kind == "timeout"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_http_500_retries_once_and_classifies():
    client = _custom_client()
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(500, text="boom")

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DecisionError) as exc:
            await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert exc.value.kind == "http"
    assert exc.value.status == 500
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_http_400_does_not_retry():
    client = _custom_client()
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DecisionError) as exc:
            await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert exc.value.kind == "http"
    assert exc.value.status == 400
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_workers_ai_envelope_is_unwrapped():
    responses = [
        httpx.Response(
            200,
            json={
                "result": {
                    "answers": {
                        "is_present": {"p": 0.42},
                        "situation": {"choice": "shifted", "probability": 0.6},
                    },
                },
                "success": True,
            },
        )
    ]
    handler, _ = _handler(responses)
    client = _custom_client()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert result.noul("is_present") == pytest.approx(0.42)
    assert result.choice("situation").chosen == "shifted"


@pytest.mark.asyncio
async def test_endpoint_reported_failure_is_protocol_error():
    client = _custom_client()

    async def handler(request):
        return httpx.Response(200, json={"success": False, "errors": [{"code": 7003}]})

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DecisionError) as exc:
            await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert exc.value.kind == "protocol"


@pytest.mark.asyncio
async def test_more_than_four_images_are_capped():
    small = _tiny_jpeg(50, 50)
    # normalize_images embeds at most MAX_IMAGES images (silent cap).
    encoded = normalize_images([small] * (MAX_IMAGES + 2))
    assert len(encoded) == MAX_IMAGES


def test_question_count_is_enforced():
    client = _custom_client()
    many = {f"q{i}": noul_question(f"q{i}?") for i in range(MAX_QUESTIONS + 1)}
    with pytest.raises(DecisionError):
        client.build_request("s", many)


def test_parse_decision_answers_accepts_bare_question_map():
    payload = {"is_present": {"answer": "yes"}, "situation": {"choice": "disappeared"}}
    answers = parse_decision_answers(_QUESTIONS, payload)
    assert answers["is_present"].probability == 1.0
    assert answers["situation"].chosen == "disappeared"


def test_parse_decision_answers_rejects_missing_answers():
    with pytest.raises(DecisionError):
        parse_decision_answers(_QUESTIONS, {"answers": {"is_present": {"p": 0.5}}})


def test_cloudflare_client_requires_credentials(monkeypatch):
    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct", raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", None, raising=False)
    cfg = DecisionModelConfig(enabled=True)  # provider=cloudflare
    with pytest.raises(DecisionError) as exc:
        DecisionClient(cfg)
    assert exc.value.kind == "config"


def test_cloudflare_endpoint_url_and_headers(monkeypatch):
    from pydantic import SecretStr

    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct-123", raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    client = DecisionClient(DecisionModelConfig(enabled=True, model="clef-flash"))
    assert client._endpoint_url() == (
        "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/run/@cf/cloudflare/clef-flash"
    )
    assert client._headers()["Authorization"] == "Bearer tok"


def test_score_question_roundtrip():
    questions = {"quality": score_question("How good?", 1, 5)}
    answers = parse_decision_answers(questions, {"answers": {"quality": {"score": 4}}})
    assert answers["quality"].score == 4.0


@pytest.mark.asyncio
async def test_fake_decision_client_scripting_and_recording():
    fake = FakeDecisionClient(
        scripted={"pixel": {"is_present": 0.11, "situation": {"shifted": 0.8, "unchanged": 0.2}}}
    )
    result = await fake.decide("s", _QUESTIONS, decision_point="pixel")
    assert result.noul("is_present") == pytest.approx(0.11)
    assert result.choice("situation").chosen == "shifted"
    assert fake.calls[0]["decision_point"] == "pixel"
    # Default (unscripted) answers: noul 0.9, first choice option.
    other = FakeDecisionClient()
    default_result = await other.decide("s", _QUESTIONS)
    assert default_result.noul("is_present") == pytest.approx(0.9)
    assert default_result.choice("situation").chosen == "unchanged"


@pytest.mark.asyncio
async def test_null_result_field_does_not_raise():
    """A `result: null` 200 payload must parse (usage extraction guards)."""

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "answers": {"is_present": {"p": 0.7}, "situation": {"choice": "unchanged"}},
                "result": None,
            },
        )

    client = _custom_client()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert result.noul("is_present") == pytest.approx(0.7)
    assert result.usage is None


@pytest.mark.parametrize(
    "bad_url",
    ["http://[::1/decide", "http://example.com:abc/decide", "://nohost"],
)
@pytest.mark.asyncio
async def test_malformed_base_url_raises_decision_error_config(bad_url):
    """httpx.InvalidURL/ValueError at request-build time must surface as
    DecisionError (never escape the DecisionError-only contract)."""

    async def handler(request):  # pragma: no cover - never reached
        return httpx.Response(200, json={"answers": {}})

    cfg = DecisionModelConfig(enabled=True, provider="custom", base_url=bad_url)
    client = DecisionClient(cfg)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DecisionError) as exc:
            await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert exc.value.kind == "config"


@pytest.mark.asyncio
async def test_http_429_is_retryable():
    client = _custom_client()
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json={"answers": {"is_present": {"p": 0.5}, "situation": {"choice": "shifted"}}},
        )

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert calls["n"] == 2
    assert result.noul("is_present") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_http_502_is_retryable():
    client = _custom_client()
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(
            200,
            json={"answers": {"is_present": {"p": 0.5}, "situation": {"choice": "shifted"}}},
        )

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("s", _QUESTIONS)
    finally:
        await client._http.aclose()
    assert calls["n"] == 2
    assert result.noul("is_present") == pytest.approx(0.5)


def _cloudflare_client(**overrides) -> DecisionClient:
    """A cloudflare-provider client with settings-shaped credentials stubbed."""
    cfg = DecisionModelConfig(
        enabled=True, provider="cloudflare", model="clef-flash", **overrides
    )
    return DecisionClient(cfg)


@pytest.mark.asyncio
async def test_cloudflare_wire_format_translates_questions_and_answers(monkeypatch):
    """The cloudflare provider speaks the Workers AI Clef schema, not Jev.

    Request: question/options -> instructions/criteria (criteria as an object
    for choice). Response: the {"result": {...}} envelope with noul/choice
    probabilities in the Workers AI answer shape.
    """
    from pydantic import SecretStr

    from artemis.config import settings as artemis_settings

    monkeypatch.setattr(artemis_settings, "CLOUDFLARE_ACCOUNT_ID", "acct123", raising=False)
    monkeypatch.setattr(
        artemis_settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False
    )
    responses = [
        httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "model": "clef-flash",
                    "answers": {
                        "is_present": {"type": "noul", "noul": 0.9551},
                        "situation": {
                            "type": "choice",
                            "choice": "shifted",
                            "probabilities": {
                                "unchanged": 0.0506,
                                "shifted": 0.9354,
                                "disappeared": 0.014,
                            },
                            "confidence": 0.8165,
                        },
                    },
                    "usage": {"input_tokens": 346, "output_tokens": 0},
                },
            },
        )
    ]
    handler, requests = _handler(responses)
    client = _cloudflare_client()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.decide("the state", _QUESTIONS, decision_point="test")
    finally:
        await client._http.aclose()

    import json

    payload = json.loads(requests[0].read())
    # Request translation: Jev question/options -> Clef instructions/criteria.
    assert payload["model"] == "clef-flash"
    assert payload["questions"]["is_present"] == {
        "type": "noul",
        "instructions": "Is it present?",
    }
    assert payload["questions"]["situation"]["criteria"] == {
        "unchanged": "unchanged",
        "shifted": "shifted",
        "disappeared": "disappeared",
    }
    assert "question" not in payload["questions"]["situation"]
    assert "options" not in payload["questions"]["situation"]
    # URL carries the @cf/cloudflare/ prefix and bearer auth.
    assert requests[0].url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
    assert requests[0].headers["authorization"] == "Bearer tok"
    # Response parsing: noul key + probabilities distribution.
    assert result.noul("is_present") == pytest.approx(0.9551)
    verdict = result.choice("situation")
    assert verdict.chosen == "shifted"
    assert verdict.distribution["shifted"] == pytest.approx(0.9354)
    assert verdict.confidence == pytest.approx(0.9354)
    assert result.usage == {"input_tokens": 346, "output_tokens": 0}


def test_cloudflare_score_question_maps_numeric_scale(monkeypatch):
    """score questions send low..high criteria labels; legend answers are lifted."""
    spec = score_question("How good?", 1, 5)
    translated = _to_cloudflare_question(spec)
    assert translated["criteria"] == ["1", "2", "3", "4", "5"]
    # Workers AI shape (weighted 0-based index + legend) -> lifted onto 1..5.
    answers = parse_decision_answers(
        {"quality": spec},
        {"answers": {"quality": {"score": 3.72, "legend": {"0": "1", "1": "2"}}}},
    )
    assert answers["quality"].score == pytest.approx(4.72)
    # Bare Jev shape answers the scale value directly.
    answers_jev = parse_decision_answers(
        {"quality": spec}, {"answers": {"quality": {"score": 4}}}
    )
    assert answers_jev["quality"].score == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_cloudflare_images_are_data_uris(monkeypatch):
    """Workers AI rejects bare base64 in images[] with 422/5012
    ("image must be an embedded base64 data URI"); the cloudflare provider
    must wrap each normalized image as a data URI. The custom (Jev) provider
    keeps bare base64."""
    from pydantic import SecretStr

    from artemis.config import settings as artemis_settings

    monkeypatch.setattr(artemis_settings, "CLOUDFLARE_ACCOUNT_ID", "acct123", raising=False)
    monkeypatch.setattr(
        artemis_settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False
    )
    responses = [
        httpx.Response(
            200, json={"answers": {"is_present": {"p": 0.9}, "situation": {"choice": "unchanged"}}}
        ),
        httpx.Response(
            200, json={"answers": {"is_present": {"p": 0.9}, "situation": {"choice": "unchanged"}}}
        ),
    ]
    handler, requests = _handler(responses)

    cf = _cloudflare_client()
    cf._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    jev = _custom_client()
    jev._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await cf.decide("s", _QUESTIONS, images=[_tiny_jpeg()], decision_point="t")
        await jev.decide("s", _QUESTIONS, images=[_tiny_jpeg()], decision_point="t")
    finally:
        await cf._http.aclose()
        await jev._http.aclose()

    import base64
    import json

    cf_payload = json.loads(requests[0].read())
    jev_payload = json.loads(requests[1].read())
    assert cf_payload["images"][0].startswith("data:image/jpeg;base64,")
    # The payload decodes back to the (re-encoded) JPEG bytes.
    decoded = base64.b64decode(cf_payload["images"][0].split(",", 1)[1])
    assert decoded.startswith(b"\xff\xd8")
    # The custom/Jev provider keeps the bare base64 shape.
    assert not jev_payload["images"][0].startswith("data:")
    assert base64.b64decode(jev_payload["images"][0]).startswith(b"\xff\xd8")
