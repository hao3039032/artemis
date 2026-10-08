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

"""Decision-model client for Clef-family scoring endpoints (Jev-compatible).

Clef (Cloudflare, Apache 2.0) is a non-autoregressive decision model: given a
``state`` (text/JSON), optional embedded ``images`` (Clef extension, max 4)
and a set of typed ``questions``, it returns a calibrated probability per
question instead of generated text:

- ``noul``   — boolean question; the answer carries P(yes).
- ``choice`` — one option among ``options``; the answer carries either a full
               probability distribution or the chosen option plus its
               probability.
- ``score``  — ordinal question; the answer carries the score value.

This module is the wire-level client only: request construction, defensive
image normalization, one retry on timeout/5xx/429, tolerant answer parsing, and a
``decision_call`` telemetry event. Call sites resolve their client through
:func:`artemis.services.decision.get_decision_client` and keep their existing
VLM/LLM path as the fallback for every ``DecisionError``.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from artemis.config import DecisionModelConfig
from artemis.utils.logger import get_logger

logger = get_logger(__name__)

#: Clef accepts at most four embedded images per request.
MAX_IMAGES = 4

#: Defensive image normalization targets (mirrors the multimodal LLM limits):
#: JPEG re-encode, long edge cap, per-image size cap, whole-request cap.
_MAX_IMAGE_LONG_EDGE = 1568
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_REQUEST_BYTES = 13 * 1024 * 1024

#: Clef answers at most 64 questions per request.
MAX_QUESTIONS = 64

_CLOUDFLARE_BASE = "https://api.cloudflare.com/client/v4/accounts"

_RETRYABLE_STATUS = frozenset({500, 502, 503, 504, 429})


class DecisionError(Exception):
    """A decision-model call failed; the caller must use its fallback path."""

    def __init__(self, message: str, *, kind: str = "error", status: int | None = None):
        super().__init__(message)
        self.kind = kind  # "timeout" | "http" | "protocol" | "config"
        self.status = status  # HTTP status for kind="http"


# --- Answers ---------------------------------------------------------------------------


@dataclass
class DecisionAnswer:
    """One question's answer as calibrated probabilities."""

    question_id: str
    kind: str  # "noul" | "choice" | "score"
    probability: float | None = None
    """noul: P(yes) in [0, 1]."""
    distribution: dict[str, float] = field(default_factory=dict)
    """choice: option -> probability (may be partial/one-hot)."""
    chosen: str | None = None
    """choice: the argmax option."""
    score: float | None = None
    """score: the answered value."""

    @property
    def confidence(self) -> float:
        """Probability of the emitted verdict (chosen option / yes / score)."""
        if self.kind == "choice":
            if self.chosen is None:
                return 0.0
            return self.distribution.get(self.chosen, 0.0)
        if self.kind == "score":
            return 1.0
        return self.probability if self.probability is not None else 0.0


@dataclass
class DecisionResult:
    """A decision call's answers plus request-level metadata."""

    answers: dict[str, DecisionAnswer] = field(default_factory=dict)
    model: str | None = None
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None

    def noul(self, question_id: str, *, default: float | None = None) -> float | None:
        answer = self.answers.get(question_id)
        if answer is None or answer.probability is None:
            return default
        return answer.probability

    def choice(self, question_id: str) -> DecisionAnswer | None:
        return self.answers.get(question_id)


# --- Question builders -------------------------------------------------------------------


def noul_question(question: str) -> dict[str, Any]:
    """A boolean question; the answer carries P(yes)."""
    return {"type": "noul", "question": question}


def choice_question(question: str, options: list[str]) -> dict[str, Any]:
    """A single-choice question over the given options."""
    return {"type": "choice", "question": question, "options": list(options)}


def score_question(question: str, low: int = 1, high: int = 10) -> dict[str, Any]:
    """An ordinal question on the [low, high] scale."""
    return {"type": "score", "question": question, "low": low, "high": high}


def _to_cloudflare_question(spec: dict[str, Any]) -> dict[str, Any]:
    """Translates one Jev question spec to the Workers AI Clef wire format.

    Workers AI Clef (``@cf/cloudflare/clef*``) validates its input against a
    strict schema: the prompt field is ``instructions`` (not ``question``)
    and the option field is ``criteria`` — an object (option -> description)
    for ``choice`` questions and an array of scale labels for ``score``.
    Unknown keys such as ``question``/``options``/``low``/``high`` fail the
    whole request with ``5006/5012 Bad input``.
    """
    kind = str((spec or {}).get("type") or "noul")
    out: dict[str, Any] = {"type": kind}
    prompt = spec.get("question")
    if isinstance(prompt, str) and prompt.strip():
        out["instructions"] = prompt
    if kind == "choice":
        options = spec.get("options")
        if isinstance(options, (list, tuple)) and options:
            # Identity map keeps the response probabilities keyed by exactly
            # the option strings the caller knows about.
            out["criteria"] = {str(o): str(o) for o in options}
        elif isinstance(spec.get("criteria"), dict):
            out["criteria"] = spec["criteria"]
    elif kind == "score":
        if isinstance(spec.get("criteria"), list):
            out["criteria"] = [str(c) for c in spec["criteria"]]
        else:
            low = spec.get("low") if isinstance(spec.get("low"), int) else 1
            high = spec.get("high") if isinstance(spec.get("high"), int) else 10
            if high < low:
                low, high = high, low
            out["criteria"] = [str(v) for v in range(low, high + 1)]
    return out


# --- Image normalization -------------------------------------------------------------


def normalize_image_bytes(image: bytes) -> bytes:
    """Re-encodes one image as a bounded JPEG for the decision request.

    Long edge capped at 1568 px and payload at 4 MiB (JPEG quality 85 then 70).
    Raises :class:`DecisionError` (kind ``protocol``) for unreadable bytes or
    an image that still exceeds the cap after re-encoding — the caller falls
    back to its VLM path rather than sending a request the endpoint must
    reject.
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - Pillow is a hard dependency
        raise DecisionError(f"image normalization unavailable: {e}", kind="protocol") from e
    try:
        with Image.open(io.BytesIO(image)) as img:
            img = img.convert("RGB")
            long_edge = max(img.size)
            if long_edge > _MAX_IMAGE_LONG_EDGE:
                scale = _MAX_IMAGE_LONG_EDGE / float(long_edge)
                img = img.resize(
                    (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            for quality in (85, 70, 55):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                data = buf.getvalue()
                if len(data) <= _MAX_IMAGE_BYTES:
                    return data
    except DecisionError:
        raise
    except Exception as e:
        raise DecisionError(f"image normalization failed: {e}", kind="protocol") from e
    raise DecisionError(f"normalized image still exceeds {_MAX_IMAGE_BYTES} bytes", kind="protocol")


def normalize_images(images: list[bytes]) -> list[str]:
    """Normalizes and base64-encodes up to :data:`MAX_IMAGES` images.

    Raises :class:`DecisionError` when the encoded payload would exceed the
    whole-request cap, so the caller can fall back instead of sending a
    request the endpoint must reject.
    """
    encoded: list[str] = []
    total = 0
    for image in images[:MAX_IMAGES]:
        data = normalize_image_bytes(image) if not _is_bounded_jpeg(image) else image
        b64 = base64.b64encode(data).decode("ascii")
        total += len(b64)
        if total > _MAX_REQUEST_BYTES:
            raise DecisionError(
                f"decision request images exceed {_MAX_REQUEST_BYTES} bytes", kind="protocol"
            )
        encoded.append(b64)
    return encoded


def _is_bounded_jpeg(image: bytes) -> bool:
    """True for an already-bounded JPEG that needs no re-encode."""
    if not image.startswith(b"\xff\xd8"):
        return False
    if len(image) > _MAX_IMAGE_BYTES:
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image)) as img:
            return max(img.size) <= _MAX_IMAGE_LONG_EDGE
    except Exception:
        return False


# --- Answer parsing ------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= f <= 1.0:
        return f
    return None


def parse_decision_answers(questions: dict[str, Any], payload: Any) -> dict[str, DecisionAnswer]:
    """Parses one decision response body into per-question answers.

    Tolerant by design — the wire shape varies between the Cloudflare Workers
    AI envelope (``{"result": {"answers": ...}}``) and bare Jev endpoints
    (``{"answers": ...}``), and answer payloads may carry either a probability
    distribution or a chosen option. Raises :class:`DecisionError` (kind
    ``protocol``) when no answer for a requested question can be parsed.
    """
    if isinstance(payload, dict):
        answers_raw = payload.get("answers")
        if not isinstance(answers_raw, dict):
            result = payload.get("result")
            if isinstance(result, dict):
                answers_raw = result.get("answers")
        if not isinstance(answers_raw, dict):
            # A bare per-question map (id -> answer) is also accepted.
            if all(qid in payload for qid in questions):
                answers_raw = payload
    else:
        answers_raw = None
    if not isinstance(answers_raw, dict):
        raise DecisionError(
            f"decision response carries no answers object: {json.dumps(payload)[:200]}",
            kind="protocol",
        )

    answers: dict[str, DecisionAnswer] = {}
    for qid, spec in questions.items():
        raw = answers_raw.get(qid)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raw = {"p": raw}
        kind = str((spec or {}).get("type") or "noul")
        if kind == "noul":
            answers[qid] = _parse_noul(qid, raw)
        elif kind == "choice":
            answers[qid] = _parse_choice(qid, spec, raw)
        elif kind == "score":
            answers[qid] = _parse_score(qid, raw, spec)
        else:
            raise DecisionError(f"unsupported question type '{kind}'", kind="protocol")

    missing = [qid for qid in questions if qid not in answers]
    if missing:
        raise DecisionError(
            f"decision response missing answers for: {', '.join(missing)}", kind="protocol"
        )
    return answers


def _parse_noul(qid: str, raw: dict[str, Any]) -> DecisionAnswer:
    for key in ("p", "probability", "yes", "noul", "confidence"):
        p = _as_float(raw.get(key))
        if p is not None:
            return DecisionAnswer(question_id=qid, kind="noul", probability=p)
    answer = raw.get("answer", raw.get("value"))
    if isinstance(answer, bool):
        return DecisionAnswer(question_id=qid, kind="noul", probability=1.0 if answer else 0.0)
    if isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in ("yes", "true"):
            return DecisionAnswer(question_id=qid, kind="noul", probability=1.0)
        if lowered in ("no", "false"):
            return DecisionAnswer(question_id=qid, kind="noul", probability=0.0)
    raise DecisionError(f"unparseable noul answer for '{qid}': {raw}", kind="protocol")


def _parse_choice(qid: str, spec: dict[str, Any], raw: dict[str, Any]) -> DecisionAnswer:
    criteria = spec.get("criteria") if isinstance(spec.get("criteria"), dict) else None
    options = [str(o) for o in (spec or {}).get("options") or []]
    if criteria:
        options = options or [str(k) for k in criteria]
    # Shape 1: an explicit option -> probability distribution. Cloudflare
    # Workers AI returns the full distribution under "probabilities".
    for dist_key in ("p", "distribution", "probabilities"):
        dist_raw = raw.get(dist_key)
        if isinstance(dist_raw, dict) and dist_raw:
            distribution = {}
            for option, prob in dist_raw.items():
                p = _as_float(prob)
                if p is not None and (not options or str(option) in options):
                    distribution[str(option)] = p
            if distribution:
                chosen = max(distribution, key=distribution.get)  # type: ignore[arg-type]
                return DecisionAnswer(
                    question_id=qid, kind="choice", chosen=chosen, distribution=distribution
                )
    # Shape 2: a chosen option (with optional probability).
    chosen_raw = None
    for key in ("choice", "option", "answer", "value"):
        candidate = raw.get(key)
        if isinstance(candidate, str) and candidate.strip():
            chosen_raw = candidate.strip()
            break
    p = None
    for key in ("p", "probability", "confidence"):
        p = _as_float(raw.get(key))
        if p is not None:
            break
    if chosen_raw is not None and (not options or chosen_raw in options):
        distribution = {option: 0.0 for option in options} if options else {}
        distribution[chosen_raw] = p if p is not None else 1.0
        if not options:
            distribution = {chosen_raw: distribution[chosen_raw]}
        return DecisionAnswer(
            question_id=qid, kind="choice", chosen=chosen_raw, distribution=distribution
        )
    raise DecisionError(f"unparseable choice answer for '{qid}': {raw}", kind="protocol")


def _parse_score(qid: str, raw: dict[str, Any], spec: dict[str, Any] | None = None) -> DecisionAnswer:
    for key in ("score", "value", "answer"):
        v = raw.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            score = float(v)
            # A "legend" (0-based index -> label map) marks the Workers AI
            # score shape: the value is a weighted criteria index, so lift it
            # back onto a numeric [low, high] scale when the spec has one.
            # Bare Jev endpoints answer the scale value directly — no offset.
            low = (spec or {}).get("low")
            has_legend = isinstance(raw.get("legend"), dict)
            if has_legend and isinstance(low, (int, float)) and not isinstance(low, bool):
                score += float(low)
            return DecisionAnswer(question_id=qid, kind="score", score=score)
    raise DecisionError(f"unparseable score answer for '{qid}': {raw}", kind="protocol")


# --- Client -----------------------------------------------------------------------------


def _probability_summary(result: DecisionResult) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for qid, answer in result.answers.items():
        if answer.kind == "choice" and answer.chosen is not None:
            summary[qid] = {"chosen": answer.chosen, "p": round(answer.confidence, 4)}
        elif answer.kind == "noul":
            summary[qid] = round(answer.probability or 0.0, 4)
        else:
            summary[qid] = answer.score
    return summary


class DecisionClient:
    """Async client for one configured decision endpoint.

    The client is provider-agnostic over two transports:

    - ``cloudflare``: Workers AI REST
      ``POST /accounts/{id}/ai/run/@cf/cloudflare/{model}`` with a bearer
      token; the response arrives in the Workers AI ``{"result": ...}``
      envelope.
    - ``custom``: ``POST {base_url}`` against any Jev/Clef-compatible
      endpoint (``base_url`` is the full decision-endpoint URL).

    Both share the Jev request body (``model`` / ``state`` / ``questions``,
    plus the Clef ``images`` extension) and the tolerant answer parser.
    """

    def __init__(self, config: DecisionModelConfig, http_client: httpx.AsyncClient | None = None):
        try:
            config.validate_runtime()
        except ValueError as e:
            raise DecisionError(str(e), kind="config") from e
        self.config = config
        self._http = http_client
        self._owns_http = http_client is None

    @property
    def model_name(self) -> str:
        return self.config.model

    def use_enabled(self, decision_point: str) -> bool:
        """Whether ``decision_point`` (a DecisionModelUseConfig field) is on."""
        return bool(getattr(self.config.use, decision_point, False))

    def _endpoint_url(self) -> str:
        if self.config.provider == "custom":
            return str(self.config.base_url or "").strip()
        from artemis.config.settings import settings

        account_id = (settings.CLOUDFLARE_ACCOUNT_ID or "").strip()
        return f"{_CLOUDFLARE_BASE}/{account_id}/ai/run/@cf/cloudflare/{self.config.model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.provider == "cloudflare":
            from artemis.config.settings import settings

            token = settings.CLOUDFLARE_AUTH_TOKEN
            secret = token.get_secret_value() if token else ""
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def build_request(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Builds the Jev-compatible request body (with the Clef images extension)."""
        if not questions:
            raise DecisionError("decision request needs at least one question", kind="protocol")
        if len(questions) > MAX_QUESTIONS:
            raise DecisionError(
                f"decision request exceeds {MAX_QUESTIONS} questions", kind="protocol"
            )
        body: dict[str, Any] = {
            "model": self.config.model,
            "state": state,
            "questions": (
                {qid: _to_cloudflare_question(spec) for qid, spec in questions.items()}
                if self.config.provider == "cloudflare"
                else questions
            ),
        }
        if images:
            encoded = normalize_images(images)
            if self.config.provider == "cloudflare":
                # Workers AI requires embedded base64 data URIs, not bare base64
                # ("image must be an embedded base64 data URI", HTTP 422/5012).
                encoded = [f"data:image/jpeg;base64,{b64}" for b64 in encoded]
            body["images"] = encoded
        return body

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        images: list[bytes] | None = None,
        *,
        decision_point: str | None = None,
    ) -> DecisionResult:
        """Runs one decision request; every failure raises :class:`DecisionError`.

        ``decision_point`` only labels the ``decision_call`` telemetry event
        (e.g. ``pixel_safety_net``); callers log their own fallback on error.
        """
        url = self._endpoint_url()
        if not url:
            raise DecisionError("decision endpoint URL is empty", kind="config")
        body = self.build_request(state, questions, images)
        started = time.perf_counter()
        last_error: DecisionError | None = None
        for attempt in (1, 2):
            try:
                result = await self._request_once(url, body, questions)
                latency_ms = (time.perf_counter() - started) * 1000.0
                result.latency_ms = latency_ms
                self._emit_telemetry(decision_point, result, status="success")
                return result
            except DecisionError as e:
                last_error = e
                retryable = e.kind == "timeout" or (
                    e.kind == "http" and e.status in _RETRYABLE_STATUS
                )
                if retryable and attempt == 1:
                    logger.warning(
                        f"Decision call failed ({e.kind}{f' {e.status}' if e.status else ''});"
                        f" retrying once: {e}"
                    )
                    continue
                raise
        raise last_error or DecisionError("decision call failed", kind="error")  # pragma: no cover

    async def _request_once(
        self,
        url: str,
        body: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> DecisionResult:
        if len(body.get("images") or []) > MAX_IMAGES:
            raise DecisionError(
                f"decision request carries more than {MAX_IMAGES} images", kind="protocol"
            )
        # One shared pooled client per DecisionClient: the decision points are
        # hot paths where a per-call TLS handshake would eat the latency win.
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout))
        timeout = httpx.Timeout(self.config.timeout)
        try:
            response = await self._http.post(
                url, json=body, headers=self._headers(), timeout=timeout
            )
        except httpx.TimeoutException as e:
            raise DecisionError(f"decision call timed out: {e}", kind="timeout") from e
        except httpx.InvalidURL as e:
            # Raised at request-build time for some malformed base_url shapes
            # (bad IPv6, bad port); it inherits from Exception (not
            # httpx.HTTPError), so it needs an explicit handler to keep the
            # DecisionError-only contract.
            raise DecisionError(f"decision endpoint URL is invalid: {e}", kind="config") from e
        except ValueError as e:
            # Other malformed base_url shapes (e.g. '://nohost') raise a bare
            # ValueError at request-build time — also configuration-origin.
            # A merely scheme-less URL takes a different path
            # (httpx.UnsupportedProtocol, an httpx.HTTPError) and is mapped
            # by the handler below as kind="http".
            if "url" in str(e).lower() or "scheme" in str(e).lower():
                raise DecisionError(f"decision endpoint URL is invalid: {e}", kind="config") from e
            raise
        except httpx.HTTPError as e:
            raise DecisionError(f"decision call transport error: {e}", kind="http") from e
        if response.status_code >= 400:
            raise DecisionError(
                f"decision call HTTP {response.status_code}: {response.text[:300]}",
                kind="http",
                status=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as e:
            raise DecisionError(
                f"decision response is not JSON: {response.text[:200]}", kind="protocol"
            ) from e
        if isinstance(payload, dict) and payload.get("success") is False:
            errors = payload.get("errors")
            raise DecisionError(
                f"decision endpoint reported failure: {json.dumps(errors)[:300]}",
                kind="protocol",
            )
        answers = parse_decision_answers(questions, payload)
        usage = None
        if isinstance(payload, dict):
            result_field = payload.get("result")
            raw_usage = payload.get("usage")
            if raw_usage is None and isinstance(result_field, dict):
                # Guarded: a "result": null payload must not raise here.
                raw_usage = result_field.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
        return DecisionResult(answers=answers, model=self.config.model, usage=usage)

    def _emit_telemetry(
        self, decision_point: str | None, result: DecisionResult, *, status: str
    ) -> None:
        """Best-effort ``decision_call`` event on the LLM telemetry bus."""
        try:
            from artemis.services.llm import _record_llm_event

            _record_llm_event(
                "decision_call",
                {
                    "decision_point": decision_point,
                    "model": result.model,
                    "latency_ms": round(result.latency_ms or 0.0, 2),
                    "answers": _probability_summary(result),
                    "usage": result.usage,
                },
                status=status,
            )
        except Exception as e:  # pragma: no cover - telemetry is best-effort
            logger.debug(f"decision_call telemetry not recorded: {e}")

    async def aclose(self) -> None:
        """Closes the pooled HTTP client when this instance owns it."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None


class FakeDecisionClient:
    """Deterministic decision double for tests (pattern: ``llm/fake_model.py``).

    Without scripting, every ``noul`` answers P(yes)=0.9 and every ``choice``
    picks the first option with p=0.9; ``decide()`` records each call for
    assertions. ``scripted`` maps a ``decision_point`` to a dict of raw
    answers; an exact ``decision_point`` match wins, and a single scripted
    entry (without a matching point) applies to every call.
    """

    def __init__(self, scripted: dict[str, dict[str, Any]] | None = None):
        self.scripted = dict(scripted or {})
        self.calls: list[dict[str, Any]] = []
        self.model_name = "fake-decision"
        self.config = DecisionModelConfig(enabled=True, provider="custom", base_url="fake://decide")

    def use_enabled(self, decision_point: str) -> bool:
        return True

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        images: list[bytes] | None = None,
        *,
        decision_point: str | None = None,
    ) -> DecisionResult:
        self.calls.append(
            {
                "state": state,
                "questions": {k: dict(v) for k, v in questions.items()},
                "images": list(images) if images else [],
                "decision_point": decision_point,
            }
        )
        raw = self.scripted.get(decision_point or "")
        if raw is None and len(self.scripted) == 1:
            # Exactly one scripted entry applies to any decision point.
            raw = next(iter(self.scripted.values()))
        answers: dict[str, DecisionAnswer] = {}
        for qid, spec in questions.items():
            if raw and qid in raw:
                answers[qid] = _fake_answer_from_raw(qid, spec, raw[qid])
                continue
            kind = str(spec.get("type") or "noul")
            if kind == "noul":
                answers[qid] = DecisionAnswer(question_id=qid, kind="noul", probability=0.9)
            elif kind == "choice":
                option = str(spec.get("options", ["unknown"])[0])
                answers[qid] = DecisionAnswer(
                    question_id=qid,
                    kind="choice",
                    chosen=option,
                    distribution={option: 0.9},
                )
            else:
                answers[qid] = DecisionAnswer(question_id=qid, kind="score", score=1.0)
        return DecisionResult(answers=answers, model=self.model_name, latency_ms=1.0)


def _fake_answer_from_raw(qid: str, spec: dict[str, Any], raw: Any) -> DecisionAnswer:
    kind = str(spec.get("type") or "noul")
    if kind == "noul":
        p = float(raw) if isinstance(raw, (int, float)) else 0.9
        return DecisionAnswer(question_id=qid, kind="noul", probability=p)
    if kind == "choice":
        options = [str(o) for o in spec.get("options") or []]
        if isinstance(raw, str):
            distribution = {option: 0.0 for option in options}
            distribution[raw] = 0.9
            return DecisionAnswer(
                question_id=qid, kind="choice", chosen=raw, distribution=distribution
            )
        if isinstance(raw, dict):
            distribution = {str(k): float(v) for k, v in raw.items()}
            chosen = max(distribution, key=distribution.get)  # type: ignore[arg-type]
            return DecisionAnswer(
                question_id=qid, kind="choice", chosen=chosen, distribution=distribution
            )
    return DecisionAnswer(
        question_id=qid, kind=kind, score=float(raw) if isinstance(raw, (int, float)) else 1.0
    )
