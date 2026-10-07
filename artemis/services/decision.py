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

"""Decision-model service: per-context client resolution.

:func:`get_decision_client` is the single resolution point every decision
call site uses. It reads the configuration in precedence order

1. ``ctx.execution_setup.decision_model`` — the SDK/builder override
   (``AgentConfigBuilder.with_decision_model``), propagated per task by
   ``_prepare_tracing``;
2. ``ctx.llm_config.decision_model`` — the file configuration
   (top-level ``decision_model`` key of ``artemis.jsonc`` / llm-config.json);
3. ``ctx.agent_config.decision_model`` — the same SDK value read off the
   agent config directly, for contexts that carry an ``agent_config`` but
   never ran ``_prepare_tracing`` (e.g. pre-task tooling); unreachable in
   normal SDK runs, where sources 1 and 3 are the same object.

It returns ``None`` unless a configuration exists, is enabled and its
credentials validate — an unconfigured or broken decision layer must behave
exactly like today's VLM/LLM-only runs. The built client is cached on the
context (lazy singleton), keyed implicitly by the context object.
"""

from __future__ import annotations

from artemis.config import DecisionModelConfig
from artemis.context import ArtemisContext
from artemis.llm.decision import DecisionClient, DecisionError
from artemis.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_decision_config(ctx: ArtemisContext) -> DecisionModelConfig | None:
    """The effective decision-model config for this context, or ``None``."""
    for source in (
        getattr(ctx, "execution_setup", None),
        getattr(ctx, "llm_config", None),
        getattr(ctx, "agent_config", None),
    ):
        cfg = getattr(source, "decision_model", None)
        if isinstance(cfg, DecisionModelConfig):
            return cfg
    return None


def get_decision_client(ctx: ArtemisContext) -> DecisionClient | None:
    """Lazily builds (and caches on ``ctx``) the decision client, or ``None``.

    ``None`` means "run the existing VLM/LLM path": the feature is disabled,
    unconfigured, or its configuration is invalid (loudly logged once per
    context — the run continues on the fallback path either way).
    """
    try:
        cached = getattr(ctx, "decision_client", None)
    except Exception:  # pragma: no cover - getattr on exotic doubles
        cached = None
    # Duck-typed probe: an unspecced mock context auto-creates attributes,
    # and anything already closed by __aexit__ must not be reused either.
    if cached is not None and callable(getattr(cached, "decide", None)):
        return cached

    cfg = resolve_decision_config(ctx)
    if cfg is None or not cfg.enabled:
        return None
    try:
        client = DecisionClient(cfg)
    except DecisionError as e:
        logger.error(f"Decision model enabled but unusable ({e}); using VLM/LLM paths.")
        return None
    try:
        ctx.decision_client = client  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - read-only context doubles in tests
        pass
    return client
