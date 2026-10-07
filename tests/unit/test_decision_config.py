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

"""Decision-model configuration: validation, jsonc parsing, service resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from artemis.config import DecisionModelConfig, DecisionModelUseConfig
from artemis.config.llm import LLMConfig, _expand_default_into_nodes
from artemis.context import ArtemisContext, ExecutionSetup
from artemis.services.decision import get_decision_client, resolve_decision_config
from artemis.utils.file import load_jsonc


def test_disabled_by_default():
    cfg = DecisionModelConfig()
    assert cfg.enabled is False
    assert cfg.provider == "cloudflare"
    assert cfg.model == "clef-flash"
    assert cfg.timeout == 5.0
    assert cfg.use == DecisionModelUseConfig()
    # Disabled configs never require credentials or base_url.
    cfg.validate_runtime()


def test_custom_provider_requires_base_url_when_enabled():
    with pytest.raises(ValidationError):
        DecisionModelConfig(enabled=True, provider="custom")
    ok = DecisionModelConfig(enabled=True, provider="custom", base_url="http://x/decide")
    ok.validate_runtime()  # custom endpoints carry no settings credentials


def test_cloudflare_enabled_requires_credentials(monkeypatch):
    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", None, raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", None, raising=False)
    cfg = DecisionModelConfig(enabled=True)
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        cfg.validate_runtime()

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct", raising=False)
    with pytest.raises(ValueError, match="CLOUDFLARE_AUTH_TOKEN"):
        cfg.validate_runtime()

    from pydantic import SecretStr

    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    cfg.validate_runtime()


def test_factory_jsonc_section_parses():
    path = (
        Path(__file__).resolve().parents[2] / "artemis" / "resources" / "config" / "artemis.jsonc"
    )
    data = load_jsonc(open(path, encoding="utf-8"))
    cfg = DecisionModelConfig.model_validate(data["decision_model"])
    assert cfg.enabled is False  # factory default keeps behavior identical
    assert cfg.use.pixel_safety_net and cfg.use.stagnation_detection


def _minimal_unified_config() -> dict:
    """A unified-format config that expands into a valid LLMConfig."""
    model = {
        "provider": "google",
        "model": "m",
        "fallback": {"provider": "google", "model": "m2"},
    }
    return {
        "default": model,
        "nodes": {},
    }


def test_unified_format_expansion_carries_decision_model():
    unified = _minimal_unified_config()
    unified["decision_model"] = {
        "enabled": True,
        "provider": "custom",
        "base_url": "http://d",
    }
    expanded = _expand_default_into_nodes(unified)
    assert expanded["decision_model"]["enabled"] is True
    llm_cfg = LLMConfig.model_validate(expanded)
    assert llm_cfg.decision_model.enabled is True


def test_llm_config_default_decision_model_is_disabled():
    llm_cfg = LLMConfig.model_validate(_expand_default_into_nodes(_minimal_unified_config()))
    assert llm_cfg.decision_model.enabled is False


# --- service resolution ----------------------------------------------------------------


def _ctx():
    return ArtemisContext.model_construct(device=MagicMock())


def test_resolution_prefers_execution_setup_over_llm_config():
    ctx = _ctx()
    ctx.llm_config = MagicMock()
    ctx.llm_config.decision_model = DecisionModelConfig(
        enabled=True, provider="custom", base_url="http://file/decide"
    )
    assert resolve_decision_config(ctx).base_url == "http://file/decide"
    ctx.execution_setup = ExecutionSetup(
        decision_model=DecisionModelConfig(
            enabled=True, provider="custom", base_url="http://sdk/decide"
        )
    )
    assert resolve_decision_config(ctx).base_url == "http://sdk/decide"


def test_resolution_falls_back_to_llm_config():
    ctx = _ctx()
    ctx.llm_config = MagicMock()
    ctx.llm_config.decision_model = DecisionModelConfig(enabled=False)
    ctx.execution_setup = ExecutionSetup()  # decision_model=None -> inherit
    assert resolve_decision_config(ctx).enabled is False


def test_get_decision_client_none_when_unconfigured():
    ctx = _ctx()
    assert get_decision_client(ctx) is None


def test_get_decision_client_none_when_disabled():
    ctx = _ctx()
    ctx.execution_setup = ExecutionSetup(decision_model=DecisionModelConfig(enabled=False))
    assert get_decision_client(ctx) is None


def test_get_decision_client_builds_and_caches(monkeypatch):
    from pydantic import SecretStr

    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct", raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    ctx = _ctx()
    ctx.execution_setup = ExecutionSetup(decision_model=DecisionModelConfig(enabled=True))
    first = get_decision_client(ctx)
    assert first is not None
    assert get_decision_client(ctx) is first  # cached singleton


def test_get_decision_client_invalid_config_returns_none(monkeypatch):
    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", None, raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", None, raising=False)
    ctx = _ctx()
    ctx.execution_setup = ExecutionSetup(decision_model=DecisionModelConfig(enabled=True))
    assert get_decision_client(ctx) is None


def test_use_flags_gates(monkeypatch):
    from pydantic import SecretStr

    from artemis.llm.decision import DecisionClient
    from artemis.config.settings import settings

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct", raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    cfg = DecisionModelConfig(enabled=True, use={"checker_verdict": False})
    client = DecisionClient(cfg)
    assert client.use_enabled("pixel_safety_net") is True
    assert client.use_enabled("checker_verdict") is False


# --- builder wiring ----------------------------------------------------------------------


def test_builder_with_decision_model():
    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    builder = AgentConfigBuilder()
    config = builder.with_decision_model(
        DecisionModelConfig(enabled=True, provider="custom", base_url="http://d")
    ).build(validate_profiles=False)
    assert config.decision_model is not None and config.decision_model.enabled

    # Also accepts field overrides.
    builder2 = AgentConfigBuilder()
    config2 = builder2.with_decision_model(
        enabled=True, provider="custom", base_url="http://d2"
    ).build(validate_profiles=False)
    assert config2.decision_model.base_url == "http://d2"

    # Default: no SDK override -> AgentConfig.decision_model stays None and the
    # file configuration inherits via ctx.llm_config.
    config3 = AgentConfigBuilder().build(validate_profiles=False)
    assert config3.decision_model is None


@pytest.mark.asyncio
async def test_context_exit_closes_the_decision_client():
    """ArtemisContext.__aexit__ best-effort closes a resolved decision client."""
    closed = []

    class _Client:
        def aclose(self):
            closed.append(1)

            async def _c():
                return None

            return _c()

    from artemis.context import DeviceContext

    ctx = _ctx()
    ctx.decision_client = _Client()
    await ctx.__aexit__(None, None, None)
    assert closed == [1]
    assert ctx.decision_client is None

    # A context without a client (or with one lacking aclose) exits cleanly.
    ctx2 = ArtemisContext(device=DeviceContext())
    ctx2.decision_client = object()  # no aclose
    await ctx2.__aexit__(None, None, None)
    ctx3 = ArtemisContext(device=DeviceContext())
    await ctx3.__aexit__(None, None, None)
    assert True


# --- with_decision_model override validation (P1 fix) ------------------------------------


def test_builder_dict_use_override_stays_a_typed_config(monkeypatch):
    """model_copy(update=...) skips validation; the builder must re-validate
    or `use` stays a plain dict and every use_enabled() lookup returns False
    (the whole layer silently no-ops)."""
    from pydantic import SecretStr

    from artemis.config.settings import settings
    from artemis.llm.decision import DecisionClient
    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "acct", raising=False)
    monkeypatch.setattr(settings, "CLOUDFLARE_AUTH_TOKEN", SecretStr("tok"), raising=False)
    config = (
        AgentConfigBuilder()
        .with_decision_model(enabled=True, use={"checker_verdict": False})
        .build(validate_profiles=False)
    )
    assert type(config.decision_model.use).__name__ == "DecisionModelUseConfig"
    client = DecisionClient(config.decision_model)
    assert client.use_enabled("pixel_safety_net") is True
    assert client.use_enabled("planner_validation") is True
    assert client.use_enabled("checker_verdict") is False


def test_builder_override_merges_nested_use_dicts():
    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    config = (
        AgentConfigBuilder()
        .with_decision_model(
            enabled=True, provider="custom", base_url="http://d", use={"checker_verdict": False}
        )
        .with_decision_model(use={"planner_validation": False})
        .build(validate_profiles=False)
    )
    use = config.decision_model.use
    assert use.checker_verdict is False  # earlier override preserved
    assert use.planner_validation is False  # new override applied
    assert use.pixel_safety_net is True  # untouched default


def test_builder_custom_override_without_base_url_raises():
    from pydantic import ValidationError

    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    with pytest.raises(ValidationError):
        AgentConfigBuilder().with_decision_model(enabled=True, provider="custom")


def test_builder_bare_call_is_a_noop_inheriting_file_config():
    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    builder = AgentConfigBuilder()
    assert builder.with_decision_model() is builder  # returns self, sets nothing
    assert builder.build(validate_profiles=False).decision_model is None


# --- SDK propagation into ExecutionSetup --------------------------------------------------


def test_prepare_tracing_propagates_decision_model():
    """AgentConfig.decision_model must reach ExecutionSetup.decision_model
    (dropping the propagation line would silently disable with_decision_model
    end-to-end)."""
    from artemis.sdk.agent import Agent

    cfg = DecisionModelConfig(enabled=True, provider="custom", base_url="http://d/decide")
    agent = Agent.__new__(Agent)  # bypass the heavy __init__
    # A real AgentConfig (a plain MagicMock cannot carry assert_failure_policy).
    from artemis.sdk.types.agent import AgentConfig
    from artemis.sdk.builders.agent_config_builder import AgentConfigBuilder

    config = AgentConfigBuilder().build(validate_profiles=False)
    config = config.model_copy(update={"decision_model": cfg})
    agent._config = config
    agent._session_id = None
    agent._tmp_traces_dir = __import__("pathlib").Path(__import__("tempfile").mkdtemp())

    task = MagicMock()
    task.id = "tid"
    task.get_name.return_value = "task-name"
    task.request.record_trace = True
    task.request.enable_remote_tracing = False
    task.request.trace_path = str(agent._tmp_traces_dir / "t")
    task.request.profile = None
    task.request.goal = "g"
    task.request.task_name = "task-name"

    context = MagicMock()
    context.device = None
    with patch("artemis.sdk.agent.DataEngine") as mock_data_engine_cls:
        mock_data_engine_cls.return_value = MagicMock()
        agent._prepare_tracing(task=task, context=context)

    assert context.execution_setup.decision_model is cfg
