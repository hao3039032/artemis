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

"""LLM provider, model hierarchy, fallback chaining, and configuration loaders."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from artemis.config.constants import (
    LLM_CONFIG_FILENAME,
    AgentNode,
)
from artemis.config.paths import ROOT_DIR, get_config_path
from artemis.config.settings import settings
from artemis.utils.cython_compat import CyFunctionDetector
from third_party.mobile_use.config import llm as base_llm_config
from third_party.mobile_use.config.llm import (
    LLM,
    AgentNodeWithFallback,
    LLMConfigBase,
    LLMConfigUtils,
    LLMUtilsNodeWithFallback,
    LLMWithFallback,
    validate_vertex_ai_credentials,
)
from third_party.mobile_use.utils.file import load_jsonc
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "LLM",
    "AgentNodeWithFallback",
    "LLMUtilsNodeWithFallback",
    "CyFunctionDetector",
    "LLMConfig",
    "LLMConfigUtils",
    "LLMWithFallback",
    "deep_merge_llm_config",
    "get_default_llm_config",
    "initialize_llm_config",
    "lightweight_judge_default",
    "load_llm_config_override",
    "parse_llm_config",
    "validate_vertex_ai_credentials",
]


class DecisionModelUseConfig(BaseModel):
    """Per-decision-point switches for the decision-model service.

    Only meaningful while ``DecisionModelConfig.enabled`` is true; every gate
    defaults to on so a single master switch turns the whole layer on."""

    model_config = {"extra": "allow"}

    pixel_safety_net: bool = Field(
        default=True,
        description=(
            "Route the pre-action pixel safety net through the decision model"
            " (image decision point: the endpoint must support the Clef"
            " images[] extension)."
        ),
    )
    planner_validation: bool = Field(
        default=True,
        description="Route the advisory planner-change review through the decision model.",
    )
    checker_verdict: bool = Field(
        default=True,
        description=(
            "Route the Checker's final per-item verdicts through the decision"
            " model (the evidence-gathering tool loop itself stays on the LLM)."
        ),
    )
    stagnation_detection: bool = Field(
        default=True,
        description=(
            "Enable the Flash stagnation advisor (dHash heuristic gate +"
            " decision-model review, advisory notice only)."
        ),
    )
    agent_tool: bool = Field(
        default=True,
        description=(
            "Expose the 'ask_decision' agent tool (Flash runner and Pro"
            " Operator): calibrated second-opinion questions answered by the"
            " decision model with the current screenshot attached."
        ),
    )


class DecisionModelConfig(BaseModel):
    """Configuration for the standalone decision-model service (Clef family).

    Clef (Cloudflare, Apache 2.0) is a non-autoregressive scoring model: it
    answers calibrated-probability questions (``noul`` boolean / ``choice``
            option / ``score`` ordinal) over a text/JSON ``state`` plus up to
    four embedded images, with median latencies far below a general VLM. This
    config mounts it as an independent decision layer for Artemis's high-
    frequency micro-decisions; every call site keeps its existing VLM/LLM path
    as the fallback, so the default (``enabled: false``) reproduces today's
    behavior byte-for-byte.
    """

    model_config = {"extra": "allow"}

    enabled: bool = Field(
        default=False,
        description="Master switch. Disabled by default: no decision-model calls are made.",
    )
    provider: Literal["cloudflare", "custom"] = Field(
        default="cloudflare",
        description=(
            "'cloudflare' uses Workers AI (account id + token from"
            " CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_AUTH_TOKEN); 'custom' posts to"
            " a self-hosted Jev/Clef-compatible endpoint given by base_url."
        ),
    )
    model: str = Field(
        default="clef-flash",
        description=(
            "Decision model name ('clef' 27B or 'clef-flash' 9B). Defaults to"
            " clef-flash: the decision points are hot paths where latency wins."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Full decision-endpoint URL for provider='custom' (Jev/Clef"
            " compatible). Ignored for provider='cloudflare'."
        ),
    )
    timeout: float = Field(
        default=5.0,
        gt=0,
        description="Per-request timeout in seconds (one retry on timeout/5xx/429).",
    )
    use: DecisionModelUseConfig = Field(
        default_factory=DecisionModelUseConfig,
        description="Per-decision-point switches (all default to on).",
    )

    @model_validator(mode="after")
    def _custom_provider_needs_base_url(self) -> "DecisionModelConfig":
        if self.enabled and self.provider == "custom" and not (self.base_url or "").strip():
            raise ValueError(
                "decision_model.provider='custom' requires decision_model.base_url"
                " (a Jev/Clef-compatible decision endpoint URL)."
            )
        return self

    def validate_runtime(self) -> None:
        """Check provider credentials that live in settings, not in the file.

        Raises:
            ValueError: when enabled with provider='cloudflare' but the
                account id or auth token is missing from settings/.env.
        """
        if not self.enabled:
            return
        if self.provider == "cloudflare":
            missing = [
                name
                for name, value in (
                    ("CLOUDFLARE_ACCOUNT_ID", settings.CLOUDFLARE_ACCOUNT_ID),
                    ("CLOUDFLARE_AUTH_TOKEN", settings.CLOUDFLARE_AUTH_TOKEN),
                )
                if not (value and str(value).strip())
            ]
            if missing:
                raise ValueError(
                    "decision_model is enabled with provider='cloudflare' but"
                    f" {', '.join(missing)} is missing from settings/.env."
                )


def lightweight_judge_default() -> "LLMWithFallback":
    """Factory default for the lightweight judge nodes (pixel safety net and
    planner validation): a flash-lite model at temperature 0."""
    return LLMWithFallback(
        provider="google",
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        fallback=LLM(
            provider="google",
            model="gemini-3.1-flash-lite",
            temperature=0.0,
        ),
    )


class LLMConfig(LLMConfigBase):
    """Comprehensive LLM configuration mapping every node to primary/fallback models."""

    summarizer: LLMWithFallback
    operator: LLMWithFallback
    operator_summarizer: LLMWithFallback
    log_reader_sub_agent: LLMWithFallback
    log_analyzer: LLMWithFallback
    diagnoser: LLMWithFallback
    checker: LLMWithFallback
    planner_avatar: LLMWithFallback
    history_analyzer_expert: LLMWithFallback
    diagnoser_expert: LLMWithFallback
    explorer: LLMWithFallback
    history_analyzer: LLMWithFallback | None = None
    validator_pixel_safety_net: LLMWithFallback | None = None
    planner_validation: LLMWithFallback | None = None
    output_analyzer: LLMWithFallback | None = None
    decision_model: DecisionModelConfig = Field(
        default_factory=DecisionModelConfig,
        description=(
            "Standalone decision-model service config (Clef family). Off by"
            " default; parsed from the top-level 'decision_model' key of"
            " artemis.jsonc / llm-config.json."
        ),
    )

    def validate_providers(self) -> None:
        """Validate credentials across all configured agent nodes."""
        super().validate_providers()
        self.summarizer.validate_provider("Summarizer")
        self.operator.validate_provider("Operator")
        self.operator_summarizer.validate_provider("OperatorSummarizer")
        self.log_reader_sub_agent.validate_provider("LogReaderSubAgent")
        self.log_analyzer.validate_provider("LogAnalyzer")
        self.diagnoser.validate_provider("Diagnoser")
        self.checker.validate_provider("Checker")
        self.planner_avatar.validate_provider("PlannerAvatar")
        self.history_analyzer_expert.validate_provider("HistoryAnalyzerExpert")
        self.diagnoser_expert.validate_provider("DiagnoserExpert")
        self.explorer.validate_provider("Explorer")
        if self.history_analyzer:
            self.history_analyzer.validate_provider("HistoryAnalyzer")
        if self.validator_pixel_safety_net:
            self.validator_pixel_safety_net.validate_provider("ValidatorPixelSafetyNet")
        if self.planner_validation:
            self.planner_validation.validate_provider("PlannerValidation")
        if self.output_analyzer:
            self.output_analyzer.validate_provider("OutputAnalyzer")

    def get_agent(self, item: AgentNode) -> LLMWithFallback:
        """Retrieve model configuration for a specific agent node with sensible defaults."""
        val = getattr(self, item)
        if val is None:
            if item == "history_analyzer":
                return self.operator
            elif item in ("validator_pixel_safety_net", "planner_validation"):
                # Both are cheap, high-frequency judges: the pixel safety net
                # runs before actions, the planner validator after every
                # milestone edit. They share one lightweight default.
                return lightweight_judge_default()
            elif item == "output_analyzer":
                return self.log_analyzer
        return val


def _expand_default_into_nodes(config_dict: dict) -> dict:
    """Expand unified config format with 'default' and 'nodes' into full LLMConfig schema."""
    if "planner" in config_dict and "utils" in config_dict:
        return config_dict

    default_model_cfg = config_dict.get(
        "default",
        {
            "provider": "google",
            "model": "gemini-3.8-flash",
            "fallback": {
                "provider": "google",
                "model": "gemini-3.7-flash",
            },
        },
    )

    nodes_override = config_dict.get("nodes", {})

    all_agent_nodes = [
        "planner",
        "summarizer",
        "operator",
        "operator_summarizer",
        "log_reader_sub_agent",
        "log_analyzer",
        "diagnoser",
        "checker",
        "planner_avatar",
        "history_analyzer_expert",
        "diagnoser_expert",
        "explorer",
    ]

    all_utils_nodes = [
        "outputter",
        "hopper",
        "video_analyzer",
        "object_detector",
    ]

    result: dict[str, Any] = {}
    for node in all_agent_nodes:
        node_cfg = dict(default_model_cfg)
        if node in nodes_override:
            for k, v in nodes_override[node].items():
                if isinstance(v, dict) and isinstance(node_cfg.get(k), dict):
                    node_cfg[k] = {**node_cfg[k], **v}
                else:
                    node_cfg[k] = v
        result[node] = node_cfg

    utils_dict: dict[str, Any] = {}
    for util in all_utils_nodes:
        util_cfg = dict(default_model_cfg)
        if util in nodes_override:
            for k, v in nodes_override[util].items():
                if isinstance(v, dict) and isinstance(util_cfg.get(k), dict):
                    util_cfg[k] = {**util_cfg[k], **v}
                else:
                    util_cfg[k] = v
        utils_dict[util] = util_cfg
    result["utils"] = utils_dict

    # Top-level non-node sections (e.g. the decision-model service config)
    # must survive the expansion into the per-node schema.
    if "decision_model" in config_dict:
        result["decision_model"] = config_dict["decision_model"]

    return result


def parse_llm_config() -> LLMConfig:
    """Parse and instantiate LLMConfig from artemis.jsonc or llm-config.json."""
    config_path = None
    for candidate in ("artemis.jsonc", "artemis.json", LLM_CONFIG_FILENAME):
        try:
            config_path = get_config_path(candidate)
            break
        except FileNotFoundError:
            continue

    if not config_path:
        config_path = get_config_path(LLM_CONFIG_FILENAME, ROOT_DIR / LLM_CONFIG_FILENAME)

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = load_jsonc(f)
            expanded_dict = _expand_default_into_nodes(config_dict)
            return LLMConfig.model_validate(expanded_dict)
    except Exception as e:
        logger.error(f"Failed to load or parse llm config: {config_path}. Error: {e}")
        raise


def initialize_llm_config() -> LLMConfig:
    """Parse and validate credentials for LLMConfig."""
    return base_llm_config.initialize_llm_config(parse_llm_config)


def get_default_llm_config() -> LLMConfig:
    """Returns default LLMConfig parsed from standard configuration file."""
    return parse_llm_config()


def deep_merge_llm_config(base: LLMConfig, overrides: dict) -> LLMConfig:
    """Recursively merge dictionary overrides into an existing LLMConfig object."""
    base_dict = base.model_dump()

    def merge(d1: dict, d2: dict) -> None:
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                merge(d1[k], v)
            else:
                d1[k] = v

    merge(base_dict, overrides)
    return LLMConfig.model_validate(base_dict)


def load_llm_config_override(path: Path | str) -> LLMConfig:
    """Load custom LLM configuration JSON/JSONC overrides on top of default configuration."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        try:
            resolved_path = get_config_path(str(path))
        except OSError:
            pass
    return base_llm_config.load_llm_config_override(
        resolved_path, get_default_llm_config, deep_merge_llm_config
    )
