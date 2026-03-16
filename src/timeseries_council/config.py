# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Configuration management for timeseries-council.
Handles loading config.yaml and creating provider instances.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .logging import get_logger
from .exceptions import ConfigurationError
from .providers import create_provider, get_available_providers
from .providers.base import BaseLLMProvider

logger = get_logger(__name__)


def find_config_file() -> Optional[Path]:
    """Find config.yaml in common locations."""
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd().parent / "config.yaml",
        Path(__file__).parent.parent.parent.parent / "config.yaml",  # Project root
        Path.home() / ".config" / "timeseries-council" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


class Config:
    """Configuration manager for timeseries-council."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml (auto-detected if not provided)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = find_config_file()

        self._config = self._load_config()
        logger.info(f"Configuration loaded from: {self.config_path or 'defaults'}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path and self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    config = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded config with keys: {list(config.keys())}")
                    return config
            except ImportError:
                logger.warning("PyYAML not installed, using defaults")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        return {}

    def _get_api_key(self, provider_name: str, provider_config: Dict[str, Any]) -> str:
        """
        Get API key from config or environment variable.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration dict

        Returns:
            API key string

        Raises:
            ConfigurationError: If no API key is found
        """
        # First check for direct api_key in config
        if "api_key" in provider_config and provider_config["api_key"]:
            logger.debug(f"Using API key from config for {provider_name}")
            return provider_config["api_key"]

        # Then check environment variable
        api_key_env = provider_config.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                logger.debug(f"Using API key from env var {api_key_env}")
                return api_key

        # Default environment variable names
        default_env_vars = {
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
        }

        default_env = default_env_vars.get(provider_name.lower())
        if default_env:
            api_key = os.environ.get(default_env)
            if api_key:
                logger.debug(f"Using API key from default env var {default_env}")
                return api_key

        raise ConfigurationError(
            f"API key not found for {provider_name}. "
            f"Set {api_key_env or default_env} environment variable "
            f"or add 'api_key' to config.yaml",
            details={"provider": provider_name}
        )

    def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """
        Get a configured LLM provider instance.

        Args:
            provider_name: Provider name (defaults to default_provider in config)

        Returns:
            Configured LLM provider instance

        Raises:
            ConfigurationError: If provider not found or not configured
        """
        provider_name = provider_name or self._config.get("default_provider", "gemini")
        provider_name = provider_name.lower().strip()

        # Handle aliases
        if provider_name == "claude":
            provider_name = "anthropic"
        elif provider_name == "gpt":
            provider_name = "openai"

        logger.info(f"Getting provider: {provider_name}")

        providers_config = self._config.get("providers", {})
        provider_config = providers_config.get(provider_name, {})

        api_key = self._get_api_key(provider_name, provider_config)

        # Extract extra kwargs
        extra_kwargs = {
            k: v for k, v in provider_config.items()
            if k not in ("api_key", "api_key_env", "model")
        }

        try:
            return create_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=provider_config.get("model"),
                **extra_kwargs
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create provider {provider_name}: {e}",
                details={"provider": provider_name}
            )

    def get_council_providers(self) -> Dict[str, BaseLLMProvider]:
        """
        Get providers for each council role.

        Returns:
            Dict mapping role names to provider instances
        """
        council_config = self._config.get("council", {})
        roles = ["forecaster", "risk_analyst", "business_explainer"]

        logger.info("Getting council providers")

        if council_config.get("use_same_provider", True):
            provider_name = council_config.get(
                "provider",
                self._config.get("default_provider", "gemini")
            )
            provider = self.get_provider(provider_name)
            return {role: provider for role in roles}
        else:
            role_providers = council_config.get("roles", {})
            default_provider = self._config.get("default_provider", "gemini")

            result = {}
            for role in roles:
                provider_name = role_providers.get(role, default_provider)
                try:
                    result[role] = self.get_provider(provider_name)
                except ConfigurationError as e:
                    logger.warning(f"Could not get provider for {role}: {e}")
                    result[role] = self.get_provider(default_provider)
            return result

    def get_advanced_council_config(self) -> Dict[str, Any]:
        """
        Get configuration for the advanced (Karpathy-style) council.

        Returns:
            Dict with 'enabled', 'chairman', 'providers' keys
        """
        adv_config = self._config.get("advanced_council", {})

        if not adv_config.get("enabled", False):
            return {"enabled": False, "providers": {}, "chairman": None}

        model_names = adv_config.get("models", [])
        providers = {}

        for name in model_names:
            try:
                providers[name] = self.get_provider(name)
            except ConfigurationError as e:
                logger.warning(f"Could not load provider '{name}': {e}")

        if not providers:
            logger.warning("No providers available for advanced council")
            return {"enabled": False, "providers": {}, "chairman": None}

        chairman = adv_config.get("chairman", list(providers.keys())[0])
        if chairman not in providers:
            chairman = list(providers.keys())[0]

        logger.info(f"Advanced council enabled with {len(providers)} providers, chairman: {chairman}")

        return {
            "enabled": True,
            "providers": providers,
            "chairman": chairman
        }

    def get_forecaster_config(self) -> Dict[str, Any]:
        """Get forecaster configuration."""
        return self._config.get("forecasters", {
            "default": "moirai",
            "model_size": "small"
        })

    def get_detector_config(self) -> Dict[str, Any]:
        """Get anomaly detector configuration."""
        return self._config.get("detectors", {
            "default": "zscore",
            "sensitivity": 2.0
        })

    def get_server_config(self) -> Dict[str, Any]:
        """Get web server configuration."""
        return self._config.get("server", {"host": "0.0.0.0", "port": 8000})

    def get_data_config(self) -> Dict[str, Any]:
        """Get default data configuration."""
        return self._config.get("data", {
            "default_csv": "data/sample_sales.csv",
            "default_target": "sales"
        })

    @property
    def default_provider(self) -> str:
        """Get the default provider name."""
        return self._config.get("default_provider", "gemini")

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available provider names."""
        return get_available_providers()
