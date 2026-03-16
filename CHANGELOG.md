# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-03-04

### Added
- Multi-model forecasting with Chronos, Moirai, TimesFM, Lag-Llama, and statistical baselines
- Anomaly detection with Merlion, Isolation Forest, LOF, MAD, Z-Score, LSTM-VAE, and PyOD detectors
- LLM orchestration with support for Anthropic, Google Gemini, OpenAI, DeepSeek, and Qwen providers
- AI Council deliberation system (single-provider and multi-provider modes)
- Detection Memory for stateful anomaly tracking across sessions
- FastAPI web interface with real-time chat and analysis
- CLI interface for command-line usage
- Backtesting and simulation tools
- Deployment security hardening (rate limiting, admin tokens, provider allowlists, upload security)
- GitHub Actions CI/CD pipeline with multi-version Python testing
