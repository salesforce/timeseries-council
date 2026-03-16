# Time Series Council

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/salesforce/timeseries-council/actions/workflows/ci.yml/badge.svg)](https://github.com/salesforce/timeseries-council/actions/workflows/ci.yml)
[![ICLR 2026](https://img.shields.io/badge/ICLR_2026-TSALM_Workshop-red.svg)](https://openreview.net/forum?id=LycMKa0o0b)

<div align="center">

### **Tune-as-Inference: Amortized Configuration Learning for Time Series Foundation Models**

[![Read the Paper](https://img.shields.io/badge/%F0%9F%93%84_Read_the_Paper-ICLR_2026_TSALM_Workshop-brightgreen?style=for-the-badge)](https://openreview.net/forum?id=LycMKa0o0b)

</div>

An AI-powered time series analysis library featuring multiple LLM providers, forecasting models, anomaly detectors, and a unique multi-agent council deliberation system.

## Features

- **Multi-Provider LLM Support**: Gemini, Claude, OpenAI, DeepSeek, Qwen
- **Multiple Forecasting Models**: Moirai2, Chronos, TimesFM, Ti-Rex, Lag-Llama, LLM-based
- **Multiple Anomaly Detectors**: Z-score, MAD, RuleDetector, Isolation Forest, LOF, LSTM-VAE
- **Three Analysis Modes**:
  - **Standard**: Single LLM with tool execution
  - **Council**: 3-role council (Forecaster, Risk Analyst, Business Explainer)
  - **Advanced Council**: Karpathy-style 3-stage deliberation with peer ranking
- **Web UI**: FastAPI-based interface with real-time progress updates
- **Structured Logging**: Configurable logging throughout

## Installation

```bash
# From source
git clone https://github.com/salesforce/timeseries-council.git
cd timeseries-council
pip install -e ".[all]"
```

## Foundation Models Setup

The library includes several foundation models for time series forecasting. Some models work out of the box, while others require additional setup.

### Models Included (work out of box)

- **Chronos** (Amazon) - `chronos-forecasting` package
- **Merlion detectors** (Salesforce) - `salesforce-merlion` package
- **Statistical baselines** - `statsmodels` package

### Models Requiring Special Installation

| Model | Requirement | Installation |
|-------|------------|--------------|
| **Moirai2** | Python 3.10+ | `pip install git+https://github.com/SalesforceAIResearch/uni2ts.git` |
| **TimesFM** | JAX runtime | `pip install timesfm` |
| **Ti-Rex** | Special install | `pip install tirex-ts` |
| **PyOD** (ECOD, COPOD, HBOS, KNN, OCSVM, LODA) | — | `pip install pyod` |

> **Note**: Without special installs, TimesFM/Ti-Rex/Moirai2 will use statistical fallback methods.

### Download Model Weights

Foundation models need to download weights from HuggingFace on first use. You can pre-download them:

```bash
# Download small model weights (recommended)
timeseries-council download-models

# Download specific sizes (tiny, small, base, large)
timeseries-council download-models --sizes small base

# Full setup: install packages + download all models
timeseries-council download-models --all

# Check model status
timeseries-council status
timeseries-council status --verbose
```

### Complete Setup (Recommended)

For the best experience with all models:

```bash
# 1. Install the package with all dependencies
pip install -e ".[all]"

# 2. Install Moirai2 (requires Python 3.10+)
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git

# 3. Install PyOD detectors (optional)
pip install pyod

# 4. Download model weights
timeseries-council download-models --sizes small

# 5. Verify all models are available
timeseries-council status
```

Expected output from `status`:
```
Forecasters (available):
  [OK] zscore_baseline
  [OK] llm
  [OK] moirai2
  [OK] chronos
  [OK] timesfm
  [OK] lag-llama

Detectors (available):
  [OK] zscore
  [OK] mad
  [OK] isolation-forest
  [OK] lof
  [OK] merlion
  ...
```

## Quick Start

### Python API

```python
from timeseries_council import Orchestrator
from timeseries_council.providers import create_provider
from timeseries_council.forecasters import create_forecaster
from timeseries_council.detectors import create_detector
import os

# Create LLM provider
api_key = os.getenv("GEMINI_API_KEY")
provider = create_provider("gemini", api_key)  # or "anthropic", "openai", etc.

# Create optional forecaster and detector
forecaster = create_forecaster("moirai")
detector = create_detector("zscore")

# Initialize orchestrator
orchestrator = Orchestrator(
    llm_provider=provider,
    csv_path="data/sample_sales.csv",
    target_col="sales",
    forecaster=forecaster,
    detector=detector
)

# Chat with your data
response = orchestrator.chat("What will sales be next week?")
print(response)

# Use council mode for multi-perspective analysis
response = orchestrator.chat_with_council("Analyze the sales trend")
print(response)
```

### Web Interface

```bash
# Start the web server
timeseries-council serve --host 127.0.0.1 --port 8000

# Or with uvicorn directly
uvicorn timeseries_council.web.app:create_app --factory --reload
```

Then open http://localhost:8000 in your browser.

### CLI

```bash
# Interactive CLI session
timeseries-council chat data/sales.csv --target sales --provider gemini

# With custom models
timeseries-council chat data/sales.csv \
    --target sales \
    --provider anthropic \
    --forecaster chronos \
    --detector isolation_forest
```

## Configuration

### Environment Variables

```bash
# LLM API Keys
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"  # For Qwen

# Deployment hardening (recommended for public hosting)
export TS_ENABLED_PROVIDERS="anthropic"       # Comma-separated allowlist (unset = allow all)
export TS_ENABLE_RAW_SESSION_PATH="false"    # Keep upload-only flow
# export TS_ADMIN_TOKEN="set-strong-token"   # Required to enable /api/models/setup*
export TS_MAX_UPLOAD_MB="20"                 # Reject larger uploads (HTTP 413)
export TS_EXPOSE_SESSION_LIST="false"        # Keep /api/sessions hidden
# export TS_SESSION_API_TOKEN="set-token"    # Optional X-Session-Token gate for session/upload APIs
export TS_ENABLE_DYNAMIC_SKILLS="false"      # Keep runtime code generation disabled
export TS_RATE_LIMIT_ENABLED="true"          # Enable in-app per-IP rate limits
export TS_RATE_LIMIT_WINDOW_SECONDS="60"     # Shared window duration
export TS_RATE_LIMIT_UPLOAD_PER_WINDOW="10"  # Upload requests/IP/window
export TS_RATE_LIMIT_SESSION_PER_WINDOW="20" # Session requests/IP/window
export TS_RATE_LIMIT_CHAT_PER_WINDOW="60"    # Chat requests/IP/window
export TS_RATE_LIMIT_DEFAULT_PER_WINDOW="120"# Other guarded endpoints/IP/window

# Logging
export TIMESERIES_COUNCIL_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Configuration File (config.yaml)

```yaml
default_provider: gemini
default_forecaster: moirai
default_detector: zscore

providers:
  gemini:
    model: gemini-2.5-flash
  anthropic:
    model: claude-sonnet-4-20250514

forecasters:
  moirai:
    context_length: 512
  chronos:
    model_size: small

detectors:
  zscore:
    sensitivity: 2.0
  isolation_forest:
    contamination: 0.1

logging:
  level: INFO
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
```

### Reverse Proxy

For production deployments, place the app behind a reverse proxy (e.g., Nginx):
- Bind the app to `127.0.0.1:8000` (default)
- Expose only the reverse proxy on ports `80`/`443`

## Modes

### Standard Mode

Single LLM analyzes your query and executes appropriate tools:

```python
response = orchestrator.chat("Forecast the next 7 days")
```

### Council Mode

Three specialized roles provide different perspectives:

- **Quantitative Analyst**: Focus on numbers and statistics
- **Risk Analyst**: Identify potential risks and uncertainties
- **Business Explainer**: Translate insights for stakeholders

```python
response = orchestrator.chat_with_council("What's the outlook for Q4?")
```

### Advanced Council Mode

Karpathy-style 3-stage deliberation:

1. **Stage 1**: All models provide initial responses
2. **Stage 2**: Models rank each other's responses (anonymized)
3. **Stage 3**: Chairman synthesizes final answer

```python
from timeseries_council.council import AdvancedCouncil

council = AdvancedCouncil(
    council_providers={
        "gemini": provider1,
        "claude": provider2,
        "gpt4": provider3
    },
    chairman_name="claude"
)

result = council.run_sync(
    user_query="What will sales be next month?",
    context="Historical data context..."
)
```

## Available Models

### Forecasters

| Name | Description | Dependencies |
|------|-------------|--------------|
| `moirai` | Salesforce Moirai2 via uni2ts | `uni2ts`, `gluonts`, `torch` |
| `chronos` | Amazon Chronos | `chronos-forecasting` |
| `timesfm` | Google TimesFM | `timesfm` |
| `tirex` | Ti-Rex | `tirex-ts` |
| `lag_llama` | Lag-Llama | `lag-llama` |
| `llm` | LLM-based forecasting | LLM provider |
| `zscore_baseline` | Simple statistical baseline | Built-in |

### Detectors

| Name | Description | Dependencies |
|------|-------------|--------------|
| `zscore` | Z-score detection | Built-in |
| `mad` | Median Absolute Deviation | Built-in |
| `ruledetector` | Rule-based Anomaly Detector | Built-in |
| `isolation_forest` | Isolation Forest | `scikit-learn` |
| `lof` | Local Outlier Factor | `scikit-learn` |
| `moirai` | Moirai2 back-prediction | `uni2ts`, `gluonts`, `torch` |
| `merlion` | Merlion ensemble | `salesforce-merlion` |
| `lstm_vae` | LSTM Variational Autoencoder | `torch` |
| `pyod` | PyOD detectors (ECOD, COPOD, HBOS, KNN, OCSVM, LODA) | `pyod` |
| `llm` | LLM-based detection | LLM provider |

### LLM Providers

| Name | Description | Environment Variable |
|------|-------------|---------------------|
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `anthropic` | Anthropic Claude | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI GPT | `OPENAI_API_KEY` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `qwen` | Alibaba Qwen | `DASHSCOPE_API_KEY` |

## Tools

The orchestrator has access to these analysis tools:

| Tool | Description |
|------|-------------|
| `forecast` | Generate time series forecasts |
| `describe_series` | Statistical summary of the series |
| `detect_anomalies` | Find anomalies in the data |
| `decompose_series` | Trend, seasonal, residual decomposition |
| `compare_series` | Compare multiple time periods |
| `what_if_simulation` | Scenario analysis |

## Detection Memory

The anomaly detection pipeline supports **stateful memory** via the `DetectionMemory` dataclass. This allows callers to pass historical context (baseline statistics, expected ranges, domain knowledge) so detectors can make more informed decisions.

### DetectionMemory Fields

| Field | Type | Description |
|-------|------|-------------|
| `baseline_stats` | `Dict[str, float]` | Known-normal statistics: `mean`, `std`, `median`, `mad` |
| `expected_range` | `Optional[List[float]]` | Expected value range `[low, high]` — values inside are filtered out |
| `context` | `Any` | Free-form domain context (used by LLM detector in its prompt) |

### How Each Detector Uses Memory

| Detector | Integration |
|----------|-------------|
| **Z-Score** | Computes z-scores against baseline mean/std instead of batch stats |
| **MAD** | Uses baseline median and MAD for modified z-score computation |
| **Isolation Forest** | Adds `baseline_zscore` as an extra feature to the model |
| **LOF** | Adds `baseline_zscore` as an extra feature to the model |
| **PyOD** (ECOD, COPOD, HBOS, KNN, OCSVM, LODA) | Adds `baseline_zscore` as an extra feature |
| **LSTM-VAE** | Normalizes data using baseline mean/std instead of batch stats |
| **Moirai2** | Boosts severity with `max(model_severity, baseline_z)` |
| **Merlion** (WindStats, Spectral, Prophet) | Scales alarm threshold by baseline/current std ratio |
| **LLM** | Injects baseline stats, expected range, and domain context into prompt |

All detectors also apply shared post-processing via `_apply_memory()`: baseline rescoring and expected-range filtering.

### Usage

```python
from timeseries_council.tools import detect_anomalies
from timeseries_council.types import DetectionMemory

# Create memory with known-normal baseline
memory = DetectionMemory(
    baseline_stats={"mean": 100.0, "std": 10.0, "median": 98.0},
    expected_range=[80, 120],
    context="Holiday season — expect higher variability",
)

# Detection uses baseline for scoring, filters values within [80, 120]
result = detect_anomalies(series=my_series, memory=memory)
```

### Orchestrator Auto-Accumulation

When using the `Orchestrator`, detection memory is automatically accumulated across calls. After each detection run, baseline stats (mean, std, median) from the result metadata are stored and passed to subsequent detection calls.

```python
orchestrator = Orchestrator(llm_provider=provider, csv_path="data/sales.csv", target_col="sales")

# First call — detects anomalies, stores baseline stats in memory
response1 = orchestrator.chat("Are there anomalies in the data?")

# Second call — detection now uses first run's stats as baseline context
response2 = orchestrator.chat("Check for anomalies in this updated data")
```

## Progress Tracking

The library supports real-time progress tracking via callbacks:

```python
from timeseries_council.types import ProgressStage

def my_callback(stage: ProgressStage, message: str, progress: float):
    print(f"[{stage.value}] {progress:.0%} - {message}")

orchestrator = Orchestrator(
    llm_provider=provider,
    csv_path="data/sales.csv",
    target_col="sales",
    progress_callback=my_callback
)
```

Progress stages:
- `INITIALIZING` - Starting up
- `TOOL_SELECTION` - LLM choosing tools
- `FORECASTING` - Running forecast
- `DETECTING` - Running anomaly detection
- `COUNCIL_STAGE_1` - Collecting opinions
- `COUNCIL_STAGE_2` - Peer ranking
- `COUNCIL_STAGE_3` - Chairman synthesis
- `COMPLETE` - Done

## Examples

See the [examples/](examples/) directory for:

- `demo.ipynb` - Interactive Jupyter notebook demo
- `sample_data.py` - Generate sample datasets
- `finetune_vertex.py` - Fine-tune models on Vertex AI
- `generate_training_data.py` - Create training data

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=timeseries_council

# Type checking
mypy src/timeseries_council

# Linting
ruff check src/
```

## Architecture

```
timeseries-council/
├── src/timeseries_council/
│   ├── providers/      # LLM provider implementations
│   ├── forecasters/    # Forecasting model implementations
│   ├── detectors/      # Anomaly detector implementations
│   ├── tools/          # Analysis tools
│   ├── council/        # Council orchestration
│   ├── web/            # Web interface
│   ├── cli/            # Command-line interface
│   ├── orchestrator.py # Main orchestration logic
│   ├── config.py       # Configuration management
│   ├── types.py        # Type definitions
│   ├── logging.py      # Logging utilities
│   └── exceptions.py   # Custom exceptions
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Development setup
- Running tests and linting
- Submitting pull requests

## Citation

If you use Time Series Council in your research, please cite our [ICLR 2026 TSALM Workshop paper](https://openreview.net/forum?id=LycMKa0o0b):

```bibtex
@inproceedings{tune_as_inference_2026,
  title={Tune-as-Inference: Amortized Configuration Learning for Time Series Foundation Models},
  author={Gupta, Piyush and Reddy, Sriteja and Singh, Manpreet and Sahoo, Doyen},
  booktitle={ICLR 2026 Workshop on Time Series and Language Models (TSALM)},
  year={2026},
  url={https://openreview.net/forum?id=LycMKa0o0b}
}
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Karpathy's LLM Council](https://github.com/karpathy/llm-council) for the advanced council concept
- [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) for forecasting
- [Google TimesFM](https://github.com/google-research/timesfm) for time series foundation model
- [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) for probabilistic forecasting
- [GluonTS](https://github.com/awslabs/gluonts) for dataset infrastructure
- [IIIT Hyderabad](https://www.iiit.ac.in/) for research collaboration
