# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Command-line interface for timeseries-council.
"""

import argparse
import sys
from pathlib import Path

from ..logging import get_logger, configure_logging
from ..version import __version__

logger = get_logger(__name__)


def serve_command(args):
    """Start the web server."""
    from ..web.app import run_server

    logger.info(f"Starting server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port, reload=args.reload)


def chat_command(args):
    """Start interactive chat session."""
    import os
    from ..providers import create_provider
    from ..forecasters import create_forecaster
    from ..detectors import create_detector
    from ..orchestrator import Orchestrator

    # Get API key from environment
    provider_env_vars = {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
    }

    env_var = provider_env_vars.get(args.provider)
    api_key = os.environ.get(env_var) if env_var else None

    if not api_key:
        print(f"Error: {env_var} environment variable not set")
        sys.exit(1)

    # Create provider
    provider = create_provider(args.provider, api_key=api_key)
    if not provider:
        print(f"Error: Failed to create provider '{args.provider}'")
        sys.exit(1)

    # Create forecaster
    forecaster = None
    if args.forecaster:
        forecaster = create_forecaster(args.forecaster)
        if not forecaster:
            print(f"Warning: Unknown forecaster '{args.forecaster}', using default")

    # Create detector
    detector = None
    if args.detector:
        detector = create_detector(args.detector)
        if not detector:
            print(f"Warning: Unknown detector '{args.detector}', using default")

    # Create orchestrator
    try:
        orchestrator = Orchestrator(
            llm_provider=provider,
            csv_path=args.csv_path,
            target_col=args.target,
            forecaster=forecaster,
            detector=detector
        )
    except Exception as e:
        print(f"Error: Failed to initialize: {e}")
        sys.exit(1)

    print(f"\nTime Series Council - Interactive Chat")
    print(f"Provider: {args.provider} | Data: {args.csv_path} | Target: {args.target}")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            response = orchestrator.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def version_command(args):
    """Show version information."""
    print(f"timeseries-council version {__version__}")


def download_models_command(args):
    """Download all foundation model weights."""
    from ..setup_models import download_all_models, setup_all

    print("Downloading foundation models from HuggingFace...")
    print(f"Sizes: {args.sizes}")
    print()

    if args.all:
        # Full setup: packages + models
        result = setup_all(auto_install=True, download_models=True, sizes=args.sizes)
    else:
        # Just download models
        result = download_all_models(sizes=args.sizes)

    if result["success"]:
        print(f"\n{result['message']}")
        if "models_downloaded" in result:
            print(f"Downloaded: {', '.join(result['models_downloaded'])}")
    else:
        print(f"\nFailed: {result['message']}")
        if "models_failed" in result:
            print(f"Failed models: {', '.join(result['models_failed'])}")
        sys.exit(1)


def status_command(args):
    """Show setup status for all models."""
    from ..setup_models import get_setup_status
    from ..forecasters import get_available_forecasters
    from ..detectors import get_available_detectors

    print("Time Series Council - Model Status\n")
    print("=" * 50)

    # Available forecasters
    forecasters = get_available_forecasters()
    print("\nForecasters (available):")
    for f in forecasters:
        print(f"  [OK] {f}")

    print("\nForecasters requiring special install:")
    special_install = {
        "moirai": "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git  (Python 3.10+)",
        "timesfm": "pip install timesfm  (requires JAX)",
        "lag-llama": "pip install lag-llama  (from git)",
    }
    for model, cmd in special_install.items():
        status = "[OK]" if model in forecasters else "[FALLBACK]"
        print(f"  {status} {model}: {cmd}")

    print()

    # Available detectors
    detectors = get_available_detectors()
    print("Detectors (available):")
    for d in detectors:
        print(f"  [OK] {d}")

    print()
    print("=" * 50)
    print("Note: Models with [FALLBACK] use statistical methods.")
    print("Install the packages above for full foundation model support.")

    # Detailed status
    if args.verbose:
        print("\nDetailed Package Status:")
        status = get_setup_status()
        for model, info in sorted(status.items()):
            pkg_status = "installed" if info["packages_installed"] else "missing"
            print(f"  {model}: packages {pkg_status}")


def main():
    """Main entry point for CLI."""
    configure_logging()

    parser = argparse.ArgumentParser(
        prog="timeseries-council",
        description="AI Council for Time Series Analysis"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    serve_parser.set_defaults(func=serve_command)

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat session")
    chat_parser.add_argument(
        "csv_path",
        help="Path to CSV file with time series data"
    )
    chat_parser.add_argument(
        "--target", "-t",
        default="sales",
        help="Target column name (default: sales)"
    )
    chat_parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "anthropic", "openai", "deepseek", "qwen"],
        help="LLM provider (default: gemini)"
    )
    chat_parser.add_argument(
        "--forecaster", "-f",
        help="Forecaster model (e.g., moirai, chronos, zscore_baseline)"
    )
    chat_parser.add_argument(
        "--detector", "-d",
        help="Anomaly detector (e.g., zscore, mad, isolation-forest)"
    )
    chat_parser.set_defaults(func=chat_command)

    # Download models command
    download_parser = subparsers.add_parser(
        "download-models",
        help="Download foundation model weights from HuggingFace"
    )
    download_parser.add_argument(
        "--sizes", "-s",
        nargs="+",
        default=["small"],
        choices=["tiny", "mini", "small", "base", "large"],
        help="Model sizes to download (default: small)"
    )
    download_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Full setup: install packages and download all models"
    )
    download_parser.set_defaults(func=download_models_command)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show available models and their status"
    )
    status_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed status"
    )
    status_parser.set_defaults(func=status_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Run the command
    args.func(args)


if __name__ == "__main__":
    main()
