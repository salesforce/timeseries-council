# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Auto-setup module for downloading and configuring models.

Handles:
- Checking if required packages are installed
- Installing missing packages via pip
- Downloading model weights from HuggingFace
"""

import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)

# Package requirements for each model/detector
PACKAGE_REQUIREMENTS = {
    # Forecasters
    "moirai": ["einops", "torch", "huggingface_hub"],
    "chronos": ["torch", "transformers"],  # Primary name
    "chronos2": ["torch", "transformers"],  # Alias for backwards compatibility
    "timesfm": ["torch", "transformers", "huggingface_hub"],
    "lag-llama": ["torch", "gluonts", "huggingface_hub"],
    "tirex": ["torch", "transformers", "huggingface_hub"],
    # Detectors
    "merlion": ["salesforce-merlion"],
    "windstats": ["salesforce-merlion"],
    "spectral": ["salesforce-merlion"],
    "prophet": ["salesforce-merlion", "prophet"],
    "isolation-forest": ["scikit-learn"],
    "lof": ["scikit-learn"],
    "lstm-vae": ["torch"],

    # PyOD detectors
    "ecod": ["pyod"],
    "copod": ["pyod"],
    "hbos": ["pyod"],
    "knn": ["pyod"],
    "ocsvm": ["pyod"],
    "loda": ["pyod"],
}

# HuggingFace model IDs
HUGGINGFACE_MODELS = {
    "moirai-small": "Salesforce/moirai-2.0-R-small",
    "moirai-base": "Salesforce/moirai-2.0-R-base",
    "moirai-large": "Salesforce/moirai-2.0-R-large",
    "chronos2-base": "amazon/chronos-2",
    "chronos2-synth": "autogluon/chronos-2-synth",
    "chronos2-small": "autogluon/chronos-2-small",
    "timesfm": "google/timesfm-1.0-200m",
    "lag-llama": "time-series-foundation-models/Lag-Llama",
}


def check_package_installed(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        # Map package names to import names
        import_name = package.replace("-", "_").replace("salesforce_merlion", "merlion")
        if package == "salesforce-merlion":
            import_name = "merlion"
        elif package == "scikit-learn":
            import_name = "sklearn"
        elif package == "huggingface_hub":
            import_name = "huggingface_hub"

        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package: str) -> Tuple[bool, str]:
    """Install a package via pip."""
    try:
        logger.info(f"Installing package: {package}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"Successfully installed: {package}")
            return True, f"Installed {package}"
        else:
            error = result.stderr or result.stdout
            logger.error(f"Failed to install {package}: {error}")
            return False, f"Failed to install {package}: {error}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout installing {package}"
    except Exception as e:
        return False, f"Error installing {package}: {str(e)}"


def ensure_packages_installed(model_name: str, auto_install: bool = True) -> Dict[str, Any]:
    """
    Ensure required packages for a model are installed.

    Args:
        model_name: Name of the model/detector
        auto_install: Whether to auto-install missing packages

    Returns:
        Dict with status and any errors
    """
    model_key = model_name.lower().strip()

    if model_key not in PACKAGE_REQUIREMENTS:
        return {
            "success": True,
            "message": f"No special requirements for {model_name}",
            "packages_installed": []
        }

    required = PACKAGE_REQUIREMENTS[model_key]
    missing = []
    installed = []

    for package in required:
        if not check_package_installed(package):
            missing.append(package)
        else:
            installed.append(package)

    if not missing:
        return {
            "success": True,
            "message": f"All packages for {model_name} are installed",
            "packages_installed": installed
        }

    if not auto_install:
        return {
            "success": False,
            "message": f"Missing packages: {', '.join(missing)}",
            "missing_packages": missing,
            "installed_packages": installed
        }

    # Auto-install missing packages
    install_results = []
    for package in missing:
        success, msg = install_package(package)
        install_results.append({
            "package": package,
            "success": success,
            "message": msg
        })
        if success:
            installed.append(package)

    all_success = all(r["success"] for r in install_results)

    return {
        "success": all_success,
        "message": "All packages installed" if all_success else "Some packages failed to install",
        "packages_installed": installed,
        "install_results": install_results
    }


def download_model_weights(model_name: str, force: bool = False) -> Dict[str, Any]:
    """
    Download model weights from HuggingFace.

    Args:
        model_name: Name of the model
        force: Force re-download even if exists

    Returns:
        Dict with download status
    """
    model_key = model_name.lower().strip()

    # Find matching HuggingFace model
    hf_model_id = None
    for key, model_id in HUGGINGFACE_MODELS.items():
        if key in model_key or model_key in key:
            hf_model_id = model_id
            break

    if not hf_model_id:
        return {
            "success": True,
            "message": f"No HuggingFace model to download for {model_name}",
            "downloaded": False
        }

    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        logger.info(f"Downloading model: {hf_model_id}")

        # Download the model (will cache automatically)
        cache_path = snapshot_download(
            repo_id=hf_model_id,
            repo_type="model",
            local_dir_use_symlinks=True
        )

        logger.info(f"Model downloaded to: {cache_path}")

        return {
            "success": True,
            "message": f"Downloaded {hf_model_id}",
            "downloaded": True,
            "cache_path": str(cache_path),
            "model_id": hf_model_id
        }

    except ImportError:
        return {
            "success": False,
            "message": "huggingface_hub not installed. Run: pip install huggingface_hub",
            "downloaded": False
        }
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return {
            "success": False,
            "message": f"Download failed: {str(e)}",
            "downloaded": False
        }


def setup_model(model_name: str, auto_install: bool = True) -> Dict[str, Any]:
    """
    Full setup for a model: install packages and download weights.

    Args:
        model_name: Name of the model/detector
        auto_install: Whether to auto-install packages

    Returns:
        Dict with full setup status
    """
    logger.info(f"Setting up model: {model_name}")

    # Step 1: Ensure packages
    package_result = ensure_packages_installed(model_name, auto_install)

    if not package_result["success"]:
        return {
            "success": False,
            "step": "packages",
            "message": package_result["message"],
            "details": package_result
        }

    # Step 2: Download weights (if applicable)
    download_result = download_model_weights(model_name)

    if not download_result["success"]:
        return {
            "success": False,
            "step": "download",
            "message": download_result["message"],
            "details": {
                "packages": package_result,
                "download": download_result
            }
        }

    return {
        "success": True,
        "message": f"Model {model_name} is ready",
        "details": {
            "packages": package_result,
            "download": download_result
        }
    }


def get_setup_status() -> Dict[str, Dict[str, Any]]:
    """Get setup status for all models."""
    status = {}

    all_models = set(list(PACKAGE_REQUIREMENTS.keys()) + list(HUGGINGFACE_MODELS.keys()))

    for model in sorted(all_models):
        # Check packages
        packages_ok = True
        if model in PACKAGE_REQUIREMENTS:
            for pkg in PACKAGE_REQUIREMENTS[model]:
                if not check_package_installed(pkg):
                    packages_ok = False
                    break

        status[model] = {
            "packages_installed": packages_ok,
            "requires": PACKAGE_REQUIREMENTS.get(model, []),
            "has_hf_model": any(model in k or k in model for k in HUGGINGFACE_MODELS)
        }

    return status


def setup_all_detectors(auto_install: bool = True) -> Dict[str, Any]:
    """Setup all anomaly detectors."""
    detectors = [
        "zscore", "mad", "isolation-forest", "lof", "windstats", "spectral", "prophet", "lstm-vae",
        "ecod", "copod", "hbos", "knn", "ocsvm", "loda",
    ]
    results = {}

    for detector in detectors:
        results[detector] = setup_model(detector, auto_install)

    success_count = sum(1 for r in results.values() if r["success"])

    return {
        "success": success_count == len(detectors),
        "message": f"Setup {success_count}/{len(detectors)} detectors successfully",
        "results": results
    }


def setup_all_forecasters(auto_install: bool = True) -> Dict[str, Any]:
    """Setup all forecasters."""
    forecasters = ["moirai", "chronos", "timesfm", "tirex", "lag-llama"]
    results = {}

    for forecaster in forecasters:
        results[forecaster] = setup_model(forecaster, auto_install)

    success_count = sum(1 for r in results.values() if r["success"])

    return {
        "success": success_count == len(forecasters),
        "message": f"Setup {success_count}/{len(forecasters)} forecasters successfully",
        "results": results
    }


def download_all_models(sizes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Download all foundation model weights from HuggingFace.

    This pre-downloads models so they're ready for use without internet.

    Args:
        sizes: List of model sizes to download. Default: ["small"]
               Options: ["tiny", "small", "base", "large"]

    Returns:
        Dict with download status for each model
    """
    if sizes is None:
        sizes = ["small"]

    models_to_download = []

    # Add Chronos2 models (base, synth, small)
    for size in sizes:
        if size in ["base", "synth", "small"]:
            models_to_download.append(f"chronos2-{size}")

    # Add Moirai models (small, base, large)
    for size in sizes:
        if size in ["small", "base", "large"]:
            models_to_download.append(f"moirai-{size}")

    # Add TimesFM (only one size)
    models_to_download.append("timesfm")

    # Add Lag-Llama (only one size)
    models_to_download.append("lag-llama")

    results = {}
    total = len(models_to_download)

    logger.info(f"Downloading {total} foundation models...")

    for i, model in enumerate(models_to_download, 1):
        logger.info(f"[{i}/{total}] Downloading {model}...")
        results[model] = download_model_weights(model)

    success_count = sum(1 for r in results.values() if r.get("success", False))

    return {
        "success": success_count == total,
        "message": f"Downloaded {success_count}/{total} models successfully",
        "models_downloaded": [k for k, v in results.items() if v.get("success")],
        "models_failed": [k for k, v in results.items() if not v.get("success")],
        "results": results
    }


def setup_all(auto_install: bool = True, download_models: bool = True, sizes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Complete setup: install packages and download all model weights.

    Args:
        auto_install: Auto-install missing packages
        download_models: Download model weights from HuggingFace
        sizes: Model sizes to download (default: ["small"])

    Returns:
        Dict with complete setup status
    """
    logger.info("Starting complete setup...")

    # Setup detectors
    detector_result = setup_all_detectors(auto_install)
    logger.info(f"Detectors: {detector_result['message']}")

    # Setup forecasters
    forecaster_result = setup_all_forecasters(auto_install)
    logger.info(f"Forecasters: {forecaster_result['message']}")

    # Download models
    download_result = {"success": True, "message": "Skipped"}
    if download_models:
        download_result = download_all_models(sizes)
        logger.info(f"Model downloads: {download_result['message']}")

    all_success = detector_result["success"] and forecaster_result["success"] and download_result["success"]

    return {
        "success": all_success,
        "message": "Setup complete" if all_success else "Setup completed with some failures",
        "detectors": detector_result,
        "forecasters": forecaster_result,
        "downloads": download_result
    }
