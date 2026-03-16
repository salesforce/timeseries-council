# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Minimal setup.py for editable installs with older pip versions.
Configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="timeseries-council",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
