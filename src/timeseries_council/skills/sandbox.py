# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Code Sandbox - Safe execution of dynamically generated code.
"""

import ast
import sys
import traceback
from typing import Dict, Any, Tuple, Optional, Set
from io import StringIO
import threading
import queue

from ..logging import get_logger

logger = get_logger(__name__)


class CodeSandbox:
    """
    Sandbox for executing dynamically generated Python code safely.

    Security measures:
    - Whitelist of allowed imports
    - No file system access
    - No network access
    - Execution timeout
    - Restricted builtins
    """

    # Allowed imports - safe for data analysis
    ALLOWED_IMPORTS: Set[str] = {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "math",
        "statistics",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "operator",
    }

    # Allowed sub-modules
    ALLOWED_SUBMODULES: Dict[str, Set[str]] = {
        "scipy": {"stats", "signal", "fft", "interpolate"},
        "sklearn": {"preprocessing", "metrics", "linear_model", "cluster"},
        "numpy": {"linalg", "random", "fft"},
        "pandas": {"api"},
    }

    # Dangerous builtins to block
    BLOCKED_BUILTINS: Set[str] = {
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "breakpoint",
    }

    # Note: We don't block __import__ because we provide a safe custom import
    # function. We block ast.Import/ImportFrom and only allow pre-approved modules.

    DEFAULT_TIMEOUT = 30  # seconds

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the sandbox.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate code before execution.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for dangerous constructs
        for node in ast.walk(tree):
            # Check for raw imports (we'll handle imports separately)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_import(alias.name):
                        return False, f"Import not allowed: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if not self._is_allowed_import(node.module, node.names):
                    return False, f"Import not allowed: {node.module}"

            # Check for dangerous attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    return False, f"Access to private attribute not allowed: {node.attr}"

            # Check for exec/eval in Call nodes
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"exec", "eval", "compile", "open"}:
                        return False, f"Function not allowed: {node.func.id}"

        return True, ""

    def _is_allowed_import(
        self,
        module: Optional[str],
        names: Optional[list] = None
    ) -> bool:
        """Check if an import is allowed."""
        if module is None:
            return False

        # Get the base module
        base = module.split(".")[0]

        if base not in self.ALLOWED_IMPORTS:
            return False

        # Check submodules
        if "." in module:
            parts = module.split(".")
            if len(parts) > 1:
                submodule = parts[1]
                allowed_subs = self.ALLOWED_SUBMODULES.get(base, set())
                if allowed_subs and submodule not in allowed_subs:
                    return False

        return True

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get a restricted set of builtins."""
        import builtins
        real_import = builtins.__import__
        allowed = self.ALLOWED_IMPORTS

        safe = {}
        for name in dir(builtins):
            if name not in self.BLOCKED_BUILTINS and not name.startswith("_"):
                safe[name] = getattr(builtins, name)

        # Add a safe __import__ that only allows whitelisted modules
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split('.')[0]
            if base_module in allowed:
                return real_import(name, globals, locals, fromlist, level)
            raise ImportError(f"Import not allowed: {name}")

        safe['__import__'] = safe_import

        return safe

    def _create_namespace(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe namespace for code execution."""
        namespace = {
            "__builtins__": self._get_safe_builtins(),
        }

        # Pre-import allowed modules
        try:
            import numpy as np
            import pandas as pd
            namespace["np"] = np
            namespace["numpy"] = np
            namespace["pd"] = pd
            namespace["pandas"] = pd
        except ImportError:
            pass

        try:
            import scipy
            namespace["scipy"] = scipy
        except ImportError:
            pass

        try:
            import sklearn
            namespace["sklearn"] = sklearn
        except ImportError:
            pass

        # Add math and statistics
        import math
        import statistics
        namespace["math"] = math
        namespace["statistics"] = statistics

        # Add datetime
        from datetime import datetime, timedelta, date
        namespace["datetime"] = datetime
        namespace["timedelta"] = timedelta
        namespace["date"] = date

        # Add inputs
        namespace.update(inputs)

        return namespace

    def execute(
        self,
        code: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in the sandbox.

        Args:
            code: Python code to execute
            inputs: Input variables available to the code
            timeout: Override default timeout

        Returns:
            Dictionary with:
                - success: bool
                - result: Any (return value if any)
                - output: str (stdout capture)
                - error: str (error message if failed)
                - namespace: dict (final namespace)
        """
        if inputs is None:
            inputs = {}

        timeout = timeout or self.timeout

        # Validate first
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "result": None,
                "output": "",
                "error": error,
                "namespace": {},
            }

        # Create namespace
        namespace = self._create_namespace(inputs)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        result_queue: queue.Queue = queue.Queue()

        def run_code():
            try:
                # Execute the code
                exec(code, namespace)

                # Try to get a result variable
                result = namespace.get("result", namespace.get("output", None))

                result_queue.put({
                    "success": True,
                    "result": result,
                    "output": sys.stdout.getvalue(),
                    "error": None,
                    "namespace": {
                        k: v for k, v in namespace.items()
                        if not k.startswith("_") and k not in {"np", "numpy", "pd", "pandas", "scipy", "sklearn", "math", "statistics", "datetime", "timedelta", "date"}
                    },
                })
            except Exception as e:
                result_queue.put({
                    "success": False,
                    "result": None,
                    "output": sys.stdout.getvalue(),
                    "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                    "namespace": {},
                })

        # Run in thread with timeout
        thread = threading.Thread(target=run_code)
        thread.start()
        thread.join(timeout=timeout)

        # Restore stdout
        sys.stdout = old_stdout

        if thread.is_alive():
            # Timeout occurred
            return {
                "success": False,
                "result": None,
                "output": "",
                "error": f"Execution timed out after {timeout} seconds",
                "namespace": {},
            }

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return {
                "success": False,
                "result": None,
                "output": "",
                "error": "No result returned from execution",
                "namespace": {},
            }

    def execute_function(
        self,
        code: str,
        function_name: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a function defined in the code.

        Args:
            code: Python code containing function definition
            function_name: Name of function to call
            args: Positional arguments
            kwargs: Keyword arguments
            inputs: Additional inputs for namespace

        Returns:
            Same as execute(), with result being the function's return value
        """
        args = args or []
        kwargs = kwargs or {}
        inputs = inputs or {}

        # First execute the code to define the function
        result = self.execute(code, inputs)

        if not result["success"]:
            return result

        # Check if function exists
        namespace = result["namespace"]
        if function_name not in namespace:
            return {
                "success": False,
                "result": None,
                "output": result["output"],
                "error": f"Function '{function_name}' not found in code",
                "namespace": namespace,
            }

        func = namespace[function_name]
        if not callable(func):
            return {
                "success": False,
                "result": None,
                "output": result["output"],
                "error": f"'{function_name}' is not callable",
                "namespace": namespace,
            }

        # Call the function
        try:
            func_result = func(*args, **kwargs)
            return {
                "success": True,
                "result": func_result,
                "output": result["output"],
                "error": None,
                "namespace": namespace,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "output": result["output"],
                "error": f"Error calling {function_name}: {type(e).__name__}: {str(e)}",
                "namespace": namespace,
            }
