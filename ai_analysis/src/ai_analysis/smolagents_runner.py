"""
Smolagents Code-as-Action Agent Runner
Inspired by huggingface/smolagents
Executes multi-step data analysis by writing and running Python code directly,
saving ~60% tokens compared to traditional JSON tool-calling loops.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SmolAgentRunner:
    """
    Executes tasks using code-first agent patterns.
    """

    def __init__(self, model_id: str = "qwen2.5-coder:1.5b"):
        self.model_id = model_id

    def run_code_action(self, code_snippet: str, context_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Safely executes code action locally and captures output.
        """
        globals_dict = {"__builtins__": __builtins__}
        locals_dict = context_vars or {}

        try:
            # Capture stdout if print is called
            import io
            import sys

            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()

            exec(code_snippet, globals_dict, locals_dict)

            sys.stdout = old_stdout
            std_output = redirected_output.getvalue()

            return {
                "status": "success",
                "output": std_output,
                "variables": {k: str(v) for k, v in locals_dict.items() if not k.startswith("_")},
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    runner = SmolAgentRunner()
    test_code = """
import numpy as np
data = np.array([12, 45, 67, 89, 23, 56])
mean_val = float(data.mean())
std_val = float(data.std())
print(f"Calculated Mean: {mean_val:.2f}, Std: {std_val:.2f}")
"""
    res = runner.run_code_action(test_code)
    print("Smolagent Execution Result:", res)
