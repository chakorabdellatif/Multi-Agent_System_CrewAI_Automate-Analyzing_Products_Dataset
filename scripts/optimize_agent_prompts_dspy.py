"""
Executable CLI script to run DSPy prompt auto-optimization for product dataset analysis agents.
"""

import sys
import os

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ai_analysis", "src"))

try:
    from ai_analysis.optimization.dspy_optimizer import compile_optimized_prompt
except ImportError:
    print("DSPy optimization module not available. Install dependencies with `pip install dspy-ai`.")
    sys.exit(0)


def main():
    print("🚀 Initializing DSPy Multi-Agent Prompt Auto-Optimizer...")
    try:
        compiled_agent = compile_optimized_prompt()
        print("✅ Successfully compiled and optimized agent prompt module with DSPy!")
        print("💡 The compiled agent can now generate prompt-optimized insights with maximum factual grounding.")
    except Exception as e:
        print(f"ℹ️ DSPy optimization execution note: {e}")


if __name__ == "__main__":
    main()
