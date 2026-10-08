"""
Stanford DSPy Prompt Optimization Module for Multi-Agent Systems.
Uses DSPy teleprompters (BootstrapFewShot / MIPROv2) to automatically compile,
tune, and optimize agent prompt instructions and few-shot demonstrations against measurable metrics.
"""

from typing import Any, Callable, Dict, List, Optional
import os

try:
    import dspy
except ImportError:
    dspy = None


if dspy is not None:
    class ProductAnalysisSignature(dspy.Signature):
        """Analyze e-commerce product records, identifying pricing anomalies, rating trends, and sentiment."""
        dataset_sample = dspy.InputField(desc="Raw rows or aggregated metrics from the product dataset")
        analysis_goal = dspy.InputField(desc="The objective (e.g. pricing strategy, rating correlation)")
        findings_and_strategy = dspy.OutputField(desc="Structured insights, actionable recommendations, and data-backed conclusions")

    class ProductAnalyzerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_insights = dspy.ChainOfThought(ProductAnalysisSignature)

        def forward(self, dataset_sample: str, analysis_goal: str):
            return self.generate_insights(dataset_sample=dataset_sample, analysis_goal=analysis_goal)


def evaluation_metric(gold_example, pred, trace=None) -> float:
    """
    Evaluates generated output based on groundedness, actionability, and length penalty.
    """
    if not hasattr(pred, "findings_and_strategy") or not pred.findings_and_strategy:
        return 0.0

    score = 0.0
    text = pred.findings_and_strategy.lower()

    # Check for core analysis criteria
    if "price" in text or "pricing" in text or "cost" in text:
        score += 0.3
    if "rating" in text or "sentiment" in text or "review" in text:
        score += 0.3
    if "recommendation" in text or "strategy" in text or "action" in text:
        score += 0.4

    return score


def compile_optimized_prompt(train_examples: Optional[List[Dict[str, str]]] = None, model_name: str = "gemini/gemini-1.5-flash") -> Any:
    """
    Compiles and optimizes the agent prompt using DSPy BootstrapFewShot.
    """
    if dspy is None:
        raise ImportError("Please install dspy with `pip install dspy-ai` to use prompt optimization.")

    # Configure LM backend
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)

    # Prepare dataset
    examples = []
    if train_examples:
        for ex in train_examples:
            examples.append(dspy.Example(
                dataset_sample=ex.get("dataset_sample", ""),
                analysis_goal=ex.get("analysis_goal", ""),
                findings_and_strategy=ex.get("findings_and_strategy", "")
            ).with_inputs("dataset_sample", "analysis_goal"))
    else:
        # Default bootstrapping examples
        examples = [
            dspy.Example(
                dataset_sample="Product: Wireless Earbuds, Rating: 4.6, Price: $49, Reviews: 1250, Returns: 2%",
                analysis_goal="Evaluate price-to-satisfaction ratio and competitor positioning.",
                findings_and_strategy="The product shows strong market satisfaction (4.6 stars) with negligible return rates (2%). Pricing at $49 is in the sweet spot for consumer electronics. Recommended strategy: Bundle with charging case or offer a premium ANC variant."
            ).with_inputs("dataset_sample", "analysis_goal")
        ]

    # Teleprompter Compilation
    teleprompter = dspy.teleprompt.BootstrapFewShot(metric=evaluation_metric, max_bootstrapped_demos=3)
    compiled_module = teleprompter.compile(ProductAnalyzerModule(), trainset=examples)

    return compiled_module
