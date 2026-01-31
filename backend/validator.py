"""
Tester Agent - Validates refined responses with auto-fix capability.

This module implements the validation phase where:
1. Tester evaluates each pair's final response
2. If validation fails (score < threshold), feedback goes back to pair
3. Pair attempts to fix based on Tester feedback
4. Loop continues until pass or max retries

The auto-fix loop is independent of the initial Creator-Critic refinement.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent_pair import PairResult, run_creator, run_critic
from .config import (
    AUTO_FIX_ON_VALIDATION_FAIL,
    MAX_VALIDATION_RETRIES,
    TESTER_MODEL,
    TESTER_TIMEOUT_MS,
    CONVERGENCE_THRESHOLD,
)
from .openrouter import query_model


@dataclass
class TestScores:
    """Detailed test scores from Tester."""
    accuracy: float = 0.0
    logic: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    overall: float = 0.0


@dataclass
class AutoFixAttempt:
    """Record of a single auto-fix attempt."""
    attempt_number: int
    tester_feedback: str
    creator_fix: str
    critic_recheck_score: float
    retest_score: float
    passed: bool
    time_ms: int = 0


@dataclass
class ValidationResult:
    """Result from tester validation for one pair."""
    pair_name: str
    passed: bool
    initial_scores: TestScores
    final_scores: TestScores
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    raw_feedback: str = ""
    auto_fix_attempts: List[AutoFixAttempt] = field(default_factory=list)
    final_response: str = ""
    total_time_ms: int = 0


@dataclass
class AggregatedValidation:
    """Aggregated validation across all pairs."""
    results: List[ValidationResult] = field(default_factory=list)
    best_pair: str = ""
    best_score: float = 0.0
    all_passed: bool = False
    recommendation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TESTER PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

TESTER_PROMPT = """You are a rigorous TESTER agent. Your job is to validate the quality and correctness of a response.

USER QUERY:
{query}

RESPONSE TO VALIDATE:
{response}

Evaluate this response with extreme rigor on these criteria (score 0-10 each):

1. **ACCURACY** - Are all facts, claims, and technical details correct and verifiable?
2. **LOGIC** - Is the reasoning sound? Are there any logical fallacies or contradictions?
3. **COMPLETENESS** - Are all aspects of the query addressed? Any missing edge cases?
4. **CLARITY** - Is the response well-structured, clear, and easy to understand?

Provide your evaluation in this EXACT JSON format:
```json
{{
    "scores": {{
        "accuracy": <0-10>,
        "logic": <0-10>,
        "completeness": <0-10>,
        "clarity": <0-10>
    }},
    "overall_score": <0.0-1.0>,
    "passed": <true if overall_score >= 0.95, false otherwise>,
    "issues": [
        "Specific issue 1 that needs to be fixed",
        "Specific issue 2 that needs to be fixed"
    ],
    "strengths": [
        "Strength 1 of the response",
        "Strength 2 of the response"
    ],
    "detailed_feedback": "Your comprehensive analysis explaining the scores and what specifically needs improvement..."
}}
```

Be extremely thorough. A passing score (0.95+) should only be given for truly excellent responses with no significant issues."""


AUTO_FIX_CREATOR_PROMPT = """You are an expert assistant. The TESTER has identified issues with your previous response. Fix these issues while maintaining the strengths.

USER QUERY:
{query}

YOUR PREVIOUS RESPONSE:
{previous_response}

TESTER'S FEEDBACK:
{tester_feedback}

ISSUES TO FIX:
{issues}

TESTER'S SCORE: {score}/1.0

Please provide a FIXED response that:
1. Addresses EVERY issue mentioned by the tester
2. Maintains all the identified strengths
3. Improves accuracy, logic, completeness, and clarity

Focus specifically on fixing the identified issues."""


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tester_response(response_text: str) -> Tuple[TestScores, bool, List[str], List[str], str]:
    """
    Parse Tester's JSON response.

    Returns:
        Tuple of (TestScores, passed, issues, strengths, raw_feedback)
    """
    default_scores = TestScores(
        accuracy=5.0, logic=5.0, completeness=5.0, clarity=5.0, overall=0.5
    )

    if not response_text:
        return default_scores, False, ["No response from tester"], [], ""

    # Try to extract JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{[^{}]*"scores"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback
            return default_scores, False, ["Could not parse tester response"], [], response_text

    try:
        data = json.loads(json_str)

        scores_data = data.get("scores", {})
        scores = TestScores(
            accuracy=float(scores_data.get("accuracy", 5)) / 10.0,
            logic=float(scores_data.get("logic", 5)) / 10.0,
            completeness=float(scores_data.get("completeness", 5)) / 10.0,
            clarity=float(scores_data.get("clarity", 5)) / 10.0,
            overall=float(data.get("overall_score", 0.5))
        )

        passed = data.get("passed", scores.overall >= CONVERGENCE_THRESHOLD)
        issues = data.get("issues", [])
        strengths = data.get("strengths", [])
        feedback = data.get("detailed_feedback", response_text)

        return scores, passed, issues, strengths, feedback

    except (json.JSONDecodeError, ValueError, TypeError):
        return default_scores, False, ["Failed to parse tester evaluation"], [], response_text


# ═══════════════════════════════════════════════════════════════════════════════
# CORE VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def run_tester(
    query: str,
    response: str,
    timeout_ms: int = TESTER_TIMEOUT_MS
) -> Tuple[TestScores, bool, List[str], List[str], str, int]:
    """
    Run Tester validation on a response.

    Returns:
        Tuple of (scores, passed, issues, strengths, feedback, time_ms)
    """
    start_time = time.time()

    prompt = TESTER_PROMPT.format(query=query, response=response)
    messages = [{"role": "user", "content": prompt}]

    result = await query_model(
        model=TESTER_MODEL,
        messages=messages,
        timeout=timeout_ms / 1000.0
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if result and result.get("content"):
        scores, passed, issues, strengths, feedback = parse_tester_response(result["content"])
        return scores, passed, issues, strengths, feedback, elapsed_ms

    return TestScores(overall=0.5), False, ["Tester failed to respond"], [], "", elapsed_ms


async def auto_fix_with_feedback(
    pair_config: Dict[str, str],
    query: str,
    original_response: str,
    tester_feedback: str,
    issues: List[str],
    tester_score: float,
    max_retries: int = MAX_VALIDATION_RETRIES
) -> Tuple[str, List[AutoFixAttempt], float]:
    """
    Auto-fix loop when Tester validation fails.

    Flow:
    1. Creator receives original response + tester issues
    2. Creator generates fix
    3. Critic validates fix
    4. Re-run Tester
    5. If still fail and retries remain, loop

    Returns:
        Tuple of (final_response, fix_attempts, final_score)
    """
    current_response = original_response
    current_score = tester_score
    fix_attempts = []

    for attempt_num in range(1, max_retries + 1):
        start_time = time.time()

        # Build fix prompt
        issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "General improvements needed"

        prompt = AUTO_FIX_CREATOR_PROMPT.format(
            query=query,
            previous_response=current_response,
            tester_feedback=tester_feedback,
            issues=issues_text,
            score=f"{current_score:.2f}"
        )

        messages = [{"role": "user", "content": prompt}]

        # Get Creator fix
        fix_result = await query_model(
            model=pair_config["creator"],
            messages=messages,
            timeout=120.0
        )

        if not fix_result or not fix_result.get("content"):
            # Creator failed, keep current response
            break

        creator_fix = fix_result["content"]

        # Run Critic recheck on fix
        from .agent_pair import run_critic
        critic_score, _, _, _ = await run_critic(
            model=pair_config["critic"],
            query=query,
            response=creator_fix
        )

        # Re-run Tester
        new_scores, passed, new_issues, _, new_feedback, _ = await run_tester(
            query=query,
            response=creator_fix
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        fix_attempt = AutoFixAttempt(
            attempt_number=attempt_num,
            tester_feedback=tester_feedback,
            creator_fix=creator_fix,
            critic_recheck_score=critic_score,
            retest_score=new_scores.overall,
            passed=passed,
            time_ms=elapsed_ms
        )
        fix_attempts.append(fix_attempt)

        # Update state
        current_response = creator_fix
        current_score = new_scores.overall
        tester_feedback = new_feedback
        issues = new_issues

        if passed:
            break

    return current_response, fix_attempts, current_score


async def validate_single_pair(
    query: str,
    pair_result: PairResult,
    on_progress: Optional[callable] = None
) -> ValidationResult:
    """
    Validate a single pair's result with auto-fix if needed.

    Args:
        query: User's original query
        pair_result: Result from agent pair
        on_progress: Optional callback for progress updates

    Returns:
        ValidationResult with all details
    """
    start_time = time.time()

    validation = ValidationResult(
        pair_name=pair_result.pair_name,
        passed=False,
        initial_scores=TestScores(),
        final_scores=TestScores(),
        final_response=pair_result.final_response
    )

    if not pair_result.final_response:
        validation.issues = ["Pair produced no response"]
        return validation

    # Initial Tester run
    scores, passed, issues, strengths, feedback, _ = await run_tester(
        query=query,
        response=pair_result.final_response
    )

    validation.initial_scores = scores
    validation.issues = issues
    validation.strengths = strengths
    validation.raw_feedback = feedback

    if passed:
        validation.passed = True
        validation.final_scores = scores
        validation.total_time_ms = int((time.time() - start_time) * 1000)
        return validation

    # Auto-fix if enabled and validation failed
    if AUTO_FIX_ON_VALIDATION_FAIL and not passed:
        pair_config = {
            "name": pair_result.pair_name,
            "creator": pair_result.creator_model,
            "critic": pair_result.critic_model
        }

        fixed_response, fix_attempts, final_score = await auto_fix_with_feedback(
            pair_config=pair_config,
            query=query,
            original_response=pair_result.final_response,
            tester_feedback=feedback,
            issues=issues,
            tester_score=scores.overall
        )

        validation.auto_fix_attempts = fix_attempts
        validation.final_response = fixed_response

        # Get final scores
        if fix_attempts:
            last_attempt = fix_attempts[-1]
            validation.passed = last_attempt.passed
            validation.final_scores = TestScores(overall=last_attempt.retest_score)
        else:
            validation.final_scores = scores
    else:
        validation.final_scores = scores

    validation.total_time_ms = int((time.time() - start_time) * 1000)
    return validation


async def validate_all_pairs(
    query: str,
    pair_results: List[PairResult],
    on_progress: Optional[callable] = None
) -> AggregatedValidation:
    """
    Validate all pair results and determine best.

    Args:
        query: User's original query
        pair_results: List of PairResult from all pairs
        on_progress: Optional callback for progress updates

    Returns:
        AggregatedValidation with all results and recommendations
    """
    aggregated = AggregatedValidation()

    # Validate each pair (sequentially to avoid overwhelming API)
    for pair_result in pair_results:
        validation = await validate_single_pair(
            query=query,
            pair_result=pair_result,
            on_progress=on_progress
        )
        aggregated.results.append(validation)

        if on_progress:
            await on_progress(validation)

    # Determine best pair
    if aggregated.results:
        best = max(aggregated.results, key=lambda r: r.final_scores.overall)
        aggregated.best_pair = best.pair_name
        aggregated.best_score = best.final_scores.overall
        aggregated.all_passed = all(r.passed for r in aggregated.results)

        # Generate recommendation
        passed_count = sum(1 for r in aggregated.results if r.passed)
        if aggregated.all_passed:
            aggregated.recommendation = f"All {len(aggregated.results)} pairs passed validation. Best: {best.pair_name} (score: {best.final_scores.overall:.2f})"
        elif passed_count > 0:
            aggregated.recommendation = f"{passed_count}/{len(aggregated.results)} pairs passed. Best: {best.pair_name} (score: {best.final_scores.overall:.2f})"
        else:
            aggregated.recommendation = f"No pairs passed validation. Highest score: {best.pair_name} ({best.final_scores.overall:.2f})"

    return aggregated


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def test_scores_to_dict(scores: TestScores) -> Dict[str, float]:
    """Convert TestScores to serializable dict."""
    return {
        "accuracy": scores.accuracy,
        "logic": scores.logic,
        "completeness": scores.completeness,
        "clarity": scores.clarity,
        "overall": scores.overall
    }


def auto_fix_attempt_to_dict(attempt: AutoFixAttempt) -> Dict[str, Any]:
    """Convert AutoFixAttempt to serializable dict."""
    return {
        "attempt_number": attempt.attempt_number,
        "tester_feedback": attempt.tester_feedback,
        "creator_fix": attempt.creator_fix,
        "critic_recheck_score": attempt.critic_recheck_score,
        "retest_score": attempt.retest_score,
        "passed": attempt.passed,
        "time_ms": attempt.time_ms
    }


def validation_result_to_dict(result: ValidationResult) -> Dict[str, Any]:
    """Convert ValidationResult to serializable dict."""
    return {
        "pair_name": result.pair_name,
        "passed": result.passed,
        "initial_scores": test_scores_to_dict(result.initial_scores),
        "final_scores": test_scores_to_dict(result.final_scores),
        "issues": result.issues,
        "strengths": result.strengths,
        "raw_feedback": result.raw_feedback,
        "auto_fix_attempts": [auto_fix_attempt_to_dict(a) for a in result.auto_fix_attempts],
        "final_response": result.final_response,
        "total_time_ms": result.total_time_ms
    }


def aggregated_validation_to_dict(agg: AggregatedValidation) -> Dict[str, Any]:
    """Convert AggregatedValidation to serializable dict."""
    return {
        "results": [validation_result_to_dict(r) for r in agg.results],
        "best_pair": agg.best_pair,
        "best_score": agg.best_score,
        "all_passed": agg.all_passed,
        "recommendation": agg.recommendation
    }
