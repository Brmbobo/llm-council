"""
Agent Pair Orchestration - Creator-Critic Iterative Refinement.

This module implements the core Creator-Critic pattern where:
1. Creator generates initial response
2. Critic evaluates and provides feedback with score
3. If score < threshold, Creator refines based on feedback
4. Loop continues until convergence or max iterations

Designed for parallel execution of multiple pairs via asyncio.gather().
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    CONVERGENCE_THRESHOLD,
    CREATOR_TIMEOUT_MS,
    CRITIC_TIMEOUT_MS,
    MAX_REFINEMENT_ITERATIONS,
    PARALLEL_PAIRS_ENABLED,
)
from .openrouter import query_model


class IterationStatus(Enum):
    """Status of a single refinement iteration."""
    CONTINUE = "continue"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


@dataclass
class RefinementIteration:
    """Result of a single refinement iteration."""
    iteration_number: int
    creator_response: str
    critic_feedback: str
    critic_score: float
    status: IterationStatus
    creator_time_ms: int = 0
    critic_time_ms: int = 0
    error_message: Optional[str] = None


@dataclass
class PairResult:
    """Complete result from one agent pair."""
    pair_name: str
    creator_model: str
    critic_model: str
    iterations: List[RefinementIteration] = field(default_factory=list)
    final_response: str = ""
    final_score: float = 0.0
    total_iterations: int = 0
    converged: bool = False
    total_time_ms: int = 0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

CREATOR_INITIAL_PROMPT = """You are an expert assistant. Provide a comprehensive, accurate, and well-structured response to the user's query.

USER QUERY:
{query}

{context}

Provide your best response. Be thorough, accurate, and clear."""

CREATOR_REFINEMENT_PROMPT = """You are an expert assistant. Your previous response received feedback from a critic. Improve your response based on this feedback.

USER QUERY:
{query}

YOUR PREVIOUS RESPONSE:
{previous_response}

CRITIC'S FEEDBACK:
{feedback}

CRITIC'S SCORE: {score}/1.0

Please provide an improved response that addresses the critic's concerns while maintaining the strengths of your original response. Focus on:
1. Addressing specific issues mentioned in the feedback
2. Improving accuracy and completeness
3. Enhancing clarity and structure"""

CRITIC_PROMPT = """You are a rigorous critic evaluating the quality of a response. Analyze the response carefully and provide constructive feedback.

USER QUERY:
{query}

RESPONSE TO EVALUATE:
{response}

Evaluate the response on these criteria:
1. ACCURACY - Are the facts and claims correct?
2. COMPLETENESS - Does it fully address the query?
3. CLARITY - Is it well-organized and easy to understand?
4. DEPTH - Does it provide sufficient detail and insight?

Provide your evaluation in this EXACT JSON format:
```json
{{
    "score": <float between 0.0 and 1.0>,
    "accuracy_score": <float 0.0-1.0>,
    "completeness_score": <float 0.0-1.0>,
    "clarity_score": <float 0.0-1.0>,
    "depth_score": <float 0.0-1.0>,
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "suggestions": ["suggestion 1", "suggestion 2", ...],
    "detailed_feedback": "Your detailed analysis here..."
}}
```

Be rigorous but fair. A score of 0.95+ means the response is excellent with only minor improvements possible."""


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_critic_response(response_text: str) -> Tuple[float, str, Dict[str, Any]]:
    """
    Parse critic's JSON response to extract score and feedback.

    Returns:
        Tuple of (score, feedback_text, full_parsed_data)
    """
    # Default values
    default_score = 0.5
    default_feedback = response_text
    default_data = {"score": default_score, "detailed_feedback": response_text}

    if not response_text:
        return default_score, "No feedback provided.", default_data

    # Try to extract JSON from the response
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: try to extract score from text
            score_match = re.search(r'score["\s:]+([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                    return score, response_text, {"score": score, "detailed_feedback": response_text}
                except ValueError:
                    pass
            return default_score, response_text, default_data

    try:
        data = json.loads(json_str)
        score = float(data.get("score", default_score))
        score = max(0.0, min(1.0, score))  # Clamp to 0-1

        # Build feedback text
        feedback_parts = []
        if data.get("detailed_feedback"):
            feedback_parts.append(data["detailed_feedback"])
        if data.get("weaknesses"):
            feedback_parts.append("Weaknesses: " + "; ".join(data["weaknesses"]))
        if data.get("suggestions"):
            feedback_parts.append("Suggestions: " + "; ".join(data["suggestions"]))

        feedback = "\n".join(feedback_parts) if feedback_parts else response_text

        return score, feedback, data

    except (json.JSONDecodeError, ValueError, TypeError):
        return default_score, response_text, default_data


async def run_creator(
    model: str,
    query: str,
    context: str = "",
    previous_response: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    previous_score: Optional[float] = None,
    timeout_ms: int = CREATOR_TIMEOUT_MS
) -> Tuple[Optional[str], int]:
    """
    Generate response using Creator model.

    Args:
        model: Creator model identifier
        query: User's original query
        context: Optional document context
        previous_response: Previous response for refinement
        previous_feedback: Critic's feedback for refinement
        previous_score: Previous critic score
        timeout_ms: Timeout in milliseconds

    Returns:
        Tuple of (response_text, time_taken_ms)
    """
    start_time = time.time()

    # Build prompt based on whether this is initial or refinement
    if previous_response and previous_feedback:
        prompt = CREATOR_REFINEMENT_PROMPT.format(
            query=query,
            previous_response=previous_response,
            feedback=previous_feedback,
            score=f"{previous_score:.2f}" if previous_score else "N/A"
        )
    else:
        context_section = f"\nCONTEXT:\n{context}" if context else ""
        prompt = CREATOR_INITIAL_PROMPT.format(
            query=query,
            context=context_section
        )

    messages = [{"role": "user", "content": prompt}]

    result = await query_model(
        model=model,
        messages=messages,
        timeout=timeout_ms / 1000.0
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if result and result.get("content"):
        return result["content"], elapsed_ms
    return None, elapsed_ms


async def run_critic(
    model: str,
    query: str,
    response: str,
    timeout_ms: int = CRITIC_TIMEOUT_MS
) -> Tuple[float, str, Dict[str, Any], int]:
    """
    Evaluate response using Critic model.

    Args:
        model: Critic model identifier
        query: User's original query
        response: Response to evaluate
        timeout_ms: Timeout in milliseconds

    Returns:
        Tuple of (score, feedback_text, full_data, time_taken_ms)
    """
    start_time = time.time()

    prompt = CRITIC_PROMPT.format(query=query, response=response)
    messages = [{"role": "user", "content": prompt}]

    result = await query_model(
        model=model,
        messages=messages,
        timeout=timeout_ms / 1000.0
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if result and result.get("content"):
        score, feedback, data = parse_critic_response(result["content"])
        return score, feedback, data, elapsed_ms

    return 0.5, "Failed to get critic evaluation.", {}, elapsed_ms


async def run_single_pair(
    pair_config: Dict[str, str],
    query: str,
    context: str = "",
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    on_iteration: Optional[callable] = None
) -> PairResult:
    """
    Run complete Creator-Critic loop for one pair.

    Args:
        pair_config: Dict with 'name', 'creator', 'critic' keys
        query: User's query
        context: Optional document context
        max_iterations: Maximum number of refinement iterations
        convergence_threshold: Score threshold for convergence
        on_iteration: Optional callback after each iteration

    Returns:
        PairResult with all iterations and final response
    """
    start_time = time.time()

    result = PairResult(
        pair_name=pair_config["name"],
        creator_model=pair_config["creator"],
        critic_model=pair_config["critic"]
    )

    current_response = None
    current_feedback = None
    current_score = None

    for iteration in range(1, max_iterations + 1):
        # Run Creator
        creator_response, creator_time = await run_creator(
            model=pair_config["creator"],
            query=query,
            context=context if iteration == 1 else "",
            previous_response=current_response,
            previous_feedback=current_feedback,
            previous_score=current_score
        )

        if not creator_response:
            iteration_result = RefinementIteration(
                iteration_number=iteration,
                creator_response="",
                critic_feedback="",
                critic_score=0.0,
                status=IterationStatus.ERROR,
                creator_time_ms=creator_time,
                error_message="Creator failed to generate response"
            )
            result.iterations.append(iteration_result)
            result.error = "Creator failed to generate response"
            break

        # Run Critic
        score, feedback, critic_data, critic_time = await run_critic(
            model=pair_config["critic"],
            query=query,
            response=creator_response
        )

        # Determine status
        if score >= convergence_threshold:
            status = IterationStatus.CONVERGED
        elif iteration >= max_iterations:
            status = IterationStatus.MAX_ITERATIONS
        else:
            status = IterationStatus.CONTINUE

        iteration_result = RefinementIteration(
            iteration_number=iteration,
            creator_response=creator_response,
            critic_feedback=feedback,
            critic_score=score,
            status=status,
            creator_time_ms=creator_time,
            critic_time_ms=critic_time
        )
        result.iterations.append(iteration_result)

        # Callback for streaming updates
        if on_iteration:
            await on_iteration(pair_config["name"], iteration_result)

        # Update state for next iteration
        current_response = creator_response
        current_feedback = feedback
        current_score = score

        # Check if we should stop
        if status in (IterationStatus.CONVERGED, IterationStatus.MAX_ITERATIONS):
            result.converged = (status == IterationStatus.CONVERGED)
            break

    # Set final result
    if result.iterations:
        last_iteration = result.iterations[-1]
        result.final_response = last_iteration.creator_response
        result.final_score = last_iteration.critic_score

    result.total_iterations = len(result.iterations)
    result.total_time_ms = int((time.time() - start_time) * 1000)

    return result


async def run_pairs_parallel(
    pairs: List[Dict[str, str]],
    query: str,
    context: str = "",
    on_iteration: Optional[callable] = None
) -> List[PairResult]:
    """
    Run multiple pairs in parallel using asyncio.gather().

    Args:
        pairs: List of pair configs
        query: User's query
        context: Optional document context
        on_iteration: Optional callback for streaming updates

    Returns:
        List of PairResult objects
    """
    if not PARALLEL_PAIRS_ENABLED or len(pairs) == 1:
        # Sequential execution
        results = []
        for pair in pairs:
            result = await run_single_pair(
                pair_config=pair,
                query=query,
                context=context,
                on_iteration=on_iteration
            )
            results.append(result)
        return results

    # Parallel execution
    tasks = [
        run_single_pair(
            pair_config=pair,
            query=query,
            context=context,
            on_iteration=on_iteration
        )
        for pair in pairs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Create error result
            error_result = PairResult(
                pair_name=pairs[i]["name"],
                creator_model=pairs[i]["creator"],
                critic_model=pairs[i]["critic"],
                error=str(result)
            )
            processed_results.append(error_result)
        else:
            processed_results.append(result)

    return processed_results


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def iteration_to_dict(iteration: RefinementIteration) -> Dict[str, Any]:
    """Convert RefinementIteration to serializable dict."""
    return {
        "iteration_number": iteration.iteration_number,
        "creator_response": iteration.creator_response,
        "critic_feedback": iteration.critic_feedback,
        "critic_score": iteration.critic_score,
        "status": iteration.status.value,
        "creator_time_ms": iteration.creator_time_ms,
        "critic_time_ms": iteration.critic_time_ms,
        "error_message": iteration.error_message
    }


def pair_result_to_dict(result: PairResult) -> Dict[str, Any]:
    """Convert PairResult to serializable dict."""
    return {
        "pair_name": result.pair_name,
        "creator_model": result.creator_model,
        "critic_model": result.critic_model,
        "iterations": [iteration_to_dict(it) for it in result.iterations],
        "final_response": result.final_response,
        "final_score": result.final_score,
        "total_iterations": result.total_iterations,
        "converged": result.converged,
        "total_time_ms": result.total_time_ms,
        "error": result.error
    }
