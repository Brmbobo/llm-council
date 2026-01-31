"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import (
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    COUNCIL_ROLES,
    DEFAULT_CHAIRMAN_PROMPT,
    DEFAULT_ACTIVE_ROLES,
    AGENT_PAIRS
)
from .services.document_storage import DocumentStorage
from .services.context_manager import ContextManager, DocumentContext

# Service instances for document context
_document_storage = DocumentStorage()
_context_manager = ContextManager()


async def _get_document_context(conversation_id: Optional[str]) -> str:
    """
    Load documents for a conversation and prepare context for injection.

    Args:
        conversation_id: The conversation ID to load documents for

    Returns:
        Formatted context string to prepend to prompts, or empty string
    """
    if not conversation_id:
        return ""

    try:
        documents = await _document_storage.list_documents(conversation_id)
        if not documents:
            return ""

        # Load extracted text for each document
        doc_contexts = []
        for doc in documents:
            text = await _document_storage.get_extracted_text(
                conversation_id,
                doc.id
            )
            if text:
                doc_contexts.append(DocumentContext(
                    document_id=doc.id,
                    filename=doc.original_filename,
                    text=text,
                    token_count=doc.token_count
                ))

        if not doc_contexts:
            return ""

        # Prepare context with token budgeting
        injection = _context_manager.prepare_context(doc_contexts)

        if injection.overflow_warning:
            print(f"[Context] {injection.overflow_warning}")

        return injection.formatted_context

    except Exception as e:
        print(f"[Context] Error loading documents: {e}")
        return ""


async def stage1_collect_responses(
    user_query: str,
    conversation_id: Optional[str] = None,
    role_configs: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        conversation_id: Optional conversation ID for document context
        role_configs: Optional list of role configurations, each with:
            - role_id: The role identifier (e.g., "technical")
            - custom_prompt: Optional custom prompt to override default

    Returns:
        List of dicts with 'model', 'response', 'role_id', 'role_name' keys
    """
    import asyncio

    # Get document context if available
    document_context = await _get_document_context(conversation_id)

    # Build the full query with context
    if document_context:
        full_query = document_context + user_query
    else:
        full_query = user_query

    # If roles are specified, use role-specific system prompts
    if role_configs:
        # Build individual queries with role-specific prompts
        query_tasks = []
        task_metadata = []

        for i, model in enumerate(COUNCIL_MODELS):
            # Assign role to model (cycle through roles if fewer roles than models)
            role_config = role_configs[i % len(role_configs)] if role_configs else None

            if role_config:
                role_id = role_config.get('role_id')
                custom_prompt = role_config.get('custom_prompt')

                # Get the prompt (custom or default from COUNCIL_ROLES)
                if custom_prompt:
                    system_prompt = custom_prompt
                elif role_id and role_id in COUNCIL_ROLES:
                    system_prompt = COUNCIL_ROLES[role_id]['default_prompt']
                else:
                    system_prompt = None

                role_name = COUNCIL_ROLES.get(role_id, {}).get('name') if role_id else None

                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_query}
                    ]
                else:
                    messages = [{"role": "user", "content": full_query}]

                task_metadata.append({
                    "model": model,
                    "role_id": role_id,
                    "role_name": role_name
                })
            else:
                messages = [{"role": "user", "content": full_query}]
                task_metadata.append({
                    "model": model,
                    "role_id": None,
                    "role_name": None
                })

            query_tasks.append(query_model(model, messages))

        # Execute all queries in parallel
        responses = await asyncio.gather(*query_tasks)

        # Format results
        stage1_results = []
        for metadata, response in zip(task_metadata, responses):
            if response is not None:
                stage1_results.append({
                    "model": metadata["model"],
                    "response": response.get('content', ''),
                    "role_id": metadata["role_id"],
                    "role_name": metadata["role_name"]
                })

        return stage1_results

    else:
        # Original behavior without roles
        messages = [{"role": "user", "content": full_query}]

        # Query all models in parallel
        responses = await query_models_parallel(COUNCIL_MODELS, messages)

        # Format results
        stage1_results = []
        for model, response in responses.items():
            if response is not None:  # Only include successful responses
                stage1_results.append({
                    "model": model,
                    "response": response.get('content', ''),
                    "role_id": None,
                    "role_name": None
                })

        return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


def _extract_feedback_for_label(
    stage2_results: List[Dict[str, Any]],
    target_label: str
) -> str:
    """
    Extract peer feedback about a specific response label from all evaluations.

    Args:
        stage2_results: All Stage 2 evaluation results
        target_label: The label to extract feedback for (e.g., "Response A")

    Returns:
        Aggregated feedback text mentioning the target label
    """
    feedback_parts = []

    for result in stage2_results:
        evaluation_text = result.get('ranking', '')
        evaluator_model = result.get('model', 'Unknown')

        # Check if this evaluation mentions the target label
        if target_label in evaluation_text:
            # Extract paragraphs mentioning the label
            paragraphs = evaluation_text.split('\n\n')
            relevant = [p for p in paragraphs if target_label in p and 'FINAL RANKING' not in p]

            if relevant:
                feedback_parts.append(
                    f"Evaluator ({evaluator_model.split('/')[-1]}):\n" +
                    '\n'.join(relevant)
                )

    return '\n\n---\n\n'.join(feedback_parts) if feedback_parts else "No specific feedback available."


async def stage2_5_collect_revisions(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Stage 2.5: Each model revises its response based on peer feedback.

    Each model receives:
    - Their original response
    - Anonymized peer feedback (what others said about their response)
    - Request to revise while maintaining strengths

    Args:
        user_query: The original user question
        stage1_results: Original responses from Stage 1
        stage2_results: Evaluations from Stage 2
        label_to_model: Mapping from "Response X" to model names

    Returns:
        List of dicts with original and revised responses
    """
    import asyncio

    # Invert the mapping: model -> label
    model_to_label = {v: k for k, v in label_to_model.items()}

    # Build revision tasks for each model
    revision_tasks = []
    task_metadata = []

    for stage1_entry in stage1_results:
        model = stage1_entry["model"]
        original_response = stage1_entry["response"]
        role_id = stage1_entry.get("role_id")
        role_name = stage1_entry.get("role_name")

        # Find this model's label
        my_label = model_to_label.get(model)
        if not my_label:
            continue

        # Extract feedback about this model from all evaluations
        feedback = _extract_feedback_for_label(stage2_results, my_label)

        revision_prompt = f"""You previously answered the following question:

QUESTION: {user_query}

YOUR ORIGINAL RESPONSE:
{original_response}

PEER FEEDBACK:
You have received anonymous peer feedback from other AI models. Your response was labeled as "{my_label}". Here is what they said about it:

{feedback}

YOUR TASK:
Please provide a REVISED response that:
1. Addresses the specific criticisms and weaknesses identified
2. Maintains the strengths that were appreciated
3. Improves clarity, accuracy, and completeness
4. Corrects any factual errors if identified

If you believe some feedback is incorrect, briefly explain why, but still consider if improvements can be made.

Provide your complete revised response:"""

        messages = [{"role": "user", "content": revision_prompt}]
        revision_tasks.append(query_model(model, messages))
        task_metadata.append({
            "model": model,
            "original_response": original_response,
            "role_id": role_id,
            "role_name": role_name,
            "label": my_label
        })

    # Execute all revisions in parallel
    responses = await asyncio.gather(*revision_tasks)

    # Format results
    stage2_5_results = []
    for metadata, response in zip(task_metadata, responses):
        if response is not None:
            stage2_5_results.append({
                "model": metadata["model"],
                "original_response": metadata["original_response"],
                "revised_response": response.get('content', ''),
                "role_id": metadata.get("role_id"),
                "role_name": metadata.get("role_name"),
                "label": metadata.get("label")
            })

    return stage2_5_results


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    stage2_5_results: Optional[List[Dict[str, Any]]] = None,
    custom_chairman_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        stage2_5_results: Optional revised responses from Stage 2.5
        custom_chairman_prompt: Optional custom prompt for the chairman

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Determine which responses to use (revised if available, otherwise original)
    if stage2_5_results:
        # Use revised responses
        responses_text = "\n\n".join([
            f"Model: {result['model']}" +
            (f" (Role: {result['role_name']})" if result.get('role_name') else "") +
            f"\nRevised Response: {result['revised_response']}"
            for result in stage2_5_results
        ])
        responses_note = "Note: Models have REVISED their responses based on peer feedback."
    else:
        # Use original responses
        responses_text = "\n\n".join([
            f"Model: {result['model']}" +
            (f" (Role: {result['role_name']})" if result.get('role_name') else "") +
            f"\nResponse: {result['response']}"
            for result in stage1_results
        ])
        responses_note = ""

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    # Use custom prompt if provided, otherwise use default
    base_prompt = custom_chairman_prompt if custom_chairman_prompt else DEFAULT_CHAIRMAN_PROMPT

    chairman_prompt = f"""{base_prompt}

Original Question: {user_query}

{responses_note}

MODEL RESPONSES:
{responses_text}

PEER RANKINGS (of original responses):
{stage2_text}

Please provide your synthesis:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    conversation_id: Optional[str] = None,
    role_configs: Optional[List[Dict[str, Any]]] = None,
    enable_revisions: bool = False,
    custom_chairman_prompt: Optional[str] = None
) -> Tuple[List, List, Optional[List], Dict, Dict]:
    """
    Run the complete council process (3 or 4 stages depending on revisions).

    Args:
        user_query: The user's question
        conversation_id: Optional conversation ID for document context
        role_configs: Optional list of role configurations for models
        enable_revisions: Whether to run Stage 2.5 (model revisions)
        custom_chairman_prompt: Optional custom prompt for the chairman

    Returns:
        Tuple of (stage1_results, stage2_results, stage2_5_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses (with document context and roles if available)
    stage1_results = await stage1_collect_responses(
        user_query,
        conversation_id,
        role_configs
    )

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], None, {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 2.5 (Optional): Collect revisions based on peer feedback
    stage2_5_results = None
    if enable_revisions and stage2_results:
        stage2_5_results = await stage2_5_collect_revisions(
            user_query,
            stage1_results,
            stage2_results,
            label_to_model
        )

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        stage2_5_results,
        custom_chairman_prompt
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage2_5_results, stage3_result, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED COUNCIL WITH AGENT PAIRS (from PR #1)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_enhanced_council(
    user_query: str,
    conversation_id: Optional[str] = None,
    on_progress: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run the enhanced 4-stage council process with Agent Pairs.

    Stage 1: Parallel Agent Pairs (Creator-Critic iterative refinement)
    Stage 2: Tester Validation + Auto-Fix
    Stage 3: Peer Rankings (anonymized)
    Stage 4: Chairman Synthesis

    Args:
        user_query: The user's question
        conversation_id: Optional conversation ID for document context
        on_progress: Optional callback for streaming progress updates

    Returns:
        Dict with all stages and metadata
    """
    from .agent_pair import run_pairs_parallel, pair_result_to_dict
    from .validator import validate_all_pairs, aggregated_validation_to_dict

    # Get document context
    document_context = await _get_document_context(conversation_id)

    # Stage 1: Run Agent Pairs in Parallel
    if on_progress:
        await on_progress({
            "type": "pairs_start",
            "data": {"pairs": [p["name"] for p in AGENT_PAIRS]}
        })

    async def on_pair_iteration(pair_name: str, iteration):
        if on_progress:
            await on_progress({
                "type": "pair_iteration",
                "data": {
                    "pair": pair_name,
                    "iteration": iteration.iteration_number,
                    "score": iteration.critic_score,
                    "status": iteration.status.value
                }
            })

    pair_results = await run_pairs_parallel(
        pairs=AGENT_PAIRS,
        query=user_query,
        context=document_context,
        on_iteration=on_pair_iteration
    )

    if on_progress:
        await on_progress({
            "type": "pairs_complete",
            "data": {
                "pairs": [
                    {
                        "name": r.pair_name,
                        "converged": r.converged,
                        "iterations": r.total_iterations,
                        "final_score": r.final_score
                    }
                    for r in pair_results
                ]
            }
        })

    # Stage 2: Tester Validation + Auto-Fix
    if on_progress:
        await on_progress({"type": "validation_start"})

    async def on_validation_progress(result):
        if on_progress:
            await on_progress({
                "type": "validation_progress",
                "data": {
                    "pair": result.pair_name,
                    "passed": result.passed,
                    "score": result.final_scores.overall,
                    "auto_fix_attempts": len(result.auto_fix_attempts)
                }
            })

    validation_results = await validate_all_pairs(
        query=user_query,
        pair_results=pair_results,
        on_progress=on_validation_progress
    )

    if on_progress:
        await on_progress({
            "type": "validation_complete",
            "data": {
                "best_pair": validation_results.best_pair,
                "best_score": validation_results.best_score,
                "all_passed": validation_results.all_passed,
                "recommendation": validation_results.recommendation
            }
        })

    # Stage 3: Peer Rankings (using validated responses)
    if on_progress:
        await on_progress({"type": "rankings_start"})

    stage1_like_results = []
    for val_result in validation_results.results:
        if val_result.final_response:
            stage1_like_results.append({
                "model": val_result.pair_name,
                "response": val_result.final_response
            })

    stage2_rankings = []
    label_to_model = {}
    aggregate_rankings = []

    if stage1_like_results:
        stage2_rankings, label_to_model = await stage2_collect_rankings(
            user_query,
            stage1_like_results
        )
        aggregate_rankings = calculate_aggregate_rankings(stage2_rankings, label_to_model)

    if on_progress:
        await on_progress({
            "type": "rankings_complete",
            "data": {"aggregate": aggregate_rankings}
        })

    # Stage 4: Chairman Synthesis
    if on_progress:
        await on_progress({"type": "synthesis_start"})

    synthesis_result = await _synthesize_with_pair_context(
        user_query=user_query,
        pair_results=pair_results,
        validation_results=validation_results,
        stage2_rankings=stage2_rankings
    )

    if on_progress:
        await on_progress({
            "type": "synthesis_complete",
            "data": synthesis_result
        })

    return {
        "stage_pairs": [pair_result_to_dict(r) for r in pair_results],
        "stage_validation": aggregated_validation_to_dict(validation_results),
        "stage_rankings": stage2_rankings,
        "stage_synthesis": synthesis_result,
        "metadata": {
            "label_to_model": label_to_model,
            "aggregate_rankings": aggregate_rankings,
            "best_pair": validation_results.best_pair,
            "all_passed": validation_results.all_passed
        }
    }


async def _synthesize_with_pair_context(
    user_query: str,
    pair_results: List,
    validation_results,
    stage2_rankings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Chairman synthesis with full pair context including iterations and fixes."""
    pair_summaries = []
    for pr in pair_results:
        summary = f"""
=== {pr.pair_name} ===
Creator: {pr.creator_model}
Critic: {pr.critic_model}
Iterations: {pr.total_iterations}
Converged: {pr.converged}
Final Score: {pr.final_score:.2f}

Final Response:
{pr.final_response[:2000]}{"..." if len(pr.final_response) > 2000 else ""}
"""
        pair_summaries.append(summary)

    validation_summaries = []
    for vr in validation_results.results:
        fix_info = ""
        if vr.auto_fix_attempts:
            fix_info = f"\nAuto-Fix Attempts: {len(vr.auto_fix_attempts)}"
            for attempt in vr.auto_fix_attempts:
                fix_info += f"\n  - Attempt {attempt.attempt_number}: Score {attempt.retest_score:.2f} ({'PASS' if attempt.passed else 'FAIL'})"

        summary = f"""
--- {vr.pair_name} Validation ---
Initial Score: {vr.initial_scores.overall:.2f}
Final Score: {vr.final_scores.overall:.2f}
Passed: {vr.passed}
Issues: {', '.join(vr.issues[:3])}
Strengths: {', '.join(vr.strengths[:3])}{fix_info}
"""
        validation_summaries.append(summary)

    rankings_summary = ""
    if stage2_rankings:
        rankings_summary = "\n".join([
            f"{r['model']}: {r['ranking'][:500]}..."
            for r in stage2_rankings[:3]
        ])

    chairman_prompt = f"""You are the Chairman of an Enhanced LLM Council. Multiple AI model PAIRS have collaboratively refined responses through Creator-Critic iterations, then validated by a Tester agent.

ORIGINAL QUESTION:
{user_query}

═══════════════════════════════════════════════════════════════════════════════
STAGE 1: AGENT PAIR RESULTS
{chr(10).join(pair_summaries)}

═══════════════════════════════════════════════════════════════════════════════
STAGE 2: TESTER VALIDATION
Best Pair: {validation_results.best_pair}
Best Score: {validation_results.best_score:.2f}
All Passed: {validation_results.all_passed}
Recommendation: {validation_results.recommendation}

{chr(10).join(validation_summaries)}

═══════════════════════════════════════════════════════════════════════════════
STAGE 3: PEER RANKINGS SUMMARY
{rankings_summary if rankings_summary else "No peer rankings available."}

═══════════════════════════════════════════════════════════════════════════════

As Chairman, synthesize all of this information into a SINGLE, COMPREHENSIVE, ACCURATE answer.

Consider:
1. Which pair produced the best response and why
2. What the validation revealed about response quality
3. Any issues that were identified and how they were resolved
4. The consensus from peer rankings

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]
    response = await query_model(CHAIRMAN_MODEL, messages, timeout=180.0)

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }
