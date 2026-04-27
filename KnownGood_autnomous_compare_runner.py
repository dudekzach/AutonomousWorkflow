import argparse
import html
import json
import os
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

from openai import OpenAI
import anthropic


# =========================
# Config
# =========================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OPENAI_MODEL)

CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "6000"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MAX_CONTINUATION_ATTEMPTS = int(os.getenv("MAX_CONTINUATION_ATTEMPTS", "3"))
ENABLE_OPTIMIZER_BY_DEFAULT = os.getenv("ENABLE_OPTIMIZER", "true").lower() == "true"
ENABLE_STITCHING_BY_DEFAULT = os.getenv("ENABLE_STITCHING", "true").lower() == "true"
LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "true").lower() == "true"
FORCE_DISABLE_OPTIMIZER = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# =========================
# Data structures
# =========================

@dataclass
class ProviderResult:
    provider: str
    model: str
    prompt: str
    text: str
    error: Optional[str] = None
    response_id: Optional[str] = None


@dataclass
class ScoreCard:
    clarity: int
    completeness: int
    practical_usefulness: int
    tone_appropriateness: int
    overall_strength: int


@dataclass
class JudgeDecision:
    winner: str
    next_action: str
    reason: str
    revised_prompt: str
    follow_up_prompt: str
    rerun_target: str
    confidence: str
    openai_scores: ScoreCard
    claude_scores: ScoreCard


@dataclass
class OpenAIChatState:
    last_response_id: Optional[str] = None


@dataclass
class ClaudeChatState:
    messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptOptimizationResult:
    original_prompt: str
    optimized_prompt: str
    annotated_explanation: str
    optional_variants: List[Any]
    provider: str
    strategy: str
    target_model: Optional[str] = None
    use_case: Optional[str] = None
    tone_style: Optional[str] = None
    output_format: Optional[str] = None
    selection_reason: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class IterationRecord:
    iteration: int
    active_prompt: str
    openai_result: ProviderResult
    claude_result: ProviderResult
    judge_decision: JudgeDecision
    post_action_result: Optional[ProviderResult] = None
    stitched_result: Optional[ProviderResult] = None
    prompt_optimization: Optional[PromptOptimizationResult] = None


# =========================
# Setup
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
ANTHROPIC_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "60"))

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=ANTHROPIC_TIMEOUT_SECONDS)


# =========================
# Logging helpers
# =========================

def add_log(logs: Optional[List[str]], message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    if logs is not None:
        logs.append(line)
    if LOG_TO_STDOUT:
        print(line, flush=True)


def capture_exception_summary(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_step(
    step_name: str,
    func: Callable[[], Any],
    logs: Optional[List[str]] = None,
    *,
    swallow: bool = False,
    fallback: Any = None,
) -> Any:
    add_log(logs, f"START {step_name}")
    started = time.time()
    try:
        result = func()
        duration = round(time.time() - started, 2)
        add_log(logs, f"DONE {step_name} ({duration}s)")
        return result
    except Exception as exc:
        duration = round(time.time() - started, 2)
        add_log(logs, f"FAIL {step_name} ({duration}s) -> {capture_exception_summary(exc)}")
        add_log(logs, traceback.format_exc())
        if swallow:
            return fallback
        raise


# =========================
# Provider calls
# =========================

def call_openai_new_chat(prompt: str, model: str = OPENAI_MODEL) -> ProviderResult:
    try:
        response = openai_client.responses.create(
            model=model,
            input=prompt,
            store=True,
        )
        return ProviderResult(
            provider="OpenAI",
            model=model,
            prompt=prompt,
            text=response.output_text,
            response_id=response.id,
        )
    except Exception as e:
        return ProviderResult(
            provider="OpenAI",
            model=model,
            prompt=prompt,
            text="",
            error=str(e),
        )


def call_openai_follow_up(
    state: OpenAIChatState,
    prompt: str,
    model: str = OPENAI_MODEL,
) -> ProviderResult:
    try:
        if not state.last_response_id:
            return call_openai_new_chat(prompt, model)

        response = openai_client.responses.create(
            model=model,
            input=prompt,
            previous_response_id=state.last_response_id,
            store=True,
        )
        state.last_response_id = response.id
        return ProviderResult(
            provider="OpenAI",
            model=model,
            prompt=prompt,
            text=response.output_text,
            response_id=response.id,
        )
    except Exception as e:
        return ProviderResult(
            provider="OpenAI",
            model=model,
            prompt=prompt,
            text="",
            error=str(e),
        )


def call_claude_new_chat(
    state: ClaudeChatState,
    prompt: str,
    model: str = CLAUDE_MODEL,
) -> ProviderResult:
    try:
        state.messages = [{"role": "user", "content": prompt}]
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=state.messages,
        )
        text_parts = [
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        ]
        text = "\n".join(text_parts).strip()
        state.messages.append({"role": "assistant", "content": text})
        return ProviderResult(
            provider="Claude",
            model=model,
            prompt=prompt,
            text=text,
            response_id=getattr(response, "id", None),
        )
    except Exception as e:
        return ProviderResult(
            provider="Claude",
            model=model,
            prompt=prompt,
            text="",
            error=str(e),
        )


def call_claude_follow_up(
    state: ClaudeChatState,
    prompt: str,
    model: str = CLAUDE_MODEL,
) -> ProviderResult:
    try:
        if not state.messages:
            return call_claude_new_chat(state, prompt, model)

        state.messages.append({"role": "user", "content": prompt})
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=state.messages,
        )
        text_parts = [
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        ]
        text = "\n".join(text_parts).strip()
        state.messages.append({"role": "assistant", "content": text})
        return ProviderResult(
            provider="Claude",
            model=model,
            prompt=prompt,
            text=text,
            response_id=getattr(response, "id", None),
        )
    except Exception as e:
        return ProviderResult(
            provider="Claude",
            model=model,
            prompt=prompt,
            text="",
            error=str(e),
        )


# =========================
# Judge
# =========================

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "winner": {
            "type": "string",
            "enum": ["OpenAI", "Claude", "Tie"],
        },
        "next_action": {
            "type": "string",
            "enum": ["accept", "revise_prompt_and_rerun", "follow_up_on_winner"],
        },
        "reason": {"type": "string"},
        "revised_prompt": {"type": "string"},
        "follow_up_prompt": {"type": "string"},
        "rerun_target": {
            "type": "string",
            "enum": ["OpenAI", "Claude", "Both", "None"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "openai_scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                "practical_usefulness": {"type": "integer", "minimum": 1, "maximum": 5},
                "tone_appropriateness": {"type": "integer", "minimum": 1, "maximum": 5},
                "overall_strength": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [
                "clarity",
                "completeness",
                "practical_usefulness",
                "tone_appropriateness",
                "overall_strength",
            ],
        },
        "claude_scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                "practical_usefulness": {"type": "integer", "minimum": 1, "maximum": 5},
                "tone_appropriateness": {"type": "integer", "minimum": 1, "maximum": 5},
                "overall_strength": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [
                "clarity",
                "completeness",
                "practical_usefulness",
                "tone_appropriateness",
                "overall_strength",
            ],
        },
    },
    "required": [
        "winner",
        "next_action",
        "reason",
        "revised_prompt",
        "follow_up_prompt",
        "rerun_target",
        "confidence",
        "openai_scores",
        "claude_scores",
    ],
}


def looks_incomplete(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()

    incomplete_endings = (
        "```",
        "<style>",
        "<script>",
        "<div",
        "<span",
        "<section",
        "<html",
        "<body",
        "function",
        "return",
        "opacity:",
        "color:",
        "background:",
        "border:",
        "padding:",
        "margin:",
        "grid-template-columns:",
    )

    if stripped.endswith(("{", "(", "[", ":", ",", "-", "/", "=")):
        return True

    for token in incomplete_endings:
        if stripped.endswith(token):
            return True

    if stripped.count("```") % 2 != 0:
        return True

    if stripped.count("<html") > stripped.count("</html>"):
        return True
    if stripped.count("<body") > stripped.count("</body>"):
        return True
    if stripped.count("<style") > stripped.count("</style>"):
        return True
    if stripped.count("<script") > stripped.count("</script>"):
        return True

    return False


def sanitize_follow_up_prompt(prompt: str) -> str:
    if not prompt:
        return prompt

    lowered = prompt.lower()

    blocked_terms = [
        "svg",
        "png",
        "powerpoint",
        "ppt",
        "pptx",
        "download",
        "file export",
        "export",
        "upload",
        "sandbox:/",
        "illustrator",
        "adobe illustrator",
        "high-resolution",
        "high resolution",
        "vector graphic file",
        "editable file",
    ]

    if any(term in lowered for term in blocked_terms):
        return (
            "Please provide the requested output as plain text only, directly in your response. "
            "Do not claim to create, attach, export, upload, or link to files. "
            "If the user asked for a visual artifact, provide the full source content as text "
            "(for example, full HTML, SVG, or structured markup) so it can be saved manually."
        )

    return prompt


def default_scorecard(value: int = 3) -> ScoreCard:
    return ScoreCard(
        clarity=value,
        completeness=value,
        practical_usefulness=value,
        tone_appropriateness=value,
        overall_strength=value,
    )


def build_fallback_judge(openai_result: ProviderResult, claude_result: ProviderResult) -> JudgeDecision:
    openai_ok = bool((openai_result.text or "").strip()) and not openai_result.error
    claude_ok = bool((claude_result.text or "").strip()) and not claude_result.error

    if openai_ok and not claude_ok:
        winner = "OpenAI"
    elif claude_ok and not openai_ok:
        winner = "Claude"
    elif len(openai_result.text or "") >= len(claude_result.text or ""):
        winner = "OpenAI"
    else:
        winner = "Claude"

    return JudgeDecision(
        winner=winner,
        next_action="accept",
        reason="Fallback judge decision used because structured judging failed. Manual scoring was skipped.",
        revised_prompt="",
        follow_up_prompt="Please continue and complete the previous answer.",
        rerun_target="None",
        confidence="low",
        openai_scores=default_scorecard(),
        claude_scores=default_scorecard(),
    )


def judge_outputs(
    original_prompt: str,
    openai_result: ProviderResult,
    claude_result: ProviderResult,
) -> JudgeDecision:
    judge_prompt = f"""
You are the orchestration layer for a multi-model workflow.

Original prompt:
{original_prompt}

OpenAI response:
{openai_result.text if openai_result.text else f"ERROR: {openai_result.error}"}

Claude response:
{claude_result.text if claude_result.text else f"ERROR: {claude_result.error}"}

Your job:
1. Pick the stronger response.
2. Score both responses from 1 to 5 in these categories:
   - clarity
   - completeness
   - practical_usefulness
   - tone_appropriateness
   - overall_strength
3. Decide the best next action.
4. If needed, produce the next prompt automatically.

Decision rules:
- Choose "accept" when one response is already strong enough to use.
- Choose "revise_prompt_and_rerun" when both outputs are weak, vague, too generic, or need tighter instructions.
- Choose "follow_up_on_winner" when one output is promising but needs another turn to improve or operationalize it.
- If a response appears cut off or incomplete, that is a strong reason to use "follow_up_on_winner".
- Keep follow-up actions inside text-only model behavior. Do not ask the winner to create, export, attach, upload, or link to files.

For rerun_target:
- Use "Both" if the revised prompt should go back to both models.
- Use "OpenAI" or "Claude" if only one should continue.
- Use "None" when next_action is "accept".

Return only valid JSON matching the schema.
""".strip()

    response = openai_client.responses.create(
        model=JUDGE_MODEL,
        input=judge_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "judge_decision",
                "strict": True,
                "schema": DECISION_SCHEMA,
            }
        },
    )

    data = json.loads(response.output_text)

    if data.get("follow_up_prompt"):
        data["follow_up_prompt"] = sanitize_follow_up_prompt(data["follow_up_prompt"])

    return JudgeDecision(
        winner=data["winner"],
        next_action=data["next_action"],
        reason=data["reason"],
        revised_prompt=data["revised_prompt"],
        follow_up_prompt=data["follow_up_prompt"],
        rerun_target=data["rerun_target"],
        confidence=data["confidence"],
        openai_scores=ScoreCard(**data["openai_scores"]),
        claude_scores=ScoreCard(**data["claude_scores"]),
    )


def judge_outputs_safe(
    original_prompt: str,
    openai_result: ProviderResult,
    claude_result: ProviderResult,
    logs: Optional[List[str]] = None,
) -> JudgeDecision:
    try:
        return judge_outputs(original_prompt, openai_result, claude_result)
    except Exception as exc:
        add_log(logs, f"Judge failed. Falling back to simple decision. {capture_exception_summary(exc)}")
        return build_fallback_judge(openai_result, claude_result)


def stitch_final_response(
    provider: str,
    model: str,
    original_prompt: str,
    base_text: str,
    continuation_text: str,
) -> ProviderResult:
    stitch_prompt = f"""
You are cleaning up a two-part draft response.

The original response was cut off.
A continuation response was generated afterward.

Your job:
1. Merge both parts into one complete final answer.
2. Remove any duplicated content.
3. Remove any cut-off fragments or incomplete sentences.
4. Preserve the strongest structure, tone, and detail.
5. Return only the final cleaned answer.

Original user prompt:
{original_prompt}

Part 1 (cut off original answer):
{base_text}

Part 2 (continuation):
{continuation_text}
""".strip()

    if provider == "Claude":
        return call_claude_new_chat(ClaudeChatState(), stitch_prompt, model=model)
    return call_openai_new_chat(stitch_prompt, model=model)


PROMPT_OPTIMIZER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "optimized_prompt": {"type": "string"},
        "annotated_explanation": {"type": "string"},
        "optional_variants": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["optimized_prompt", "annotated_explanation", "optional_variants"],
}

PROMPT_OPTIMIZER_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_provider": {
            "type": "string",
            "enum": ["OpenAI", "Claude"],
        },
        "selected_optimized_prompt": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["selected_provider", "selected_optimized_prompt", "reason"],
}


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def build_optimizer_prompt(
    original_prompt: str,
    target_model: Optional[str] = None,
    use_case: Optional[str] = None,
    tone_style: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    details = []
    if target_model:
        details.append(f"Target model: {target_model}")
    if use_case:
        details.append(f"Use case: {use_case}")
    if tone_style:
        details.append(f"Desired tone/style: {tone_style}")
    if output_format:
        details.append(f"Preferred output format: {output_format}")

    context_block = "\n".join(details) if details else "No additional optimization context provided."

    return f"""
You are a prompt optimization expert.

Optimize the user's prompt for maximum clarity, specificity, structure, efficiency, and creativity control.

Evaluate the prompt across:
- Clarity
- Specificity
- Structure
- Efficiency
- Creativity control

Additional optimization context:
{context_block}

Return only valid JSON matching the schema with:
1. optimized_prompt
2. annotated_explanation
3. optional_variants (0-3 concise alternatives)

User prompt:
{original_prompt}
""".strip()


def optimize_prompt_with_openai(
    original_prompt: str,
    *,
    target_model: Optional[str] = None,
    use_case: Optional[str] = None,
    tone_style: Optional[str] = None,
    output_format: Optional[str] = None,
) -> PromptOptimizationResult:
    prompt = build_optimizer_prompt(
        original_prompt=original_prompt,
        target_model=target_model,
        use_case=use_case,
        tone_style=tone_style,
        output_format=output_format,
    )

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "prompt_optimizer",
                "strict": True,
                "schema": PROMPT_OPTIMIZER_SCHEMA,
            }
        },
    )

    data = json.loads(response.output_text)
    return PromptOptimizationResult(
        original_prompt=original_prompt,
        optimized_prompt=data["optimized_prompt"],
        annotated_explanation=data["annotated_explanation"],
        optional_variants=data["optional_variants"],
        provider="OpenAI",
        strategy="single_openai",
        target_model=target_model,
        use_case=use_case,
        tone_style=tone_style,
        output_format=output_format,
        raw_response=response.output_text,
    )


def optimize_prompt_with_claude(
    original_prompt: str,
    *,
    target_model: Optional[str] = None,
    use_case: Optional[str] = None,
    tone_style: Optional[str] = None,
    output_format: Optional[str] = None,
) -> PromptOptimizationResult:
    prompt = build_optimizer_prompt(
        original_prompt=original_prompt,
        target_model=target_model,
        use_case=use_case,
        tone_style=tone_style,
        output_format=output_format,
    )

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [
        block.text for block in response.content
        if getattr(block, "type", "") == "text"
    ]
    text = "\n".join(text_parts).strip()
    data = extract_json_object(text)

    return PromptOptimizationResult(
        original_prompt=original_prompt,
        optimized_prompt=data["optimized_prompt"],
        annotated_explanation=data["annotated_explanation"],
        optional_variants=data["optional_variants"],
        provider="Claude",
        strategy="single_claude",
        target_model=target_model,
        use_case=use_case,
        tone_style=tone_style,
        output_format=output_format,
        raw_response=text,
    )


def select_best_optimized_prompt(
    original_prompt: str,
    openai_optimized: PromptOptimizationResult,
    claude_optimized: PromptOptimizationResult,
) -> Dict[str, str]:
    selection_prompt = f"""
You are evaluating two optimized prompts for the same original prompt.

Choose the optimized prompt that will most likely produce the best downstream model performance.

Selection criteria:
- clarity
- specificity
- structure
- efficiency
- preservation of user intent
- usefulness across strong LLMs

Original prompt:
{original_prompt}

OpenAI optimized prompt:
{openai_optimized.optimized_prompt}

Claude optimized prompt:
{claude_optimized.optimized_prompt}

Return only valid JSON matching the schema.
""".strip()

    response = openai_client.responses.create(
        model=JUDGE_MODEL,
        input=selection_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "prompt_optimizer_selection",
                "strict": True,
                "schema": PROMPT_OPTIMIZER_SELECTION_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def optimize_prompt(
    original_prompt: str,
    options: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None,
) -> PromptOptimizationResult:
    options = options or {}

    optimize_enabled = False if FORCE_DISABLE_OPTIMIZER else options.get("optimize_prompt", ENABLE_OPTIMIZER_BY_DEFAULT)
    strategy = options.get("optimizer_strategy", "compare_both")
    target_model = options.get("optimizer_target_model")
    use_case = options.get("optimizer_use_case")
    tone_style = options.get("optimizer_tone_style")
    output_format = options.get("optimizer_output_format")

    if not optimize_enabled:
        add_log(logs, "Prompt optimizer disabled for this run")
        return PromptOptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=original_prompt,
            annotated_explanation="Prompt optimization was disabled for this run.",
            optional_variants=[],
            provider="None",
            strategy="disabled",
            target_model=target_model,
            use_case=use_case,
            tone_style=tone_style,
            output_format=output_format,
        )

    if strategy == "single_claude":
        result = run_step(
            "prompt_optimizer_single_claude",
            lambda: optimize_prompt_with_claude(
                original_prompt,
                target_model=target_model,
                use_case=use_case,
                tone_style=tone_style,
                output_format=output_format,
            ),
            logs,
            swallow=True,
            fallback=None,
        )
        if result:
            result.strategy = strategy
            return result

    elif strategy == "single_openai":
        result = run_step(
            "prompt_optimizer_single_openai",
            lambda: optimize_prompt_with_openai(
                original_prompt,
                target_model=target_model,
                use_case=use_case,
                tone_style=tone_style,
                output_format=output_format,
            ),
            logs,
            swallow=True,
            fallback=None,
        )
        if result:
            result.strategy = strategy
            return result

    else:
        openai_result = run_step(
            "prompt_optimizer_openai",
            lambda: optimize_prompt_with_openai(
                original_prompt,
                target_model=target_model,
                use_case=use_case,
                tone_style=tone_style,
                output_format=output_format,
            ),
            logs,
            swallow=True,
            fallback=None,
        )
        claude_result = run_step(
            "prompt_optimizer_claude",
            lambda: optimize_prompt_with_claude(
                original_prompt,
                target_model=target_model,
                use_case=use_case,
                tone_style=tone_style,
                output_format=output_format,
            ),
            logs,
            swallow=True,
            fallback=None,
        )

        if openai_result and claude_result:
            selection = run_step(
                "prompt_optimizer_selection",
                lambda: select_best_optimized_prompt(
                    original_prompt=original_prompt,
                    openai_optimized=openai_result,
                    claude_optimized=claude_result,
                ),
                logs,
                swallow=True,
                fallback=None,
            )
            if selection:
                selected_provider = selection["selected_provider"]
                selected_result = openai_result if selected_provider == "OpenAI" else claude_result
                selected_result.strategy = "compare_both"
                selected_result.selection_reason = selection["reason"]
                selected_result.raw_response = json.dumps(
                    {
                        "selected_provider": selected_provider,
                        "selection_reason": selection["reason"],
                        "openai_optimizer": openai_result.raw_response,
                        "claude_optimizer": claude_result.raw_response,
                    },
                    indent=2,
                )
                selected_result.optimized_prompt = selection["selected_optimized_prompt"]
                return selected_result

        if openai_result:
            openai_result.strategy = "fallback_openai"
            openai_result.selection_reason = "Claude optimizer or selection step failed, so OpenAI optimizer was used."
            return openai_result
        if claude_result:
            claude_result.strategy = "fallback_claude"
            claude_result.selection_reason = "OpenAI optimizer or selection step failed, so Claude optimizer was used."
            return claude_result

    add_log(logs, "All optimizer paths failed. Falling back to original prompt.")
    return PromptOptimizationResult(
        original_prompt=original_prompt,
        optimized_prompt=original_prompt,
        annotated_explanation="Optimizer failed, so the original prompt was used as a safe fallback.",
        optional_variants=[],
        provider="None",
        strategy="fallback_original_prompt",
        target_model=target_model,
        use_case=use_case,
        tone_style=tone_style,
        output_format=output_format,
    )


# =========================
# Orchestrator
# =========================

def run_autonomous_loop(
    initial_prompt: str,
    max_iterations: int = MAX_ITERATIONS,
    options: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None,
) -> List[IterationRecord]:
    records: List[IterationRecord] = []

    openai_state = OpenAIChatState()
    claude_state = ClaudeChatState()

    options = options or {}
    optimization_result = optimize_prompt(initial_prompt, options, logs)
    active_prompt = optimization_result.optimized_prompt.strip() or initial_prompt
    add_log(logs, f"Active prompt ready. Length={len(active_prompt)}")

    for iteration in range(1, max_iterations + 1):
        add_log(logs, f"Iteration {iteration} started")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_step,
                    f"iteration_{iteration}_openai_initial",
                    lambda: call_openai_new_chat(active_prompt),
                    logs,
                ): "openai",
                executor.submit(
                    run_step,
                    f"iteration_{iteration}_claude_initial",
                    lambda: call_claude_new_chat(claude_state, active_prompt),
                    logs,
                ): "claude",
            }

            openai_result: Optional[ProviderResult] = None
            claude_result: Optional[ProviderResult] = None

            for future in as_completed(futures):
                model = futures[future]
                result = future.result()

                if model == "openai":
                    openai_result = result
                    if openai_result.response_id:
                        openai_state.last_response_id = openai_result.response_id
                elif model == "claude":
                    claude_result = result

        if openai_result is None or claude_result is None:
            raise RuntimeError("Parallel model execution did not return both provider results")

        judge = run_step(
            f"iteration_{iteration}_judge",
            lambda: judge_outputs_safe(active_prompt, openai_result, claude_result, logs),
            logs,
        )

        post_action_result: Optional[ProviderResult] = None

        if judge.next_action == "accept":
            add_log(logs, f"Iteration {iteration} accepted by judge")
            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=None,
                    prompt_optimization=optimization_result,
                )
            )
            break

        if judge.next_action == "revise_prompt_and_rerun":
            add_log(logs, f"Iteration {iteration} requested revised prompt rerun")
            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=None,
                    prompt_optimization=optimization_result,
                )
            )
            active_prompt = judge.revised_prompt.strip() or active_prompt
            continue

        if judge.next_action == "follow_up_on_winner":
            raw_follow_up_prompt = (
                judge.follow_up_prompt.strip()
                or "Please continue and complete the previous answer. Finish any incomplete sections and provide a complete final version."
            )
            follow_up_prompt = sanitize_follow_up_prompt(raw_follow_up_prompt)

            target_provider = judge.rerun_target
            if target_provider == "None":
                target_provider = judge.winner

            add_log(logs, f"Iteration {iteration} follow-up target: {target_provider}")

            if target_provider == "OpenAI":
                post_action_result = run_step(
                    f"iteration_{iteration}_openai_follow_up",
                    lambda: call_openai_follow_up(openai_state, follow_up_prompt),
                    logs,
                )

                continuation_attempts = 0
                while (
                    post_action_result
                    and looks_incomplete(post_action_result.text or "")
                    and continuation_attempts < MAX_CONTINUATION_ATTEMPTS
                ):
                    continuation_attempts += 1
                    add_log(logs, f"OpenAI continuation attempt {continuation_attempts}")
                    follow_up_prompt = (
                        "Please continue from where you left off and complete the answer. "
                        "Do not restart. Finish any incomplete sections and provide the full final version."
                    )
                    post_action_result = run_step(
                        f"iteration_{iteration}_openai_continuation_{continuation_attempts}",
                        lambda: call_openai_follow_up(openai_state, follow_up_prompt),
                        logs,
                    )

            elif target_provider == "Claude":
                post_action_result = run_step(
                    f"iteration_{iteration}_claude_follow_up",
                    lambda: call_claude_follow_up(claude_state, follow_up_prompt),
                    logs,
                )

                continuation_attempts = 0
                while (
                    post_action_result
                    and looks_incomplete(post_action_result.text or "")
                    and continuation_attempts < MAX_CONTINUATION_ATTEMPTS
                ):
                    continuation_attempts += 1
                    add_log(logs, f"Claude continuation attempt {continuation_attempts}")
                    follow_up_prompt = (
                        "Please continue from where you left off and complete the answer. "
                        "Do not restart. Finish any incomplete sections and provide the full final version."
                    )
                    post_action_result = run_step(
                        f"iteration_{iteration}_claude_continuation_{continuation_attempts}",
                        lambda: call_claude_follow_up(claude_state, follow_up_prompt),
                        logs,
                    )

            elif target_provider == "Both":
                add_log(logs, f"Iteration {iteration} requested Both. Using revised prompt for next loop.")
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        active_prompt=active_prompt,
                        openai_result=openai_result,
                        claude_result=claude_result,
                        judge_decision=judge,
                        post_action_result=None,
                        prompt_optimization=optimization_result,
                    )
                )
                active_prompt = judge.revised_prompt.strip() or follow_up_prompt
                continue

            else:
                add_log(logs, f"Unexpected rerun_target={target_provider}. Falling back safely.")
                if judge.revised_prompt.strip():
                    records.append(
                        IterationRecord(
                            iteration=iteration,
                            active_prompt=active_prompt,
                            openai_result=openai_result,
                            claude_result=claude_result,
                            judge_decision=judge,
                            post_action_result=None,
                            prompt_optimization=optimization_result,
                        )
                    )
                    active_prompt = judge.revised_prompt.strip()
                    continue

                records.append(
                    IterationRecord(
                        iteration=iteration,
                        active_prompt=active_prompt,
                        openai_result=openai_result,
                        claude_result=claude_result,
                        judge_decision=judge,
                        post_action_result=None,
                        prompt_optimization=optimization_result,
                    )
                )
                break

            stitched_result: Optional[ProviderResult] = None
            stitching_enabled = options.get("enable_stitching", ENABLE_STITCHING_BY_DEFAULT)

            if post_action_result and stitching_enabled:
                if target_provider == "Claude":
                    base_text = claude_result.text or ""
                    stitched_result = run_step(
                        f"iteration_{iteration}_stitch_claude",
                        lambda: stitch_final_response(
                            provider="Claude",
                            model=CLAUDE_MODEL,
                            original_prompt=active_prompt,
                            base_text=base_text,
                            continuation_text=post_action_result.text or "",
                        ),
                        logs,
                        swallow=True,
                        fallback=None,
                    )
                elif target_provider == "OpenAI":
                    base_text = openai_result.text or ""
                    stitched_result = run_step(
                        f"iteration_{iteration}_stitch_openai",
                        lambda: stitch_final_response(
                            provider="OpenAI",
                            model=OPENAI_MODEL,
                            original_prompt=active_prompt,
                            base_text=base_text,
                            continuation_text=post_action_result.text or "",
                        ),
                        logs,
                        swallow=True,
                        fallback=None,
                    )
            elif not stitching_enabled:
                add_log(logs, "Stitching disabled for this run")

            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=post_action_result,
                    stitched_result=stitched_result,
                    prompt_optimization=optimization_result,
                )
            )
            break

        records.append(
            IterationRecord(
                iteration=iteration,
                active_prompt=active_prompt,
                openai_result=openai_result,
                claude_result=claude_result,
                judge_decision=judge,
                post_action_result=None,
                prompt_optimization=optimization_result,
            )
        )
        break

    return records


# =========================
# Reporting helpers
# =========================

def esc(text: Optional[Any]) -> str:
    if text is None:
        return ""
    if isinstance(text, (dict, list)):
        return html.escape(json.dumps(text, indent=2, ensure_ascii=False))
    return html.escape(str(text))


def render_optional_variants(optional_variants: Optional[List[Any]]) -> str:
    if not optional_variants:
        return "N/A"

    rendered_items: List[str] = []
    for item in optional_variants:
        if isinstance(item, str):
            rendered_items.append(item)
        else:
            rendered_items.append(json.dumps(item, indent=2, ensure_ascii=False))

    return "\n\n".join(rendered_items)
    

def score_total(score: ScoreCard) -> int:
    return (
        score.clarity
        + score.completeness
        + score.practical_usefulness
        + score.tone_appropriateness
        + score.overall_strength
    )


def render_score_table(openai_scores: ScoreCard, claude_scores: ScoreCard) -> str:
    rows = [
        ("Clarity", openai_scores.clarity, claude_scores.clarity),
        ("Completeness", openai_scores.completeness, claude_scores.completeness),
        ("Practical Usefulness", openai_scores.practical_usefulness, claude_scores.practical_usefulness),
        ("Tone Appropriateness", openai_scores.tone_appropriateness, claude_scores.tone_appropriateness),
        ("Overall Strength", openai_scores.overall_strength, claude_scores.overall_strength),
        ("Total", score_total(openai_scores), score_total(claude_scores)),
    ]

    html_rows = []
    for category, openai_val, claude_val in rows:
        openai_class = "winner-cell" if openai_val > claude_val else ""
        claude_class = "winner-cell" if claude_val > openai_val else ""
        if openai_val == claude_val:
            openai_class = "tie-cell"
            claude_class = "tie-cell"

        html_rows.append(
            f"""
        <tr>
            <td>{esc(category)}</td>
            <td class=\"{openai_class}\">{openai_val}</td>
            <td class=\"{claude_class}\">{claude_val}</td>
        </tr>
        """
        )

    return f"""
    <table class=\"score-table\">
        <thead>
            <tr>
                <th>Category</th>
                <th>OpenAI</th>
                <th>Claude</th>
            </tr>
        </thead>
        <tbody>
            {''.join(html_rows)}
        </tbody>
    </table>
    """


def get_final_output_text(last_record: IterationRecord) -> str:
    if last_record.stitched_result and (last_record.stitched_result.text or "").strip():
        return last_record.stitched_result.text

    if last_record.post_action_result and (last_record.post_action_result.text or "").strip():
        return last_record.post_action_result.text

    winner = last_record.judge_decision.winner
    if winner == "OpenAI":
        return last_record.openai_result.text or last_record.openai_result.error or ""
    if winner == "Claude":
        return last_record.claude_result.text or last_record.claude_result.error or ""

    if (last_record.openai_result.text or "").strip():
        return last_record.openai_result.text
    if (last_record.claude_result.text or "").strip():
        return last_record.claude_result.text

    return "Tie detected. Review manually."


def build_html(records: List[IterationRecord], initial_prompt: str) -> str:
    last_record = records[-1]
    final_output = get_final_output_text(last_record)

    final_score_table = render_score_table(
        last_record.judge_decision.openai_scores,
        last_record.judge_decision.claude_scores,
    )

    sections = []
    for record in records:
        post_action_html = ""
        if record.post_action_result:
            post_action_html = f"""
            <div class=\"section action\">
                <div class=\"title\">Automated Next Step Result</div>
                <div class=\"subtitle\">
                    Provider: {esc(record.post_action_result.provider)} |
                    Model: {esc(record.post_action_result.model)}
                </div>
                <pre>{esc(record.post_action_result.text or record.post_action_result.error)}</pre>
            </div>
            """

        stitched_html = ""
        if record.stitched_result:
            stitched_html = f"""
            <div class=\"section action\">
                <div class=\"title\">Stitched Final Result</div>
                <div class=\"subtitle\">
                    Provider: {esc(record.stitched_result.provider)} |
                    Model: {esc(record.stitched_result.model)}
                </div>
                <pre>{esc(record.stitched_result.text or record.stitched_result.error)}</pre>
            </div>
            """

        score_table = render_score_table(
            record.judge_decision.openai_scores,
            record.judge_decision.claude_scores,
        )

        prompt_optimization_html = ""
        if record.prompt_optimization:
            prompt_optimization_html = f"""
            <div class=\"prompt-block\">
                <strong>Prompt Optimization</strong>
                <div class=\"subtitle\">
                    Provider: {esc(record.prompt_optimization.provider)} |
                    Strategy: {esc(record.prompt_optimization.strategy)} |
                    Target Model: {esc(record.prompt_optimization.target_model or 'N/A')} |
                    Use Case: {esc(record.prompt_optimization.use_case or 'N/A')} |
                    Tone/Style: {esc(record.prompt_optimization.tone_style or 'N/A')} |
                    Output Format: {esc(record.prompt_optimization.output_format or 'N/A')}
                </div>
                <strong>Optimized Prompt</strong>
                <pre>{esc(record.prompt_optimization.optimized_prompt)}</pre>
                <strong>What Changed and Why</strong>
                <pre>{esc(record.prompt_optimization.annotated_explanation)}</pre>
                <strong>Optional Variants</strong>
                <pre>{esc(render_optional_variants(record.prompt_optimization.optional_variants))}</pre>
                <strong>Optimizer Selection Reason</strong>
                <pre>{esc(record.prompt_optimization.selection_reason or 'N/A')}</pre>
            </div>
            """

        sections.append(
            f"""
        <div class=\"section\">
            <div class=\"title\">Iteration {record.iteration}</div>

            <div class=\"prompt-block\">
                <strong>Active Prompt</strong>
                <pre>{esc(record.active_prompt)}</pre>
            </div>

            {prompt_optimization_html}

            <div class=\"judge-summary\">
                <div><strong>Winner:</strong> {esc(record.judge_decision.winner)}</div>
                <div><strong>Next Action:</strong> {esc(record.judge_decision.next_action)}</div>
                <div><strong>Confidence:</strong> {esc(record.judge_decision.confidence)}</div>
            </div>

            <div class=\"judge-block\">
                <strong>Judge Reasoning</strong>
                <pre>{esc(record.judge_decision.reason)}</pre>
            </div>

            <div class=\"score-block\">
                <strong>Scorecard (1–5)</strong>
                {score_table}
            </div>

            <div class=\"container\">
                <div class=\"card\">
                    <div class=\"title\">OpenAI</div>
                    <div class=\"subtitle\">{esc(record.openai_result.model)}</div>
                    <pre>{esc(record.openai_result.text or record.openai_result.error)}</pre>
                </div>

                <div class=\"card\">
                    <div class=\"title\">Claude</div>
                    <div class=\"subtitle\">{esc(record.claude_result.model)}</div>
                    <pre>{esc(record.claude_result.text or record.claude_result.error)}</pre>
                </div>
            </div>

            <div class=\"judge-block\">
                <strong>Generated Follow-Up Prompt</strong>
                <pre>{esc(record.judge_decision.follow_up_prompt or 'N/A')}</pre>
            </div>

            <div class=\"judge-block\">
                <strong>Generated Revised Prompt</strong>
                <pre>{esc(record.judge_decision.revised_prompt or 'N/A')}</pre>
            </div>

            {post_action_html}
            {stitched_html}
        </div>
        """
        )

    return f"""
    <html>
    <head>
        <title>Autonomous Multi-Model Workflow</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
                background: #f5f7fa;
                color: #1f2937;
            }}
            h1 {{
                margin-bottom: 8px;
            }}
            .lead {{
                margin-bottom: 24px;
                color: #4b5563;
            }}
            .section {{
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                padding: 20px;
                margin-bottom: 24px;
            }}
            .hero {{
                border-left: 6px solid #16a34a;
                background: #f0fdf4;
            }}
            .meta {{
                display: flex;
                gap: 24px;
                flex-wrap: wrap;
                margin-top: 12px;
                margin-bottom: 12px;
            }}
            .meta div {{
                background: #ecfdf5;
                border: 1px solid #bbf7d0;
                border-radius: 8px;
                padding: 10px 12px;
            }}
            .container {{
                display: flex;
                gap: 20px;
                align-items: stretch;
                margin-top: 18px;
                flex-wrap: wrap;
            }}
            .card {{
                flex: 1;
                min-width: 280px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 16px;
            }}
            .title {{
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: #6b7280;
                margin-bottom: 12px;
                font-size: 13px;
            }}
            .prompt-block, .judge-block, .score-block, .action, .judge-summary {{
                margin-top: 16px;
            }}
            .judge-summary {{
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }}
            .judge-summary div {{
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 10px 12px;
            }}
            .score-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            .score-table th,
            .score-table td {{
                border: 1px solid #e5e7eb;
                padding: 10px;
                text-align: left;
            }}
            .score-table th {{
                background: #f9fafb;
            }}
            .winner-cell {{
                background: #dcfce7;
                font-weight: bold;
            }}
            .tie-cell {{
                background: #f3f4f6;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 0;
                line-height: 1.45;
                font-family: Arial, sans-serif;
            }}
        </style>
    </head>
    <body>
        <h1>Autonomous Multi-Model Workflow</h1>
        <div class=\"lead\">Initial Prompt</div>
        <div class=\"section\">
            <pre>{esc(initial_prompt)}</pre>
        </div>

        <div class=\"section hero\">
            <div class=\"title\">Final Output (Recommended)</div>
            <div class=\"meta\">
                <div><strong>Winner:</strong> {esc(last_record.judge_decision.winner)}</div>
                <div><strong>Next Action Chosen:</strong> {esc(last_record.judge_decision.next_action)}</div>
                <div><strong>Confidence:</strong> {esc(last_record.judge_decision.confidence)}</div>
            </div>
            <div class=\"judge-block\">
                <strong>Why This Won</strong>
                <pre>{esc(last_record.judge_decision.reason)}</pre>
            </div>
            <div class=\"score-block\">
                <strong>Final Scorecard (1–5)</strong>
                {final_score_table}
            </div>
            <div class=\"judge-block\">
                <strong>Recommended Output</strong>
                <pre>{esc(final_output)}</pre>
            </div>
        </div>

        {''.join(sections)}
    </body>
    </html>
    """


def save_outputs(
    records: List[IterationRecord],
    initial_prompt: str,
    open_browser: bool = True,
) -> Dict[str, str]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_filename = f"autonomous_run_{timestamp}.html"
    json_filename = f"autonomous_run_{timestamp}.json"

    html_path = os.path.join(OUTPUTS_DIR, html_filename)
    json_path = os.path.join(OUTPUTS_DIR, json_filename)

    payload = {
        "initial_prompt": initial_prompt,
        "records": [
            {
                "iteration": r.iteration,
                "active_prompt": r.active_prompt,
                "openai_result": r.openai_result.__dict__,
                "claude_result": r.claude_result.__dict__,
                "judge_decision": {
                    "winner": r.judge_decision.winner,
                    "next_action": r.judge_decision.next_action,
                    "reason": r.judge_decision.reason,
                    "revised_prompt": r.judge_decision.revised_prompt,
                    "follow_up_prompt": r.judge_decision.follow_up_prompt,
                    "rerun_target": r.judge_decision.rerun_target,
                    "confidence": r.judge_decision.confidence,
                    "openai_scores": r.judge_decision.openai_scores.__dict__,
                    "claude_scores": r.judge_decision.claude_scores.__dict__,
                },
                "post_action_result": r.post_action_result.__dict__ if r.post_action_result else None,
                "stitched_result": r.stitched_result.__dict__ if r.stitched_result else None,
                "prompt_optimization": r.prompt_optimization.__dict__ if r.prompt_optimization else None,
            }
            for r in records
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(records, initial_prompt))

    print("\nRun complete.", flush=True)
    print(f"SAVE_OUTPUTS: BASE_DIR={BASE_DIR}", flush=True)
    print(f"SAVE_OUTPUTS: OUTPUTS_DIR={OUTPUTS_DIR}", flush=True)
    print(f"SAVE_OUTPUTS: html_file_path={html_path}", flush=True)
    print(f"SAVE_OUTPUTS: json_file_path={json_path}", flush=True)
    print(f"SAVE_OUTPUTS: html_exists={os.path.exists(html_path)}", flush=True)
    print(f"SAVE_OUTPUTS: json_exists={os.path.exists(json_path)}", flush=True)
    print(f"SAVE_OUTPUTS: cwd={os.getcwd()}", flush=True)

    if open_browser:
        try:
            webbrowser.open(os.path.abspath(html_path))
        except Exception:
            pass

    return {
        "html_path": f"outputs/{html_filename}",
        "json_path": f"outputs/{json_filename}",
    }

# =========================
# Wrapper
# =========================

def run_autonomous_compare(
    prompt: str,
    max_iterations: int = MAX_ITERATIONS,
    user_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    options = options or {}

    start_time = time.time()
    logs: List[str] = []
    artifacts: List[Dict[str, str]] = []

    try:
        add_log(logs, "Workflow request received")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        add_log(logs, f"Prompt length={len(prompt)}")
        add_log(logs, f"Prompt optimization enabled: {options.get('optimize_prompt', ENABLE_OPTIMIZER_BY_DEFAULT)}")
        add_log(logs, f"Prompt optimizer strategy: {options.get('optimizer_strategy', 'compare_both')}")
        add_log(logs, f"Stitching enabled: {options.get('enable_stitching', ENABLE_STITCHING_BY_DEFAULT)}")

        records = run_step(
            "run_autonomous_loop",
            lambda: run_autonomous_loop(
                initial_prompt=prompt,
                max_iterations=max_iterations,
                options=options,
                logs=logs,
            ),
            logs,
        )

        if not records:
            raise RuntimeError("No iteration records were produced")

        add_log(logs, f"Completed {len(records)} iteration(s)")

        last = records[-1]
        final_output = get_final_output_text(last)
        add_log(logs, "Final output extracted")
        if last.prompt_optimization:
            add_log(logs, f"Optimizer provider: {last.prompt_optimization.provider}")

        output_paths = run_step(
            "save_outputs",
            lambda: save_outputs(records, prompt, open_browser=False),
            logs,
        )
        add_log(logs, "Outputs saved successfully")

        artifacts.append({
            "type": "html",
            "path": output_paths["html_path"],
        })
        artifacts.append({
            "type": "json",
            "path": output_paths["json_path"],
        })

        runtime_seconds = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "summary": f"{last.judge_decision.winner} selected as best response",
            "full_output": final_output,
            "artifacts": artifacts,
            "logs": logs,
            "runtime_seconds": runtime_seconds,
            "error": None,
        }

    except Exception as e:
        runtime_seconds = round(time.time() - start_time, 2)
        add_log(logs, f"Workflow failed: {str(e)}")

        return {
            "status": "error",
            "summary": "Autonomous workflow failed",
            "full_output": None,
            "artifacts": artifacts,
            "logs": logs,
            "runtime_seconds": runtime_seconds,
            "error": str(e),
        }


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous multi-model workflow runner.")
    parser.add_argument("--prompt", required=True, help="Initial prompt")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Max autonomous refinement loops",
    )
    parser.add_argument(
        "--disable-optimizer",
        action="store_true",
        help="Disable prompt optimization for this run",
    )
    parser.add_argument(
        "--disable-stitching",
        action="store_true",
        help="Disable stitched final output for this run",
    )
    parser.add_argument(
        "--optimizer-strategy",
        default="compare_both",
        choices=["compare_both", "single_openai", "single_claude"],
        help="Prompt optimizer strategy",
    )
    args = parser.parse_args()

    result = run_autonomous_compare(
        prompt=args.prompt,
        max_iterations=args.max_iterations,
        user_id="local_user",
        options={
            "optimize_prompt": not args.disable_optimizer,
            "optimizer_strategy": args.optimizer_strategy,
            "optimizer_target_model": None,
            "optimizer_use_case": None,
            "optimizer_tone_style": None,
            "optimizer_output_format": None,
            "enable_stitching": not args.disable_stitching,
        },
    )

    if result["status"] == "success":
        html_artifact = next((a for a in result["artifacts"] if a["type"] == "html"), None)
        if html_artifact:
            try:
                webbrowser.open(os.path.abspath(html_artifact["path"]))
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("WORKFLOW RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
