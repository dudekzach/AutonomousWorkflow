from __future__ import annotations

import argparse
import html
import json
import os
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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

OPENAI_MODEL = "gpt-4.1-mini"
CLAUDE_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "gpt-4.1-mini"

CLAUDE_MAX_TOKENS = 6000
MAX_ITERATIONS = 3
MAX_CONTINUATION_ATTEMPTS = 3


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
class IterationRecord:
    iteration: int
    active_prompt: str
    openai_result: ProviderResult
    claude_result: ProviderResult
    judge_decision: JudgeDecision
    post_action_result: Optional[ProviderResult] = None
    stitched_result: Optional[ProviderResult] = None


# =========================
# Setup
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


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
    """
    Heuristic check for likely truncated or incomplete outputs.
    This is intentionally simple and conservative.
    """
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
    """
    Rewrites follow-up prompts so they stay within what this workflow can
    actually execute: text-only model responses, not real file generation.
    """
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
    else:
        return call_openai_new_chat(stitch_prompt, model=model)


# =========================
# Orchestrator
# =========================

def run_autonomous_loop(initial_prompt: str, max_iterations: int = MAX_ITERATIONS) -> List[IterationRecord]:
    records: List[IterationRecord] = []

    openai_state = OpenAIChatState()
    claude_state = ClaudeChatState()

    active_prompt = initial_prompt

    for iteration in range(1, max_iterations + 1):
        openai_result = call_openai_new_chat(active_prompt)
        if openai_result.response_id:
            openai_state.last_response_id = openai_result.response_id

        claude_result = call_claude_new_chat(claude_state, active_prompt)

        judge = judge_outputs(active_prompt, openai_result, claude_result)

        post_action_result: Optional[ProviderResult] = None

        if judge.next_action == "accept":
            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=None,
                )
            )
            break

        if judge.next_action == "revise_prompt_and_rerun":
            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=None,
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

            if target_provider == "OpenAI":
                post_action_result = call_openai_follow_up(openai_state, follow_up_prompt)

                continuation_attempts = 0
                while (
                    post_action_result
                    and looks_incomplete(post_action_result.text or "")
                    and continuation_attempts < MAX_CONTINUATION_ATTEMPTS
                ):
                    continuation_attempts += 1
                    follow_up_prompt = (
                        "Please continue from where you left off and complete the answer. "
                        "Do not restart. Finish any incomplete sections and provide the full final version."
                    )
                    post_action_result = call_openai_follow_up(openai_state, follow_up_prompt)

            elif target_provider == "Claude":
                post_action_result = call_claude_follow_up(claude_state, follow_up_prompt)

                continuation_attempts = 0
                while (
                    post_action_result
                    and looks_incomplete(post_action_result.text or "")
                    and continuation_attempts < MAX_CONTINUATION_ATTEMPTS
                ):
                    continuation_attempts += 1
                    follow_up_prompt = (
                        "Please continue from where you left off and complete the answer. "
                        "Do not restart. Finish any incomplete sections and provide the full final version."
                    )
                    post_action_result = call_claude_follow_up(claude_state, follow_up_prompt)

            elif target_provider == "Both":
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        active_prompt=active_prompt,
                        openai_result=openai_result,
                        claude_result=claude_result,
                        judge_decision=judge,
                        post_action_result=None,
                    )
                )
                active_prompt = judge.revised_prompt.strip() or follow_up_prompt
                continue

            else:
                if judge.revised_prompt.strip():
                    records.append(
                        IterationRecord(
                            iteration=iteration,
                            active_prompt=active_prompt,
                            openai_result=openai_result,
                            claude_result=claude_result,
                            judge_decision=judge,
                            post_action_result=None,
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
                    )
                )
                break

            stitched_result: Optional[ProviderResult] = None

            if post_action_result:
                if target_provider == "Claude":
                    base_text = claude_result.text or ""
                    stitched_result = stitch_final_response(
                        provider="Claude",
                        model=CLAUDE_MODEL,
                        original_prompt=active_prompt,
                        base_text=base_text,
                        continuation_text=post_action_result.text or "",
                    )
                elif target_provider == "OpenAI":
                    base_text = openai_result.text or ""
                    stitched_result = stitch_final_response(
                        provider="OpenAI",
                        model=OPENAI_MODEL,
                        original_prompt=active_prompt,
                        base_text=base_text,
                        continuation_text=post_action_result.text or "",
                    )

            records.append(
                IterationRecord(
                    iteration=iteration,
                    active_prompt=active_prompt,
                    openai_result=openai_result,
                    claude_result=claude_result,
                    judge_decision=judge,
                    post_action_result=post_action_result,
                    stitched_result=stitched_result,
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
            )
        )
        break

    return records


# =========================
# Reporting helpers
# =========================

def esc(text: Optional[str]) -> str:
    return html.escape(text or "")


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
            <td class="{openai_class}">{openai_val}</td>
            <td class="{claude_class}">{claude_val}</td>
        </tr>
        """
        )

    return f"""
    <table class="score-table">
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
    """
    Determines the best final output.
    Combines outputs when follow-up is an enhancement, not a replacement.
    """
    if not last_record.post_action_result:
        winner = last_record.judge_decision.winner
        if winner == "OpenAI":
            return last_record.openai_result.text or last_record.openai_result.error or ""
        if winner == "Claude":
            return last_record.claude_result.text or last_record.claude_result.error or ""
        return "Tie detected. Review manually."

    follow_up_text = last_record.post_action_result.text or ""
    winner = last_record.judge_decision.winner

    if winner == "OpenAI":
        base_output = last_record.openai_result.text or ""
    elif winner == "Claude":
        base_output = last_record.claude_result.text or ""
    else:
        base_output = ""

    format_indicators = ["```", "mermaid", "<html", "<svg", "flowchart", "diagram"]
    is_format_output = any(indicator in follow_up_text.lower() for indicator in format_indicators)

    if is_format_output:
        if looks_incomplete(base_output):
            return follow_up_text

        return (
            base_output
            + "\n\n---\n\n"
            + "### Enhanced Output (Generated Artifact)\n\n"
            + follow_up_text
        )

    return follow_up_text


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
            <div class="section action">
                <div class="title">Automated Next Step Result</div>
                <div class="subtitle">
                    Provider: {esc(record.post_action_result.provider)} |
                    Model: {esc(record.post_action_result.model)}
                </div>
                <pre>{esc(record.post_action_result.text or record.post_action_result.error)}</pre>
            </div>
            """

        score_table = render_score_table(
            record.judge_decision.openai_scores,
            record.judge_decision.claude_scores,
        )

        sections.append(
            f"""
        <div class="section">
            <div class="title">Iteration {record.iteration}</div>

            <div class="prompt-block">
                <strong>Active Prompt</strong>
                <pre>{esc(record.active_prompt)}</pre>
            </div>

            <div class="judge-summary">
                <div><strong>Winner:</strong> {esc(record.judge_decision.winner)}</div>
                <div><strong>Next Action:</strong> {esc(record.judge_decision.next_action)}</div>
                <div><strong>Confidence:</strong> {esc(record.judge_decision.confidence)}</div>
            </div>

            <div class="judge-block">
                <strong>Judge Reasoning</strong>
                <pre>{esc(record.judge_decision.reason)}</pre>
            </div>

            <div class="score-block">
                <strong>Scorecard (1–5)</strong>
                {score_table}
            </div>

            <div class="container">
                <div class="card">
                    <div class="title">OpenAI</div>
                    <div class="subtitle">{esc(record.openai_result.model)}</div>
                    <pre>{esc(record.openai_result.text or record.openai_result.error)}</pre>
                </div>

                <div class="card">
                    <div class="title">Claude</div>
                    <div class="subtitle">{esc(record.claude_result.model)}</div>
                    <pre>{esc(record.claude_result.text or record.claude_result.error)}</pre>
                </div>
            </div>

            <div class="judge-block">
                <strong>Generated Follow-Up Prompt</strong>
                <pre>{esc(record.judge_decision.follow_up_prompt or 'N/A')}</pre>
            </div>

            <div class="judge-block">
                <strong>Generated Revised Prompt</strong>
                <pre>{esc(record.judge_decision.revised_prompt or 'N/A')}</pre>
            </div>

            {post_action_html}
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
            }}
            .card {{
                flex: 1;
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
        <div class="lead">Initial Prompt</div>
        <div class="section">
            <pre>{esc(initial_prompt)}</pre>
        </div>

        <div class="section hero">
            <div class="title">Final Output (Recommended)</div>
            <div class="meta">
                <div><strong>Winner:</strong> {esc(last_record.judge_decision.winner)}</div>
                <div><strong>Next Action Chosen:</strong> {esc(last_record.judge_decision.next_action)}</div>
                <div><strong>Confidence:</strong> {esc(last_record.judge_decision.confidence)}</div>
            </div>
            <div class="judge-block">
                <strong>Why This Won</strong>
                <pre>{esc(last_record.judge_decision.reason)}</pre>
            </div>
            <div class="score-block">
                <strong>Final Scorecard (1–5)</strong>
                {final_score_table}
            </div>
            <div class="judge-block">
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
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = f"outputs/autonomous_run_{timestamp}.html"
    json_path = f"outputs/autonomous_run_{timestamp}.json"

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
            }
            for r in records
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(records, initial_prompt))

    print("\nRun complete.")
    print(f"HTML report: {html_path}")
    print(f"JSON log:    {json_path}")

    if open_browser:
        try:
            webbrowser.open(os.path.abspath(html_path))
        except Exception:
            pass

    return {
        "html_path": html_path,
        "json_path": json_path,
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
    del user_id
    del options

    start_time = time.time()
    logs: List[str] = []
    artifacts: List[Dict[str, str]] = []

    try:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logs.append("Starting autonomous workflow")

        records = run_autonomous_loop(
            initial_prompt=prompt,
            max_iterations=max_iterations,
        )

        logs.append(f"Completed {len(records)} iteration(s)")

        last = records[-1]
        final_output = get_final_output_text(last)
        logs.append("Final output extracted")

        output_paths = save_outputs(records, prompt, open_browser=False)
        logs.append("Outputs saved successfully")

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
        logs.append(f"Workflow failed: {str(e)}")

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
    args = parser.parse_args()

    result = run_autonomous_compare(
        prompt=args.prompt,
        max_iterations=args.max_iterations,
        user_id="local_user",
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
