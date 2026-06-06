# -*- coding: utf-8 -*-
"""Prompt templates and parsers used to build the implicit-premise data."""

from __future__ import annotations

import os
from typing import Dict, Optional, Union


REASONING_SEPARATOR = " \u2192 "
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _topic_text(topic: str) -> str:
    return str(topic).replace("_", " ")


def build_reasoning_chain_prompt(
    premise: str,
    claim: str,
    num_statements: Union[int, str],
    topic: str,
    label: str,
) -> str:
    """Build the chain prompt used for Exp3 implicit-premise generation."""
    return f"""Generate {num_statements} reasoning statements that connect the premise to the claim based on the provided label, specifically related to the topic provided.


**Premise:** {premise}
**Claim:** {claim}

**Topic:** {_topic_text(topic)}
**Label:** {label}


**Instructions:**
- If the label is "contradiction," provide statements that are implied by the premise but contradict the claim, while relating to the topic.
- If the label is "entailment," provide statements that logically links the premise to the claim, while relating to the topic.
- Ensure exactly {num_statements} statements are generated to establish a coherent connection.

- Each statement must be concise, logically follow from the previous one, and be limited to 10 words or fewer.
- Use clear, direct language without pronouns.
- Do not repeat the premise or claim verbatim.
- Separate multiple statements with "{REASONING_SEPARATOR.strip()}" in your output.


**Output Format:**
Your output must follow this structure precisely. No additional text, headers, or explanations.

Premise: {premise}
Claim: {claim}
Helpful: [insert reasoning chain here]
"""


def build_single_implicit_prompt(
    premise: str,
    claim: str,
    topic: str,
    label: str,
) -> str:
    """Build the single implicit-premise prompt used for Exp2 data."""
    return f"""Generate a reasoning statement that connects the premise to the claim based on the label provided, specifically related to the topic provided.


**Premise:** {premise}
**Claim:** {claim}
**Topic:** {_topic_text(topic)}
**Label:** {label}


**Instructions:**
- If the label is "contradiction," provide a statement that is implied by the premise but contradicts the claim, while relating to the topic.

- If the label is "entailment," provide a statement that logically links the premise to the claim, while relating to the topic.
- Limit the statement to 10 words or fewer.

- Use clear, direct language without pronouns.
- Do not repeat the premise or claim verbatim.


**Output Format:**
Your output must match the following structure exactly. No additional text, headers, or explanations.

Premise: {premise}
Claim: {claim}
Helpful: [insert the single helpful reasoning statement here]
"""


def parse_chain_output(
    output_string: str,
    premise: str,
    claim: str,
    label: str,
) -> Dict[str, Optional[str]]:
    """Parse a generated prompt response into the Exp3 CSV row format."""
    helpful: Optional[str] = None

    for line in output_string.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Helpful: "):
            helpful = stripped[len("Helpful: ") :].strip()

    return {
        "Premise": premise,
        "Claim": claim,
        "Helpful": helpful,
        "label": label,
    }


def ask_deepseek(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
) -> str:
    """Call DeepSeek through the OpenAI-compatible API.

    The API key is read from ``DEEPSEEK_API_KEY`` unless passed explicitly.
    No key is stored in this repository.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or pass api_key explicitly.")

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content
