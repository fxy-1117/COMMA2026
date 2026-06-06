# Reasoning-Chain Prompt

This prompt is used to ask an LLM for multiple implicit reasoning statements
that connect a premise to a claim. The generated text is stored in the `Helpful`
column used by Experiment 3.

## Chain Prompt With Topic

```text
Generate {num_statements} reasoning statements that connect the premise to the claim based on the provided label, specifically related to the topic provided.


**Premise:** {premise}
**Claim:** {claim}

**Topic:** {topic}
**Label:** {label}


**Instructions:**
- If the label is "contradiction," provide statements that are implied by the premise but contradict the claim, while relating to the topic.
- If the label is "entailment," provide statements that logically links the premise to the claim, while relating to the topic.
- Ensure exactly {num_statements} statements are generated to establish a coherent connection.

- Each statement must be concise, logically follow from the previous one, and be limited to 10 words or fewer.
- Use clear, direct language without pronouns.
- Do not repeat the premise or claim verbatim.
- Separate multiple statements with "→" in your output.


**Output Format:**
Your output must follow this structure precisely. No additional text, headers, or explanations.

Premise: {premise}
Claim: {claim}
Helpful: [insert reasoning chain here]
```

The run that produced `data/rte_pairs_exp3.csv` used `num_statements="six"` and
then parsed the response by reading the line that starts with `Helpful:`.

## Python Entry Points

Use `comma_core.prompting.build_reasoning_chain_prompt` to render the prompt and
`comma_core.prompting.parse_chain_output` to convert model output into the CSV
row shape.

API keys are intentionally not stored in the repository. To call DeepSeek:

```powershell
$env:DEEPSEEK_API_KEY='...'
python -c "from comma_core.prompting import ask_deepseek; print(ask_deepseek('hello'))"
```
