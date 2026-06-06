# Single Implicit-Premise Prompt

This prompt is used to ask an LLM for one implicit reasoning statement that
connects a premise to a claim. The generated text is stored in the `Helpful`
column used by Experiment 2.

## Single Prompt With Topic

```text
Generate a reasoning statement that connects the premise to the claim based on the label provided, specifically related to the topic provided.


**Premise:** {premise}
**Claim:** {claim}
**Topic:** {topic}
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
```

## Python Entry Points

Use `comma_core.prompting.build_single_implicit_prompt` to render the prompt and
`comma_core.prompting.parse_chain_output` to convert model output into the CSV
row shape. The parser is shared because both prompt outputs use the same
`Premise`/`Claim`/`Helpful` format.
