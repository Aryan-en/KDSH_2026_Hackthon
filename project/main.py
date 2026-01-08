import pathway as pw
from typing import Dict
from openai import OpenAI

# =========================
# OPENAI CLIENT (API KEY HERE)
# =========================

client = OpenAI(
    api_key=""   # <-- PUT YOUR API KEY HERE LATER
)

# =========================
# LLM HELPERS
# =========================

def llm_call(prompt: str) -> str:
    """
    Generic OpenAI call wrapper.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a precise information extraction system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def extract_characteristics_from_text(text: str) -> Dict[str, str]:
    """
    Extract character traits from novel text.
    """

    prompt = f"""
Extract ONLY explicit character traits from the text.
Return a Python dictionary.

Rules:
- Use short keys (violence, authority, morality, fear, ambition)
- Values must be single words
- If unclear, omit the trait
- Do not infer beyond text

Text:
{text}

Return format:
{{"trait": "value"}}
"""

    output = llm_call(prompt)

    try:
        return eval(output)
    except:
        return {}


def extract_backstory_claims(text: str) -> Dict[str, str]:
    """
    Extract explicit claims from backstory.
    """

    prompt = f"""
Extract explicit character claims from this backstory.
Return a Python dictionary.

Rules:
- Claims must be testable
- Use same keys as novel traits if possible
- No interpretation beyond text

Backstory:
{text}

Return format:
{{"trait": "value"}}
"""

    output = llm_call(prompt)

    try:
        return eval(output)
    except:
        return {}


def check_consistency(
    novel_traits: Dict[str, str],
    backstory_traits: Dict[str, str]
) -> Dict[str, str]:
    """
    Uses LLM to check SUPPORTS / CONTRADICTS / NEUTRAL
    """

    results = {}

    for trait, backstory_value in backstory_traits.items():
        novel_value = novel_traits.get(trait, "unknown")

        prompt = f"""
Trait: {trait}

Backstory claim: {backstory_value}
Novel evidence: {novel_value}

Does the novel contradict the backstory?

Answer ONLY one word:
SUPPORTS
CONTRADICTS
NEUTRAL
"""

        verdict = llm_call(prompt)
        results[trait] = verdict

    return results


# =========================
# MAIN PIPELINE
# =========================

def main():

    NOVEL_PATH = "novel.txt"
    BACKSTORY_PATH = "backstory.txt"

    # --------
    # LOAD NOVEL USING PATHWAY
    # --------
    novel_table = pw.io.fs.read(
        NOVEL_PATH,
        format="text"
    )

    novel_chunks = novel_table.select(text=pw.this.data)

    # --------
    # EXTRACT NOVEL CHARACTERISTICS
    # --------
    novel_characteristics = {}

    for row in novel_chunks:
        traits = extract_characteristics_from_text(row["text"])
        for k, v in traits.items():
            novel_characteristics[k] = v

    # --------
    # LOAD BACKSTORY
    # --------
    with open(BACKSTORY_PATH, "r", encoding="utf-8") as f:
        backstory_text = f.read()

    backstory_characteristics = extract_backstory_claims(backstory_text)

    # --------
    # CONSISTENCY CHECK
    # --------
    consistency_results = check_consistency(
        novel_characteristics,
        backstory_characteristics
    )

    # --------
    # SCORE
    # --------
    total = len(consistency_results)
    supports = sum(1 for v in consistency_results.values() if v == "SUPPORTS")
    contradicts = sum(1 for v in consistency_results.values() if v == "CONTRADICTS")

    consistency_percentage = (supports / total) * 100 if total else 0
    final_label = 0 if contradicts > 0 else 1

    # --------
    # OUTPUT
    # --------
    print("\n=== NOVEL CHARACTERISTICS ===")
    print(novel_characteristics)

    print("\n=== BACKSTORY CHARACTERISTICS ===")
    print(backstory_characteristics)

    print("\n=== CONSISTENCY ANALYSIS ===")
    print(consistency_results)

    print("\nConsistency Percentage:", round(consistency_percentage, 2), "%")
    print("FINAL PREDICTION:", final_label)


if __name__ == "__main__":
    main()
