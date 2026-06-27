"""
Iran Insight Agent
==================
Reads classified_posts.csv, filters to Relevant == True posts, then sends
posts to Gemini in batches of 3. Each post is analysed for 3 outputs:
    1. War_State          — current state of the Iran conflict in this post
    2. Oil_Supply_Chain   — oil supply chain status signals in this post
    3. Energy_Companies   — motions of other energy companies in this post

After all batches are processed, one final conclusion call synthesises
everything into a single intelligence finding printed to stdout.

All per-post results are saved to a single timestamped CSV.
Token usage and estimated cost are printed at the end.

Usage:

Dependencies:
    pip install pandas google-genai
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

from google import genai
from google.genai import types
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME       = "gemini-3.1-flash-lite"
POST_TEXT_COLUMN = "Post_Text"
RELEVANT_COLUMN  = "Relevant"
BATCH_SIZE       = 5       # posts per API call
API_DELAY        = 2.0     # seconds between calls to avoid rate limiting
PRICE_PER_1M     = 0.25    # USD per 1 million tokens


# ─────────────────────────────────────────────────────────────────────────────
# AGENT PROMPT
# Fill in each section below. Do not remove {post_text}, {total}, or {digest}.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_PROMPT = {

    # ── SECTION 1: SYSTEM ────────────────────────────────────────────────────
    # Defines the agent's identity, role, expertise, and tone.
    # Sent once as the system instruction before any analysis begins.
    # ↓ Type your system prompt inside the triple quotes below ↓
    "system": """
    You are a intelligence analyst specializing in global oil energy markets for ExxonMobil. Your task is to analyse
    social media posts and extract insight, conside intelligence signals related to Iran conflict and ExxonMobil's specific interests. 
    
    """,
    # ↑ End of system prompt ↑


    # ── SECTION 2: BATCH (PER-POST INSTRUCTIONS) ─────────────────────────────
    # Instructions for how to analyse each post.
    # {post_text} is replaced automatically — do not remove it.
    # ↓ Type your analysis instructions above {post_text} below ↓
    "batch": """
    Analyse only the provided what is written in the posts and extract intelligence signals about the Iran conflict. For each post extract insights related to: 
    1. War State: what is the current state of the Iran conflictas described in this post? Escalateing, stalled, ceasefire, de-escalating, or unclear?
    2. Oil Supply Chain: output these options as applicable: in usual route, routes as changed, halt due to impact on coming strike, strikes on transportation or supply prouduction,
    or unclear?
    3. Energy Companies: In one sentence, summarize any motions of other energy companies mentioned in this post, such as in production, supply, or transportation.
    With unveried rumors or unclear signals, Be concise of what is the rumor claimed to be and its source.
    IMPORTANT: Analyse only what is explicitly written in the post. Do not bring in external context, background knowledge, or assumptions about the Iran conflict beyond what the post states.
    {post_text}

    """,
    # ↑ End of batch prompt ↑


    # ── SECTION 3: CONCLUSION ────────────────────────────────────────────────
    # Instructions for the final conclusion across all posts.
    # {total} = number of posts analysed — do not remove.
    # {digest} = combined outputs of all posts — do not remove.
    # ↓ Type your conclusion framing above {total} and {digest} below ↓
    "conclusion": """

    You have analysed {total} posts. Synthesize a final intelligence assessment brefing covering related to ExxonMobil: 
    1. War State Conclusion: state overall state of the Iran conflict across all posts, and predict how it might evolve the next day or hours.
    2. Oil Supply Chain Conclusion: state overall impact on oil supply chain across all posts, and predict how it might evolve the next day or hours.
    3. Energy Companies Conclusion: state overall motions of energy companies across all posts, and predict how they might evolve the next day or hours.
    Give a final assessment overall what does this mean for ExxonMobil and predict its implications.


    {digest}

    """,
    # ↑ End of conclusion prompt ↑

}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL PROMPT BUILDERS
# Wires AGENT_PROMPT sections into final prompts sent to Gemini.
# Do not edit below this line.
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    base = AGENT_PROMPT["system"].strip()
    if USER_SYSTEM_INPUT.strip():
        return f"{base}\n\n{USER_SYSTEM_INPUT.strip()}"
    return base


def _build_batch_prompt(batch: list[dict]) -> str:
    posts_block = "\n\n".join(
        f'POST_{item["index"]}: """{item["text"].strip()}"""'
        for item in batch
    )
    user_instructions = AGENT_PROMPT["batch"].format(
        post_text=posts_block
    ).strip()
    return f"""{user_instructions}

Respond ONLY in a valid JSON array — no markdown, no preamble.
One object per post in the same order as above:
[{{"post_index":1,"war_state":"...","oil_supply_chain":"...","energy_companies":"..."}}]""".strip()


def _build_conclusion_prompt(total: int, digest: str) -> str:
    user_framing = AGENT_PROMPT["conclusion"].format(
        total=total,
        digest=digest.strip()
    ).strip()
    return f"""{user_framing}

Respond ONLY in valid JSON — no markdown, no preamble:
{{"war_state_conclusion":"...","oil_supply_chain_conclusion":"...","energy_companies_conclusion":"...","final_assessment":"..."}}""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN TRACKER
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_USAGE = {"input": 0, "output": 0}

# ── User input added to the system prompt at runtime ──────────────────────
USER_SYSTEM_INPUT = ""  # set via --context CLI arg in __main__ below; never blocks on stdin


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def init_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def call_gemini(client: genai.Client, prompt: str, expect_list: bool = False, retries: int = 3):
    """
    Call Gemini and parse JSON response. Retries on failure.
    expect_list=True  → returns a list  (batch calls)
    expect_list=False → returns a dict  (conclusion call)
    """
    system = _build_system_prompt()

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        response_mime_type="application/json",
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            raw = response.text.strip()

            # ── Track token usage ─────────────────────────────────────────
            if response.usage_metadata:
                TOKEN_USAGE["input"]  += response.usage_metadata.prompt_token_count or 0
                TOKEN_USAGE["output"] += response.usage_metadata.candidates_token_count or 0

            # Strip markdown fences if model wraps in ```json
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())

        except json.JSONDecodeError as e:
            print(f"    [Attempt {attempt+1}] JSON parse error: {e}. Retrying...")
            time.sleep(API_DELAY)
        except Exception as e:
            print(f"    [Attempt {attempt+1}] API error: {e}. Retrying...")
            time.sleep(API_DELAY * 2)

    # Return safe empty structure on total failure
    if expect_list:
        return [
            {
                "post_index":      0,
                "war_state":       "ERROR: could not parse response",
                "oil_supply_chain":"ERROR: could not parse response",
                "energy_companies":"ERROR: could not parse response",
            }
        ]
    return {
        "war_state_conclusion":        "ERROR: could not parse response",
        "oil_supply_chain_conclusion": "ERROR: could not parse response",
        "energy_companies_conclusion": "ERROR: could not parse response",
        "final_assessment":            "ERROR: could not parse response",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_insight(input_path: str, output_path: str, api_key: str) -> None:
    print("=" * 65)
    print("  Iran Insight Agent")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Batch size : {BATCH_SIZE} posts per call")
    print(f"  Input      : {input_path}")
    print("=" * 65)

    # ── Load & filter ─────────────────────────────────────────────────────────
    df = pd.read_csv(input_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")

    total_raw = len(df)

    if RELEVANT_COLUMN in df.columns:
        df = df[df[RELEVANT_COLUMN].str.strip().str.lower() == "true"].reset_index(drop=True)
        print(f"\n  Loaded {total_raw} rows → {len(df)} relevant posts after filter.\n")
    else:
        print(f"\n  No '{RELEVANT_COLUMN}' column found. Processing all {total_raw} rows.\n")

    total = len(df)

    if total == 0:
        print("  No relevant posts to process. Check the CSV file.")
        return

    # ── Build post list & batches ─────────────────────────────────────────────
    posts = [
        {
            "index": idx + 1,
            "text":  row[POST_TEXT_COLUMN].strip(),
            "row":   row,
        }
        for idx, row in df.iterrows()
    ]
    batches     = [posts[i:i + BATCH_SIZE] for i in range(0, len(posts), BATCH_SIZE)]
    total_batch = len(batches)

    print(f"  {total} posts → {total_batch} batches of up to {BATCH_SIZE}\n")

    # ── Init Gemini client ────────────────────────────────────────────────────
    client = init_client(api_key)

    # ── Batch analysis ────────────────────────────────────────────────────────
    results      = []
    digest_lines = []

    for b_idx, batch in enumerate(batches):
        print(f"  [Batch {b_idx+1}/{total_batch}] Sending {len(batch)} posts...")

        prompt      = _build_batch_prompt(batch)
        raw_results = call_gemini(client, prompt, expect_list=True)

        # Ensure raw_results is a list (safety check)
        if isinstance(raw_results, dict):
            raw_results = [raw_results]

        # Map each result back to its original row
        for item, analysis in zip(batch, raw_results):
            row = item["row"]

            war_state  = analysis.get("war_state", "")
            oil_supply = analysis.get("oil_supply_chain", "")
            energy_cos = analysis.get("energy_companies", "")

            result = {
                "Post_Index":       item["index"],
                "Date_Posted":      row.get("Date_Posted", ""),
                "Author_DID":       row.get("Author_DID", ""),
                "Source_URI":       row.get("Source_URI", ""),
                "Confidence":       row.get("Confidence", ""),
                "Matched_Topics":   row.get("Matched_Topics", ""),
                "Post_Text":        item["text"],
                # ── 3 outputs ──
                "War_State":        war_state,
                "Oil_Supply_Chain": oil_supply,
                "Energy_Companies": energy_cos,
            }
            results.append(result)

            digest_lines.append(
                f"POST {item['index']}:\n"
                f"  War State       : {war_state}\n"
                f"  Oil Supply Chain: {oil_supply}\n"
                f"  Energy Companies: {energy_cos}"
            )

            print(f"    Post {item['index']:>2} | War State       : {war_state[:70]}")
            print(f"           | Oil Supply Chain: {oil_supply[:70]}")
            print(f"           | Energy Companies: {energy_cos}")
            print()

        time.sleep(API_DELAY)

    # ── Final conclusion ───────────────────────────────────────────────────────
    print("─" * 65)
    print("  Generating final conclusion across all posts...")
    print("─" * 65)

    conclusion_prompt = _build_conclusion_prompt(
        total=total,
        digest="\n\n".join(digest_lines),
    )
    conclusion = call_gemini(client, conclusion_prompt, expect_list=False)

    print("\n  ── CONCLUSION ────────────────────────────────────────────")
    print(f"  War State        : {conclusion.get('war_state_conclusion','')}")
    print(f"  Oil Supply Chain : {conclusion.get('oil_supply_chain_conclusion','')}")
    print(f"  Energy Companies : {conclusion.get('energy_companies_conclusion','')}")
    print()
    print("  ── FINAL ASSESSMENT ──────────────────────────────────────")
    print(f"  {conclusion.get('final_assessment','')}")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("CSV_Files", exist_ok=True)
    timestamp    = datetime.now().strftime("%B%d_%Y_%I-%M%p")
    base         = os.path.splitext(os.path.basename(output_path))[0]
    combined_csv = os.path.join("CSV_Files", f"{base}_ExxonMobil.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(combined_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

# ── Save conclusion CSV ───────────────────────────────────────────────────
    os.makedirs("Final_Assessment", exist_ok=True)
    conclusion_csv = os.path.join("Final_Assessment", f"{base}_conclusion_{timestamp}_ExxonMobil.csv")
    conclusion_row = {
        "Timestamp":                   timestamp,
        "Total_Posts_Analysed":        total,
        "War_State_Conclusion":        conclusion.get("war_state_conclusion", ""),
        "Oil_Supply_Chain_Conclusion": conclusion.get("oil_supply_chain_conclusion", ""),
        "Energy_Companies_Conclusion": conclusion.get("energy_companies_conclusion", ""),
        "Final_Assessment":            conclusion.get("final_assessment", ""),
    }
    pd.DataFrame([conclusion_row]).to_csv(
        conclusion_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8"
    )

    # ── Token usage & cost ────────────────────────────────────────────────────
    total_tokens = TOKEN_USAGE["input"] + TOKEN_USAGE["output"]
    cost         = (total_tokens / 1_000_000) * PRICE_PER_1M

    print("=" * 65)
    print(f"  Combined CSV saved   : {combined_csv}")
    print(f"  Conclusion CSV saved : {conclusion_csv}")
    print("─" * 65)
    print("  ── TOKEN USAGE ───────────────────────────────────────────")
    print(f"  Input tokens         : {TOKEN_USAGE['input']:,}")
    print(f"  Output tokens        : {TOKEN_USAGE['output']:,}")
    print(f"  Total tokens         : {total_tokens:,}")
    print(f"  Estimated cost       : ${cost:.6f}  (@ ${PRICE_PER_1M} / 1M tokens)")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iran Insight Agent — analyses relevant posts in batches of 3 with Gemini. "
            "Outputs war state, oil supply chain status, and energy company motions "
            "per post, plus a final conclusion printed to stdout."
        )
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to classified CSV (must have Post_Text and Relevant columns).")
    parser.add_argument("--output", "-o", default="iran_insight",
                        help="Base name for output CSV. Default: iran_insight")
    parser.add_argument("--api-key", "-k", required=True,
                        help="Gemini API key.")
    parser.add_argument("--context", "-c", default="",
                        help="Additional system prompt context (non-interactive, replaces input() prompt).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    USER_SYSTEM_INPUT = args.context
    run_agent_insight(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
    )