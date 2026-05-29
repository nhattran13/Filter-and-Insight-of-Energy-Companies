"""
Iran Conflict / Oil & Energy Filter Agent
==========================================
Reads a Bluesky CSV export, classifies each post using the Google Gemini API,
and writes ALL posts back to a single output CSV with classification columns
appended.

Input CSV columns (required header):
    Headline (CID), Date_Posted, Post_Text, Source_URI, Author_DID

Output CSV columns (all input posts retained):
    Headline (CID), Date_Posted, Post_Text, Source_URI, Author_DID,
    Relevant, Confidence, Matched_Topics, Reasoning

Usage:
    python Filter_Agent.py --input posts.csv
    python Filter_Agent.py --input posts.csv --output classified_posts.csv
    python Filter_Agent.py --input posts.csv --limit 50

Dependencies:
    pip install google-genai pandas
"""

import argparse
import csv
import json
import os
import time

from google import genai
from google.genai import types
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL       = "gemini-3.1-flash-lite"    # change to gemini-1.5-pro for higher accuracy
BATCH_PAUSE = 6.0                    # seconds between API calls (rate-limit buffer)

OUTPUT_FOLDER = "CSV_Files"      # all output CSVs are saved into this folder

PRICE_PER_MILLION_TOKENS = 0.25      # USD per 1 million tokens

# Input column names (exactly as they appear in the CSV header)
COL_CID    = "Headline (CID)"
COL_DATE   = "Date_Posted"
COL_TEXT   = "Post_Text"
COL_SOURCE = "Source_URI"
COL_AUTHOR = "Author_DID"

REQUIRED_COLUMNS = [COL_CID, COL_DATE, COL_TEXT, COL_SOURCE, COL_AUTHOR]

SYSTEM_PROMPT = """You are a content classifier.

Your task is to decide whether a social-media post is relevant to ANY of
the following topics:
  1. The Iran conflict — military operations, diplomacy, sanctions, nuclear
     programme, or Iran's role in
     regional conflicts (Gaza, Lebanon, Yemen, Iraq, Syria).
  2. Oil supply chain / energy markets — OPEC decisions, Strait of Hormuz,
     oil/gas prices, LNG, petroleum exports, or market disruptions linked
     to Middle-East tensions.

Return ONLY a JSON object — no markdown fences, no extra text — with
exactly these keys:
  {
    "relevant": true | false,
    "confidence": "high" | "medium" | "low",
    "topics": ["<matched topic label>", ...],
    "reasoning": "<one concise sentence>"
  }

If the post does not clearly relate to any topic above, set relevant to false.
"""

USER_TEMPLATE = "Post text:\n{text}"


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load and validate the input CSV."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.fillna("").replace({"N/A": "", "n/a": "", "NA": ""})
    return df


# ---------------------------------------------------------------------------
# Gemini classifier
# ---------------------------------------------------------------------------

def classify_post(client: genai.Client, text: str) -> tuple[dict, int, int]:
    """Send a post to Gemini and return (classification dict, input_tokens, output_tokens)."""
    if not text.strip():
        return {
            "relevant":   False,
            "confidence": "high",
            "topics":     [],
            "reasoning":  "Empty post text — skipped.",
        }, 0, 0

    max_retries = 5
    raw = ""
    response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=USER_TEMPLATE.format(text=text[:2000]),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            raw = response.text.strip()
            break  # Success! Exit the retry loop
        except Exception as e:
            err_str = str(e)
            if "ResourceExhausted" in err_str or "429" in err_str:
                if "GenerateRequestsPerDay" in err_str:
                    print("\n[!!!] DAILY QUOTA EXHAUSTED. Stopping script to prevent further errors.")
                    exit()
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10 + 2
                    print(f"\n[!] Quota exceeded. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print("\n[!] Max retries reached. Skipping this post.")
                    return {
                        "relevant": False,
                        "confidence": "low",
                        "topics": [],
                        "reasoning": "Quota limit reached after multiple retries."
                    }, 0, 0
            else:
                raise

    # Extract token usage from response metadata
    input_tokens  = 0
    output_tokens = 0
    if response and hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens  = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw), input_tokens, output_tokens
    except json.JSONDecodeError:
        return {
            "relevant":   False,
            "confidence": "low",
            "topics":     [],
            "reasoning":  f"Could not parse model response: {raw[:120]}",
        }, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_agent(input_path: str, output_path: str, limit: int | None, api_key: str = None) -> None:
    # Resolve API key: parameter → environment variable
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        raise EnvironmentError(
            "Gemini API key not set.\n"
            "Either set GEMINI_API_KEY in main.py, or:\n"
            "  export GEMINI_API_KEY=your-key-from-aistudio.google.com"
        )

    client = genai.Client(api_key=key)

    print(f"Loading posts from : {input_path}")
    df = load_csv(input_path)
    print(f"  -> {len(df)} posts loaded.")

    if limit:
        df = df.head(limit)
        print(f"  -> Processing first {limit} posts only.")

    total = len(df)
    relevant_count      = 0
    total_input_tokens  = 0
    total_output_tokens = 0

    # Classification result columns — pre-filled with defaults
    relevants   = []
    confidences = []
    topics_list = []
    reasonings  = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = row[COL_TEXT]
        cid  = row[COL_CID]

        print(f"\n[{i}/{total}] {cid[:40]}")
        print(f"  Text : {text[:90]}{'...' if len(text) > 90 else ''}")

        result, in_tok, out_tok = classify_post(client, text)

        total_input_tokens  += in_tok
        total_output_tokens += out_tok

        relevants.append(result["relevant"])
        confidences.append(result.get("confidence", ""))
        topics_list.append("; ".join(result.get("topics", [])))
        reasonings.append(result.get("reasoning", ""))

        if result["relevant"]:
            relevant_count += 1
            print(f"  RELEVANT  | confidence={result['confidence']}")
            print(f"  topics    : {'; '.join(result.get('topics', []))}")
            print(f"  reasoning : {result.get('reasoning', '')}")
        else:
            print(f"  not relevant -- {result.get('reasoning', '')[:90]}")

        print(f"  tokens    : {in_tok} in / {out_tok} out")

        time.sleep(BATCH_PAUSE)

    # Append classification columns to the original dataframe
    df["Relevant"]       = relevants
    df["Confidence"]     = confidences
    df["Matched_Topics"] = topics_list
    df["Reasoning"]      = reasonings

    # Ensure output folder exists and resolve full output path
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    final_output_path = os.path.join(OUTPUT_FOLDER, output_path)

    # Write all posts to output CSV
    df.to_csv(final_output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    # Token & cost summary
    total_tokens = total_input_tokens + total_output_tokens
    total_cost   = (total_tokens / 1_000_000) * PRICE_PER_MILLION_TOKENS

    print("\n" + "=" * 60)
    print(f"Processed : {total} posts")
    print(f"Relevant  : {relevant_count} posts  ({relevant_count / total * 100:.1f}%)")
    print(f"Exported  : {final_output_path}  ({total} rows total)")
    print("-" * 60)
    print(f"Tokens used  : {total_input_tokens:,} input  +  {total_output_tokens:,} output  =  {total_tokens:,} total")
    print(f"Estimated cost : ${total_cost:.6f}  (@ ${PRICE_PER_MILLION_TOKENS}/1M tokens)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Bluesky posts for relevance to the Iran conflict, "
            "oil supply chain, and energy companies. All posts are kept; "
            "a Relevant column is added to the output."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="classified_posts.csv",
        help="Path for the output CSV (all posts + classification). Default: classified_posts.csv",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of posts to process (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        api_key=os.environ.get("GEMINI_API_KEY"),
    )