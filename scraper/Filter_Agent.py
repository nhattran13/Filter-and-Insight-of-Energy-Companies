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
BATCH_SIZE  = 10                     # number of posts sent to the LLM per API call

OUTPUT_FOLDER = "CSV_Files"      # all output CSVs are saved into this folder

PRICE_PER_MILLION_TOKENS = 0.25      # USD per 1 million tokens

# Input column names (exactly as they appear in the CSV header)
COL_CID    = "Headline (CID)"
COL_DATE   = "Date_Posted"
COL_TEXT   = "Post_Text"
COL_SOURCE = "Source_URI"
COL_AUTHOR = "Author_DID"

REQUIRED_COLUMNS = [COL_CID, COL_DATE, COL_TEXT, COL_SOURCE, COL_AUTHOR]

OUTPUT_COLUMNS = [
    COL_CID, COL_DATE, COL_TEXT, COL_SOURCE, COL_AUTHOR,
    "Relevant", "Confidence", "Matched_Topics", "Reasoning",
]

SYSTEM_PROMPT = """You are a content classifier.

You will be given a numbered batch of social-media posts. For EACH post,
decide whether it is relevant to ANY of the following topics:
  1. The Iran conflict — military operations, diplomacy, sanctions, nuclear
     programme, or Iran's role in
     regional conflicts (Gaza, Lebanon, Yemen, Iraq, Syria).
  2. Oil supply chain / energy markets — OPEC decisions, Strait of Hormuz,
     oil/gas prices, LNG, petroleum exports, or market disruptions linked
     to Middle-East tensions.

Return ONLY a JSON array — no markdown fences, no extra text — with one
object per post, in the SAME ORDER as the input posts. Each object must
have exactly these keys:
  {
    "index": <the post number as given in the input>,
    "relevant": true | false,
    "confidence": "high" | "medium" | "low",
    "topics": ["<matched topic label>", ...],
    "reasoning": "<one concise sentence>"
  }

The output array must contain exactly as many objects as there were posts
in the input batch. If a post does not clearly relate to any topic above,
set relevant to false for that post.
"""

USER_TEMPLATE = "Post text:\n{text}"

BATCH_USER_TEMPLATE = "Classify the following {count} posts.\n\n{posts_block}"


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

def _empty_result(reasoning: str, confidence: str = "low") -> dict:
    return {
        "relevant":   False,
        "confidence": confidence,
        "topics":     [],
        "reasoning":  reasoning,
    }


def classify_batch(client: genai.Client, texts: list[str]) -> tuple[list[dict], int, int]:
    """Send a batch of posts to Gemini and return (list of classification dicts, input_tokens, output_tokens).

    The returned list is always the same length and order as `texts`.
    Empty-text posts are classified locally (skipped) without using the API.
    """
    # Map local batch positions -> texts that actually need to be sent to the model
    indices_to_send = [i for i, t in enumerate(texts) if t.strip()]

    # All posts in this batch are empty — nothing to send
    if not indices_to_send:
        results = [_empty_result("Empty post text — skipped.", "high") for _ in texts]
        return results, 0, 0

    posts_block = "\n\n".join(
        f"Post {pos + 1}:\n{texts[pos][:2000]}" for pos in indices_to_send
    )
    user_message = BATCH_USER_TEMPLATE.format(count=len(indices_to_send), posts_block=posts_block)

    max_retries = 5
    raw = ""
    response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_message,
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
                    print("\n[!] Max retries reached. Skipping this batch.")
                    results = [_empty_result("Quota limit reached after multiple retries.") for _ in texts]
                    return results, 0, 0
            else:
                # Non-quota error — report it and signal caller to continue
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

    # Start with a default "could not parse" result for every post in the batch
    results = [
        _empty_result(f"Could not parse model response: {raw[:70]}")
        for _ in texts
    ]
    # Pre-fill empty posts (not sent to the model) with their skip reasoning
    for i, t in enumerate(texts):
        if not t.strip():
            results[i] = _empty_result("Empty post text — skipped.", "high")

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Model response was not a JSON array")

        for item in parsed:
            idx = item.get("index")
            if idx is None:
                continue
            pos = idx - 1  # convert 1-based "Post N" back to 0-based list position
            if 0 <= pos < len(texts):
                results[pos] = {
                    "relevant":   bool(item.get("relevant", False)),
                    "confidence": item.get("confidence", ""),
                    "topics":     item.get("topics", []),
                    "reasoning":  item.get("reasoning", ""),
                }
    except (json.JSONDecodeError, ValueError):
        pass  # leave the default "could not parse" results in place

    return results, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_agent_filter(input_path: str, output_path: str, limit: int | None, api_key: str = None) -> None:
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
    error_count         = 0
    total_input_tokens  = 0
    total_output_tokens = 0

    # Ensure output folder exists and resolve full output path
    final_output_path = os.path.join(OUTPUT_FOLDER, output_path)
    os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)

    # Open the output CSV once and write rows immediately after each batch is scanned
    with open(final_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        rows = list(df.iterrows())  # list of (orig_index, row)
        num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(num_batches):
            start = batch_num * BATCH_SIZE
            end   = min(start + BATCH_SIZE, total)
            batch_rows = rows[start:end]
            batch_texts = [row[COL_TEXT] for _, row in batch_rows]

            print(f"\n[Batch {batch_num + 1}/{num_batches}] posts {start + 1}-{end} of {total}")
            for offset, (_, row) in enumerate(batch_rows):
                cid  = row[COL_CID]
                text = row[COL_TEXT]
                print(f"  ({start + offset + 1}/{total}) {cid[:40]} -- {text[:70]}{'...' if len(text) > 70 else ''}")

            try:
                results, in_tok, out_tok = classify_batch(client, batch_texts)

            except Exception as e:
                error_count += len(batch_rows)
                print(f"  [ERROR] {type(e).__name__}: {e}")
                print(f"  -> Skipping batch and continuing...")
                results = [
                    {
                        "relevant":   False,
                        "confidence": "low",
                        "topics":     [],
                        "reasoning":  f"Error during classification: {type(e).__name__}: {str(e)[:70]}",
                    }
                    for _ in batch_rows
                ]
                in_tok  = 0
                out_tok = 0

            total_input_tokens  += in_tok
            total_output_tokens += out_tok

            for (_, row), result in zip(batch_rows, results):
                text = row[COL_TEXT]

                if result["relevant"]:
                    relevant_count += 1
                    print(f"  RELEVANT  | {row[COL_CID][:30]:30s} confidence={result['confidence']}")
                    print(f"      topics    : {'; '.join(result.get('topics', []))}")
                    print(f"      reasoning : {result.get('reasoning', '')}")
                else:
                    print(f"  not relevant | {row[COL_CID][:30]:30s} confidence={result['confidence']}")

                # Write this row immediately to disk — safe even if script crashes
                writer.writerow({
                    COL_CID:           row[COL_CID],
                    COL_DATE:          row[COL_DATE],
                    COL_TEXT:          text,
                    COL_SOURCE:        row[COL_SOURCE],
                    COL_AUTHOR:        row[COL_AUTHOR],
                    "Relevant":        result["relevant"],
                    "Confidence":      result.get("confidence", ""),
                    "Matched_Topics":  "; ".join(result.get("topics", [])),
                    "Reasoning":       result.get("reasoning", ""),
                })

            csv_file.flush()  # force write to disk immediately
            print(f"  tokens    : {in_tok} in / {out_tok} out (batch total)")

            time.sleep(BATCH_PAUSE)

    # Token & cost summary
    total_tokens = total_input_tokens + total_output_tokens
    total_cost   = (total_tokens / 1_000_000) * PRICE_PER_MILLION_TOKENS

    print("\n" + "=" * 60)
    print(f"Processed : {total} posts")
    print(f"Relevant  : {relevant_count} posts  ({relevant_count / total * 100:.1f}%)")
    print(f"Errors    : {error_count} posts skipped due to errors")
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