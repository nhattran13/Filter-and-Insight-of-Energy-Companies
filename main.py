"""
main.py
=======
Runs the full pipeline using the existing files:
  1. data_ingest_test.py    — streams Bluesky for 60s, saves bluesky_stream_data.csv
  2. Filter_CSV.py          — keyword-filters that CSV, saves filtered_output.csv
  3. Filter_Agent.py        — AI classifies filtered posts, saves classified_posts.csv
  4. Insight_Categorize.py  — categorizes classified posts, saves categorized_posts.csv
"""

import asyncio
from data_ingest import stream_continuous
from Filter_CSV import filter_csv, OUTPUT_FILE
from Filter_Agent import run_agent
from Insight_Categorize import categorize
import os
from dotenv import load_dotenv
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI API KEY  ← paste your key from aistudio.google.com here
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <-- Make sure to set this in your .env file

script_dir = os.path.dirname(os.path.abspath(__file__))

CLASSIFIED_FILE  = os.path.join(script_dir, "CSV_Files", "classified_posts.csv")
CATEGORIZED_FILE = os.path.join(script_dir, "CSV_Files", "categorized_posts.csv")


def main():
    print("=" * 60)
    print(" Step 1: Streaming posts from Bluesky...")
    print("=" * 60)
#    asyncio.run(stream_continuous())

    print("\n" + "=" * 60)
    print(" Step 2: Filtering posts by keywords...")
    print("=" * 60)
    filter_csv()

    print("\n" + "=" * 60)
    print(" Step 3: AI classification with Gemini...")
    print("=" * 60)
    run_agent(input_path=OUTPUT_FILE, output_path=CLASSIFIED_FILE, limit=None, api_key=GEMINI_API_KEY)

    print("\n" + "=" * 60)
    print(" Step 4: Categorizing classified posts...")
    print("=" * 60)
    categorize(input_path=CLASSIFIED_FILE, output_path=CATEGORIZED_FILE)

    print("\nPipeline complete.")
    print(f"  Filtered data  : {OUTPUT_FILE}")
    print(f"  Classified data: {CLASSIFIED_FILE}")
    print(f"  Categorized    : {CATEGORIZED_FILE}")


if __name__ == "__main__":
    main()