import os
import pandas as pd
import re

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Insert your CSV file name/path here when switching to a new file.
# Example: INPUT_FILE = "bluesky_stream_data.csv"
#          INPUT_FILE = "/path/to/your/file.csv"
INPUT_FILE = "bluesky_stream_data.csv"  # <-- INSERT FILE NAME HERE

OUTPUT_FOLDER = "CSV_Files"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "filtered_output.csv")

# Bluesky CSV columns (from bluesky_stream_data.csv):
#   Headline (CID) | Date_Posted | Post_Text | Source_URI | Author_DID
#
# The filter searches only the columns listed below.
# Adjust this list if your CSV has different column names.
SEARCH_COLUMNS = ["Post_Text", "Headline (CID)"]  # <-- CHANGE IF NEEDED

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# Case-insensitive. Matched as whole words or phrases.
# Add / remove freely.
# ─────────────────────────────────────────────────────────────────────────────

KEYWORDS = [
    # ── Oil & Energy ──────────────────────────────────────────────────────────
    "oil",
    "crude oil",
    "crude",
    "petroleum",
    "gasoline",
    "fuel",
    "energy prices",
    "oil prices",
    "oil supply",
    "oil production",
    "oil market",
    "oil exports",
    "oil imports",
    "brent crude",
    "WTI",
    "OPEC",
    "barrel",
    "refinery",
    "natural gas",
    "LNG",

    # ── Supply Chain ──────────────────────────────────────────────────────────
    "supply chain",
    "supply chains",
    "logistics",
    "shipping",
    "cargo",
    "freight",
    "tanker",
    "oil tanker",
    "disruption",
    "shortage",
    "trade route",

    # ── Oil Companies ─────────────────────────────────────────────────────────
    "Exxon",
    "ExxonMobil",
    "Chevron",
    "BP",
    "Shell",
    "TotalEnergies",
    "ConocoPhillips",
    "Halliburton",
    "Schlumberger",
    "SLB",

    # ── Iran Conflict & Geopolitics ───────────────────────────────────────────
    "Iran",
    "Iranian",
    "Iran war",
    "Iran conflict",
    "Iran sanctions",
    "Iran nuclear",
    "IRGC",
    "Strait of Hormuz",
    "Straits of Hormuz",
    "Hormuz",
    "Persian Gulf",
    "Gulf of Oman",
    "Middle East conflict",
    "Middle East war",
    "Gulf conflict",
    "Gulf war",
    "proxy war",
    "Houthi",
    "Houthis",
    "Yemen",
    "Red Sea",
    "missile attack",
    "drone attack",
    "naval blockade",
    "tanker attack",

    # ── Political Figures / Policy ────────────────────────────────────────────
    "Trump",
    "sanctions",
    "embargo",
    "geopolitical risk",
    "geopolitical",
]

# ─────────────────────────────────────────────────────────────────────────────
# FILTER LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def build_pattern(keywords):
    """Compile a single regex from all keywords (longest phrases matched first)."""
    escaped = sorted([re.escape(kw) for kw in keywords], key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def filter_csv():
    df = pd.read_csv(INPUT_FILE, dtype=str)
    df.fillna("", inplace=True)

    # Only keep columns that actually exist in this file
    cols_to_search = [c for c in SEARCH_COLUMNS if c in df.columns]
    if not cols_to_search:
        raise ValueError(
            f"None of the SEARCH_COLUMNS {SEARCH_COLUMNS} were found in the file.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    pattern = build_pattern(KEYWORDS)

    mask = df[cols_to_search].apply(
        lambda col: col.str.contains(pattern, regex=True)
    ).any(axis=1)

    filtered_df = df[mask]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Creates folder if it doesn't exist
    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Columns searched   : {cols_to_search}")
    print(f"Total rows read    : {len(df)}")
    print(f"Rows matched       : {len(filtered_df)}")
    print(f"Filtered file saved: {OUTPUT_FILE}")

