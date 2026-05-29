"""
Post Categorization & Pattern Intelligence
==========================================
Reads classified CSV output, filters to Relevant == True posts, then:
  1. Runs keyword matching (geo-stakeholder vs other)
  2. Detects 5 named signal patterns per post:
       - AIRSTRIKE        : airstrike / bombing / strike language
       - OPEN_FOR_WAR     : open war escalation signals
       - OIL_MOVEMENT     : oil price / supply / market signals
       - GEO_MOVEMENT     : geopolitical shifts, alliances, treaties
       - INFLUENCER_RUMOR : named figures pushing narratives / rumors
  3. Buckets keyword counts (1-5, 6-10, …)
  4. Flags Highlight if geo-stakeholder + any other keyword co-occur
  5. Writes enriched CSV + prints summary

Usage:
    python Insight_Categorize.py --input classified_posts.csv
    python Insight_Categorize.py --input classified_posts.csv --output categorized.csv
"""

import argparse
import csv
import os
import re

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CORE KEYWORD GROUPS (from Filter_CSV.py)
# ─────────────────────────────────────────────────────────────────────────────

GEO_STAKEHOLDER_KEYWORDS = [
    "Iran", "Iranian", "Iran war", "Iran conflict", "Iran sanctions",
    "Iran nuclear", "IRGC", "Strait of Hormuz", "Straits of Hormuz",
    "Hormuz", "Persian Gulf", "Gulf of Oman", "Middle East conflict",
    "Middle East war", "Gulf conflict", "Gulf war", "proxy war",
    "Houthi", "Houthis", "Yemen", "Red Sea", "missile attack",
    "drone attack", "naval blockade", "tanker attack", "Trump",
    "sanctions", "embargo", "geopolitical risk", "geopolitical",
]

OTHER_KEYWORDS = [
    "oil", "crude oil", "crude", "petroleum", "gasoline", "fuel",
    "energy prices", "oil prices", "oil supply", "oil production",
    "oil market", "oil exports", "oil imports", "brent crude", "WTI",
    "OPEC", "barrel", "refinery", "natural gas", "LNG",
    "supply chain", "supply chains", "logistics", "shipping", "cargo",
    "freight", "tanker", "oil tanker", "disruption", "shortage",
    "trade route",
    "Exxon", "ExxonMobil", "Chevron", "BP", "Shell", "TotalEnergies",
    "ConocoPhillips", "Halliburton", "Schlumberger", "SLB",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN SIGNAL DICTIONARIES
# Each key = pattern name, value = list of trigger phrases (whole-word, case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────

PATTERN_KEYWORDS = {
    "AIRSTRIKE": [
        "airstrike", "air strike", "air strikes", "airstrikes",
        "bombing", "bombed", "bomb", "bombers", "strike",
        "strikes", "struck", "warplane", "warplanes", "fighter jet",
        "fighter jets", "F-35", "B-2", "cruise missile", "cruise missiles",
        "precision strike", "surgical strike", "targeted strike",
        "attack on Iran", "hit Iran", "Iran hit", "Iran bombed",
        "Iran struck", "attack Iran", "attacked Iran",
        "nuclear site", "nuclear facility", "Fordow", "Natanz",
        "military strike", "military attack",
    ],
    "OPEN_FOR_WAR": [
        "open war", "full scale war", "full-scale war", "war with Iran",
        "go to war", "declare war", "war declaration", "escalation",
        "escalate", "escalating", "all-out war", "all out war",
        "boots on the ground", "ground invasion", "ground troops",
        "war hawks", "war hawk", "warmonger", "warmongering",
        "Gulf War", "civil war", "war drums", "brink of war",
        "edge of war", "regime change", "regime collapse",
        "imminent war", "imminent threat", "bloody", "conflict",
        "military action", "military campaign", "military operation",
        "invasion", "combat", "battle", "confrontation",
    ],
    "OIL_MOVEMENT": [
        "oil price", "oil prices", "crude oil", "crude", "brent",
        "WTI", "OPEC", "barrel", "petroleum", "gasoline", "fuel price",
        "oil supply", "oil market", "oil exports", "oil imports",
        "energy prices", "energy market", "natural gas", "LNG",
        "refinery", "oil shock", "oil spike", "oil surge",
        "tanker", "oil tanker", "Strait of Hormuz", "Hormuz",
        "oil oligarch", "oil oligarchs", "oil company",
        "Exxon", "Chevron", "BP", "Shell", "TotalEnergies",
        "supply chain", "shipping", "freight", "trade route",
        "oil production", "oil output", "oil disruption",
        "energy security", "strategic reserve", "SPR",
    ],
    "GEO_MOVEMENT": [
        "geopolitical", "geopolitics", "diplomacy", "diplomatic",
        "treaty", "deal", "agreement", "alliance", "allied",
        "coalition", "multilateral", "UN", "United Nations",
        "NATO", "G7", "G20", "sanctions", "embargo", "nuclear deal",
        "JCPOA", "Middle East", "Persian Gulf", "Gulf of Oman",
        "proxy war", "proxy", "regional power", "sphere of influence",
        "normalization", "Abraham Accords", "ceasefire", "peace deal",
        "deterrence", "deterrent", "strategic", "Red Sea",
        "Houthi", "Houthis", "Yemen", "IRGC", "regime",
        "Tehran", "Jerusalem", "Tel Aviv", "Israel", "Netanyahu",
        "Pentagon", "State Department", "White House",
        "Congress", "Senate", "bipartisan",
    ],
    "INFLUENCER_RUMOR": [
        "Trump", "Biden", "Netanyahu", "Khamenei", "Graham",
        "Lindsey Graham", "Kushner", "Pompeo", "Blinken",
        "Rubio", "Carney", "Obama",
        "rumor", "rumours", "narrative", "playbook", "allegation",
        "claims", "accused", "alleges", "allegedly", "interfering",
        "interference", "conspiracy", "propaganda", "disinformation",
        "misinformation", "fake", "manipulate", "manipulation",
        "nutso", "nutsos", "warn", "warning", "ominous",
        "push the narrative", "gin up", "distraction",
        "loyalty test", "slap in the face",
        "idiocy", "corruption", "lies",
    ],
    "PAUSE_STALL": [
        # Diplomatic holds
        "ceasefire", "cease-fire", "negotiations", "negotiation",
        "back-channel", "back channel", "diplomatic solution",
        "talks", "de-escalate", "de-escalation", "stand down",
        "hold fire", "pause the strike", "delay", "hold off",
        "cooling off", "cooling-off period", "diplomatic pause",
        # Legal / political blockers
        "War Powers Act", "war powers", "Congress must authorize",
        "no authorization", "unconstitutional", "bipartisan opposition",
        "Dems demand answers", "veto", "blocked", "filibuster",
        # Deterrence working
        "Iran backed down", "Iran retreated", "Iran stands down",
        "avoided escalation", "prevented war", "pulled back",
        "walked back", "backed off", "defused",
        # Negotiation progress
        "deal reached", "agreement signed", "nuclear talks",
        "IAEA", "uranium freeze", "enrichment freeze",
        "sanctions relief", "prisoner swap", "hostage deal",
        # Public / political resistance
        "anti-war", "antiwar", "oppose the strike", "no war with Iran",
        "war is not the answer", "protest the war", "peace talks",
        "diplomatic channel", "diplomatic channels", "restraint",
        "measured response", "proportional response",
        # Deal suspension language
        "on hold", "now on hold", "deal on hold", "put on hold",
        "suspended", "suspension", "stalled", "stall", "freeze",
        "frozen", "paused", "pause", "halted", "halt",
        "Iran deal", "the deal", "deal off", "deal collapsed",
        "collapsed", "breakdown", "broke down", "fell apart",
        # Uncertainty / limbo framing
        "uncertain", "uncertainty", "in limbo", "unclear",
        "no deal", "deal dead", "back to square one",
    ],
}

POST_TEXT_COLUMN = "Post_Text"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compile(keywords):
    return [(kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in keywords]


GEO_PATTERNS   = _compile(GEO_STAKEHOLDER_KEYWORDS)
OTHER_PATTERNS = _compile(OTHER_KEYWORDS)

PATTERN_COMPILED = {
    name: _compile(kws) for name, kws in PATTERN_KEYWORDS.items()
}


def match_core_keywords(text):
    geo   = [kw for kw, pat in GEO_PATTERNS   if pat.search(text)]
    other = [kw for kw, pat in OTHER_PATTERNS if pat.search(text)]
    return geo, other


def match_patterns(text):
    """Returns dict: pattern_name -> list of matched trigger words."""
    return {
        name: [kw for kw, pat in patterns if pat.search(text)]
        for name, patterns in PATTERN_COMPILED.items()
    }


def bucket_label(n):
    if n == 0:
        return "Bucket_0"
    low  = ((n - 1) // 5) * 5 + 1
    high = low + 4
    return f"Bucket_{low}_{high}"


def all_bucket_labels(max_count):
    labels = ["Bucket_0"]
    n = 1
    while n <= max_count:
        low  = ((n - 1) // 5) * 5 + 1
        high = low + 4
        lbl  = f"Bucket_{low}_{high}"
        if lbl not in labels:
            labels.append(lbl)
        n = high + 1
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def categorize(input_path, output_path):
    print(f"Reading : {input_path}")
    df = pd.read_csv(input_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")

    if POST_TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{POST_TEXT_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    total_raw = len(df)

    # ── Filter to Relevant == True only ──────────────────────────────────────
    if "Relevant" in df.columns:
        df = df[df["Relevant"].str.strip().str.lower() == "true"].reset_index(drop=True)
        print(f"  -> {len(df)} relevant rows (filtered from {total_raw} total).\n")
    else:
        print(f"  -> No 'Relevant' column found; processing all {total_raw} rows.\n")

    total = len(df)

    # ── Per-row analysis ──────────────────────────────────────────────────────
    all_geo_hits   = []
    all_other_hits = []
    all_counts     = []
    all_kw_strings = []
    all_pattern_hits = {name: [] for name in PATTERN_KEYWORDS}
    all_pattern_kws  = {name: [] for name in PATTERN_KEYWORDS}

    for text in df[POST_TEXT_COLUMN]:
        geo, other = match_core_keywords(text)
        count = len(geo) + len(other)
        all_geo_hits.append(geo)
        all_other_hits.append(other)
        all_counts.append(count)
        all_kw_strings.append("; ".join(sorted(set(geo + other))))

        pattern_results = match_patterns(text)
        for name in PATTERN_KEYWORDS:
            hits = pattern_results[name]
            all_pattern_hits[name].append(1 if hits else 0)
            all_pattern_kws[name].append("; ".join(sorted(set(hits))))

    # ── Bucket columns ────────────────────────────────────────────────────────
    max_count   = max(all_counts) if all_counts else 0
    bucket_lbls = all_bucket_labels(max_count)
    for lbl in bucket_lbls:
        df[lbl] = 0
    for idx, count in enumerate(all_counts):
        df.at[idx, bucket_label(count)] = 1

    # ── Core summary columns ──────────────────────────────────────────────────
    df["Keyword_Match_Count"] = all_counts
    df["Matched_Keywords"]    = all_kw_strings
    df["Geo_Stakeholder_Hit"] = [1 if g else 0 for g in all_geo_hits]
    df["Other_Keywords_Hit"]  = [1 if o else 0 for o in all_other_hits]
    df["Highlight"]           = [
        g == 1 and o == 1
        for g, o in zip(df["Geo_Stakeholder_Hit"], df["Other_Keywords_Hit"])
    ]

    # ── Pattern signal columns ────────────────────────────────────────────────
    pattern_cols = []
    for name in PATTERN_KEYWORDS:
        hit_col = f"Pattern_{name}"
        kw_col  = f"Pattern_{name}_Keywords"
        df[hit_col] = all_pattern_hits[name]
        df[kw_col]  = all_pattern_kws[name]
        pattern_cols += [hit_col, kw_col]

    # Combined signal: how many patterns fired on this post
    df["Pattern_Signal_Count"] = df[[f"Pattern_{n}" for n in PATTERN_KEYWORDS]].sum(axis=1)

    # ── Column ordering ───────────────────────────────────────────────────────
    original_cols = list(df.columns[:df.columns.tolist().index(POST_TEXT_COLUMN) + 1])
    new_cols = (
        ["Keyword_Match_Count", "Matched_Keywords"]
        + bucket_lbls
        + ["Geo_Stakeholder_Hit", "Other_Keywords_Hit", "Highlight"]
        + pattern_cols
        + ["Pattern_Signal_Count"]
    )
    remaining = [c for c in df.columns if c not in original_cols and c not in new_cols]
    df = df[original_cols + new_cols + remaining]

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs("CSV_Files", exist_ok=True)
    out = os.path.join("CSV_Files", os.path.basename(output_path))
    df.to_csv(out, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    # ── Summary report ────────────────────────────────────────────────────────
    highlighted  = int(df["Highlight"].sum())
    geo_only     = int(((df["Geo_Stakeholder_Hit"] == 1) & (df["Other_Keywords_Hit"] == 0)).sum())
    other_only   = int(((df["Geo_Stakeholder_Hit"] == 0) & (df["Other_Keywords_Hit"] == 1)).sum())
    zero_matches = int((df["Keyword_Match_Count"] == 0).sum())

    print("=" * 65)
    print(f"Total relevant posts processed : {total}")
    print(f"Zero keyword matches           : {zero_matches}")
    print(f"Geo-stakeholder hits only      : {geo_only}")
    print(f"Other keyword hits only        : {other_only}")
    print(f"HIGHLIGHTED (both groups)      : {highlighted}  ({highlighted / total * 100:.1f}%)")

    print()
    print("── PATTERN SIGNAL BREAKDOWN ──────────────────────────────────")
    for name in PATTERN_KEYWORDS:
        col   = f"Pattern_{name}"
        count = int(df[col].sum())
        bar   = "█" * count
        print(f"  {name:<20} {count:>3} / {total}  {bar}")

    print()
    print("── POSTS WITH MULTIPLE PATTERNS FIRING ───────────────────────")
    multi = df[df["Pattern_Signal_Count"] >= 2]
    if len(multi):
        for _, row in multi.iterrows():
            active = [n for n in PATTERN_KEYWORDS if row[f"Pattern_{n}"] == 1]
            snippet = row[POST_TEXT_COLUMN][:120].replace("\n", " ")
            print(f"  [{', '.join(active)}]")
            print(f"    \"{snippet}...\"")
            print()
    else:
        print("  None found.")

    print()
    print("── KEYWORD-COUNT BUCKETS ─────────────────────────────────────")
    for lbl in bucket_lbls:
        count = int(df[lbl].sum())
        bar   = "█" * min(count, 40)
        print(f"  {lbl:<18} {count:>5}  {bar}")

    print()
    print(f"Output written to : {out}")
    print("=" * 65)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Categorize relevant posts by keyword match + 5 named pattern signals: "
            "AIRSTRIKE, OPEN_FOR_WAR, OIL_MOVEMENT, GEO_MOVEMENT, INFLUENCER_RUMOR."
        )
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to classified CSV from Filter_Agent.py.")
    parser.add_argument("--output", "-o", default="categorized_posts.csv",
                        help="Output CSV filename. Default: categorized_posts.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    categorize(input_path=args.input, output_path=args.output)