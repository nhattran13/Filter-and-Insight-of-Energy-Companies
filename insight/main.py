"""
main.py
=======
Runs two Insight Agent scripts simultaneously, one for Chevron and one
for ExxonMobil, both reading from the same classified_posts.csv.

Each script runs as a separate subprocess in its own thread, so both
execute in parallel rather than one after another.

NOTE on input():
Both Insight_Agent scripts call input() at startup to collect an optional
system prompt addition. Since two subprocesses cannot share one terminal's
stdin cleanly, this script reads ONE shared context string from the user
up front, then passes it to both subprocesses non-interactively via the
--context flag (see "--context" in parse_args() inside each Insight_Agent
script). This avoids stdin collisions between the two parallel processes.

Usage:
    python main.py --input classified_posts.csv --api-key YOUR_GEMINI_KEY
"""

import argparse
import os
import subprocess
import sys
import threading
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <-- Make sure to set this in your .env file

SCRIPTS = {
    "Chevron":    "Insight_Agent-Chevron.py",
    "ExxonMobil": "Insight_Agent-ExxonMobil.py",
}


# ─────────────────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────────────────

def run_script(name: str, script_path: str, input_path: str, api_key: str, context: str) -> None:
    """Runs one Insight Agent script as a subprocess and streams its output live."""
    print(f"[{name}] Starting {script_path}...")

    cmd = [
        sys.executable, script_path,
        "--input", input_path,
        "--api-key", api_key,
    ]
    if context:
        cmd += ["--context", context]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output live, prefixed with the company name so both
    # processes' output can be told apart in the shared terminal.
    for line in process.stdout:
        print(f"[{name}] {line}", end="")

    process.wait()
    print(f"[{name}] Finished with exit code {process.returncode}.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chevron and ExxonMobil Insight Agents simultaneously."
    )
    # Changed default to "classified_posts.csv" and removed required=True
    parser.add_argument("--input", "-i", default="CSV_Files/classified_posts.csv", required=False,
                        help="Path to classified_posts.csv (shared by both scripts).")
    parser.add_argument("--context","-c",default="",help="Additional system prompt context")
    args = parser.parse_args()

    print("=" * 65)
    print("  Iran Insight Agent — Parallel Runner")
    print("  Running: Chevron + ExxonMobil simultaneously")
    print("=" * 65)
    print()
    
    # ── Hardcoded Context ──────────────────────────────────────────────────────
    # Bypassed the input() interactive prompt by hardcoding an empty string
    context = args.context

    if context:
        print(f"Using context: {context}")
    else:
        print("No additional system context provided.")
        print("Skipping additional system prompt input...")
    print()

    # ── Launch both scripts in parallel threads ─────────────────────────────────
    threads = []
    for name, script_path in SCRIPTS.items():
        t = threading.Thread(
            target=run_script,
            args=(name, script_path, args.input, GEMINI_API_KEY, context),
        )
        threads.append(t)
        t.start()

    # ── Wait for both to finish ───────────────────────────────────────────────
    for t in threads:
        t.join()

    print("=" * 65)
    print("  Both Insight Agents Completed Successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()