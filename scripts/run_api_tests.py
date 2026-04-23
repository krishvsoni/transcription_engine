#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows so translated text in responses doesn't crash
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""API test runner.

Hits every endpoint in the BitScribe API, records pass/fail, response
time, HTTP status, and a sample of the response body, then writes two
output files:

  api_test_results.csv   — full row per endpoint test
  api_test_report.md     — human-readable pass/fail report

Usage:
    python scripts/run_api_tests.py
    python scripts/run_api_tests.py --base-url http://localhost:8000
    python scripts/run_api_tests.py --base-url http://localhost:8000 --output my_results
    python scripts/run_api_tests.py --skip-slow     # skip transcription (slow)
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


TEST_CASES = [
   
    ("Health", "Health check", "GET", "/health", None, 200, False),
    ("Scheduler", "Scheduler status", "GET", "/ingestion/scheduler/status", None, 200, False),
    ("Translation", "List supported languages", "GET", "/translation/languages", None, 200, False),
    ("Translation", "Translate to French (Gemini)", "POST", "/translation/text",
     {"text": "The Lightning Network enables instant Bitcoin payments with very low fees.", "target_language": "fr"},
     200, False),

    ("Translation", "Translate to German (Gemini)", "POST", "/translation/text",
     {"text": "Bitcoin mining uses proof of work. The hashrate secures the network.", "target_language": "de"},
     200, False),

    ("Translation", "Translate to Spanish (Gemini)", "POST", "/translation/text",
     {"text": "Taproot activates Schnorr signatures, enabling more private Bitcoin transactions.", "target_language": "es"},
     200, False),

  
    ("Translation", "Translate to Hindi (Sarvam)", "POST", "/translation/text",
     {"text": "Bitcoin uses a proof-of-work consensus mechanism. The UTXO model tracks unspent outputs.", "target_language": "hi"},
     200, False),

    ("Translation", "Translate to Tamil (Sarvam)", "POST", "/translation/text",
     {"text": "The Lightning Network is a layer-2 payment protocol built on Bitcoin.", "target_language": "ta"},
     200, False),

    ("Translation", "Translate to Telugu (Sarvam)", "POST", "/translation/text",
     {"text": "Taproot and Schnorr signatures improve Bitcoin privacy and efficiency.", "target_language": "te"},
     200, False),

  
    ("Translation", "Translate transcript to hi+es+fr", "POST", "/translation/transcript",
     {
         "raw_text": (
             "Speaker 0: Today we are discussing the Lightning Network and its HTLC mechanism. "
             "The key insight is that payment channels allow off-chain settlement. "
             "Speaker 1: Right. And with Taproot we can make channel opens and closes indistinguishable "
             "from regular transactions, improving privacy significantly."
         ),
         "target_languages": ["hi", "es", "fr"],
     },
     200, False),

    
    ("RSS", "List RSS feeds", "GET", "/ingestion/rss/feeds", None, 200, False),

    ("RSS", "Poll bitcoin_optech feed", "POST", "/ingestion/rss/poll/bitcoin_optech", None, 200, False),

    ("RSS", "Add custom RSS feed", "POST", "/ingestion/rss/feeds",
     {"name": "Test Feed Delete Me", "url": "https://bitcoinops.org/feed.xml",
      "loc": "podcast", "tags": ["bitcoin"], "category": ["bitcoin"]},
     [200, 409], False),  # 409 is ok if already exists

    
    ("Conference", "Conference supported languages", "GET", "/conference/supported-languages", None, 200, False),

    ("Conference", "Discover conferences (1 page)", "POST", "/conference/discover",
     {"max_pages": 1}, 200, False),

    
    ("Ingestion", "List channels", "GET", "/ingestion/channels", None, [200, 503], False),
    ("Ingestion", "List videos", "GET", "/ingestion/videos", None, [200, 503], False),
    ("Ingestion", "List ingestion runs", "GET", "/ingestion/runs", None, [200, 503], False),

  
    ("Transcription", "View queue", "GET", "/transcription/queue/", None, 200, False),
    ("Transcription", "View corrected transcripts", "GET", "/transcription/corrected/", None, 200, False),
    ("Transcription", "View summaries", "GET", "/transcription/summaries/", None, 200, False),
    ("Transcription", "Submit YouTube job (Deepgram+correct+summarize)", "POST_FORM",
     "/transcription/add_to_queue/",
     [
         ("source", "https://www.youtube.com/watch?v=33i1PdSJgwA"),
         ("loc", "tabconf"),
         ("username", "test_runner"),
         ("deepgram", "true"),
         ("diarize", "true"),
         ("markdown", "true"),
         ("correct", "true"),
         ("summarize", "true"),
         ("llm_provider", "google"),
     ],
     200, True),  # marked slow=True
]


class APITestRunner:
    def __init__(self, base_url: str, skip_slow: bool = False):
        self.base_url = base_url.rstrip("/")
        self.skip_slow = skip_slow
        self.results: list[dict] = []
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "BitScribe-TestRunner/1.0"

    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  BitScribe API Test Runner")
        print(f"  Base URL : {self.base_url}")
        print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        for group, name, method, path, body, expected, slow in TEST_CASES:
            if slow and self.skip_slow:
                print(f"  [SKIP] {group} / {name}")
                self.results.append(self._skipped_row(group, name, method, path))
                continue

            row = self._run_one(group, name, method, path, body, expected)
            self.results.append(row)

            status_icon = "PASS" if row["passed"] else "FAIL"
            print(
                f"  [{status_icon}] {group:15s} | {name[:45]:45s} | "
                f"HTTP {row['http_status']} | {row['elapsed_ms']:.0f}ms"
            )
            if not row["passed"]:
                print(f"        ERROR: {row['error']}")

        passed = sum(1 for r in self.results if r["passed"])
        skipped = sum(1 for r in self.results if r.get("skipped"))
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total - skipped} passed  ({skipped} skipped)")
        print(f"{'='*60}\n")

    def _run_one(self, group, name, method, path, body, expected) -> dict:
        url = self.base_url + path
        expected_statuses = expected if isinstance(expected, list) else [expected]
        start = time.time()
        http_status = 0
        response_preview = ""
        error = ""
        passed = False

        try:
            if method == "GET":
                resp = self._session.get(url, timeout=60)
            elif method == "POST":
                resp = self._session.post(url, json=body, timeout=120)
            elif method == "POST_FORM":
                resp = self._session.post(url, data=body, timeout=600)
            elif method == "DELETE":
                resp = self._session.delete(url, timeout=30)
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed_ms = (time.time() - start) * 1000
            http_status = resp.status_code
            passed = http_status in expected_statuses

            try:
                data = resp.json()
                response_preview = json.dumps(data)[:300]
                if not passed:
                    error = data.get("detail", str(data))[:200]
            except Exception:
                response_preview = resp.text[:300]
                if not passed:
                    error = resp.text[:200]

        except requests.exceptions.ConnectionError:
            elapsed_ms = (time.time() - start) * 1000
            error = "Connection refused — is the server running?"
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start) * 1000
            error = "Request timed out"
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            error = str(e)[:200]

        return {
            "group": group,
            "test_name": name,
            "method": method.replace("_FORM", ""),
            "endpoint": path,
            "http_status": http_status,
            "expected_status": str(expected_statuses),
            "passed": passed,
            "skipped": False,
            "elapsed_ms": round(elapsed_ms, 1),
            "response_preview": response_preview,
            "error": error,
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _skipped_row(group, name, method, path) -> dict:
        return {
            "group": group,
            "test_name": name,
            "method": method,
            "endpoint": path,
            "http_status": 0,
            "expected_status": "",
            "passed": False,
            "skipped": True,
            "elapsed_ms": 0,
            "response_preview": "",
            "error": "Skipped (--skip-slow)",
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }



def write_csv(results: list[dict], output_path: str):
    path = Path(output_path)
    fieldnames = [
        "group", "test_name", "method", "endpoint",
        "http_status", "expected_status", "passed", "skipped",
        "elapsed_ms", "error", "response_preview", "tested_at",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"CSV  saved -> {path.resolve()}")


def write_markdown(results: list[dict], output_path: str, base_url: str):
    path = Path(output_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(1 for r in results if r["passed"])
    skipped = sum(1 for r in results if r.get("skipped"))
    total = len(results)
    ran = total - skipped

    lines = [
        "# BitScribe API Test Report",
        "",
        f"**Server:** `{base_url}`  ",
        f"**Run at:** {now}  ",
        f"**Result:** {passed}/{ran} passed ({skipped} skipped)  ",
        "",
        "---",
        "",
        "## Summary by Group",
        "",
        "| Group | Passed | Total | Status |",
        "|-------|--------|-------|--------|",
    ]

    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)

    for group, rows in groups.items():
        ran_in_group = [r for r in rows if not r.get("skipped")]
        p = sum(1 for r in ran_in_group if r["passed"])
        t = len(ran_in_group)
        icon = "PASS" if p == t else ("PARTIAL" if p > 0 else "FAIL")
        lines.append(f"| {group} | {p} | {t} | {icon} |")

    lines += [
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    for group, rows in groups.items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Test | Method | Endpoint | Status | Time | Result |")
        lines.append("|------|--------|----------|--------|------|--------|")
        for r in rows:
            if r.get("skipped"):
                icon = "⏭ skip"
            elif r["passed"]:
                icon = "pass"
            else:
                icon = "fail"
            lines.append(
                f"| {r['test_name']} "
                f"| `{r['method']}` "
                f"| `{r['endpoint']}` "
                f"| {r['http_status']} "
                f"| {r['elapsed_ms']:.0f}ms "
                f"| {icon} |"
            )
            if not r["passed"] and not r.get("skipped") and r["error"]:
                lines.append(f"|  | | _Error: {r['error'][:100]}_ | | | |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Sample Responses",
        "",
    ]

    for r in results:
        if r["passed"] and r["response_preview"] and not r.get("skipped"):
            lines.append(f"### `{r['method']} {r['endpoint']}`")
            lines.append("")
            lines.append("```json")
            try:
                pretty = json.dumps(json.loads(r["response_preview"]), indent=2)[:600]
                lines.append(pretty)
            except Exception:
                lines.append(r["response_preview"][:400])
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown saved -> {path.resolve()}")



def main():
    parser = argparse.ArgumentParser(description="BitScribe API test runner")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output", default="api_test_results",
        help="Output file base name (default: api_test_results → .csv + .md)"
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Skip tests that hit external services (transcription jobs)"
    )
    args = parser.parse_args()

    runner = APITestRunner(base_url=args.base_url, skip_slow=args.skip_slow)
    runner.run_all()

    csv_path = args.output + ".csv"
    md_path = args.output + ".md"

    write_csv(runner.results, csv_path)
    write_markdown(runner.results, md_path, args.base_url)


if __name__ == "__main__":
    main()
