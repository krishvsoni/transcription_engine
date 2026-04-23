#!/usr/bin/env python3
"""STT provider evaluation script.

Benchmarks Whisper, Deepgram, and Gemini STT against the same Bitcoin
conference audio files and produces a CSV + console summary ranked by
Word Error Rate (WER), Character Error Rate (CER), and domain accuracy.

Usage:
    python scripts/evaluate_stt.py --audio path/to/audio.mp3 [--reference path/to/ref.txt]
    python scripts/evaluate_stt.py --batch path/to/manifest.json
    python scripts/evaluate_stt.py --youtube VIDEO_ID

Manifest format (JSON list):
    [
      {
        "video_id": "abc123",
        "title": "Taproot Explained",
        "speaker": "Pieter Wuille",
        "conference": "Bitcoin 2022",
        "reference_transcript": "path/to/ref.txt"   // optional
      }
    ]

Outputs:
    evaluation_results.csv   — full row per (video, provider)
    evaluation_summary.txt   — ranked leaderboard
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


BITCOIN_DOMAIN_TERMS = [
    "utxo", "taproot", "schnorr", "lightning", "segwit", "timelock",
    "multisig", "mempool", "hashrate", "difficulty", "proof of work",
    "merkle", "coinbase", "bip", "script", "witness", "p2wpkh", "p2tr",
    "musig", "miniscript", "descriptor", "psbt", "channel", "htlc",
    "lnurl", "bolt", "watchtower", "submarine swap", "splicing",
    "fedimint", "cashu", "ecash", "nostr", "covenant", "op_return",
    "op_csv", "op_ctv", "bip32", "bip39", "bip340", "bip341",
    "stratum", "getblocktemplate", "mining pool", "full node",
    "bitcoin core", "lnd", "cln", "eclair", "ldk",
]

WHISPER_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
DEFAULT_WHISPER_MODEL = "small.en"



def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein distance at word level
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref)][len(hyp)] / len(ref)


def compute_domain_accuracy(hypothesis: str) -> float:
    """Fraction of known Bitcoin terms correctly present in the transcript."""
    hyp_lower = _normalize(hypothesis)
    found = sum(1 for term in BITCOIN_DOMAIN_TERMS if term in hyp_lower)
    return found / len(BITCOIN_DOMAIN_TERMS)



def _transcribe_whisper(audio_path: str, model_name: str = DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe with local Whisper model."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Whisper not installed. Run: pip install .[whisper]")

    print(f"  [whisper/{model_name}] transcribing...")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result.get("text", "")


def _transcribe_deepgram(audio_path: str) -> str:
    """Transcribe via Deepgram Nova-2."""
    try:
        import deepgram as dg_sdk
    except ImportError:
        raise RuntimeError("deepgram-sdk not installed.")

    from app.config import settings

    print("  [deepgram] transcribing...")
    client = dg_sdk.Deepgram(settings.DEEPGRAM_API_KEY)
    with open(audio_path, "rb") as f:
        mime = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
        source = {"buffer": f, "mimetype": mime}
        response = client.transcription.sync_prerecorded(
            source,
            {
                "model": "nova-2",
                "punctuate": True,
                "smart_formatting": True,
                "language": "en",
            },
        )
    transcript = (
        response.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    return transcript


def _transcribe_gemini(audio_path: str) -> str:
    """Transcribe via Gemini STT service."""
    from app.data_writer import DataWriter
    from app.services.gemini_stt import GeminiSTT
    from app.transcript import Transcript, Source

    print("  [gemini] transcribing...")
    # We only need the audio_to_text method; create a minimal stub
    writer = DataWriter(tempfile.mkdtemp())
    service = GeminiSTT(diarize=False, upload=False, data_writer=writer)
    result = service.audio_to_text(audio_path)
    return result.get("text", "")


PROVIDERS = {
    "whisper": _transcribe_whisper,
    "deepgram": _transcribe_deepgram,
    "gemini": _transcribe_gemini,
}




def download_audio(youtube_url: str, output_dir: str) -> str:
    """Download audio from YouTube using yt-dlp. Returns local file path."""
    import yt_dlp

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id", "unknown")

    audio_path = os.path.join(output_dir, f"{video_id}.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Downloaded audio not found at {audio_path}")
    return audio_path


def evaluate_audio(
    audio_path: str,
    reference_text: str = "",
    providers: list[str] = None,
    metadata: dict = None,
) -> list[dict]:
    """Run all providers against one audio file and return result rows."""
    if providers is None:
        providers = list(PROVIDERS.keys())

    rows = []
    meta = metadata or {}

    for provider_name in providers:
        runner = PROVIDERS.get(provider_name)
        if runner is None:
            print(f"  Unknown provider: {provider_name} — skipping.")
            continue

        start = time.time()
        try:
            hypothesis = runner(audio_path)
            elapsed = time.time() - start
            error = ""
        except Exception as e:
            hypothesis = ""
            elapsed = time.time() - start
            error = str(e)
            print(f"  [{provider_name}] ERROR: {e}")

        wer = compute_wer(reference_text, hypothesis) if reference_text else None
        cer = compute_cer(reference_text, hypothesis) if reference_text else None
        domain_acc = compute_domain_accuracy(hypothesis)

        row = {
            "video_id": meta.get("video_id", Path(audio_path).stem),
            "title": meta.get("title", ""),
            "speaker": meta.get("speaker", ""),
            "conference": meta.get("conference", ""),
            "duration_min": meta.get("duration_min", ""),
            "provider": provider_name,
            "wer": f"{wer:.4f}" if wer is not None else "N/A",
            "cer": f"{cer:.4f}" if cer is not None else "N/A",
            "domain_accuracy_score": f"{domain_acc:.4f}",
            "elapsed_seconds": f"{elapsed:.1f}",
            "error": error,
            "transcript_length": len(hypothesis),
        }
        rows.append(row)
        print(
            f"  [{provider_name}] WER={row['wer']} CER={row['cer']} "
            f"domain={row['domain_accuracy_score']} ({elapsed:.1f}s)"
        )

    return rows




def write_csv(rows: list[dict], output_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to: {output_path}")


def print_leaderboard(rows: list[dict]):
    """Print a ranked summary table grouped by provider."""
    from collections import defaultdict

    provider_stats: dict[str, dict] = defaultdict(
        lambda: {"wer_sum": 0, "cer_sum": 0, "domain_sum": 0, "count": 0, "errors": 0}
    )

    for row in rows:
        p = row["provider"]
        stats = provider_stats[p]
        stats["count"] += 1
        if row["error"]:
            stats["errors"] += 1
        if row["wer"] != "N/A":
            stats["wer_sum"] += float(row["wer"])
            stats["cer_sum"] += float(row["cer"])
        stats["domain_sum"] += float(row["domain_accuracy_score"])

    print("\n" + "=" * 60)
    print("STT PROVIDER LEADERBOARD")
    print("=" * 60)
    print(f"{'Provider':<15} {'Avg WER':>8} {'Avg CER':>8} {'Domain Acc':>10} {'Errors':>7}")
    print("-" * 60)

    ranked = sorted(
        provider_stats.items(),
        key=lambda x: (
            x[1]["wer_sum"] / max(x[1]["count"] - x[1]["errors"], 1)
            if x[1]["count"] > x[1]["errors"]
            else 9999
        ),
    )

    for provider, s in ranked:
        n = s["count"] - s["errors"]
        avg_wer = s["wer_sum"] / n if n else float("nan")
        avg_cer = s["cer_sum"] / n if n else float("nan")
        avg_dom = s["domain_sum"] / s["count"] if s["count"] else 0
        print(
            f"{provider:<15} {avg_wer:>8.4f} {avg_cer:>8.4f} "
            f"{avg_dom:>10.4f} {s['errors']:>7}"
        )

    print("=" * 60)



def main():
    parser = argparse.ArgumentParser(description="Evaluate STT providers on Bitcoin audio.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="Path to a local audio file.")
    group.add_argument("--youtube", help="YouTube video ID or URL.")
    group.add_argument("--batch", help="Path to a JSON manifest file.")

    parser.add_argument("--reference", help="Path to reference transcript (plain text).")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(PROVIDERS.keys()),
        choices=list(PROVIDERS.keys()),
        help="Which providers to benchmark (default: all).",
    )
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV path.")
    parser.add_argument("--title", default="", help="Talk title (for metadata).")
    parser.add_argument("--speaker", default="", help="Speaker name.")
    parser.add_argument("--conference", default="", help="Conference name.")
    args = parser.parse_args()

    # Ensure app config is importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

    all_rows = []

    if args.audio:
        reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""
        meta = {"title": args.title, "speaker": args.speaker, "conference": args.conference}
        print(f"\nEvaluating: {args.audio}")
        rows = evaluate_audio(args.audio, reference, args.providers, meta)
        all_rows.extend(rows)

    elif args.youtube:
        url = args.youtube if args.youtube.startswith("http") else f"https://www.youtube.com/watch?v={args.youtube}"
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nDownloading: {url}")
            audio_path = download_audio(url, tmpdir)
            reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""
            meta = {
                "video_id": args.youtube,
                "title": args.title,
                "speaker": args.speaker,
                "conference": args.conference,
            }
            print(f"Evaluating: {audio_path}")
            rows = evaluate_audio(audio_path, reference, args.providers, meta)
            all_rows.extend(rows)

    elif args.batch:
        manifest = json.loads(Path(args.batch).read_text(encoding="utf-8"))
        for item in manifest:
            video_id = item.get("video_id", "")
            url = item.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            ref_path = item.get("reference_transcript")
            reference = Path(ref_path).read_text(encoding="utf-8") if ref_path else ""

            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"\nDownloading: {item.get('title', video_id)}")
                try:
                    audio_path = download_audio(url, tmpdir)
                except Exception as e:
                    print(f"  Download failed: {e}")
                    continue

                rows = evaluate_audio(audio_path, reference, args.providers, item)
                all_rows.extend(rows)

            # Be polite between videos
            time.sleep(2)

    write_csv(all_rows, args.output)
    print_leaderboard(all_rows)


if __name__ == "__main__":
    main()
