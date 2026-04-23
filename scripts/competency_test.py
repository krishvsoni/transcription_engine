#!/usr/bin/env python3
"""Competency Test — Bitcoin Conference Transcription Benchmark.

Transcribes 10 Bitcoin conference videos (2024/2025) using Deepgram and
OpenAI Whisper, applies correction, computes WER/CER/domain accuracy,
and saves results to competency_test_results.csv.

Usage:
    python scripts/competency_test.py
    python scripts/competency_test.py --max-duration 180   # first 3 min only
    python scripts/competency_test.py --providers deepgram whisper
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv()


VIDEOS = [
    {"id": "33i1PdSJgwA",  "title": "How Bitcoin Mining Actually Works",         "speaker": "Unknown",            "conference": "Bitcoin Education", "year": 2024},
    {"id": "ceWhyoIDO_M",  "title": "The Future of Finance Runs on Bitcoin",     "speaker": "Adam Back",           "conference": "Bitcoin 2025",      "year": 2025},
    {"id": "Giuzcd4oxIk",  "title": "Nothing Stops This Train",                  "speaker": "Lyn Alden",           "conference": "Bitcoin 2025",      "year": 2025},
    {"id": "hMK2ULrVq6A",  "title": "JD Vance Bitcoin 2025 Keynote",             "speaker": "JD Vance",            "conference": "Bitcoin 2025",      "year": 2025},
    {"id": "6fgFyQEWiK4",  "title": "Saifedean Ammous Bitcoin Amsterdam 2025",   "speaker": "Saifedean Ammous",    "conference": "Bitcoin Amsterdam 2025", "year": 2025},
    {"id": "xm50QauXqJo",  "title": "CZ Fireside Chat",                          "speaker": "Changpeng Zhao",      "conference": "Bitcoin MENA 2025", "year": 2025},
    {"id": "48xggm4IEIM",  "title": "Bitcoin as Collateral in Real Estate",      "speaker": "Leon Wankum",         "conference": "Bitcoin 2025",      "year": 2025},
    {"id": "S9JGmA5_unY",  "title": "How Secure is 256-bit Security?",           "speaker": "3Blue1Brown",         "conference": "Education",         "year": 2017},
    {"id": "Gs9lJTRZCDc",  "title": "Schnorr Signatures & MuSig",               "speaker": "Greg Maxwell",        "conference": "SF Bitcoin Devs",   "year": 2019},
    {"id": "UlKZ83REIkA",  "title": "Bitcoin Explained for Beginners",           "speaker": "Various",             "conference": "Bitcoin Education", "year": 2023},
]

# Bitcoin domain terms for accuracy scoring
BITCOIN_TERMS = [
    "bitcoin", "blockchain", "utxo", "taproot", "schnorr", "lightning",
    "segwit", "mempool", "hashrate", "proof of work", "mining", "wallet",
    "private key", "public key", "signature", "transaction", "block",
    "consensus", "node", "peer-to-peer", "satoshi", "bip", "script",
    "multisig", "timelock", "channel", "routing", "htlc", "lnp", "dlc",
    "musig", "mast", "tapscript", "sighash", "descriptor", "psbt",
]


def download_audio(video_id: str, max_duration: int, output_dir: str) -> str:
    """Download audio from YouTube, optionally trimmed to max_duration seconds."""
    import yt_dlp

    output_path = os.path.join(output_dir, f"{video_id}.mp3")
    if os.path.exists(output_path):
        return output_path

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, f"{video_id}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "quiet": True,
        "no_warnings": True,
    }

    if max_duration:
        ydl_opts["download_ranges"] = lambda info, ydl: [{"start_time": 0, "end_time": max_duration}]
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    return output_path


def transcribe_deepgram(audio_path: str, diarize: bool = True) -> dict:
    """Transcribe with Deepgram Nova-2."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY not set")

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "true" if diarize else "false",
        "language": "en",
    }
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/mpeg"}

    with open(audio_path, "rb") as f:
        resp = requests.post(url, params=params, headers=headers, data=f, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    words = data["results"]["channels"][0]["alternatives"][0].get("words", [])
    confidence = data["results"]["channels"][0]["alternatives"][0].get("confidence", 0)

    return {
        "text": transcript,
        "words": len(transcript.split()),
        "confidence": round(confidence, 3),
        "word_details": words[:5],
    }


def transcribe_whisper(audio_path: str) -> dict:
    """Transcribe with OpenAI Whisper-1 API."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    return {
        "text": resp.text,
        "words": len(resp.text.split()),
        "confidence": None,
        "language": getattr(resp, "language", "en"),
    }


def correct_transcript(raw_text: str) -> str:
    """Apply Bitcoin-domain correction via OpenAI."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return raw_text

    client = openai.OpenAI(api_key=api_key)
    prompt = (
        "You are a Bitcoin conference transcript corrector.\n"
        "Fix ASR errors in the following transcript, paying special attention to:\n"
        "- Bitcoin technical terms: UTXO, Taproot, Schnorr, Lightning, SegWit, mempool, "
        "  hashrate, BIP, HTLC, MuSig, Tapscript, PSBT, descriptor\n"
        "- Speaker names common in Bitcoin: Satoshi, Nakamoto, Szabo, Back, Wuille, Maxwell\n"
        "- Do NOT change meaning, add content, or summarize.\n\n"
        f"Transcript:\n{raw_text[:6000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=120,
    )
    return resp.choices[0].message.content.strip()


_SARVAM_LANGUAGES = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN",
    "pa": "pa-IN", "or": "or-IN",
}
_SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


def translate_sarvam(text: str, target_lang: str = "hi") -> str:
    """Translate text to Indian language via Sarvam AI."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return ""
    sarvam_lang = _SARVAM_LANGUAGES.get(target_lang, "hi-IN")
    payload = {
        "input": text[:2000],
        "source_language_code": "en-IN",
        "target_language_code": sarvam_lang,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }
    headers = {"Content-Type": "application/json", "api-subscription-key": api_key}
    try:
        resp = requests.post(_SARVAM_TRANSLATE_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("translated_text") or data.get("output", "")
    except Exception as e:
        return f"[translation error: {e}]"


def translate_openai(text: str, language: str = "French") -> str:
    """Translate text via OpenAI gpt-4o-mini."""
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    client = openai.OpenAI(api_key=api_key)
    prompt = (
        f"Translate the following English Bitcoin conference text into {language}. "
        "Preserve Bitcoin technical terms (UTXO, Lightning, Taproot, etc.) in English. "
        "Return ONLY the translated text.\n\n"
        f"{text[:2000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


def summarize_transcript(text: str, title: str) -> str:
    """Generate structured summary via OpenAI."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    client = openai.OpenAI(api_key=api_key)
    prompt = (
        f'Summarize this Bitcoin conference talk titled "{title}" in 2-3 sentences. '
        "Focus on main argument, key Bitcoin concepts discussed, and conclusion.\n\n"
        f"{text[:4000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return round(d[len(ref_words)][len(hyp_words)] / len(ref_words), 4)


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate."""
    ref_c = list(ref.lower().replace(" ", ""))
    hyp_c = list(hyp.lower().replace(" ", ""))
    if not ref_c:
        return 0.0

    d = [[0] * (len(hyp_c) + 1) for _ in range(len(ref_c) + 1)]
    for i in range(len(ref_c) + 1):
        d[i][0] = i
    for j in range(len(hyp_c) + 1):
        d[0][j] = j
    for i in range(1, len(ref_c) + 1):
        for j in range(1, len(hyp_c) + 1):
            cost = 0 if ref_c[i-1] == hyp_c[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return round(d[len(ref_c)][len(hyp_c)] / len(ref_c), 4)


def domain_accuracy(text: str) -> float:
    """Fraction of expected Bitcoin terms present in the transcript."""
    text_lower = text.lower()
    found = sum(1 for term in BITCOIN_TERMS if term in text_lower)
    return round(found / len(BITCOIN_TERMS), 4)


def get_video_duration(video_id: str) -> int:
    """Get video duration in seconds via yt-dlp."""
    import yt_dlp
    ydl_opts = {"quiet": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info.get("duration", 0)
    except Exception:
        return 0


def run_competency_test(providers: list, max_duration: int, output_csv: str):
    print(f"\n{'='*65}")
    print(f"  Bitcoin STT Competency Test")
    print(f"  Providers  : {', '.join(providers)}")
    print(f"  Max audio  : {max_duration}s per video")
    print(f"  Videos     : {len(VIDEOS)}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    rows = []
    audio_dir = str(Path(__file__).parent.parent / "audio" / "competency")
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Audio dir  : {audio_dir}")

    for i, video in enumerate(VIDEOS, 1):
        vid_id = video["id"]
        print(f"[{i:02d}/{len(VIDEOS)}] {video['title'][:50]} ({vid_id})")

        # Get duration
        duration_s = get_video_duration(vid_id)
        duration_min = round(duration_s / 60, 1)

        # Download audio
        print(f"         Downloading audio (max {max_duration}s)...")
        try:
            audio_path = download_audio(vid_id, max_duration, audio_dir)
        except Exception as e:
            print(f"         ERROR downloading: {e}")
            for provider in providers:
                rows.append(_error_row(video, provider, duration_min, str(e)))
            continue

        for provider in providers:
            print(f"         Transcribing with {provider}...")
            t_start = time.time()

            try:
                if provider == "deepgram":
                    result = transcribe_deepgram(audio_path)
                elif provider == "whisper":
                    result = transcribe_whisper(audio_path)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                raw_text = result["text"]
                t_transcribe = round(time.time() - t_start, 1)

                print(f"         Correcting ({len(raw_text)} chars)...")
                corrected = correct_transcript(raw_text)

                print(f"         Summarizing...")
                summary = summarize_transcript(corrected, video["title"])

                # Metrics
                wer = compute_wer(raw_text, corrected)          # raw vs corrected
                cer = compute_cer(raw_text, corrected)
                domain_raw = domain_accuracy(raw_text)
                domain_corrected = domain_accuracy(corrected)

                print(f"         domain_raw={domain_raw:.2%}  domain_corrected={domain_corrected:.2%}  WER={wer:.4f}")

                # Translate corrected transcript (use first 500 chars for speed)
                sample = corrected[:500]
                print(f"         Translating to Hindi (Sarvam)...")
                translation_hi = translate_sarvam(sample, "hi")

                print(f"         Translating to French (OpenAI)...")
                translation_fr = translate_openai(sample, "French")

                print(f"         Translating to Tamil (Sarvam)...")
                translation_ta = translate_sarvam(sample, "ta")

                rows.append({
                    "video_id":              vid_id,
                    "title":                 video["title"],
                    "speaker":               video["speaker"],
                    "conference":            video["conference"],
                    "year":                  video["year"],
                    "duration_min":          duration_min,
                    "provider":              provider,
                    "model":                 "nova-2" if provider == "deepgram" else "whisper-1",
                    "confidence":            result.get("confidence", "N/A"),
                    "word_count_raw":        result.get("words", 0),
                    "raw_transcript":        raw_text[:500].replace("\n", " "),
                    "corrected_transcript":  corrected[:500].replace("\n", " "),
                    "summary":               summary.replace("\n", " "),
                    "translation_hi":        translation_hi[:300].replace("\n", " "),
                    "translation_fr":        translation_fr[:300].replace("\n", " "),
                    "translation_ta":        translation_ta[:300].replace("\n", " "),
                    "wer_raw_vs_corrected":  wer,
                    "cer_raw_vs_corrected":  cer,
                    "domain_accuracy_raw":    domain_raw,
                    "domain_accuracy_corrected": domain_corrected,
                    "domain_accuracy_score": domain_corrected,  # proposal-spec alias
                    "transcription_time_s":  t_transcribe,
                    "status":                "success",
                    "error":                 "",
                })

            except Exception as e:
                print(f"         ERROR: {e}")
                rows.append(_error_row(video, provider, duration_min, str(e)))

        print()

    # Write CSV
    _write_csv(rows, output_csv)

    # Print leaderboard
    _print_leaderboard(rows)


def _error_row(video, provider, duration_min, error):
    return {
        "video_id": video["id"], "title": video["title"],
        "speaker": video["speaker"], "conference": video["conference"],
        "year": video["year"], "duration_min": duration_min,
        "provider": provider, "model": "", "confidence": "",
        "word_count_raw": 0, "raw_transcript": "", "corrected_transcript": "",
        "summary": "", "translation_hi": "", "translation_fr": "", "translation_ta": "",
        "wer_raw_vs_corrected": "", "cer_raw_vs_corrected": "",
        "domain_accuracy_raw": "", "domain_accuracy_corrected": "",
        "transcription_time_s": "", "status": "error", "error": error[:200],
    }


def _write_csv(rows, output_csv):
    fieldnames = [
        "video_id", "title", "speaker", "conference", "year", "duration_min",
        "provider", "model", "confidence", "word_count_raw",
        "raw_transcript", "corrected_transcript", "summary",
        "translation_hi", "translation_fr", "translation_ta",
        "wer_raw_vs_corrected", "cer_raw_vs_corrected",
        "domain_accuracy_raw", "domain_accuracy_corrected", "domain_accuracy_score",
        "transcription_time_s", "status", "error",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nCSV saved -> {Path(output_csv).resolve()}")


def _print_leaderboard(rows):
    from collections import defaultdict
    print(f"\n{'='*65}")
    print("  STT Provider Leaderboard")
    print(f"{'='*65}")
    print(f"  {'Provider':<12} {'Avg Domain Raw':>15} {'Avg Domain Corr':>16} {'Avg Time(s)':>12} {'Success':>8}")
    print(f"  {'-'*63}")

    by_provider = defaultdict(list)
    for r in rows:
        if r["status"] == "success":
            by_provider[r["provider"]].append(r)

    for prov, prows in sorted(by_provider.items()):
        avg_dom_raw  = sum(float(r["domain_accuracy_raw"]) for r in prows) / len(prows)
        avg_dom_corr = sum(float(r["domain_accuracy_corrected"]) for r in prows) / len(prows)
        avg_time     = sum(float(r["transcription_time_s"]) for r in prows) / len(prows)
        print(f"  {prov:<12} {avg_dom_raw:>14.2%}  {avg_dom_corr:>14.2%}  {avg_time:>11.1f}  {len(prows):>6}/{len(prows)}")

    total = len(rows)
    success = sum(1 for r in rows if r["status"] == "success")
    print(f"\n  Total: {success}/{total} successful")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Bitcoin STT Competency Test")
    parser.add_argument("--providers", nargs="+", default=["deepgram", "whisper"],
                        choices=["deepgram", "whisper"])
    parser.add_argument("--max-duration", type=int, default=240,
                        help="Max audio seconds to download per video (default: 240 = 4 min)")
    parser.add_argument("--output", default="competency_test_results.csv")
    args = parser.parse_args()

    run_competency_test(
        providers=args.providers,
        max_duration=args.max_duration,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
