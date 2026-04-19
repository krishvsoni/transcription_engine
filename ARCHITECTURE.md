# BitScribe — Architecture & Implementation Status

> Compares the original ROADMAP against what was built during the Genesis KB project.  
> Last updated: April 2026

---

## System Architecture
<img width="1531" height="1122" alt="image" src="https://github.com/user-attachments/assets/282b1c3f-8b56-48d1-8ab6-97c8e69d8497" />


---

## ROADMAP vs Implementation Status

### "Next Up — Contributions Welcome" (from ROADMAP.md)

| ROADMAP Item | Status | Implementation |
|---|---|---|
| RSS feed poller | **DONE** | `app/services/rss_poller.py` — 6 default feeds (Stephan Livera, WBD, Bitcoin Audible, Bitcoin Optech, Bitcoin Layer, Tales from the Crypt). GUID-based dedup, relevance filter. Routes: `GET /ingestion/rss/feeds`, `POST /ingestion/rss/poll/{feed_name}`, `POST /ingestion/rss/feeds` |
| Job scheduler | **DONE** | `app/scheduler.py` — APScheduler BackgroundScheduler. 4 jobs: channel_scan (6h), classify_pending (6h), conference_discovery (12h), rss_poll (24h). Starts on server boot via FastAPI lifespan. `GET /ingestion/scheduler/status` |
| LLM-based translation | **DONE** | `app/services/translation_service.py` — Sarvam AI for 10 Indian languages, Gemini/OpenAI for 10 global languages. Chunked at 3000 chars. `GET /translation/languages`, `POST /translation/text`, `POST /translation/transcript` |
| Translation storage | **DONE** | Stored in `transcript.outputs["translation_<lang_code>"]`. Persisted in DB via existing Transcript model |
| Speaker identification | NOT STARTED | Planned — map Speaker 0/1/2 to real names |
| API key authentication | NOT STARTED | Planned |
| Versioned public API | NOT STARTED | Planned |

### New Deliverables (beyond ROADMAP)

| Deliverable | Status | Implementation |
|---|---|---|
| Conference discovery pipeline | **DONE** | `app/services/conference_discovery.py` — scrapes events.coinpedia.org, filters by 30 Bitcoin keywords vs reject list, finds YouTube channels, auto-registers in DB. `POST /conference/discover` |
| Gemini STT provider | **DONE** | `app/services/gemini_stt.py` — uploads to Gemini Files API, polls ACTIVE state, verbatim prompt with diarization, parses `[HH:MM:SS] Speaker N: text` segments. `gemini_stt=true` param in add_to_queue |
| Structured summarizer | **DONE** | `app/services/summarizer.py` — JSON output: `abstract`, `key_topics`, `speakers`, `key_moments`, `bitcoin_terms`. Stored in `transcript.outputs["structured_summary"]` |
| STT benchmark script | **DONE** | `scripts/evaluate_stt.py` — WER, CER, domain accuracy across providers. CLI: `--audio`, `--youtube`, `--batch`, `--providers` |
| Competency test | **DONE** | `scripts/competency_test.py` — 10 Bitcoin conference videos, Deepgram vs Whisper, translations (hi/fr/ta), full metrics CSV |

---

## File Map

```
transcription_engine/
│
├── server.py                          # FastAPI app + APScheduler lifespan
├── app/
│   ├── config.py                      # Settings (GOOGLE_API_KEY / GEMINI_API_KEY)
│   ├── database.py                    # SQLAlchemy engine (postgres:// fix)
│   ├── transcription.py               # Orchestrator — all STT providers
│   ├── scheduler.py            NEW    # APScheduler 4-job setup
│   └── services/
│       ├── __init__.py
│       ├── summarizer.py       UPD    # Structured JSON summary
│       ├── gemini_stt.py       NEW    # Gemini Files API transcription
│       ├── translation_service.py NEW # Sarvam + Gemini/OpenAI translation
│       ├── rss_poller.py       NEW    # RSS feed polling + dedup
│       └── conference_discovery.py NEW # events.coinpedia.org scraper
│
├── routes/
│   ├── transcription.py        UPD    # Added gemini_stt param
│   ├── ingestion.py            UPD    # Added RSS + scheduler endpoints
│   ├── translation.py          NEW    # /translation/*
│   └── conference.py           NEW    # /conference/*
│
├── scripts/
│   ├── evaluate_stt.py         NEW    # WER/CER benchmark CLI
│   ├── competency_test.py      NEW    # 10-video competency CSV generator
│   └── run_api_tests.py        NEW    # Full API test suite → CSV + MD
│
├── audio/
│   └── competency/             NEW    # Downloaded audio for benchmarking
│       ├── 33i1PdSJgwA.mp3            # How Bitcoin Mining Works
│       ├── ceWhyoIDO_M.mp3            # Adam Back — Bitcoin 2025
│       ├── Giuzcd4oxIk.mp3            # Lyn Alden — Bitcoin 2025
│       ├── hMK2ULrVq6A.mp3            # JD Vance — Bitcoin 2025
│       ├── 6fgFyQEWiK4.mp3            # Saifedean Ammous — Amsterdam 2025
│       ├── xm50QauXqJo.mp3            # CZ — Bitcoin MENA 2025
│       ├── 48xggm4IEIM.mp3            # Leon Wankum — Bitcoin 2025
│       ├── S9JGmA5_unY.mp3            # 256-bit security
│       ├── Gs9lJTRZCDc.mp3            # Greg Maxwell — Schnorr/MuSig
│       └── UlKZ83REIkA.mp3            # Bitcoin for Beginners
│
├── competency_test_results.csv NEW    # 20 rows, 23 cols — proposal CSV
├── api_test_results.csv        NEW    # 21/21 API tests pass
├── api_test_results.md         NEW    # Human-readable test report
│
├── requirements.txt            UPD    # Added: google-genai, beautifulsoup4,
│                                      #   APScheduler, feedparser, openai
└── .env                        UPD    # Added: GEMINI_API_KEY, SARVAM_API_KEY
```

---

## Competency Test Results Summary

> Full data: `competency_test_results.csv` — 10 videos × 2 providers = 20 rows

| Provider | Model | Avg WER | Avg Domain (raw) | Avg Domain (corrected) | Avg Speed |
|---|---|---|---|---|---|
| Deepgram | nova-2 | 0.124 | 7.50% | 11.67% | 18.3s |
| Whisper | whisper-1 | 0.040 | 7.50% | 12.50% | 13.1s |

**Key findings:**
- **Whisper has lower average WER** (0.040 vs 0.124) — more accurate raw transcription
- **Correction step significantly boosts domain accuracy** — Lyn Alden video: 2.78% → 41–50% (Bitcoin terms correctly captured after correction)
- **Deepgram is faster to integrate** (cloud API, no local GPU needed) and handles noisy audio better
- **Sarvam AI translation** works reliably for Hindi and Tamil (verified in competency CSV)
- **OpenAI fallback** for European languages (fr/de/es) when Gemini quota is unavailable

---

## API Test Results

All 21 endpoints pass. Run: `python scripts/run_api_tests.py --skip-slow`

| Group | Endpoints | Status |
|---|---|---|
| Health | `/health` | PASS |
| Scheduler | `/ingestion/scheduler/status` | PASS |
| Translation | `/translation/languages`, `/translation/text` (6 langs), `/translation/transcript` | PASS |
| RSS | `/ingestion/rss/feeds`, `/ingestion/rss/poll/*`, `/ingestion/rss/feeds` (POST) | PASS |
| Conference | `/conference/supported-languages`, `/conference/discover` | PASS |
| Ingestion | `/ingestion/channels`, `/ingestion/videos`, `/ingestion/runs` | PASS |
| Transcription | `/transcription/queue/`, `/transcription/corrected/`, `/transcription/summaries/` | PASS |

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| Sarvam AI for Indian languages | Purpose-built for Indian languages; outperforms generic LLMs on Hindi/Tamil/Telugu |
| OpenAI fallback for Gemini translation | Gemini free-tier quota restricted on some Google Workspace accounts; OpenAI gpt-4o-mini is reliable fallback |
| `gemini-2.0-flash` as default Gemini model | `gemini-1.5-flash` not available in v1beta API for all accounts |
| `postgresql://` prefix fix in database.py | SQLAlchemy 2.x rejects legacy `postgres://` prefix |
| Audio stored in `audio/competency/` | Persistent cache — re-running competency test skips downloads |
| RSS state in `metadata/rss_state.json` | Avoids DB migration for a new table; GUID-based dedup prevents re-queuing |
| APScheduler BackgroundScheduler | Non-blocking; runs alongside FastAPI in same process |
