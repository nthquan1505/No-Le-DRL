### No-Le-DRL

## System Architecture Description for No-Le-DRL

# Database Schema (`drl_database.db`)
> Hosted as a single central SQLite database on the Server (`S`); client/local modules (`U`) interact with it via Server APIs while reading user-local data (e.g., external Google Calendar / `.ics` files) on the client side.
> All dates and times follow ISO-8601 format in UTC+7 (`Asia/Ho_Chi_Minh`). Lists of values are stored as JSON-encoded `TEXT` arrays in SQLite. Map coordinates use `"lat,lng"` `TEXT` format.

1. `events` table: contains list of events collected by the server.
   - `event_id` (INT, primary key): id for event record.
   - `title` (TEXT): the extracted title.
   - `description` (TEXT): the summarized description of the event's content and benefits for participants.
   - `keywords` (TEXT, JSON list of TEXT): list of keywords extracted directly from the raw text that describe the event instead of a whole sentence (used for the classification module).
   - `is_online` (BOOL): mark whether the event is held online.
   - `is_cancelled` (BOOL): soft-delete / cancellation flag when an event is cancelled or removed (keeps `event_id` valid for notifications and subscriptions).
   - `reg_start_datetime` (TEXT): the date and time that the event opens for registrations.
   - `reg_end_datetime` (TEXT): the date and time that the event stops allowing registrations.
   - `event_start_datetime` (TEXT): the date and time that the event occurs.
   - `event_end_datetime` (TEXT): the date and time that the event ends.
   - `location` (TEXT): the location that the event takes place.
   - `coordinate` (TEXT): the map coordinate of the simplified location in `"lat,lng"` format.
   - `fee` (TEXT): the fee of the event.
   - `target_audience` (TEXT, JSON list of TEXT): the target audiences for the event.
   - `topic` (TEXT, JSON list of TEXT): the topics to decide which category the event will be classified as (academic, sports, community, historical, coding, geography,...).
   - `format` (TEXT, JSON list of TEXT): the formats of how the event is held and how participants take part in (workshop, seminar, volunteering, contest, field trip, survey,...).
   - `award` (TEXT): how participants would be awarded if they attend or complete a specific task (bonus point, certificates,...).
   - `registration_url` (TEXT): registration form/page link if required to register beforehand (`NULL` if no registration link).
   - `created_at` (TEXT): the date and time when the record is added.
   - `updated_at` (TEXT): the date and time when the record is last updated.
   - `sources` (TEXT): facebook fanpage name or email from organizer or both.
   - `post_url` (TEXT): the facebook post link if it was from a facebook post.
   - `images_url` (TEXT, JSON list of TEXT): media image urls if it was from a facebook post.
   - `category` (INT, `1..5`, nullable): the 1-based category in the school's rubric that the event fits most (`NULL` until classified; maps to index `category - 1` in 5-element credit arrays).
   - `min_credit` (INT, nullable): the minimum estimated bonus credit granted to participants (`NULL` until calculated).

2. `users` table: contains list of users.
   - `user_id` (TEXT, primary key): id for user account.
   - `username` (TEXT, unique): account username.
   - `password` (TEXT): hashed password.
   - `email` (TEXT): user's email.
   - `school_id` (INT, foreign key -> `rubric.school_id`, nullable): id of the user's school rubric.
   - `default_coordinate` (TEXT): the map coordinate of the user's default location in `"lat,lng"` format.
   - `start_year` (INT): year that the user started as a freshman.
   - `weights` (TEXT, JSON list of 4 FLOAT): weights `[w1, w2, w3, w4]` for the 4 parameters in the personal recommendation function, total value of 1.
   - `credits` (TEXT, JSON list of 5 INT): current total credit for each of the 5 categories in the school's rubric (supports manual baseline credits + finalized `done` event credits).
   - `preferred_topics` (TEXT, JSON list of TEXT): list of ideal topics to prioritize (used in calculation for `interested` parameter in the personal recommendation function).
   - `preferred_formats` (TEXT, JSON list of TEXT): list of ideal formats to prioritize (used in calculation for `interested` parameter in the personal recommendation function).

3. `event_subscription` table: contains records of each event subscription from every user.
   - `user_id` (TEXT, primary key, foreign key -> `users.user_id`): id for user account.
   - `event_id` (INT, primary key, foreign key -> `events.event_id`): id for event record.
   - `gain` (FLOAT, `[0..1]`): normalized estimated credit value produced by `gain_calc` in `CRD`.
   - `fit` (FLOAT, `[0..1]`): schedule fitting and travel distance value.
   - `interested` (FLOAT, `[0..1]`): topics and formats matching value.
   - `urgency` (FLOAT, `[0..1]`): deadline to registration or start time value.
   - `role` (TEXT): the role that the user participates as (`participant`, `host`, `collaborator`).
   - `status` (TEXT): subscription lifecycle state (`interested`, `registered`, `in_progress`, `done`).
   - `credit` (INT, nullable): finalized credit calculated from event's title, description, category, and user's role (`NULL` until `status = 'done'` and `CRD(U)` calculates it once, preventing double-counting).
   - `other_awards` (TEXT): other awards (certificates, titles,...) or none.

4. `fanpages` table: contains list of facebook fanpages.
   - `fanpage_id` (INT, primary key): id for fanpage.
   - `fanpage_name` (TEXT): fanpage name.
   - `facebook_url` (TEXT, unique): URL of the facebook fanpage crawled by `ING(S)`.
   - `number_of_events` (INT): number of events in `events` table.

5. `notifications` table: contains list of notifications.
   - `notification_id` (INT, primary key): id for notification.
   - `user_id` (TEXT, foreign key -> `users.user_id`): id for receiver.
   - `type` (TEXT): notification type (update for events, cancellation, reminder,...).
   - `content` (TEXT): notification content.
   - `event_id` (INT, foreign key -> `events.event_id`, nullable): event id if the notification relates to an event (e.g., delayed, cancelled/soft-deleted via `events.is_cancelled`, or reminder).
   - `is_read` (BOOL): whether the user has read the notification (`0` unread, `1` read).
   - `created_at` (TEXT): ISO-8601 UTC+7 timestamp when the notification was created.

6. `rubric` table: contains each school's max credit for each category.
   - `school_id` (INT, primary key): id for school.
   - `school_name` (TEXT): name of the school/university.
   - `max_credit` (TEXT, JSON list of 5 INT): max credit for each of the 5 categories, total of 100.

---

# System Modules (9 Modules)
1. **ING - Ingestion**: crawls facebook posts through the list of fanpages (`facebook_url`) in `fanpages` table (3 times a day at 08:00, 14:00, 20:00 UTC+7) and receives inbox submissions from clients as email posts.
2. **EXT - Extraction**: calls LLM models to extract structured fields from raw post text and media (including a heuristic keyword extractor fallback), checking whether the outcome is a new event, an update/cancellation for an existing event, or a duplicate. For a new event, a record is inserted into `events` with `category` and `min_credit` left as `NULL`.
3. **CLF - Classification**: utilizes an event record's `title`, `keywords`, `topic`, and `format` to decide which rubric category (`1..5`) it fits best, updating `events.category`.
4. **SCH - Schedule detection**: compares `event_start_datetime` and `event_end_datetime` against the user's external local calendar (Google Calendar / `.ics`) and uses `coordinate` vs `users.default_coordinate` (plus `location`) to calculate travel distance and schedule compatibility into `fit` (`[0..1]`).
5. **CRD - Credit calculation**: estimates and finalizes event credits (`credit_calc` for integer credits and `gain_calc` in `CRD` to produce a normalized `gain` value in `[0..1]`).
6. **RCM - Recommendation**: uses `reg_start_datetime`, `reg_end_datetime`, and `event_start_datetime` to calculate `urgency` (`[0..1]`), compares `events.topic` & `events.format` against `users.preferred_topics` & `users.preferred_formats` to calculate `interested` (`[0..1]`), combines `[gain, fit, interested, urgency]` with `users.weights` (`[w1, w2, w3, w4]`), and applies MMR re-ranking for feed diversity.
7. **PRG - Progress tracking**: prompts the user after registration deadlines or event end times to verify and update subscription `status` and `role` in `event_subscription`.
8. **NTF - Notification**: creates and delivers reminders and event-update notifications to users in `notifications`.
9. **DSH - Dashboard**: interacts with users, displays personalized feeds, notifications, and summarized credit progress charts.

---

# Overall Workflow (Server `S` & Client/Local `U`)

### On Server (3 times a day)
1. **ING(S)** crawls fanpage posts and receives client inbox emails, forwarding raw text/media to **EXT(S)**, which inserts/updates records in `events` (or sets `is_cancelled = 1` if cancelled).
2. **CLF(S)** queries active records in `events` where `category IS NULL` and runs the classification function to set `category` (`1..5`).
3. **CRD(S)** queries active records in `events` where `min_credit IS NULL` and runs `credit_calc(event_id, role='participant')` to populate `min_credit`.

### On Local / Client (when user opens the dashboard)
1. **SCH(U)** matches active candidate events against the user's external calendar and coordinates, producing `(schedule_free_type, distance, fit)` in-memory.
2. **CRD(U)** estimates raw Remaining-Rubric-Aware credit `min(event.min_credit, rubric.max_credit[category - 1] - user.credits[category - 1])` and runs `gain_calc(...)` to bind `gain` to `[0..1]`.
3. **RCM(U)** computes `interested` and `urgency` in `[0..1]`, computes priority score `w = w1 * gain + w2 * fit + w3 * interested + w4 * urgency` using `users.weights`, and runs MMR to diversify the feed. (Candidate scores live in-memory until the user subscribes.)
4. User browses the feed and can adjust `weights`. When the user clicks **Interested**, a row is inserted into `event_subscription` (saving `gain`, `fit`, `interested`, `urgency`, and `credit = NULL`) with `status = 'registered'` if `events.registration_url IS NULL`, otherwise `status = 'interested'`.
5. At `reg_start_datetime`, **NTF(U)** reminds the user to register. After `reg_end_datetime`, **PRG(U)** asks whether the user registered and updates `status` to `'registered'` or removes the subscription row.
6. At `event_start_datetime`, **NTF(U)** reminds the user to attend; once past `event_start_datetime`, `status` becomes `'in_progress'`. After `event_end_datetime`, **PRG(U)** asks whether the user completed the event and confirms their `role`, updating `status` to `'done'` or removing the subscription row.
7. **CRD(U)** queries rows in `event_subscription` where `status = 'done'` **and `credit IS NULL`**, calls `credit_calc(event_id, role)`, writes the finalized integer into `event_subscription.credit`, and updates `users.credits[category - 1] = min(rubric.max_credit[category - 1], users.credits[category - 1] + credit)` so each completed event is counted exactly once while preserving manual baseline credits.
8. **DSH(U)** renders the updated feed, notifications, and rubric progress charts.

---

# Project Directory Structure

```text
No-Le-DRL/
├── app.py                                   # [DSH/CLF/SCH/CRD/RCM] Interactive 4-tab DRL Recommendation & Classification web app (:8765)
├── main.py                                  # Unified CLI entrypoint (`serve`, `fetch`, `extract`, `extract-post`, `classify`, ...)
├── server.py                                # FastAPI + WebSocket MVC server setup & lifespan (:8000)
├── scheduler.py                             # Cron/interval background scheduler (08:00, 14:00, 20:00 UTC+7) for [ING]
├── config.py                                # Environment loader, DB paths (`DRL_DB_PATH`), UTC+7 timezone & constants
├── drl_database.db                          # Central SQLite database (6 tables: events, users, event_subscription, fanpages, notifications, rubric)
├── pyproject.toml                           # Python package metadata & optional extras (`[ml]`, `[opt]`, `[api]`, `[dev]`)
├── requirements.txt                         # Core backend, DB, extraction, scraper & FastAPI dependencies
├── requirements-calendar.txt                # Optional Google Calendar OAuth dependencies for `src/schedule`
├── .env.example                             # Template for `.env` secrets & runtime configuration
│
├── configs/                                 # YAML & model configurations
│   ├── model_config.py                      # E5 embedding constants (`intfloat/multilingual-e5-small`, 384d) & repo paths
│   ├── classification_config.yaml           # [CLF] Dual-branch 2-stage kNN parameters, fusion weights & decision guards
│   ├── recommendation_config.yaml           # [RCM] Interest chips, weight presets, P5 filter, P6 scoring & P7 MMR settings
│   └── schedule.yaml                        # [SCH] Travel buffers, campus travel matrix & 5-step `compute_fit` weights
│
├── database/                                # Database persistence & migration layer
│   ├── __init__.py                          # Exports `DRLDatabaseManager`, `get_drl_db` & session helpers
│   ├── drl_db_manager.py                    # Thread-safe Singleton SQLite / PostgreSQL manager (WAL, FKs, retries, CRUD)
│   ├── migrate_sqlite_to_postgres.py        # Idempotent 6-table SQLite -> PostgreSQL migration utility
│   ├── db.py                                # SQLAlchemy engine/session helpers for scraper & portal tables
│   └── models.py                            # ORM models for scraped posts, fanpages & inbox records
│
├── scraper/                                 # [ING] Facebook fanpage crawler
│   └── fb_scraper.py                        # Playwright headless Chromium fanpage post & media scraper (with proxy support)
├── events/                                  # [ING/EXT] Scraper-to-extractor & email inbox orchestration bridges
│   ├── extractor.py                         # Fanpage post batch analyzer, single-URL extractor & email inbox handler
│   └── classifier.py                        # Lightweight tag classifier for portal feed folders
├── rss/                                     # [ING] RSS XML feed generator (`feeds/`)
│   └── generator.py
│
├── prototype/                               # [DSH] Frontend UI served by `app.py` (:8765)
│   ├── index.html                           # 4-tab workspace (Live Classifier, DB Browser, P5-P6 Scoring, P7 MMR Selector)
│   ├── app.js                               # Frontend state, API client, reasoning breakdown & schedule preset controls
│   └── styles.css                           # Responsive UI styling
│
├── src/                                     # Core modular domain & web packages
│   ├── schemas.py                           # Canonical data contracts (`EventNormalized`, `CreditEstimate`, `ScoredEvent`, ...)
│   ├── contract_validator.py                # Schema, rubric taxonomy (`<major>.<minor>`) & ISO-8601 validators
│   ├── db_event_adapter.py                  # Adapter bridging `drl_database.db` rows <-> `EventNormalized` & classification inputs
│   ├── embedder.py                          # Unified `intfloat/multilingual-e5-small` (`query:` / `passage:`, 384d L2) embedder
│   ├── event_cache.py                       # `.npy` + `.json` event vector & metadata cache
│   ├── signals.py                           # Thread-safe pub/sub signal bus & webhook notifier (`[EXT]` -> `[CLF]`)
│   ├── vector_ops.py                        # L2 normalization & cosine similarity operations
│   │
│   ├── extraction/                          # [EXT] 4-stage event extraction & deduplication pipeline (`drl_database.db`)
│   │   ├── contracts.py                     # `RawIngestionItem`, `ExtractedEventDraft` & controlled vocabularies
│   │   ├── prefilter.py                     # Stage 1: Deterministic non-event / recap / noise pre-filter
│   │   ├── prompts.py                       # Stage 2: System prompt & strict JSON schema for LLM extraction
│   │   ├── llm_client.py                    # Stage 2: Direct REST Gemini key-pool + Groq / custom LLM failover client
│   │   ├── heuristics.py                    # Stage 2: Deterministic field sanitizer & offline heuristic fallback
│   │   ├── media_qr.py                      # Stage 3: Local OpenCV QR-code poster registration link decoder
│   │   ├── geocoder.py                      # Stage 3: Campus-first & HCMC/VNU-HCM `"lat,lng"` coordinate resolver
│   │   ├── deduplicator.py                  # Stage 4: Anchor deduplication, updates, soft-cancel (`is_cancelled=1`) & notifications
│   │   └── pipeline.py                      # End-to-end `ExtractionPipeline` orchestrator
│   │
│   ├── classification/                      # [CLF] Dual-Branch Hierarchical 2-Stage kNN classification system
│   │   ├── data/                            # Asset loaders & integrity validators (`loaders.py`, `validators.py`)
│   │   ├── embeddings/                      # Multi-profile query text builder & cached embedder
│   │   ├── retrieval/                       # Sample branch (`sample_retriever.py`) & Rubric branch (`rubric_retriever.py`)
│   │   ├── fusion/                          # Stage 1 (group) & Stage 2 (criterion) score normalization & fusion
│   │   ├── algorithms/                      # `HierarchicalTwoStageKNN` & extensible `AlgorithmRegistry`
│   │   ├── decision/                        # Confidence thresholds & rubric boundary guards (`PREDICTED` / `NEEDS_REVIEW`)
│   │   ├── output/                          # JSON & Markdown run report writers
│   │   └── pipeline/                        # `ClassificationPipeline` (DB & JSON modes, writes `events.category`)
│   │
│   ├── schedule/                            # [SCH] Personal schedule conflict detection, routing & `Fit` calculation
│   │   ├── engine.py                        # `ScheduleEngine`: UTC+7 normalization, buffer conflict check & 5-step `compute_fit`
│   │   ├── evaluation.py                    # `FitEvaluator`: batch candidate evaluator with per-run route deduplication
│   │   ├── fit_cache.py                     # `FitCache`: privacy-preserving SQLite cache with daily TTL cleanup
│   │   ├── calendar.py                      # Google Calendar `Events.list` reader & busy/flexible interval merger
│   │   ├── oauth.py                         # Desktop OAuth2 setup & token refresh CLI (`python -m src.schedule.oauth`)
│   │   └── routing.py                       # `CarRouting`: campus catalog -> `"lat,lng"` / OSRM -> Google Routes (quota-capped)
│   │
│   ├── credit_calculation/                  # [CRD] DRL credit estimation, finalization & `Gain` calculation
│   │   ├── credit_calculator.py             # `credit_calc(event, role)`, `gain_calc(...)` & DB batch/subscription finalizers
│   │   ├── credit_engine.py                 # `RubricCreditEngine`: deterministic rubric evaluator (`data/raw/rubric.yaml`)
│   │   ├── attribute_extractor.py           # Level, role, duration bucket & explicit-point text extractors
│   │   └── interface.py                     # `CreditCalculatorProtocol`, `UserLedger` & `LedgerEvaluation` contracts
│   │
│   ├── recommendation/                      # [RCM] Multi-criteria personalized recommendation & MMR pipeline
│   │   ├── config.py                        # Merged loader for `recommendation_config.yaml` + `schedule.yaml`
│   │   ├── user_profile.py                  # P4 cold-start & behavioral user profile vector/weight adaptation
│   │   ├── rec_embedder.py                  # 4-field (`title + description + topic + format`) `passage:` event embedder
│   │   ├── schedule_adapter.py              # `ScheduleProvider` protocol & RFC 5545 `.ics` timetable parser
│   │   ├── filter.py                        # P5 `HardConstraintFilter` (`PREDICTED` gate + 8 non-short-circuiting rules)
│   │   ├── scoring.py                       # P6 `MultiCriteriaScorer` (`w1*Gain + w2*Interest + w3*Fit + w4*Urgency + Penalty`) & P8 explanations
│   │   ├── selector.py                      # P7 `MMRSelector` (hybrid semantic+rubric diversity + exploration slot)
│   │   ├── demo_profile.py                  # Demo student personas & mandatory `6.4` SHCD fixture
│   │   ├── demo_module4.py                  # Timetable preset templates (`light_schedule`, `busy_weekend`, `exam_week_heavy`)
│   │   ├── demo_module6.py                  # `DemoCreditProvider` adapter wrapping `RubricCreditEngine`
│   │   └── pipeline.py                      # End-to-end DB & demo recommendation orchestrator
│   │
│   ├── backend/                             # FastAPI MVC controllers & WebSocket runtime (`server.py` :8000)
│   │   ├── auth_controller.py               # User/admin login, registration & session endpoints
│   │   ├── admin_controller.py              # Fanpage management, crawl import & admin operations
│   │   ├── event_controller.py              # Email inbox ingestion, post extraction, feeds & WebSocket channels
│   │   ├── page_controller.py               # HTML route handlers (`/`, `/login`, `/client`, `/admin`)
│   │   └── runtime.py                       # Connection manager, background scrape lock & scheduler sync
│   ├── frontend/                            # HTML views for the FastAPI portal (`login.html`, `client.html`, `admin.html`)
│   └── model/                               # Portal repositories, Pydantic request/response schemas & session tokens
│
├── data/                                    # Rubric, reference indices & cached embeddings
│   ├── raw/rubric.yaml                      # Machine-readable HCMUS DRL rubric (51 criteria, QĐ 1028/QĐ-KHTN-CTSV)
│   ├── processed/cls_index/                 # Precomputed Sample & Rubric reference embeddings (`.npy` + `.parquet`)
│   └── generated/                           # Cached DB multivector classification (`.npz`) & 4-field rec embeddings (`.npy`)
├── res/                                     # Official HCMUS DRL regulations (`drl2025.pdf`, `drl2025.md`) & keyword rules
├── scripts/                                 # Offline dataset builders & reference embedding generators
├── docs/                                    # Architecture & module specifications (`SYSTEM_ARCHITECTURE.md`, ...)
├── integration/                             # Cross-module smoke test (`smoke_module3.py`) & historical handoff patch
└── tests/                                   # Unit & integration test suites (`pytest` / `unittest`)
```

---

# Localhost Deployment Guide

## 1. Prerequisites
- **Python**: `3.10+` (`3.11`–`3.13` recommended on Windows, macOS, or Linux).
- **Database**: `drl_database.db` (SQLite with WAL mode) is bundled in the repository root and works out of the box with zero external database setup. PostgreSQL is optionally supported via `DRL_DATABASE_URL`.

## 2. Environment & Dependency Setup

From the repository root (`No-Le-DRL/`):

```powershell
# 1. Create and activate a virtual environment (Windows PowerShell; Python 3.11-3.13 recommended)
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# (On macOS / Linux: python3 -m venv .venv && source .venv/bin/activate)

# 2. Install core backend, database, extraction, and FastAPI dependencies
pip install -r requirements.txt

# 3. Install the E5 embedding ML stack (required by app.py for live query/profile encoding)
pip install -e ".[ml,dev]"
```

**Optional integrations** (only install if needed):
- **Facebook Scraper Browser (`ING`)**: `playwright install chromium`
- **Google Calendar Live Sync (`SCH` / `src.schedule`)**:
  ```powershell
  pip install -r requirements-calendar.txt
  python -m src.schedule.oauth --client-secrets "C:/path/to/client_secret.json"
  ```

## 3. Environment Variables (`.env`)

Copy `.env.example` to `.env` in the project root:

```powershell
Copy-Item .env.example .env
```

Key settings (all external API keys are optional for local demo/offline operation):
- `DRL_DATABASE_URL`: Leave empty to use local SQLite (`drl_database.db`), or set `postgresql://user:pass@127.0.0.1:5432/nole_drl`.
- `ADMIN_TOKEN`: Token/password for signing into the `/admin` console.
- `GEMINI_API_KEY` / `CUSTOM_LLM_URL`: Used by `EXT` when extracting new raw posts/emails (falls back to deterministic heuristics offline).
- `FB_PROXY_URL`: Optional proxy URL for Playwright fanpage scraping.
- **Schedule (`SCH` / `src/schedule`) optional routing & calendar variables** (runs offline by default using campus travel matrix + coordinate distance + `configs/schedule.yaml`):
  - `MODULE4_OSRM_URL` / `MODULE4_NOMINATIM_URL`: Private OSRM / Nominatim endpoints for road routing.
  - `MODULE4_GOOGLE_ROUTES_KEY`: Google Compute Routes API key (capped at 250 calls/UTC day via `MODULE4_QUOTA_DB`).
  - `MODULE4_CALENDAR_TOKEN`: Custom path to the OAuth token file (default: `~/.no-le-drl/calendar-token.json`).

## 4. Running the Web Applications on Localhost

The repository provides two localhost web entrypoints depending on which workflow you want to run:

### A. Classification, Schedule (`src.schedule`), Credit (`src.credit_calculation`), Scoring & MMR Web App (`app.py`)
Runs the interactive 4-tab DRL Recommendation & Classification UI (`prototype/`) directly on `drl_database.db` using `intfloat/multilingual-e5-small` (`384d` L2), `src/classification/`, `src/schedule/` (`ScheduleEngine`), `src/credit_calculation/` (`RubricCreditEngine`), and `src/recommendation/` (`P5` Filter → `P6` Scoring → `P7` MMR):

```powershell
python app.py --host 127.0.0.1 --port 8765
```

- **Web UI**: `http://127.0.0.1:8765`
  - **Tab 1**: Live Dual-Branch 2-Stage kNN Classifier & `[CRD]` credit estimator
  - **Tab 2**: `drl_database.db` batch event browser & re-classification trigger
  - **Tab 3**: Personalized Recommendation (`P5` hard filter diagnostics + `P6` multi-criteria scoring with `[SCH]` `Fit` breakdown)
  - **Tab 4**: `P7` MMR diversity & exploration selector

### B. Ingestion, Extraction, RSS & Admin/Client MVC Server (`main.py serve` / `server.py`)
Runs the FastAPI + WebSocket server (`src/backend/`, `src/frontend/`) for user/admin authentication, email inbox ingestion, single-post URL extraction, fanpage management, and scheduled scraping:

```powershell
# Recommended for local testing (disables automatic background Facebook scraping):
python main.py serve --host 127.0.0.1 --port 8000 --no-scrape

# Or with scheduled scraping enabled (08:00, 14:00, 20:00 UTC+7):
python main.py serve --host 127.0.0.1 --port 8000
```

*(You can also launch FastAPI directly with `uvicorn server:app --host 127.0.0.1 --port 8000 --reload`.)*

- **Login Page**: `http://127.0.0.1:8000/` (`/login`)
- **Client Workspace**: `http://127.0.0.1:8000/client`
- **Admin Console**: `http://127.0.0.1:8000/admin`
- **Interactive API Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`

## 5. Local Verification & Health Checks

```powershell
# Verify Schedule (src.schedule) <-> Recommendation (P5/P6 Fit & routing) end-to-end smoke test
python integration/smoke_module3.py

# Run the unit & integration test suite
pytest -q
```

