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
├── config.py                        # Environment config, paths (DRL_DB_PATH), timezone (UTC+7), constants
├── main.py                          # Application entrypoint (starts FastAPI server & background scheduler)
├── drl_database.db                  # Central SQLite database (6 tables)
├── requirements.txt                 # Python dependencies
├── .env                             # Secrets & API keys (LLM keys, session secret, proxy URL)
├── SYSTEM_ARCHITECTURE.md           # System architecture, DB schema, workflow & directory guide
│
├── frontend/                        # Frontend UI/UX (DSH module & client-side interactions)
│   ├── login.html                   # Login & account registration page
│   ├── dashboard.html               # Main student dashboard (feed, rubric progress, weight sliders, notifications)
│   ├── admin.html                   # Admin panel (manage tracked fanpages, school rubrics, manual inbox)
│   ├── css/
│   │   └── styles.css               # Shared layout, feed cards, charts & modal styling
│   └── js/
│       ├── api_client.js            # HTTP/WebSocket wrapper for communicating with backend APIs
│       ├── dashboard_ui.js          # Feed rendering, weight adjustment, rubric progress charts (DSH)
│       ├── calendar_sync.js         # Reads local .ics / Google Calendar & coordinates on client PC
│       └── progress_prompts.js      # Registration/completion verification popups & notification badges
│
├── backend/                         # Server, session management & client request handlers
│   ├── __init__.py
│   ├── server.py                    # FastAPI app setup, CORS, static file mounting & route registration
│   ├── scheduler.py                 # Cron scheduler (08:00, 14:00, 20:00 UTC+7) running ING -> EXT -> CLF -> CRD
│   ├── session_manager.py           # Password hashing, session tokens/cookies & auth middleware
│   ├── schemas.py                   # Pydantic request/response validation models
│   └── routes/
│       ├── __init__.py
│       ├── auth_routes.py           # Login, logout, register & user profile/preferences/weights endpoints
│       ├── event_routes.py          # Feed retrieval, client-scored ranking & inbox email submission endpoints
│       ├── subscription_routes.py   # Subscribe ("Interested"), status/role updates & credit finalization
│       ├── notification_routes.py   # Fetch user notifications & mark-as-read endpoints
│       └── admin_routes.py          # Fanpage CRUD & school rubric management endpoints
│
├── database/                        # Singleton DB connection, query execution & error fallbacks
│   ├── __init__.py                  # Exports DRLDatabaseManager, get_drl_db & helper functions
│   └── drl_db_manager.py            # Thread-safe Singleton SQLite manager (WAL, FKs, retries, CRUD, fallbacks)
│
└── utility/                         # Core domain module implementations & shared helper functions
    ├── __init__.py
    ├── ingestion.py                 # [ING] Facebook fanpage crawler & client email inbox collector
    ├── extraction.py                # [EXT] LLM post/media extractor, keyword fallback & deduplication/updates
    ├── classification.py            # [CLF] Rubric category (1..5) classifier from title/keywords/topic/format
    ├── schedule_detector.py         # [SCH] Calendar conflict detection, coordinate distance & `fit` [0..1]
    ├── credit_calculator.py         # [CRD] `credit_calc(event_id, role)` & `gain_calc(...)` -> `gain` [0..1]
    ├── recommender.py               # [RCM] `interested`, `urgency`, weighted priority score & MMR re-ranking
    ├── progress_tracker.py          # [PRG] Deadline checks & status transitions (interested -> registered -> in_progress -> done)
    ├── notifier.py                  # [NTF] Notification builder & dispatcher for reminders/updates/cancellations
    └── geo_time_helpers.py          # Shared UTC+7 ISO datetime parsing & "lat,lng" Haversine distance helpers
```

