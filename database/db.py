import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from sqlalchemy import create_engine, delete, event, select, func, text
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_URL, LOCAL_TIMEZONE, WEBAPP_DATABASE_URL
from .models import (
    Base,
    EmailMessage,
    EventRecord,
    EventSubscription,
    Fanpage,
    Post,
    PostAnalysis,
    User,
    UserSubscription,
    WebAppBase,
    ensure_local_tz,
    local_now_iso,
)

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

webapp_engine = create_engine(WEBAPP_DATABASE_URL, echo=False, future=True)
WebAppSessionLocal = sessionmaker(bind=webapp_engine, autoflush=False, autocommit=False, expire_on_commit=False)


@event.listens_for(webapp_engine, "connect")
def _set_sqlite_fk_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


_DB_INITIALIZED = False

_INVALID_FANPAGE_EXACT_NAMES = {
    "facebook",
    "log in",
    "login",
    "log into facebook",
    "log in to facebook",
    "log in or sign up",
    "sign up",
    "sign up for facebook",
    "đăng nhập",
    "đăng nhập facebook",
    "đăng nhập hoặc đăng ký",
    "đăng ký",
    "create new account",
    "watch",
    "notifications",
    "home",
    "unknown",
    "error",
    "page not found",
    "content not found",
    "security check",
    "security check required",
    "checkpoint",
    "just a moment",
    "just a moment...",
    "something went wrong",
    "you're temporarily blocked",
    "bạn tạm thời bị chặn",
    "this content isn't available right now",
    "confirm your identity",
    "meta",
}

_INVALID_FANPAGE_NAME_PATTERNS = re.compile(
    r"(?:"
    r"^log\s*in(?:to)?(?:\s+to)?\s+facebook\b"
    r"|^facebook\s*[-–|]"
    r"|^đăng\s*nhập\b"
    r"|tạm thời bị chặn"
    r"|temporarily blocked"
    r"|security check"
    r"|confirm your identity"
    r"|this content isn't available"
    r"|page isn't available"
    r")",
    re.IGNORECASE,
)


def clean_fanpage_name(raw_name: Optional[str]) -> Optional[str]:
    """Normalize a scraped fanpage title and reject generic Facebook/login/error placeholders."""
    if not raw_name or not isinstance(raw_name, str):
        return None
    text = re.sub(r"\s+", " ", raw_name).strip()
    text = re.sub(r"^\(\d+\)\s*", "", text).strip()
    text = re.sub(r"\s*[\|\-–]\s*Facebook.*$", "", text, flags=re.IGNORECASE).strip()
    if not text or not re.search(r"\w", text, flags=re.UNICODE):
        return None
    lowered = text.lower()
    if lowered in _INVALID_FANPAGE_EXACT_NAMES or _INVALID_FANPAGE_NAME_PATTERNS.search(text):
        return None
    return text


def is_valid_fanpage_name(name: Optional[str]) -> bool:
    """Return True if `name` is a genuine fanpage title rather than a Facebook placeholder."""
    return clean_fanpage_name(name) is not None


def _table_columns(conn, table_name: str) -> List[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return [r[1] for r in rows]


def _migrate_sqlite_schema() -> None:
    """Safely migrate legacy SQLite tables (`users`, `user_subscriptions`, `events`) in place."""
    with engine.begin() as conn:
        existing_tables = {
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }

        # 1. Migrate legacy `users` & `user_subscriptions` if `user_id` is not yet the PK
        if "users" in existing_tables:
            u_cols = _table_columns(conn, "users")
            if "user_id" not in u_cols or "username" not in u_cols or "password" not in u_cols:
                old_users = conn.execute(
                    text("SELECT id, client_id, created_at FROM users")
                ).fetchall()
                old_subs = []
                if "user_subscriptions" in existing_tables:
                    old_subs = conn.execute(
                        text("SELECT user_id, fanpage_id, created_at FROM user_subscriptions")
                    ).fetchall()
                    conn.execute(text("DROP TABLE user_subscriptions"))
                conn.execute(text("DROP TABLE users"))
                conn.execute(
                    text(
                        """
                        CREATE TABLE users (
                            user_id VARCHAR(128) NOT NULL PRIMARY KEY,
                            username VARCHAR(128) NOT NULL UNIQUE,
                            password VARCHAR(255) NOT NULL DEFAULT '',
                            start_year INTEGER NOT NULL DEFAULT 2024,
                            created_at DATETIME
                        )
                        """
                    )
                )
                id_map: Dict[int, str] = {}
                for old_id, client_id, created_at in old_users:
                    new_uid = uuid.uuid4().hex
                    id_map[old_id] = new_uid
                    conn.execute(
                        text(
                            "INSERT INTO users (user_id, username, password, start_year, created_at) "
                            "VALUES (:uid, :uname, '', 2024, :cat)"
                        ),
                        {"uid": new_uid, "uname": client_id, "cat": created_at},
                    )
                conn.execute(
                    text(
                        """
                        CREATE TABLE user_subscriptions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id VARCHAR(128) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                            fanpage_id INTEGER NOT NULL REFERENCES fanpages(id) ON DELETE CASCADE,
                            created_at DATETIME,
                            CONSTRAINT uq_user_fanpage UNIQUE (user_id, fanpage_id)
                        )
                        """
                    )
                )
                for old_uid, fp_id, created_at in old_subs:
                    new_uid = id_map.get(old_uid)
                    if new_uid:
                        conn.execute(
                            text(
                                "INSERT OR IGNORE INTO user_subscriptions (user_id, fanpage_id, created_at) "
                                "VALUES (:uid, :fpid, :cat)"
                            ),
                            {"uid": new_uid, "fpid": fp_id, "cat": created_at},
                        )

        # 2. Migrate legacy `events` table (id -> event_id + new structured columns)
        if "events" in existing_tables:
            e_cols = _table_columns(conn, "events")
            if "event_id" not in e_cols:
                fanpage_names: Dict[int, str] = {}
                if "fanpages" in existing_tables:
                    for fid, fname, fslug in conn.execute(
                        text("SELECT id, name, slug FROM fanpages")
                    ).fetchall():
                        fanpage_names[fid] = fname or fslug or ""

                old_events = conn.execute(text("SELECT * FROM events")).mappings().all()
                conn.execute(text("ALTER TABLE events RENAME TO events_legacy_backup"))
                conn.execute(
                    text(
                        """
                        CREATE TABLE events (
                            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title VARCHAR(512) NOT NULL,
                            description TEXT NOT NULL DEFAULT '',
                            organizer VARCHAR(512) NOT NULL DEFAULT '',
                            is_online BOOLEAN NOT NULL DEFAULT 0,
                            event_start_datetime VARCHAR(64),
                            reg_start_datetime VARCHAR(64),
                            reg_end_datetime VARCHAR(64),
                            event_end_datetime VARCHAR(64),
                            location VARCHAR(512) NOT NULL DEFAULT '',
                            source_type VARCHAR(255) NOT NULL DEFAULT 'facebook',
                            created_at VARCHAR(64) NOT NULL,
                            updated_at VARCHAR(64) NOT NULL,
                            fee VARCHAR(255) NOT NULL DEFAULT '',
                            target_audience TEXT NOT NULL DEFAULT '[]',
                            award TEXT NOT NULL DEFAULT '',
                            registration VARCHAR(1024),
                            category TEXT NOT NULL DEFAULT '[]',
                            format TEXT NOT NULL DEFAULT '[]',
                            post_url VARCHAR(1024),
                            normalized_title VARCHAR(512) NOT NULL DEFAULT '',
                            online_mode VARCHAR(32),
                            event_date VARCHAR(32),
                            event_time VARCHAR(32),
                            start_datetime VARCHAR(64),
                            end_datetime VARCHAR(64),
                            normalized_location VARCHAR(512),
                            source_id VARCHAR(255) NOT NULL DEFAULT '',
                            fanpage_id INTEGER REFERENCES fanpages(id) ON DELETE SET NULL,
                            client_id VARCHAR(128),
                            event_json TEXT NOT NULL DEFAULT '{}',
                            version INTEGER NOT NULL DEFAULT 1
                        )
                        """
                    )
                )
                for row in old_events:
                    try:
                        extra = json.loads(row.get("event_json") or "{}")
                    except Exception:
                        extra = {}
                    fp_id = row.get("fanpage_id")
                    raw_st = (row.get("source_type") or "facebook").strip()
                    if raw_st == "facebook" and fp_id and fanpage_names.get(fp_id):
                        src_type = f"facebook: {fanpage_names[fp_id]}"
                    else:
                        src_type = raw_st

                    ev_start = extra.get("event_start_datetime") or row.get("start_datetime")
                    ev_end = extra.get("event_end_datetime") or row.get("end_datetime")
                    reg_start = extra.get("reg_start_datetime")
                    reg_end = extra.get("reg_end_datetime")
                    reg_url = extra.get("registration_url") or extra.get("registration")
                    p_url = extra.get("post_url")
                    cat = extra.get("category") or []
                    fmt = extra.get("format") or []
                    aud = extra.get("target_audience") or []
                    if isinstance(aud, str):
                        aud = [aud] if aud.strip() else []

                    conn.execute(
                        text(
                            """
                            INSERT INTO events (
                                event_id, title, description, organizer, is_online,
                                event_start_datetime, reg_start_datetime, reg_end_datetime, event_end_datetime,
                                location, source_type, created_at, updated_at,
                                fee, target_audience, award, registration, category, format, post_url,
                                normalized_title, online_mode, event_date, event_time,
                                start_datetime, end_datetime, normalized_location,
                                source_id, fanpage_id, client_id, event_json, version
                            ) VALUES (
                                :event_id, :title, :description, :organizer, :is_online,
                                :event_start_datetime, :reg_start_datetime, :reg_end_datetime, :event_end_datetime,
                                :location, :source_type, :created_at, :updated_at,
                                :fee, :target_audience, :award, :registration, :category, :format, :post_url,
                                :normalized_title, :online_mode, :event_date, :event_time,
                                :start_datetime, :end_datetime, :normalized_location,
                                :source_id, :fanpage_id, :client_id, :event_json, :version
                            )
                            """
                        ),
                        {
                            "event_id": row["id"],
                            "title": row["title"],
                            "description": extra.get("description") or "",
                            "organizer": extra.get("organizer") or "",
                            "is_online": 1 if row.get("is_online") else 0,
                            "event_start_datetime": ev_start,
                            "reg_start_datetime": reg_start,
                            "reg_end_datetime": reg_end,
                            "event_end_datetime": ev_end,
                            "location": row.get("location") or "",
                            "source_type": src_type,
                            "created_at": str(row.get("created_at") or local_now_iso()),
                            "updated_at": str(row.get("updated_at") or local_now_iso()),
                            "fee": extra.get("fee") or "",
                            "target_audience": json.dumps(aud, ensure_ascii=False),
                            "award": extra.get("award") or "",
                            "registration": reg_url,
                            "category": json.dumps(cat if isinstance(cat, list) else [], ensure_ascii=False),
                            "format": json.dumps(fmt if isinstance(fmt, list) else [], ensure_ascii=False),
                            "post_url": p_url,
                            "normalized_title": row.get("normalized_title") or "",
                            "online_mode": row.get("online_mode"),
                            "event_date": row.get("event_date"),
                            "event_time": row.get("event_time"),
                            "start_datetime": row.get("start_datetime"),
                            "end_datetime": row.get("end_datetime"),
                            "normalized_location": row.get("normalized_location"),
                            "source_id": row.get("source_id") or "",
                            "fanpage_id": fp_id,
                            "client_id": row.get("client_id"),
                            "event_json": row.get("event_json") or "{}",
                            "version": row.get("version") or 1,
                        },
                    )
                conn.execute(text("DROP TABLE events_legacy_backup"))


def _seed_webapp_fanpages_from_legacy() -> None:
    """Seed `webapp.db.fanpages` from legacy `rss_bot.db.fanpages` when first creating the table."""
    with webapp_engine.begin() as wconn:
        cnt = wconn.execute(text("SELECT COUNT(*) FROM fanpages")).scalar() or 0
        if cnt > 0:
            return

        legacy_rows = []
        try:
            with engine.connect() as iconn:
                has_legacy = iconn.execute(
                    text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fanpages'")
                ).fetchone()
                if has_legacy:
                    legacy_rows = iconn.execute(
                        text("SELECT id, name, slug, url FROM fanpages WHERE is_active = 1 ORDER BY id ASC")
                    ).fetchall()
        except Exception:
            legacy_rows = []

        for fid, fname, fslug, furl in legacy_rows:
            if not furl:
                continue
            wconn.execute(
                text(
                    "INSERT OR IGNORE INTO fanpages (fanpage_id, fanpage_name, facebook_url) "
                    "VALUES (:fid, :fname, :furl)"
                ),
                {
                    "fid": int(fid),
                    "fname": (str(fname).strip() if fname else "") or (str(fslug).strip() if fslug else ""),
                    "furl": str(furl).strip(),
                },
            )


def init_db() -> None:
    """Create database tables in rss_bot.db (ingestion) and webapp.db (events, users, event_subscriptions, fanpages)."""
    global _DB_INITIALIZED
    had_webapp_fanpages = True
    if not _DB_INITIALIZED:
        with webapp_engine.connect() as wconn:
            had_webapp_fanpages = bool(
                wconn.execute(
                    text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fanpages'")
                ).fetchone()
            )
    Base.metadata.create_all(bind=engine)
    WebAppBase.metadata.create_all(bind=webapp_engine)
    if not _DB_INITIALIZED:
        _DB_INITIALIZED = True
        if not had_webapp_fanpages:
            _seed_webapp_fanpages_from_legacy()


def get_session() -> Session:
    return SessionLocal()


def get_webapp_session() -> Session:
    return WebAppSessionLocal()


def _find_fanpage_in_session(wsession: Session, identifier: str) -> Optional[Fanpage]:
    """Locate a `Fanpage` row in `webapp.db` by `fanpage_id`, `facebook_url`, or derived `slug`."""
    ident = (identifier or "").strip()
    if not ident:
        return None
    if ident.isdigit():
        found = wsession.get(Fanpage, int(ident))
        if found:
            return found

    try:
        canonical_url, target_slug = normalize_fb_url(ident)
        found = wsession.scalar(select(Fanpage).where(Fanpage.facebook_url == canonical_url))
        if found:
            return found
    except ValueError:
        target_slug = ident

    ident_low = ident.lower()
    slug_low = target_slug.lower()
    for fp in wsession.scalars(select(Fanpage).order_by(Fanpage.fanpage_id.asc())).all():
        if fp.slug.lower() in (ident_low, slug_low):
            return fp
    return None


def _attach_fanpages_to_posts(posts: List[Post]) -> List[Post]:
    """Attach transient `Fanpage` objects from `webapp.db` onto `Post` instances loaded from `rss_bot.db`."""
    if not posts:
        return posts
    fp_ids = list({p.fanpage_id for p in posts if p.fanpage_id is not None})
    if not fp_ids:
        return posts
    with get_webapp_session() as wsession:
        fp_map = {
            fp.fanpage_id: fp
            for fp in wsession.scalars(select(Fanpage).where(Fanpage.fanpage_id.in_(fp_ids))).all()
        }
    for p in posts:
        p.fanpage = fp_map.get(p.fanpage_id)
    return posts


def normalize_fb_url(raw_url: str) -> Tuple[str, str]:
    """
    Normalize a Facebook fanpage URL or username into a canonical URL and file-safe slug.
    """
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        if "facebook.com" in raw_url or "fb.com" in raw_url:
            raw_url = "https://" + raw_url
        else:
            slug = re.sub(r"[^a-zA-Z0-9._-]", "", raw_url.strip("/"))
            if not slug:
                raise ValueError(f"Invalid Facebook Fanpage identifier: {raw_url}")
            return f"https://www.facebook.com/{slug}", slug

    parsed = urlparse(raw_url)
    path = parsed.path.strip("/")

    if path == "profile.php":
        qs = parse_qs(parsed.query)
        page_id = qs.get("id", ["unknown"])[0]
        canonical = f"https://www.facebook.com/profile.php?id={page_id}"
        slug = f"profile_{page_id}"
        return canonical, slug

    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError(f"Invalid Facebook Fanpage URL: {raw_url}")

    if parts[0] == "pages" and len(parts) >= 3:
        slug = f"{parts[1]}_{parts[2]}"
        canonical = f"https://www.facebook.com/pages/{parts[1]}/{parts[2]}"
    elif parts[0] == "people" and len(parts) >= 3:
        slug = f"{parts[1]}_{parts[2]}"
        canonical = f"https://www.facebook.com/people/{parts[1]}/{parts[2]}"
    else:
        slug = parts[0]
        canonical = f"https://www.facebook.com/{slug}"

    safe_slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return canonical, safe_slug


def add_fanpage(raw_url: str, name: Optional[str] = None) -> Tuple[Fanpage, bool]:
    """
    Add a fanpage to the `fanpages` table in `webapp.db`.
    Returns (Fanpage, created_bool).
    """
    init_db()
    canonical_url, slug = normalize_fb_url(raw_url)
    cleaned_name = clean_fanpage_name(name)
    with get_webapp_session() as wsession:
        existing = wsession.scalar(select(Fanpage).where(Fanpage.facebook_url == canonical_url))
        if existing:
            changed = False
            if cleaned_name and (
                not is_valid_fanpage_name(existing.fanpage_name) or existing.fanpage_name == existing.slug
            ):
                existing.fanpage_name = cleaned_name
                changed = True
            elif not is_valid_fanpage_name(existing.fanpage_name):
                existing.fanpage_name = existing.slug
                changed = True
            if changed:
                wsession.commit()
                wsession.refresh(existing)
            return existing, False

        fanpage = Fanpage(
            fanpage_name=cleaned_name or slug,
            facebook_url=canonical_url,
        )
        wsession.add(fanpage)
        wsession.commit()
        wsession.refresh(fanpage)
        return fanpage, True


def list_fanpages(active_only: bool = False) -> List[Tuple[Fanpage, int]]:
    """Return list of `(Fanpage, post_count)` from `webapp.db` paired with post stats from `rss_bot.db`."""
    init_db()
    with get_webapp_session() as wsession:
        fanpages = list(wsession.scalars(select(Fanpage).order_by(Fanpage.fanpage_id.asc())).all())
    if not fanpages:
        return []

    post_stats: Dict[int, Tuple[int, Optional[datetime]]] = {}
    with get_session() as session:
        stmt = select(Post.fanpage_id, func.count(Post.id), func.max(Post.scraped_at)).group_by(Post.fanpage_id)
        for fp_id, cnt, last_scr in session.execute(stmt).all():
            post_stats[int(fp_id)] = (int(cnt or 0), last_scr)

    result: List[Tuple[Fanpage, int]] = []
    for fp in fanpages:
        cnt, last_scr = post_stats.get(fp.fanpage_id, (0, None))
        fp.last_checked_at = last_scr
        result.append((fp, cnt))
    return result


def remove_fanpage(identifier: str) -> Optional[Fanpage]:
    """Remove a fanpage by ID, slug, or URL from `webapp.db` and purge its cached posts in `rss_bot.db`."""
    init_db()
    removed: Optional[Fanpage] = None
    with get_webapp_session() as wsession:
        fanpage = _find_fanpage_in_session(wsession, identifier)
        if fanpage:
            removed = fanpage
            wsession.delete(fanpage)
            wsession.commit()

    if removed is not None:
        with get_session() as session:
            session.execute(delete(Post).where(Post.fanpage_id == removed.fanpage_id))
            session.execute(delete(PostAnalysis).where(PostAnalysis.fanpage_id == removed.fanpage_id))
            session.execute(delete(UserSubscription).where(UserSubscription.fanpage_id == removed.fanpage_id))
            session.commit()
    return removed


def update_fanpage_metadata(fanpage_id: int, name: Optional[str] = None) -> Optional[str]:
    """Update `fanpage_name` in `webapp.db` when valid (healing bad names to slug).
    Returns the resulting stored name, or None if the fanpage does not exist.
    """
    cleaned_name = clean_fanpage_name(name)
    with get_webapp_session() as wsession:
        fanpage = wsession.get(Fanpage, fanpage_id)
        if not fanpage:
            return None
        fanpage.last_checked_at = datetime.now(LOCAL_TIMEZONE)
        if cleaned_name:
            fanpage.fanpage_name = cleaned_name
        elif not is_valid_fanpage_name(fanpage.fanpage_name):
            fanpage.fanpage_name = fanpage.slug
        resolved = fanpage.fanpage_name
        wsession.commit()
        return resolved


def save_posts(fanpage_id: int, scraped_items: List[Dict]) -> List[Post]:
    """
    Save newly scraped posts for a fanpage, skipping any whose post_id already exists.
    Returns list of newly inserted Post records (with fanpage attached).
    """
    if not scraped_items:
        return []

    new_post_ids: List[int] = []
    with get_session() as session:
        for item in scraped_items:
            post_id = str(item.get("post_id", "")).strip()
            if not post_id:
                continue

            exists = session.scalar(select(Post.id).where(Post.post_id == post_id))
            if exists:
                continue

            published_at = item.get("published_at")
            if not isinstance(published_at, datetime):
                published_at = datetime.now(LOCAL_TIMEZONE)
            else:
                published_at = ensure_local_tz(published_at)

            media_urls = item.get("media_urls") or []
            post = Post(
                post_id=post_id,
                fanpage_id=fanpage_id,
                post_url=item.get("post_url", ""),
                content=item.get("content", "").strip(),
                media_urls_json=json.dumps(media_urls, ensure_ascii=False),
                published_at=published_at,
                scraped_at=datetime.now(LOCAL_TIMEZONE),
            )
            session.add(post)
            try:
                session.flush()
                new_post_ids.append(post.id)
            except Exception:
                session.rollback()

        session.commit()

        if not new_post_ids:
            return []

        stmt = (
            select(Post)
            .where(Post.id.in_(new_post_ids))
            .order_by(Post.published_at.desc(), Post.id.desc())
        )
        posts = list(session.scalars(stmt).all())
    return _attach_fanpages_to_posts(posts)


def get_posts_for_fanpage(fanpage_id: int, limit: int = 50) -> List[Post]:
    """Retrieve recent posts for a specific fanpage ordered by published_at desc."""
    init_db()
    with get_session() as session:
        stmt = (
            select(Post)
            .where(Post.fanpage_id == fanpage_id)
            .order_by(Post.published_at.desc(), Post.id.desc())
            .limit(limit)
        )
        posts = list(session.scalars(stmt).all())
    return _attach_fanpages_to_posts(posts)


def get_all_recent_posts(limit: int = 100) -> List[Post]:
    """Retrieve recent posts across all fanpages."""
    init_db()
    with get_session() as session:
        stmt = (
            select(Post)
            .order_by(Post.published_at.desc(), Post.id.desc())
            .limit(limit)
        )
        posts = list(session.scalars(stmt).all())
    return _attach_fanpages_to_posts(posts)


# --- User & Subscription Management ---


def get_or_create_user(client_id: str) -> User:
    init_db()
    clean_id = client_id.strip()
    with get_webapp_session() as session:
        user = session.scalar(
            select(User).where((User.username == clean_id) | (User.user_id == clean_id))
        )
        if not user:
            user = User(
                user_id=uuid.uuid4().hex,
                username=clean_id,
                password="",
                start_year=2024,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
        return user


def create_user(username: str, hashed_password: str, start_year: int) -> User:
    """Create a registered client user account in webapp.db."""
    init_db()
    clean_username = username.strip()
    with get_webapp_session() as session:
        existing = session.scalar(select(User).where(User.username == clean_username))
        if existing:
            if not existing.password:
                existing.password = hashed_password
                existing.start_year = int(start_year)
                session.commit()
                session.refresh(existing)
                return existing
            raise ValueError(f"Username '{clean_username}' is already taken.")
        user = User(
            user_id=uuid.uuid4().hex,
            username=clean_username,
            password=hashed_password,
            start_year=int(start_year),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def get_user_by_username(username: str) -> Optional[User]:
    init_db()
    with get_webapp_session() as session:
        return session.scalar(select(User).where(User.username == username.strip()))


def get_user_by_id(user_id: str) -> Optional[User]:
    init_db()
    with get_webapp_session() as session:
        return session.get(User, user_id.strip())


def list_users() -> List[Dict[str, Any]]:
    """Return all users with their event subscription counts and total credits."""
    init_db()
    with get_webapp_session() as session:
        users = list(session.scalars(select(User).order_by(User.username.asc())).all())
        subs = list(session.scalars(select(EventSubscription)).all())
        stats: Dict[str, Dict[str, int]] = {}
        for s in subs:
            entry = stats.setdefault(s.user_id, {"event_count": 0, "total_credit": 0})
            entry["event_count"] += 1
            entry["total_credit"] += int(s.credit or 0)
        result = []
        for u in users:
            st = stats.get(u.user_id, {"event_count": 0, "total_credit": 0})
            result.append(
                {
                    **u.to_dict(),
                    "has_password": bool(u.password),
                    "subscribed_events_count": st["event_count"],
                    "total_credit": st["total_credit"],
                }
            )
        return result


def subscribe_user_to_fanpages(
    client_id: str, raw_urls: List[str]
) -> Tuple[List[Fanpage], List[Fanpage]]:
    """
    Subscribe a user (`client_id`) to one or more fanpage URLs.
    Automatically creates the user and any fanpage not yet in DB.
    Returns (user_subscribed_fanpages, newly_created_or_unscraped_fanpages).
    """
    init_db()
    clean_client = client_id.strip()
    get_or_create_user(clean_client)
    needs_scrape: List[Fanpage] = []

    for raw_url in raw_urls:
        fp, created = add_fanpage(raw_url)
        with get_session() as session:
            has_posts = session.scalar(select(Post.id).where(Post.fanpage_id == fp.fanpage_id).limit(1))
            if created or has_posts is None:
                needs_scrape.append(fp)

            existing_sub = session.scalar(
                select(UserSubscription).where(
                    UserSubscription.client_id == clean_client,
                    UserSubscription.fanpage_id == fp.fanpage_id,
                )
            )
            if not existing_sub:
                sub = UserSubscription(client_id=clean_client, fanpage_id=fp.fanpage_id)
                session.add(sub)
                session.commit()

    return get_user_subscriptions(clean_client), needs_scrape


def unsubscribe_user_from_fanpage(client_id: str, identifier: str) -> Optional[Fanpage]:
    """Unsubscribe `client_id` from a fanpage by ID, slug, or URL."""
    init_db()
    clean_client = client_id.strip()
    with get_webapp_session() as wsession:
        fanpage = _find_fanpage_in_session(wsession, identifier)
    if not fanpage:
        return None

    with get_session() as session:
        sub = session.scalar(
            select(UserSubscription).where(
                UserSubscription.client_id == clean_client,
                UserSubscription.fanpage_id == fanpage.fanpage_id,
            )
        )
        if sub:
            session.delete(sub)
            session.commit()
            return fanpage
        return None


def get_user_subscriptions(client_id: str) -> List[Fanpage]:
    """Return all Fanpages that `client_id` is currently subscribed to."""
    init_db()
    with get_session() as session:
        sub_ids = list(
            session.scalars(
                select(UserSubscription.fanpage_id).where(UserSubscription.client_id == client_id.strip())
            ).all()
        )
    if not sub_ids:
        return []
    with get_webapp_session() as wsession:
        stmt = (
            select(Fanpage)
            .where(Fanpage.fanpage_id.in_(sub_ids))
            .order_by(Fanpage.fanpage_id.asc())
        )
        return list(wsession.scalars(stmt).all())


def get_posts_for_user(client_id: str, limit: int = 50) -> List[Post]:
    """Return recent posts across all fanpages that `client_id` is subscribed to."""
    init_db()
    with get_session() as session:
        sub_fanpage_ids = list(
            session.scalars(
                select(UserSubscription.fanpage_id)
                .where(UserSubscription.client_id == client_id.strip())
            ).all()
        )
        if not sub_fanpage_ids:
            return []

        stmt = (
            select(Post)
            .where(Post.fanpage_id.in_(sub_fanpage_ids))
            .order_by(Post.published_at.desc(), Post.id.desc())
            .limit(limit)
        )
        posts = list(session.scalars(stmt).all())
    return _attach_fanpages_to_posts(posts)


def get_subscribers_for_fanpage(fanpage_id: int) -> List[str]:
    """Return list of `client_id`s subscribed to a given `fanpage_id`."""
    init_db()
    with get_session() as session:
        stmt = (
            select(UserSubscription.client_id)
            .where(UserSubscription.fanpage_id == fanpage_id)
        )
        return list(session.scalars(stmt).all())


def get_inbox_messages(client_id: Optional[str] = None, limit: int = 100) -> List[EmailMessage]:
    """Retrieve recent email entries received in the server inbox."""
    init_db()
    with get_session() as session:
        stmt = select(EmailMessage).order_by(EmailMessage.ingested_at.desc(), EmailMessage.id.desc())
        if client_id:
            stmt = stmt.where(EmailMessage.client_id == client_id.strip())
        stmt = stmt.limit(limit)
        return list(session.scalars(stmt).all())


def get_all_events(
    source_type: Optional[str] = None,
    client_id: Optional[str] = None,
    fanpage_id: Optional[int] = None,
    limit: int = 200,
) -> List[EventRecord]:
    """Retrieve deduplicated events from the unified EventRecord table in webapp.db."""
    init_db()
    with get_webapp_session() as session:
        stmt = select(EventRecord).order_by(EventRecord.updated_at.desc(), EventRecord.event_id.desc())
        if source_type:
            st_clean = source_type.strip()
            if st_clean.lower() == "facebook":
                stmt = stmt.where(EventRecord.source_type.ilike("facebook%"))
            else:
                stmt = stmt.where(EventRecord.source_type == st_clean)
        stmt = stmt.limit(limit)
        return list(session.scalars(stmt).all())


def get_event_by_id(event_id: int) -> Optional[EventRecord]:
    init_db()
    with get_webapp_session() as session:
        return session.get(EventRecord, event_id)


def subscribe_user_to_event(
    user_id: str,
    event_id: int,
    audience: str = "",
    progress: str = "interested",
    evidence: str = "",
    credit: int = 0,
) -> EventSubscription:
    """Create or update a user's subscription to an event in webapp.db."""
    init_db()
    with get_webapp_session() as session:
        ev = session.get(EventRecord, event_id)
        if not ev:
            raise ValueError(f"Event {event_id} not found.")
        sub = session.get(EventSubscription, (user_id, event_id))
        if sub:
            if audience:
                sub.audience = audience.strip()
            if progress:
                sub.progress = progress.strip()
            if evidence:
                sub.evidence = evidence.strip()
            if credit is not None:
                sub.credit = int(credit)
        else:
            sub = EventSubscription(
                user_id=user_id,
                event_id=event_id,
                audience=(audience or "").strip(),
                progress=(progress or "interested").strip(),
                evidence=(evidence or "").strip(),
                credit=int(credit or 0),
            )
            session.add(sub)
        session.commit()
        session.refresh(sub)
        return sub


def update_event_subscription(
    user_id: str,
    event_id: int,
    audience: Optional[str] = None,
    progress: Optional[str] = None,
    evidence: Optional[str] = None,
    credit: Optional[int] = None,
) -> Optional[EventSubscription]:
    """Update fields on an existing (user_id, event_id) subscription in webapp.db."""
    init_db()
    with get_webapp_session() as session:
        sub = session.get(EventSubscription, (user_id, event_id))
        if not sub:
            return None
        if audience is not None:
            sub.audience = audience.strip()
        if progress is not None:
            sub.progress = progress.strip()
        if evidence is not None:
            sub.evidence = evidence.strip()
        if credit is not None:
            sub.credit = int(credit)
        session.commit()
        session.refresh(sub)
        return sub


def unsubscribe_user_from_event(user_id: str, event_id: int) -> bool:
    """Remove a user's subscription to an event from webapp.db."""
    init_db()
    with get_webapp_session() as session:
        sub = session.get(EventSubscription, (user_id, event_id))
        if not sub:
            return False
        session.delete(sub)
        session.commit()
        return True


def get_user_event_subscriptions(user_id: str) -> List[Dict[str, Any]]:
    """Return all event subscriptions (with nested event details) for a given user_id."""
    init_db()
    with get_webapp_session() as session:
        stmt = (
            select(EventSubscription)
            .options(joinedload(EventSubscription.event))
            .where(EventSubscription.user_id == user_id)
            .order_by(EventSubscription.event_id.desc())
        )
        rows = list(session.scalars(stmt).all())
        return [r.to_dict(include_event=True) for r in rows]


def list_all_event_subscriptions(limit: int = 200) -> List[Dict[str, Any]]:
    """Return recent event subscriptions across all users (for admin view)."""
    init_db()
    with get_webapp_session() as session:
        stmt = (
            select(EventSubscription)
            .options(joinedload(EventSubscription.event), joinedload(EventSubscription.user))
            .order_by(EventSubscription.event_id.desc())
            .limit(limit)
        )
        rows = list(session.scalars(stmt).all())
        out = []
        for r in rows:
            item = r.to_dict(include_event=True)
            item["username"] = r.user.username if r.user else r.user_id
            out.append(item)
        return out
