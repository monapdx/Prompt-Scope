# apps/prompt_scope_app.py
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

# NOTE: Do NOT call st.set_page_config() here (suite_home owns it)

APP_DIR = Path(__file__).resolve().parent
# Avoid Path(__file__).resolve().parents[1] unless the package layout truly requires it.
# Most local runs expect imports to work relative to the app directory and its parent.
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from conversation_insights import analyze_chat, generate_global_insights, guess_topics
from chat_patterns import analyze_message_patterns, parse_mapping_export_bytes

# Store the DB alongside this app file (local-first, predictable).
DB_PATH = str((APP_DIR / "promptscope.db").resolve())

# ---------- PERF LIMITS ----------
MAX_ANALYSIS_CHARS = 20_000
MAX_PREVIEW_CHARS = 4_000

_PATTERNS_JSON_NOISE_TERMS = {
    "null",
    "type",
    "content",
    "parts",
    "message",
    "mapping",
    "author",
    "metadata",
    "children",
    "parent",
    "create_time",
    "update_time",
    "conversation_id",
    "recipient",
    "status",
    "finished_successfully",
}

_ROLE_LINE_RE = re.compile(r"^\s*\[(user|assistant|system|tool)\]\s*", re.I)


def _clean_text_for_token_analysis(text: str) -> str:
    """
    Remove common JSON-structural noise terms so word/phrase stats reflect user content.
    Keeps display text separate (we store raw in `text_raw`).
    """
    if not text:
        return ""
    cleaned = str(text)
    for term in _PATTERNS_JSON_NOISE_TERMS:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_role(value: Any) -> str:
    v = (str(value or "").strip().lower()) if value is not None else ""
    if "assistant" in v or v == "gpt":
        return "assistant"
    if "user" in v or "human" in v:
        return "user"
    if "system" in v:
        return "system"
    if "tool" in v:
        return "tool"
    return v or "unknown"


def _flatten_text_parts(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if "text" in part and isinstance(part["text"], str):
            return part["text"]
        if "content" in part and isinstance(part["content"], str):
            return part["content"]
        if "result" in part and isinstance(part["result"], str):
            return part["result"]
        if "parts" in part and isinstance(part["parts"], list):
            return " ".join(_flatten_text_parts(p) for p in part["parts"])
        return ""
    if isinstance(part, list):
        return " ".join(_flatten_text_parts(p) for p in part)
    return ""


def _extract_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, dict):
        if "parts" in content:
            return _flatten_text_parts(content.get("parts")).strip()
        for k in ("text", "result", "content"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        flattened = _flatten_text_parts(content)
        if flattened.strip():
            return flattened.strip()
    v = message.get("text")
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(content, str) and content.strip():
        return content.strip()
    if content is not None:
        flattened = _flatten_text_parts(content)
        if flattened.strip():
            return flattened.strip()
    return ""


def _extract_author_role(message: Dict[str, Any], node: Dict[str, Any] | None = None) -> str:
    author = message.get("author") or {}
    raw = (author.get("role") or author.get("name") or "").strip()
    if not raw:
        meta = message.get("metadata") or {}
        raw = (meta.get("author_role") or meta.get("role") or "").strip()
    if not raw and node:
        raw = ((node.get("author") or {}).get("role") or "").strip()
    return _normalize_role(raw)


def _split_transcript_with_role_markers(content: str) -> List[Dict[str, Any]]:
    if not content:
        return []
    lines = content.splitlines()
    current_role: Optional[str] = None
    buf: List[str] = []
    out: List[Dict[str, Any]] = []

    def flush():
        nonlocal current_role, buf
        if current_role and buf:
            text = "\n".join(buf).strip()
            if text:
                out.append({"author_role": current_role, "text": text})
        buf = []

    for line in lines:
        m = _ROLE_LINE_RE.match(line)
        if m:
            flush()
            current_role = m.group(1).strip().lower()
            remainder = _ROLE_LINE_RE.sub("", line).strip()
            if remainder:
                buf.append(remainder)
            continue
        buf.append(line)
    flush()
    return out


def extract_messages_from_chat_content(chat: dict) -> list[dict]:
    """
    Parse a stored chat row into message rows.

    Handles:
    - Transcript text with `[user] ...` / `[assistant] ...` markers
    - JSON string containing ChatGPT export objects
    - mapping-node exports (ChatGPT `conversations.json` shape)
    - `messages`, `turns`, or `conversation` lists
    """
    cid = chat.get("id") or chat.get("conversation_id") or ""
    title = chat.get("title") or "(untitled)"
    created_at = chat.get("created_at")
    raw_content = chat.get("content") or ""

    rows: List[Dict[str, Any]] = []

    def add_row(role: str, text: str, ts: Any = None, msg_id: str | None = None):
        text = (text or "").strip()
        if not text:
            return
        role_n = _normalize_role(role)
        ts_val = ts if ts is not None else created_at
        rows.append(
            {
                "conversation_id": cid,
                "conversation_title": title,
                "author_role": role_n,
                "created_at": ts_val,
                "text_raw": text,
                "text": _clean_text_for_token_analysis(text),
                "word_count": len(text.split()),
                "char_count": len(text),
                "message_id": msg_id or "",
            }
        )

    # 1) If it looks like a transcript with role markers, parse that first.
    transcript_parts = _split_transcript_with_role_markers(str(raw_content))
    if transcript_parts:
        for idx, p in enumerate(transcript_parts):
            add_row(p.get("author_role") or "unknown", p.get("text") or "", created_at, msg_id=f"{cid}:{idx}")
        # If we got real roles, return immediately.
        if any(r["author_role"] in ("user", "assistant", "system", "tool") for r in rows):
            return rows
        # Otherwise we still fall through to try structured JSON.

    # 2) Try JSON parse if content is JSON-ish
    obj: Any = None
    s = str(raw_content).strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
        except Exception:
            obj = None

    # If JSON parsed, attempt multiple shapes.
    if obj is not None:
        conv = obj
        # Sometimes content is a list with a single conversation.
        if isinstance(obj, list) and obj:
            # pick the first plausible dict
            conv = obj[0] if isinstance(obj[0], dict) else obj

        if isinstance(conv, dict):
            # mapping shape
            mapping = conv.get("mapping")
            if isinstance(mapping, dict) and mapping:
                for node_id, node in mapping.items():
                    message = (node or {}).get("message")
                    if not isinstance(message, dict):
                        continue
                    text = _extract_message_text(message)
                    if not text:
                        continue
                    role = _extract_author_role(message, node)
                    ts = message.get("create_time") or message.get("created_at") or created_at
                    add_row(role, text, ts, msg_id=message.get("id") or node_id)
                if rows:
                    return rows

            # messages / turns / conversation list shapes
            for k in ("messages", "turns", "conversation"):
                seq = conv.get(k)
                if isinstance(seq, list) and seq:
                    for idx, m in enumerate(seq):
                        if isinstance(m, dict):
                            role = _normalize_role(m.get("role") or m.get("author_role") or (m.get("author") or {}).get("role"))
                            text = m.get("content") or m.get("text") or ""
                            if isinstance(text, (dict, list)):
                                # ChatGPT export message objects
                                text = _extract_message_text(m)
                            ts = m.get("create_time") or m.get("created_at") or created_at
                            add_row(role, str(text), ts, msg_id=str(m.get("id") or f"{cid}:{idx}"))
                        else:
                            add_row("unknown", str(m), created_at, msg_id=f"{cid}:{idx}")
                    if rows:
                        return rows

    # 3) Final fallback: do not treat entire conversation as one message unless we have no structure.
    text_fallback = s if isinstance(raw_content, str) else str(raw_content)
    if text_fallback.strip():
        add_row("unknown", text_fallback.strip(), created_at, msg_id=f"{cid}:fallback")
    return rows

# ---------- DB LAYER ----------


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            model TEXT,
            content TEXT
        );
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_categories (
            chat_id TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            PRIMARY KEY (chat_id, category_id),
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def count_chats() -> int:
    conn = get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM chats").fetchone()
        return int(row[0] if row and row[0] is not None else 0)
    finally:
        conn.close()


def upsert_chat(chat: Dict[str, Any]):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO chats (id, title, created_at, model, content)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
           title=excluded.title,
           created_at=excluded.created_at,
           model=excluded.model,
           content=excluded.content
        """,
        (chat["id"], chat.get("title"), chat.get("created_at"), chat.get("model"), chat.get("content")),
    )
    conn.commit()
    conn.close()


def list_chats(search: str = "", category_ids: Optional[List[int]] = None, sort: str = "newest"):
    conn = get_conn()
    q = """
        SELECT c.id, c.title, c.created_at, c.model,
               COALESCE(GROUP_CONCAT(cat.name, ', '), '') as categories
        FROM chats c
        LEFT JOIN chat_categories cc ON c.id = cc.chat_id
        LEFT JOIN categories cat ON cc.category_id = cat.id
    """
    params = []
    where = []
    if search:
        where.append("(LOWER(c.title) LIKE ? OR LOWER(c.content) LIKE ?)")
        s = f"%{search.lower()}%"
        params.extend([s, s])
    if category_ids:
        placeholders = ",".join("?" * len(category_ids))
        q += f"""
            JOIN (
                SELECT chat_id
                FROM chat_categories
                WHERE category_id IN ({placeholders})
                GROUP BY chat_id
                HAVING COUNT(DISTINCT category_id) = {len(category_ids)}
            ) AS must ON must.chat_id = c.id
        """
        params.extend(category_ids)

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " GROUP BY c.id "

    if sort == "newest":
        q += " ORDER BY datetime(c.created_at) DESC NULLS LAST, c.title COLLATE NOCASE ASC"
    elif sort == "oldest":
        q += " ORDER BY datetime(c.created_at) ASC NULLS LAST, c.title COLLATE NOCASE ASC"
    else:
        q += " ORDER BY c.title COLLATE NOCASE ASC"

    rows = conn.execute(q, params).fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append(
            {"id": row[0], "title": row[1], "created_at": row[2], "model": row[3], "categories": row[4]}
        )
    return result


def list_chats_with_content(
    search: str = "",
    category_ids: Optional[List[int]] = None,
    sort: str = "newest",
    *,
    analysis_chars: int = MAX_ANALYSIS_CHARS,
    preview_chars: int = MAX_PREVIEW_CHARS,
):
    """
    Bulk query that returns everything the UI/analysis needs in one pass.
    We intentionally truncate content for performance (analysis + preview),
    because Streamlit reruns the script on every interaction.
    """
    conn = get_conn()
    q = """
        SELECT
            c.id,
            c.title,
            c.created_at,
            c.model,
            COALESCE(GROUP_CONCAT(cat.name, ', '), '') as categories,
            SUBSTR(COALESCE(c.content, ''), 1, ?) as content_analysis,
            SUBSTR(COALESCE(c.content, ''), 1, ?) as content_preview,
            LENGTH(COALESCE(c.content, '')) as content_len
        FROM chats c
        LEFT JOIN chat_categories cc ON c.id = cc.chat_id
        LEFT JOIN categories cat ON cc.category_id = cat.id
    """
    params: List[Any] = [int(analysis_chars), int(preview_chars)]
    where: List[str] = []
    if search:
        where.append("(LOWER(c.title) LIKE ? OR LOWER(c.content) LIKE ?)")
        s = f"%{search.lower()}%"
        params.extend([s, s])
    if category_ids:
        placeholders = ",".join("?" * len(category_ids))
        q += f"""
            JOIN (
                SELECT chat_id
                FROM chat_categories
                WHERE category_id IN ({placeholders})
                GROUP BY chat_id
                HAVING COUNT(DISTINCT category_id) = {len(category_ids)}
            ) AS must ON must.chat_id = c.id
        """
        params.extend(category_ids)

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " GROUP BY c.id "

    if sort == "newest":
        q += " ORDER BY datetime(c.created_at) DESC NULLS LAST, c.title COLLATE NOCASE ASC"
    elif sort == "oldest":
        q += " ORDER BY datetime(c.created_at) ASC NULLS LAST, c.title COLLATE NOCASE ASC"
    else:
        q += " ORDER BY c.title COLLATE NOCASE ASC"

    rows = conn.execute(q, params).fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "model": row[3],
                "categories": row[4],
                "content_analysis": row[5] or "",
                "content_preview": row[6] or "",
                "content_len": int(row[7] or 0),
            }
        )
    return result


def get_chat(chat_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, created_at, model, content FROM chats WHERE id = ?", (chat_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "created_at": row[2], "model": row[3], "content": row[4]}


def get_chats(chat_ids: List[str]) -> List[Dict[str, Any]]:
    if not chat_ids:
        return []
    conn = get_conn()
    qmarks = ",".join(["?"] * len(chat_ids))
    rows = conn.execute(
        f"SELECT id, title, created_at, model, content FROM chats WHERE id IN ({qmarks})",
        chat_ids,
    ).fetchall()
    conn.close()
    out = [{"id": r[0], "title": r[1], "created_at": r[2], "model": r[3], "content": r[4]} for r in rows]
    by_id = {c["id"]: c for c in out}
    return [by_id[cid] for cid in chat_ids if cid in by_id]


def list_categories() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, (SELECT COUNT(*) FROM chat_categories WHERE category_id = categories.id) as n "
        "FROM categories ORDER BY name COLLATE NOCASE ASC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "count": r[2]} for r in rows]


def assign_categories(chat_ids: List[str], category_names: List[str]):
    if not chat_ids or not category_names:
        return
    conn = get_conn()
    names = [n.strip() for n in category_names if n and n.strip()]
    for name in names:
        conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,))
    conn.commit()
    if not names:
        conn.close()
        return
    qmarks = ",".join(["?"] * len(names))
    cat_rows = conn.execute(f"SELECT id FROM categories WHERE name IN ({qmarks})", names).fetchall()
    cat_ids = [r[0] for r in cat_rows]
    for chat_id in chat_ids:
        for cid in cat_ids:
            conn.execute(
                "INSERT OR IGNORE INTO chat_categories(chat_id, category_id) VALUES (?, ?)",
                (chat_id, cid),
            )
    conn.commit()
    conn.close()


def remove_categories(chat_ids: List[str], category_names: List[str]):
    if not chat_ids or not category_names:
        return
    conn = get_conn()
    qmarks = ",".join(["?"] * len(category_names))
    cat_rows = conn.execute(f"SELECT id FROM categories WHERE name IN ({qmarks})", category_names).fetchall()
    cat_ids = [r[0] for r in cat_rows]
    for chat_id in chat_ids:
        for cid in cat_ids:
            conn.execute(
                "DELETE FROM chat_categories WHERE chat_id = ? AND category_id = ?",
                (chat_id, cid),
            )
    conn.commit()
    conn.close()


def export_categorized() -> Dict[str, Any]:
    conn = get_conn()
    chats = conn.execute("SELECT id, title, created_at, model, content FROM chats").fetchall()
    cats = conn.execute("SELECT id, name FROM categories").fetchall()
    cc = conn.execute("SELECT chat_id, category_id FROM chat_categories").fetchall()
    conn.close()
    cat_lookup = {cid: name for cid, name in cats}
    chat_map = {}
    for row in chats:
        chat_map[row[0]] = {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "model": row[3],
            "content": row[4],
            "categories": [],
        }
    for chat_id, cid in cc:
        if chat_id in chat_map and cid in cat_lookup:
            chat_map[chat_id]["categories"].append(cat_lookup[cid])
    return {
        "schema_version": 1,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "categories": [name for _, name in cats],
        "chats": list(chat_map.values()),
    }


# ---------- JSON PARSING & MERGE ----------


def hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _coerce_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt).isoformat()
        except Exception:
            pass
    try:
        iv = int(float(value))
        return datetime.utcfromtimestamp(iv).isoformat()
    except Exception:
        pass
    return str(value)


def normalize_chats(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        raw = obj.get("chats") if isinstance(obj.get("chats"), list) else obj.get("items") or [obj]
    elif isinstance(obj, list):
        raw = obj
    else:
        return []
    normalized = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or item.get("conversation_id") or item.get("uuid") or "")
        title = item.get("title") or item.get("name") or item.get("summary") or f"Chat #{i+1}"
        created = item.get("created_at") or item.get("create_time") or item.get("timestamp") or item.get("date")
        model = item.get("model") or item.get("model_slug") or item.get("engine") or ""
        content = item.get("content")
        if not content:
            msgs = item.get("messages") or item.get("turns") or item.get("conversation")
            if isinstance(msgs, list):
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        role = m.get("role") or m.get("sender") or ""
                        text = m.get("content") or m.get("text") or ""
                        parts.append(f"[{role}] {text}".strip())
                    else:
                        parts.append(str(m))
                content = "\n\n".join(parts)
            else:
                content = json.dumps(item, ensure_ascii=False)
        if not cid:
            cid = hash_id(f"{title}|{created}|{content[:80]}")
        categories = item.get("categories") or []
        if isinstance(categories, str):
            categories = [x.strip() for x in categories.split(",") if x.strip()]
        normalized.append(
            {
                "id": cid,
                "title": str(title),
                "created_at": _coerce_datetime(created),
                "model": str(model),
                "content": str(content),
                "categories": categories,
            }
        )
    return normalized


def import_json_file(json_bytes: bytes, merge_mode: str = "additive") -> int:
    try:
        obj = json.loads(json_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        obj = json.loads(json_bytes.decode("utf-16"))
    chats = normalize_chats(obj)
    for c in chats:
        upsert_chat(c)
        cats = c.get("categories") or []
        if cats:
            assign_categories([c["id"]], cats)
    return len(chats)


# ---------- UI ----------


def _tag_badge(name: str):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"border:1px solid #ddd;font-size:12px;margin-right:6px'>{name}</span>",
        unsafe_allow_html=True,
    )


def _chat_row(chat: Dict[str, Any], *, checkbox_key: str, preview_key: str):
    cols = st.columns([0.06, 0.44, 0.22, 0.28])
    with cols[0]:
        checked = st.checkbox("", key=checkbox_key)
    with cols[1]:
        title = chat["title"] or "(untitled)"
        st.markdown(f"**{title}**")
        if chat.get("created_at"):
            st.caption(chat["created_at"])
    with cols[2]:
        cats = [c.strip() for c in (chat.get("categories") or "").split(",") if c.strip()]
        if cats:
            for c in cats:
                _tag_badge(c)
        else:
            st.caption("—")
    with cols[3]:
        if st.button("Preview", key=preview_key, use_container_width=True):
            st.session_state.preview_chat_id = chat["id"] if st.session_state.get("preview_chat_id") != chat["id"] else None

        if st.session_state.get("preview_chat_id") == chat["id"]:
            details = get_chat(chat["id"])
            content = (details.get("content") if details else "") or ""
            shown = content[:MAX_PREVIEW_CHARS]
            st.text_area("Content", shown, height=160, label_visibility="collapsed")
            if len(content) > len(shown):
                st.caption(f"Preview truncated ({len(shown):,} / {len(content):,} chars).")
    return checked


@st.cache_data(show_spinner=False)
def _cached_analyze_chat(chat_id: str, title: str, created_at: Optional[str], model: str, content_analysis: str):
    # Deterministic per-chat analysis (content already truncated upstream).
    details = {"id": chat_id, "title": title, "created_at": created_at, "model": model, "content": content_analysis}
    return analyze_chat(details)


@st.cache_data(show_spinner=False)
def _cached_global_insights(analyzed_min: List[Dict[str, Any]]):
    return generate_global_insights(analyzed_min)


@st.cache_data(show_spinner=False)
def _cached_message_patterns(msg_df: pd.DataFrame, speaker_filter: str, min_len: int):
    return analyze_message_patterns(msg_df, speaker_filter=speaker_filter, min_len=min_len)


def main(go_home: Callable[[], None] | None = None):
    # Header + Back button
    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        st.title("📁 Prompt Scope")
        st.caption("Import ChatGPT exports, categorize conversations, and export back — local-only.")
    with top_right:
        if go_home is not None:
            if st.button("← Back to Home", use_container_width=True):
                go_home()

    with st.sidebar:
        st.subheader("Import / Export")
        # Ensure DB exists before any reads/writes.
        init_db()

        with st.expander("Debug (temporary)", expanded=False):
            st.write(
                {
                    "app_file": str(Path(__file__).resolve()),
                    "app_dir": str(APP_DIR),
                    "db_path": DB_PATH,
                    "db_exists": Path(DB_PATH).exists(),
                    "db_chat_count": count_chats(),
                }
            )

        file = st.file_uploader(
            "Import new chat JSON (optional; stored in SQLite for future sessions)",
            type=["json"],
            accept_multiple_files=False,
        )
        if file is not None:
            n = import_json_file(file.getvalue(), merge_mode="additive")
            st.success(f"Merged {n} chats from JSON. Categories present in the file were added (no removals).")

        st.divider()
        if st.button("Export as messages.json"):
            data = export_categorized()
            st.download_button(
                "Download messages.json",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="messages.json",
                mime="application/json",
            )

        st.caption(f"DB: {Path(DB_PATH).name} • Export includes schema_version + exported_at")

    c1, c2, c3, c4 = st.columns([0.32, 0.34, 0.17, 0.17])
    with c1:
        search = st.text_input(
            "Search",
            placeholder="Search title/content",
            help="Search across title and content.",
        )
    with c2:
        all_cats = list_categories()
        cat_name_to_id = {c["name"]: c["id"] for c in all_cats}
        cat_filter_names = st.multiselect(
            "Categories",
            options=[c["name"] for c in all_cats],
            help="Filter by category. Selected categories must all match.",
        )
        cat_filter_ids = [cat_name_to_id[name] for name in cat_filter_names] if cat_filter_names else None
    with c3:
        sort = st.selectbox(
            "Sort",
            ["newest", "oldest", "title A→Z"],
            help="Change ordering for the current results.",
        )
    with c4:
        page_size = st.selectbox(
            "Page size",
            [10, 20, 50, 100],
            index=1,
            help="How many conversations to show per page.",
        )

    # Lightweight list view: do NOT pull full content unless needed (preview/explicit analysis).
    chats = list_chats(
        search=search,
        category_ids=cat_filter_ids,
        sort={"newest": "newest", "oldest": "oldest", "title A→Z": "title"}[sort],
    )

    if "selection_reset_counter" not in st.session_state:
        st.session_state.selection_reset_counter = 0

    if not chats:
        st.info(
            "Your SQLite database is empty. Use the sidebar **Import** to upload your ChatGPT export JSON "
            "(e.g. `conversations.json`). Imported chats will persist locally and be available next time."
        )

    # IMPORTANT: Do not run heavy analysis during normal list interactions
    # (e.g. checkbox selection). Analysis is behind explicit buttons below.
    analysis_s = 0.0
    analyzed_n = 0

    tab_insights, tab_patterns = st.tabs(["Insights", "Patterns"])

    with tab_insights:
        st.subheader("Insights")
        if st.button("Run insights on current filter", key="run_insights_btn"):
            t0 = time.perf_counter()
            chats_full = list_chats_with_content(
                search=search,
                category_ids=cat_filter_ids,
                sort={"newest": "newest", "oldest": "oldest", "title A→Z": "title"}[sort],
            )
            analyzed: List[Dict[str, Any]] = []
            for c in chats_full:
                row = _cached_analyze_chat(
                    c["id"],
                    c.get("title") or "",
                    c.get("created_at"),
                    c.get("model") or "",
                    c.get("content_analysis") or "",
                )
                row = dict(row)
                row["id"] = c.get("id")
                row["title"] = c.get("title")
                row["created_at"] = c.get("created_at")
                analyzed.append(row)
            st.session_state._last_insights_analyzed_n = len(analyzed)
            st.session_state._last_insights_seconds = time.perf_counter() - t0
            st.session_state._last_insights = _cached_global_insights(analyzed)

        insights = st.session_state.get("_last_insights") or []
        if not insights:
            st.caption("Click **Run insights on current filter** to compute insights. This keeps chat selection fast.")
        else:
            for insight in insights:
                desc = insight.get("description") or ""
                n_lines = desc.count("\n") + 1 if desc else 0
                is_long = (len(desc) > 500) or (n_lines > 10)
                if is_long:
                    with st.expander(insight["title"], expanded=False):
                        st.write(desc)
                else:
                    st.markdown(f"### {insight['title']}")
                    st.write(desc)

    with tab_patterns:
        st.subheader("Patterns")
        st.caption("Message-level patterns based on conversations in your Prompt Scope database.")

        p1, p2, p3 = st.columns([0.35, 0.35, 0.3])
        with p1:
            speaker_filter = st.selectbox(
                "Speaker",
                ["all", "user", "assistant", "system", "tool", "unknown"],
                key="patterns_speaker_filter",
            )
        with p2:
            min_len = st.slider(
                "Minimum token length",
                min_value=2,
                max_value=8,
                value=3,
                key="patterns_min_token_len",
            )
        with p3:
            msg_search = st.text_input("Search messages", key="patterns_search_messages")

        if st.button("Run patterns on current filter", key="run_patterns_btn"):
            # Fetch from SQLite (source of truth), then expand each chat into message rows.
            chats_full = list_chats_with_content(
                search=search,
                category_ids=cat_filter_ids,
                sort={"newest": "newest", "oldest": "oldest", "title A→Z": "title"}[sort],
                analysis_chars=MAX_ANALYSIS_CHARS,
                preview_chars=MAX_PREVIEW_CHARS,
            )
            rows: List[Dict[str, Any]] = []
            for c in chats_full:
                chat_obj = {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "created_at": c.get("created_at"),
                    # Use analysis text (bounded) for parsing; preview/full is loaded only on demand elsewhere.
                    "content": c.get("content_analysis") or "",
                }
                rows.extend(extract_messages_from_chat_content(chat_obj))

            msg_df = pd.DataFrame(rows)
            if not msg_df.empty:
                # Normalize timestamps if possible.
                msg_df["created_at"] = pd.to_datetime(msg_df.get("created_at"), errors="coerce")
                msg_df["source_file"] = "SQLite"
            st.session_state._patterns_msg_df = msg_df
            st.session_state._patterns_filter = {"speaker_filter": speaker_filter, "min_len": min_len}

        msg_df = st.session_state.get("_patterns_msg_df")
        if msg_df is None:
            st.caption("Click **Run patterns on current filter** to compute message-level patterns.")
            msg_df = pd.DataFrame()

        # Optional fallback: allow ad-hoc raw export analysis without affecting DB.
        with st.expander("Optional: analyze raw `conversations.json` mapping exports (not imported)"):
            raw_files = st.file_uploader(
                "Upload conversations.json (mapping export) for message-level analysis",
                type=["json"],
                accept_multiple_files=True,
                key="patterns_raw_mapping_upload",
            )
            if raw_files:
                frames = []
                for f in raw_files:
                    try:
                        frames.append(parse_mapping_export_bytes(f.name, f.getvalue()))
                    except Exception as exc:
                        st.error(f"Could not parse {f.name}: {exc}")
                if frames:
                    raw_df = pd.concat(frames, ignore_index=True)
                    msg_df = pd.concat([msg_df, raw_df], ignore_index=True)

        if msg_df.empty:
            st.info("No message-level data found for the current view. Try importing a richer export or using the optional mapping parser above.")
        else:
            if msg_search:
                msg_df = msg_df[msg_df["text_raw"].astype(str).str.contains(msg_search, case=False, na=False)]

            # Avoid letting unknown/structural rows contaminate token stats by default.
            df_for_tokens = msg_df.copy()
            if speaker_filter == "all":
                df_for_tokens = df_for_tokens[df_for_tokens["author_role"].isin(["user", "assistant"])]
            analysis = _cached_message_patterns(df_for_tokens, speaker_filter=speaker_filter, min_len=min_len)
            working = analysis["working"]

            # Metrics must reflect actual parsed message rows (not conversations).
            total_messages = int(len(msg_df))
            total_words = int(msg_df["word_count"].sum()) if "word_count" in msg_df else 0
            total_conversations = int(msg_df["conversation_id"].nunique()) if "conversation_id" in msg_df else 0
            total_files = int(msg_df["source_file"].nunique()) if "source_file" in msg_df else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Messages", f"{total_messages:,}")
            m2.metric("Conversations", f"{total_conversations:,}")
            m3.metric("Words", f"{total_words:,}")
            m4.metric("Files", f"{total_files:,}")

            st.subheader("You vs Assistant")
            # Unknown rows should not contaminate these metrics.
            user_messages = int((msg_df["author_role"] == "user").sum())
            assistant_messages = int((msg_df["author_role"] == "assistant").sum())
            user_words = int(msg_df.loc[msg_df["author_role"] == "user", "word_count"].sum()) if total_words else 0
            assistant_words = int(msg_df.loc[msg_df["author_role"] == "assistant", "word_count"].sum()) if total_words else 0

            user_message_pct = (user_messages / total_messages * 100) if total_messages else 0
            assistant_message_pct = (assistant_messages / total_messages * 100) if total_messages else 0
            user_word_pct = (user_words / total_words * 100) if total_words else 0
            assistant_word_pct = (assistant_words / total_words * 100) if total_words else 0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Your messages", f"{user_messages:,}", f"{user_message_pct:.1f}%")
            r2.metric("Assistant messages", f"{assistant_messages:,}", f"{assistant_message_pct:.1f}%")
            r3.metric("Your words", f"{user_words:,}", f"{user_word_pct:.1f}%")
            r4.metric("Assistant words", f"{assistant_words:,}", f"{assistant_word_pct:.1f}%")

            t1, t2, t3, t4, t5 = st.tabs(
                [
                    "Words that keep coming up",
                    "Phrases you repeat",
                    "When you used ChatGPT most",
                    "Most active conversations",
                    "Message browser",
                ]
            )

            with t1:
                st.dataframe(analysis["words"], use_container_width=True, height=520)

            with t2:
                left, right = st.columns(2)
                with left:
                    st.caption("Top bigrams")
                    st.dataframe(analysis["bigrams"], use_container_width=True, height=420)
                with right:
                    st.caption("Top trigrams")
                    st.dataframe(analysis["trigrams"], use_container_width=True, height=420)

            with t3:
                months_df = analysis["months"]
                if months_df.empty:
                    st.info("No timestamps were available.")
                else:
                    st.line_chart(months_df.set_index("month"))
                    st.dataframe(months_df, use_container_width=True)

            with t4:
                st.dataframe(analysis["conversations"], use_container_width=True, height=520)

            with t5:
                display_cols = ["source_file", "conversation_title", "author_role", "created_at", "word_count", "text_raw"]
                st.dataframe(
                    msg_df.sort_values("created_at", na_position="last")[display_cols],
                    use_container_width=True,
                    height=650,
                )

            csv_data = msg_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download parsed messages as CSV",
                data=csv_data,
                file_name="prompt_scope_message_patterns.csv",
                mime="text/csv",
            )

            with st.expander("Patterns debug", expanded=False):
                st.caption("Helps verify message parsing quality from stored SQLite chats.")
                st.write({"author_role_counts": msg_df["author_role"].value_counts(dropna=False).to_dict()})
                fallback_unknown = int((msg_df["author_role"] == "unknown").sum())
                st.write({"fallback_unknown_rows": fallback_unknown})
                sample_cols = ["conversation_title", "author_role", "created_at", "text_raw"]
                sample = msg_df[sample_cols].head(20).copy()
                sample["text_raw"] = sample["text_raw"].astype(str).str.slice(0, 160)
                st.dataframe(sample, use_container_width=True, height=360)

    total = len(chats)
    if "page" not in st.session_state:
        st.session_state.page = 0
    max_page = max(0, (total - 1) // page_size)

    nav1, nav2, nav3 = st.columns([0.15, 0.7, 0.15])
    with nav1:
        if st.button("⟵ Prev", disabled=(st.session_state.page <= 0)):
            st.session_state.page = max(0, st.session_state.page - 1)
            st.rerun()
    with nav2:
        st.write(f"Page {st.session_state.page + 1} / {max_page + 1} • {total} chats")
    with nav3:
        if st.button("Next ⟶", disabled=(st.session_state.page >= max_page)):
            st.session_state.page = min(max_page, st.session_state.page + 1)
            st.rerun()

    start = st.session_state.page * page_size
    end = start + page_size
    visible = chats[start:end]

    st.markdown("#### Conversations")
    header = st.columns([0.06, 0.44, 0.22, 0.28])
    header[0].markdown("**✓**")
    header[1].markdown("**Title / Date**")
    header[2].markdown("**Categories**")
    header[3].markdown("**Preview**")

    selected_ids: List[str] = []
    for idx, chat in enumerate(visible):
        checkbox_key = f"row{st.session_state.selection_reset_counter}_{start+idx}_{chat['id']}"
        preview_key = f"preview_btn_{st.session_state.selection_reset_counter}_{start+idx}_{chat['id']}"
        if _chat_row(chat, checkbox_key=checkbox_key, preview_key=preview_key):
            selected_ids.append(chat["id"])

    st.markdown("---")
    st.markdown("### Categorize Selected")

    if selected_ids:
        if st.button("Suggest tags for selected", key="suggest_tags_btn"):
            combined_parts: List[str] = []
            for ch in get_chats(selected_ids):
                combined_parts.append(f"{ch.get('title') or ''}\n{(ch.get('content') or '')[:MAX_ANALYSIS_CHARS]}")
            combined_text = "\n\n".join(combined_parts)[:MAX_ANALYSIS_CHARS]
            suggested, _scores = _cached_guess_topics(combined_text)
            st.session_state.prompt_scope_suggested_options = suggested or []
            st.session_state.prompt_scope_suggested_selected = []

        suggested_opts = st.session_state.get("prompt_scope_suggested_options") or []
        if suggested_opts:
            st.markdown("### Suggested tags")
            st.caption("Suggestions only. Nothing is created/applied unless you explicitly assign it.")
            st.multiselect(
                "Choose suggested tags to apply",
                options=suggested_opts,
                default=st.session_state.get("prompt_scope_suggested_selected") or [],
                key="prompt_scope_suggested_selected",
            )

    cat_left, cat_right = st.columns([0.6, 0.4])
    with cat_left:
        all_cats = list_categories()  # refresh counts/options
        new_cat = st.text_input("Create new category (or reuse existing)", placeholder="e.g., Writing, Health, Coding")
        assign_existing = st.multiselect("Or assign existing categories", options=[c["name"] for c in all_cats])
        if st.button("Assign to selected"):
            names = []
            if new_cat.strip():
                names.append(new_cat.strip())
            names += assign_existing
            # Suggested tags are ONLY applied if user explicitly selected them above.
            names += list(st.session_state.get("prompt_scope_suggested_selected") or [])
            names = [n.strip() for n in names if n and n.strip()]
            names = list(dict.fromkeys(names))  # stable de-dupe
            assign_categories(selected_ids, names)
            st.success(f"Assigned {', '.join(names)} to {len(selected_ids)} chat(s).")
            # Reset selection by changing checkbox keys (no direct session_state writes).
            st.session_state.selection_reset_counter += 1
            st.session_state.preview_chat_id = None
            st.rerun()

    with cat_right:
        all_cats = list_categories()
        remove_existing = st.multiselect("Remove categories from selected", options=[c["name"] for c in all_cats])
        if st.button("Remove from selected"):
            remove_categories(selected_ids, remove_existing)
            st.warning(f"Removed {', '.join(remove_existing)} from {len(selected_ids)} chat(s).")
            # Reset selection by changing checkbox keys (no direct session_state writes).
            st.session_state.selection_reset_counter += 1
            st.session_state.preview_chat_id = None
            st.rerun()

    st.markdown("### Category Summary")
    cats = list_categories()
    cols = st.columns(4)
    for i, cat in enumerate(cats):
        with cols[i % 4]:
            st.metric(cat["name"], f"{cat['count']} chats")

    with st.expander("Performance / debug", expanded=False):
        st.caption("Lightweight timing for reruns. Values exclude Streamlit rendering time.")
        st.write(
            {
                "chats_loaded": len(chats),
                "chats_analyzed": analyzed_n,
                "analysis_seconds": round(analysis_s, 3),
                "analysis_chars_per_chat": MAX_ANALYSIS_CHARS,
                "preview_chars_per_chat": MAX_PREVIEW_CHARS,
                "cache_note": "Deterministic analysis is cached via st.cache_data; changes in DB/content invalidate cache automatically.",
            }
        )


@st.cache_data(show_spinner=False)
def _cached_guess_topics(text: str):
    return guess_topics(text)


if __name__ == "__main__":
    # Allows running this module directly too:
    main(go_home=None)
