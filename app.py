# apps/inboxGPT_app.py
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

# NOTE: Do NOT call st.set_page_config() here (suite_home owns it)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from conversation_insights import analyze_chat, find_similar_conversations, generate_global_insights, guess_topics
from chat_patterns import analyze_message_patterns, messages_from_stored_chats, parse_mapping_export_bytes

DB_PATH = str((PROJECT_ROOT / "chat_categorizer.db").resolve())

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


def get_chat(chat_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, created_at, model, content FROM chats WHERE id = ?", (chat_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "created_at": row[2], "model": row[3], "content": row[4]}


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


def _chat_row(chat: Dict[str, Any], key_prefix: str, analyzed_pool: List[Dict[str, Any]]):
    cols = st.columns([0.06, 0.44, 0.22, 0.28])
    with cols[0]:
        checked = st.checkbox("", key=f"{key_prefix}_{chat['id']}")
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
        with st.expander("Preview"):
            details = get_chat(chat["id"])
            st.text_area(
                "Content",
                details.get("content") if details else "",
                height=160,
                label_visibility="collapsed",
            )
            if details:
                # "You've explored this before" (deterministic matching)
                target = analyze_chat(details)
                target["id"] = details.get("id")
                target["title"] = details.get("title")
                target["created_at"] = details.get("created_at")
                similars = find_similar_conversations(target, analyzed_pool, top_k=3)
                if similars:
                    st.markdown("#### You’ve explored this before")
                    for s in similars:
                        st.markdown(f"**{s['title']}**")
                        if s.get("created_at"):
                            st.caption(s["created_at"])
                        if s.get("overlap_topics"):
                            st.write("Overlapping topics: " + ", ".join(s["overlap_topics"]))
                        if s.get("overlap_keywords"):
                            st.write("Overlapping keywords: " + ", ".join([f"`{k}`" for k in s["overlap_keywords"]]))
    return checked


def main(go_home: Callable[[], None] | None = None):
    # Header + Back button
    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        st.title("📁 InboxGPT")
        st.caption("Import ChatGPT exports, categorize conversations, and export back — local-only.")
    with top_right:
        if go_home is not None:
            if st.button("← Back to Home", use_container_width=True):
                go_home()

    init_db()

    with st.sidebar:
        st.subheader("Import / Export")

        file = st.file_uploader(
            "Import chat JSON (raw export / conversations.json / previously categorized)",
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

    chats = list_chats(
        search=search,
        category_ids=cat_filter_ids,
        sort={"newest": "newest", "oldest": "oldest", "title A→Z": "title"}[sort],
    )

    analyzed = []
    for c in chats:
        full = get_chat(c["id"])
        if not full:
            continue
        row = analyze_chat(full)
        row["id"] = full.get("id")
        row["title"] = full.get("title")
        row["created_at"] = full.get("created_at")
        analyzed.append(row)

    tab_insights, tab_patterns = st.tabs(["Insights", "Patterns"])

    with tab_insights:
        st.subheader("Insights")
        insights = generate_global_insights(analyzed)
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
        st.caption("Message-level patterns based on conversations in your InboxGPT database.")

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

        # Build message-level rows from the currently loaded chat set (filtered by search/category).
        full_chats: List[Dict[str, Any]] = []
        for c in chats:
            full = get_chat(c["id"])
            if full:
                full_chats.append(full)

        msg_df = messages_from_stored_chats(full_chats, source_name="SQLite")

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
                msg_df = msg_df[msg_df["text"].astype(str).str.contains(msg_search, case=False, na=False)]

            analysis = analyze_message_patterns(msg_df, speaker_filter=speaker_filter, min_len=min_len)
            working = analysis["working"]

            total_messages = len(working)
            total_words = int(working["word_count"].sum()) if "word_count" in working else 0
            total_conversations = int(working["conversation_id"].nunique()) if "conversation_id" in working else 0
            total_files = int(working["source_file"].nunique()) if "source_file" in working else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Messages", f"{total_messages:,}")
            m2.metric("Conversations", f"{total_conversations:,}")
            m3.metric("Words", f"{total_words:,}")
            m4.metric("Files", f"{total_files:,}")

            st.subheader("You vs Assistant")
            user_messages = int((working["author_role"] == "user").sum())
            assistant_messages = int((working["author_role"] == "assistant").sum())
            user_words = int(working.loc[working["author_role"] == "user", "word_count"].sum()) if total_words else 0
            assistant_words = int(working.loc[working["author_role"] == "assistant", "word_count"].sum()) if total_words else 0

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
                display_cols = ["source_file", "conversation_title", "author_role", "created_at", "word_count", "text"]
                st.dataframe(
                    working.sort_values("created_at", na_position="last")[display_cols],
                    use_container_width=True,
                    height=650,
                )

            csv_data = working.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download parsed messages as CSV",
                data=csv_data,
                file_name="inboxgpt_message_patterns.csv",
                mime="text/csv",
            )

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

    selected_ids = []
    for idx, chat in enumerate(visible):
        if _chat_row(chat, key_prefix=f"row{start+idx}", analyzed_pool=analyzed):
            selected_ids.append(chat["id"])

    st.markdown("---")
    st.markdown("### Categorize Selected")

    if selected_ids:
        combined_parts: List[str] = []
        for cid in selected_ids:
            ch = get_chat(cid)
            if ch:
                combined_parts.append(f"{ch.get('title') or ''}\n{ch.get('content') or ''}")
        combined_text = "\n\n".join(combined_parts)
        suggested, _scores = guess_topics(combined_text)
        if suggested:
            st.markdown("### Suggested tags")
            st.caption("Based on the selected chats’ topics (deterministic rules).")
            st.multiselect(
                "Suggested tags",
                options=suggested,
                default=suggested,
                key="inboxgpt_suggested_tags",
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
            suggested_selected = st.session_state.get("inboxgpt_suggested_tags") or []
            names += list(suggested_selected)
            names = [n.strip() for n in names if n and n.strip()]
            names = list(dict.fromkeys(names))  # stable de-dupe
            assign_categories(selected_ids, names)
            st.success(f"Assigned {', '.join(names)} to {len(selected_ids)} chat(s).")
            st.rerun()

    with cat_right:
        all_cats = list_categories()
        remove_existing = st.multiselect("Remove categories from selected", options=[c["name"] for c in all_cats])
        if st.button("Remove from selected"):
            remove_categories(selected_ids, remove_existing)
            st.warning(f"Removed {', '.join(remove_existing)} from {len(selected_ids)} chat(s).")
            st.rerun()

    st.markdown("### Category Summary")
    cats = list_categories()
    cols = st.columns(4)
    for i, cat in enumerate(cats):
        with cols[i % 4]:
            st.metric(cat["name"], f"{cat['count']} chats")


if __name__ == "__main__":
    # Allows running this module directly too:
    main(go_home=None)
