"""
Message-level parsing + deterministic pattern analysis for InboxGPT.

Designed to work local-first:
- Prefer analyzing chats already stored in SQLite (via InboxGPT's `content` field)
- Provide an optional parser for raw `conversations.json` mapping exports (no DB required)

No external APIs, no AI calls.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

# --- stopwords / tokenization (ported from chat_pattern_explorer.py) ---

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


def remove_code(text: str) -> str:
    """
    Remove common code formatting so token analysis reflects natural language.

    - Strips triple-backtick fenced blocks: ``` ... ```
    - Strips inline code spans: `...`
    """
    if not text:
        return ""
    s = str(text)
    # Triple-backtick blocks (including optional language).
    s = re.sub(r"```[\s\S]*?```", " ", s)
    # Inline backticks.
    s = re.sub(r"`[^`]*`", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_code_token(token: str) -> bool:
    """
    Heuristic: identify tokens that are more likely code-ish than natural language.
    """
    if not token:
        return False
    t = token.strip().lower()
    if "_" in t:
        return True
    if any(ch.isdigit() for ch in t):
        return True
    if len(t) > 20:
        return True
    if t.startswith(("def", "class", "return", "import")):
        return True
    return False


def tokenize(text: str, min_len: int = 3, drop_stopwords: bool = True) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def build_ngrams(tokens: List[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


# --- defensive raw-export parsing helpers (ported + adapted) ---

_ROLE_LINE_RE = re.compile(r"^\s*\[(user|assistant|system|tool)\]\s*", re.I)


def flatten_text_parts(part: Any) -> str:
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
            return " ".join(flatten_text_parts(p) for p in part["parts"])
        return ""
    if isinstance(part, list):
        return " ".join(flatten_text_parts(p) for p in part)
    return ""


def extract_author_role(message: Dict[str, Any], node: Dict[str, Any] | None = None) -> str:
    """
    Defensively extract a message author role from multiple export shapes.

    Checks, in order:
    - message["author"]["role"]
    - message["author"]["name"]
    - message["metadata"]["author_role"]
    - message["metadata"]["role"]
    - node["author"]["role"]
    - fallback "unknown"
    """

    def _get_str(v: Any) -> str:
        return v.strip() if isinstance(v, str) else ""

    raw = ""
    author = message.get("author") or {}
    raw = _get_str((author or {}).get("role"))
    if not raw:
        raw = _get_str((author or {}).get("name"))
    if not raw:
        meta = message.get("metadata") or {}
        raw = _get_str((meta or {}).get("author_role"))
    if not raw:
        meta = message.get("metadata") or {}
        raw = _get_str((meta or {}).get("role"))
    if not raw and node:
        raw = _get_str(((node.get("author") or {}).get("role")))

    v = (raw or "unknown").strip().lower()
    if "assistant" in v or "gpt" in v:
        return "assistant"
    if "user" in v or "human" in v:
        return "user"
    if "system" in v:
        return "system"
    if "tool" in v:
        return "tool"
    return v or "unknown"


def extract_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")

    if isinstance(content, dict):
        if "parts" in content:
            return flatten_text_parts(content.get("parts")).strip()
        for k in ("text", "result", "content"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        flattened = flatten_text_parts(content)
        if flattened.strip():
            return flattened.strip()

    v = message.get("text")
    if isinstance(v, str) and v.strip():
        return v.strip()

    if isinstance(content, str) and content.strip():
        return content.strip()

    if content is not None:
        flattened = flatten_text_parts(content)
        if flattened.strip():
            return flattened.strip()

    return ""


def parse_mapping_export_bytes(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    Parse raw ChatGPT `conversations.json`-style mapping exports into message rows.

    Output columns:
    - source_file, conversation_id, conversation_title, message_id, author_role, created_at, text, char_count, word_count
    """
    data = json.loads(raw_bytes.decode("utf-8"))
    rows: List[Dict[str, Any]] = []

    conversations = data if isinstance(data, list) else [data]
    for conv_idx, conv in enumerate(conversations):
        title = conv.get("title") or f"Conversation {conv_idx + 1}"
        mapping = conv.get("mapping", {}) or {}

        for node_id, node in mapping.items():
            message = (node or {}).get("message")
            if not message:
                continue

            author = extract_author_role(message, node)
            text = extract_message_text(message)
            if not text:
                continue

            created_at = message.get("create_time")
            rows.append(
                {
                    "source_file": file_name,
                    "conversation_id": conv.get("conversation_id") or conv.get("id") or f"{file_name}:{conv_idx}",
                    "conversation_title": title,
                    "message_id": message.get("id") or node_id,
                    "author_role": author,
                    "created_at": pd.to_datetime(created_at, unit="s", errors="coerce") if created_at else pd.NaT,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )

    return pd.DataFrame(rows)


# --- inboxgpt stored-chat -> message rows ---

def _split_normalized_content_into_messages(content: str) -> List[Tuple[str, str]]:
    """
    Split InboxGPT-normalized content that looks like:
      [user] ...

      [assistant] ...
    into [(role, text), ...]

    If markers are absent, returns a single ('unknown', content) message.
    """
    if not content:
        return []

    lines = content.splitlines()
    current_role = None
    buf: List[str] = []
    out: List[Tuple[str, str]] = []

    def flush():
        nonlocal buf, current_role
        if current_role and buf:
            text = "\n".join(buf).strip()
            if text:
                out.append((current_role, text))
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
    if out:
        return out

    text = content.strip()
    return [("unknown", text)] if text else []


def messages_from_stored_chats(chats: List[Dict[str, Any]], source_name: str = "SQLite") -> pd.DataFrame:
    """
    Build a message-level dataframe from InboxGPT's stored chats.

    Expects each chat dict to include: id, title, created_at, content.
    """
    rows: List[Dict[str, Any]] = []
    for chat in chats:
        cid = chat.get("id")
        title = chat.get("title") or "(untitled)"
        created_at = pd.to_datetime(chat.get("created_at"), errors="coerce")
        content = str(chat.get("content") or "")

        parts = _split_normalized_content_into_messages(content)
        for idx, (role, text) in enumerate(parts):
            rows.append(
                {
                    "source_file": source_name,
                    "conversation_id": cid,
                    "conversation_title": title,
                    "message_id": f"{cid}:{idx}",
                    "author_role": role,
                    "created_at": created_at if pd.notna(created_at) else pd.NaT,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )

    return pd.DataFrame(rows)


def analyze_message_patterns(
    df: pd.DataFrame,
    speaker_filter: str = "all",
    min_len: int = 3,
    *,
    exclude_code: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute deterministic message-level pattern tables.
    Returns:
      words (top 100), bigrams (top 50), trigrams (top 50),
      conversations (top 50 by message_count),
      months (sorted by YYYY-MM),
      working (filtered df)
    """
    if df.empty:
        return {
            "words": pd.DataFrame(columns=["term", "count"]),
            "bigrams": pd.DataFrame(columns=["term", "count"]),
            "trigrams": pd.DataFrame(columns=["term", "count"]),
            "conversations": pd.DataFrame(columns=["conversation", "message_count"]),
            "months": pd.DataFrame(columns=["month", "message_count"]),
            "working": df.copy(),
        }

    working = df.copy()
    if speaker_filter != "all":
        working = working[working["author_role"] == speaker_filter]

    token_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    trigram_counter: Counter = Counter()
    convo_counter: Counter = Counter()
    monthly_counter: Counter = Counter()

    for _, row in working.iterrows():
        text = str(row.get("text") or "")
        if exclude_code:
            text = remove_code(text)
        tokens = tokenize(text, min_len=min_len, drop_stopwords=True)
        if exclude_code:
            tokens = [t for t in tokens if not is_code_token(t)]
        token_counter.update(tokens)
        bigram_counter.update(build_ngrams(tokens, 2))
        trigram_counter.update(build_ngrams(tokens, 3))
        convo_counter.update([row.get("conversation_title") or "(untitled)"])
        if pd.notna(row.get("created_at")):
            try:
                monthly_counter.update([pd.to_datetime(row["created_at"]).strftime("%Y-%m")])
            except Exception:
                pass

    months_df = (
        pd.DataFrame(monthly_counter.most_common(), columns=["month", "message_count"]).sort_values("month")
        if monthly_counter
        else pd.DataFrame(columns=["month", "message_count"])
    )

    return {
        "words": pd.DataFrame(token_counter.most_common(100), columns=["term", "count"]),
        "bigrams": pd.DataFrame(bigram_counter.most_common(50), columns=["term", "count"]),
        "trigrams": pd.DataFrame(trigram_counter.most_common(50), columns=["term", "count"]),
        "conversations": pd.DataFrame(convo_counter.most_common(50), columns=["conversation", "message_count"]),
        "months": months_df,
        "working": working,
    }

