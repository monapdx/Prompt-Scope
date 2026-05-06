"""
Deterministic conversation insights and topic detection (no external APIs).

This module is intentionally lightweight and beginner-friendly:
- keyword extraction uses simple frequency + stopword filtering
- topic detection uses rule-based keyword matching (no AI calls)
- similarity matching is deterministic (shared topics/keywords/title words)
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "us",
        "my",
        "your",
        "their",
        "our",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "then",
        "once",
        "if",
        "because",
        "until",
        "while",
        "although",
        "though",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "again",
        "any",
        "own",
        "off",
        "out",
        "up",
        "down",
        "over",
        "am",
        "im",
        "dont",
        "didnt",
        "wont",
        "cant",
        "isnt",
        "wasnt",
        "arent",
        "theyre",
        "youre",
        "thats",
        "theres",
        "heres",
        "lets",
        "etc",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.M)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_CODEY_LINE_RE = re.compile(
    r"^\s*(def |class |import |from |const |let |var |function |if\s*\(|for\s*\(|while\s*\(|return\b|SELECT\b|INSERT\b|UPDATE\b|DELETE\b)",
    re.I,
)
_ROLE_LINE_RE = re.compile(r"^\s*\[(user|assistant|system|tool)\]\s*", re.I | re.M)


def _strip_code(text: str) -> str:
    """
    Remove common code formatting so keyword frequency reflects natural language.

    This is best-effort (not a full markdown parser).
    """
    if not text:
        return ""
    text = _FENCED_CODE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    # Drop lines that look like code, so we don't get tokens like "classmethod" or "useeffect".
    kept: list[str] = []
    for line in text.splitlines():
        if _CODEY_LINE_RE.search(line):
            continue
        symbol_heavy = sum(1 for ch in line if ch in "{}[]();=<>/*+-_")
        if len(line) >= 40 and (symbol_heavy / max(1, len(line))) > 0.18:
            continue
        kept.append(line)
    return "\n".join(kept)


def _title_tokens(title: str) -> set[str]:
    t = (title or "").lower()
    toks = {m.group(0) for m in _WORD_RE.finditer(t) if len(m.group(0)) >= 4 and m.group(0) not in _STOPWORDS}
    return toks


def extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction: split, frequency, stopwords removed; returns top 5–10 tokens."""
    if not text:
        return []
    text = _strip_code(text)
    counts: Counter[str] = Counter()
    for m in _WORD_RE.finditer(text.lower()):
        w = m.group(0)
        if len(w) < 3 or w in _STOPWORDS:
            continue
        # Heuristics to reduce code identifiers.
        if "_" in w:
            continue
        if any(ch.isdigit() for ch in w) and len(w) > 4:
            continue
        counts[w] += 1
    if not counts:
        return []
    top = [w for w, _ in counts.most_common(10)]
    return top[:10] if len(top) >= 10 else top[: max(5, len(top))]


def _score_topic(text_lc: str, patterns: Iterable[str]) -> int:
    """
    Count how many distinct keyword patterns appear in the text (word-boundary where sensible).

    A topic is included when it has enough supporting matches (not a single fragile keyword).
    """
    score = 0
    for p in patterns:
        if p.startswith("re:"):
            if re.search(p[3:], text_lc, re.I):
                score += 1
        else:
            # Prefer word boundaries for single words. For phrases, substring is fine.
            if " " in p:
                if p in text_lc:
                    score += 1
            else:
                if re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text_lc):
                    score += 1
    return score


_TOPIC_RULES: dict[str, dict[str, Any]] = {
    "AI / Prompting": {
        "min_score": 2,
        "patterns": [
            "prompt",
            "prompting",
            "system prompt",
            "instruction",
            "few-shot",
            "chain of thought",
            "rag",
            "retrieval",
            "llm",
            "gpt",
            "openai",
            "anthropic",
            "claude",
            "token",
            "temperature",
            "model",
            "fine-tune",
            "re:((?<![a-z0-9])ai(?![a-z0-9]))",
        ],
    },
    "Writing": {
        "min_score": 2,
        "patterns": ["write", "rewrite", "edit", "draft", "outline", "chapter", "story", "narrative", "tone", "voice"],
    },
    "Memoir / Personal Writing": {
        "min_score": 2,
        "patterns": ["memoir", "journal", "diary", "personal essay", "my story", "childhood", "family", "relationship"],
    },
    "Frontend": {
        "min_score": 2,
        "patterns": ["react", "component", "jsx", "tsx", "css", "tailwind", "frontend", "ui", "dom", "nextjs", "hooks"],
    },
    "Backend": {
        "min_score": 2,
        "patterns": ["api", "endpoint", "backend", "server", "auth", "jwt", "oauth", "middleware", "flask", "fastapi", "django"],
    },
    "Data / SQL": {
        "min_score": 2,
        "patterns": ["sql", "sqlite", "postgres", "database", "query", "schema", "index", "join", "table", "etl"],
    },
    "GitHub / Dev Tools": {
        "min_score": 2,
        "patterns": ["github", "git", "pull request", "pr", "commit", "branch", "merge", "rebase", "ci", "actions", "docker", "kubernetes"],
    },
    "Streamlit / Python Apps": {
        "min_score": 2,
        "patterns": ["streamlit", "st.", "python", "pandas", "venv", "pip", "requirements.txt", "app.py", "sidebar"],
    },
    "Product Design": {
        "min_score": 2,
        "patterns": ["product", "ux", "ui", "user flow", "onboarding", "wireframe", "persona", "roadmap", "feature"],
    },
    "Marketing / Sales Copy": {
        "min_score": 2,
        "patterns": ["marketing", "landing page", "sales", "copy", "headline", "cta", "value prop", "positioning", "newsletter", "seo"],
    },
    "Personal Knowledge Management": {
        "min_score": 2,
        "patterns": ["pkm", "zettelkasten", "notes", "second brain", "obsidian", "notion", "knowledge base", "tags"],
    },
    "Research": {
        "min_score": 2,
        "patterns": ["research", "paper", "literature", "citations", "study", "hypothesis", "methodology", "sources", "experiment"],
    },
    "Legal / Policy": {
        "min_score": 2,
        "patterns": ["legal", "contract", "terms", "policy", "privacy", "gdpr", "compliance", "liability", "nda"],
    },
    "Health / Medical": {
        "min_score": 2,
        "patterns": ["health", "medical", "symptom", "diagnosis", "treatment", "doctor", "medication", "sleep", "diet", "pain"],
    },
    "Finance / Money": {
        "min_score": 2,
        "patterns": ["finance", "money", "budget", "invest", "investment", "stocks", "tax", "income", "savings", "interest"],
    },
    "Images / Design Assets": {
        "min_score": 2,
        "patterns": ["image", "images", "logo", "icon", "illustration", "design asset", "png", "svg", "photoshop", "figma"],
    },
    "Audio / Video / Media": {
        "min_score": 2,
        "patterns": ["audio", "video", "podcast", "transcript", "subtitle", "caption", "mp3", "wav", "mp4", "edit video"],
    },
    "Productivity / Planning": {
        "min_score": 2,
        "patterns": ["plan", "planning", "schedule", "weekly", "todo", "task", "prioritize", "goals", "habit", "routine"],
    },
    "Troubleshooting / Debugging": {
        "min_score": 2,
        "patterns": ["error", "bug", "debug", "fix", "stack trace", "exception", "traceback", "crash", "not working", "issue"],
    },
}


def guess_topics(text: str, title: str = "") -> tuple[list[str], dict[str, int]]:
    """
    Deterministic topic detection.

    Returns (topics, topic_scores). Topics are included when enough distinct patterns matched.
    """
    blob = f"{title}\n{text}"
    t = _strip_code(blob).lower()
    scores: dict[str, int] = {}
    for topic, rule in _TOPIC_RULES.items():
        score = _score_topic(t, rule["patterns"])
        if score:
            scores[topic] = score
    topics = [tp for tp, sc in scores.items() if sc >= int(_TOPIC_RULES[tp]["min_score"])]
    topics.sort(key=lambda tp: (-scores.get(tp, 0), tp))
    return topics, scores


def _estimate_turn_count(content: str) -> int:
    if not content:
        return 0
    # Prefer explicit [user]/[assistant] markers (our normalizer emits these).
    role_hits = _ROLE_LINE_RE.findall(content)
    if role_hits:
        return len(role_hits)
    # Fallback: estimate by paragraph breaks.
    chunks = [c for c in re.split(r"\n\s*\n+", content) if c.strip()]
    return max(1, min(200, len(chunks)))


def analyze_chat(chat: dict) -> dict:
    """
    Analyze one chat dict and return richer metadata:

    {
        "topics": list[str],
        "topic_scores": dict[str, int],
        "keywords": list[str],
        "length": int,
        "turn_count": int
    }
    """
    title = str(chat.get("title") or "")
    content = str(chat.get("content") or "")
    blob = f"{title}\n{content}"

    topics, topic_scores = guess_topics(content, title=title)
    keywords = extract_keywords(blob)
    turn_count = _estimate_turn_count(content)
    return {
        "topics": topics,
        "topic_scores": topic_scores,
        "keywords": keywords,
        "length": len(blob),
        "turn_count": turn_count,
    }


def _parse_created_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        iv = int(float(s))
        return datetime.utcfromtimestamp(iv)
    except (ValueError, TypeError, OverflowError, OSError):
        pass
    return None


def generate_top_summary(chats: list[dict]) -> str:
    """
    Deterministic 2–4 sentence summary about the user's ChatGPT history.
    Expects analyzed rows (optionally including title/created_at for context).
    """
    if not chats:
        return "There’s nothing to summarize yet. Import chats or relax filters to see insights."

    topic_counts: Counter[str] = Counter()
    lengths: list[int] = []
    turns: list[int] = []
    kw_chats: dict[str, set[Any]] = {}

    for idx, row in enumerate(chats):
        for tp in row.get("topics") or []:
            topic_counts[tp] += 1
        lengths.append(int(row.get("length") or 0))
        turns.append(int(row.get("turn_count") or 0))
        chat_key: Any = row.get("id", idx)
        for w in set(row.get("keywords") or []):
            kw_chats.setdefault(w, set()).add(chat_key)

    top_topics = [t for t, _ in topic_counts.most_common(3)]
    if top_topics:
        topic_phrase = ", ".join(top_topics[:2]) + (f", and {top_topics[2]}" if len(top_topics) >= 3 else "")
        s1 = f"You primarily use ChatGPT for {topic_phrase}."
    else:
        s1 = "You use ChatGPT across a mix of conversations, but the current view doesn’t strongly match any topic category."

    avg_len = sum(lengths) / max(1, len(lengths))
    top_len = max(lengths) if lengths else 0
    avg_turn = sum(turns) / max(1, len(turns))

    deep = (top_len >= 20000) or (avg_len >= 6000) or (avg_turn >= 18)
    if deep:
        s2 = "Your archive includes several longer, multi-turn threads, which suggests you often use it as a thinking partner—not just for quick answers."
    else:
        s2 = "Many conversations are relatively compact, which suggests you often use it for quick iterations and targeted help."

    recurring = sorted(((w, len(ids)) for w, ids in kw_chats.items() if len(ids) >= 3), key=lambda x: -x[1])[:5]
    if recurring:
        s3 = "A few themes repeat across your chats, which suggests recurring interests or ongoing projects you revisit over time."
        return " ".join([s1, s2, s3])
    return " ".join([s1, s2])


def generate_global_insights(chats: list[dict]) -> list[dict]:
    """
    Build product-style insights from analyzed rows.

    Expects each dict to include: topics, keywords, length, turn_count.
    Optional: id, title, created_at for richer outputs.
    """
    insights: list[dict] = []
    if not chats:
        return [
            {
                "title": "No conversations to analyze",
                "description": "Import chats or relax filters to populate insights.",
            }
        ]

    insights.append(
        {
            "title": "What this says about your ChatGPT history",
            "description": generate_top_summary(chats),
        }
    )

    topic_counts: Counter[str] = Counter()
    for row in chats:
        for tp in row.get("topics") or []:
            topic_counts[tp] += 1
    if topic_counts:
        top = topic_counts.most_common(6)
        lines = [f"- You use ChatGPT most often for **{name}**, which appears in **{cnt}** conversations." for name, cnt in top]
        insights.append(
            {
                "title": "What you use ChatGPT for",
                "description": "\n".join(lines),
            }
        )
    else:
        insights.append(
            {
                "title": "What you use ChatGPT for",
                "description": "No topic categories strongly matched in the current view. (Tip: topics are triggered by multiple supporting keywords, not a single word.)",
            }
        )

    longest = sorted(chats, key=lambda r: (r.get("turn_count") or 0, r.get("length") or 0), reverse=True)[:5]
    long_lines = []
    for row in longest:
        title = row.get("title") or "(untitled)"
        turns = int(row.get("turn_count") or 0)
        long_lines.append(f"- **{title}**: ~{turns} turns")
    insights.append(
        {
            "title": "Where you go deep",
            "description": "Your longest conversations tend to be where you work through complex projects or refine ideas over many turns.\n\n"
            + "\n".join(long_lines),
        }
    )

    kw_freq = Counter()
    for row in chats:
        for w in row.get("keywords") or []:
            kw_freq[w] += 1
    if kw_freq:
        lines = [f"- `{word}`: {cnt}" for word, cnt in kw_freq.most_common(15)]
        insights.append(
            {
                "title": "Words that keep coming up",
                "description": "These words show up repeatedly across your conversations (excluding code blocks where possible):\n\n"
                + "\n".join(lines),
            }
        )
    else:
        insights.append(
            {
                "title": "Words that keep coming up",
                "description": "No keywords met the frequency/stopword filters for this view.",
            }
        )

    month_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    for row in chats:
        dt = _parse_created_at(row.get("created_at"))
        if not dt:
            continue
        month_counts[dt.strftime("%Y-%m")] += 1
        year_counts[str(dt.year)] += 1
    if month_counts or year_counts:
        lines_m = [f"- {m}: {c} conversations" for m, c in sorted(month_counts.items())]
        lines_y = [f"- {y}: {c} conversations" for y, c in sorted(year_counts.items())]
        desc = "**By month**\n\n" + ("\n".join(lines_m) if lines_m else "_No monthly timestamps._")
        desc += "\n\n**By year**\n\n" + ("\n".join(lines_y) if lines_y else "_No yearly timestamps._")
        insights.append({"title": "When you used ChatGPT most", "description": desc})
    else:
        insights.append(
            {
                "title": "When you used ChatGPT most",
                "description": "No parsable `created_at` values on these rows (some exports omit timestamps).",
            }
        )

    kw_chats: dict[str, set[Any]] = {}
    for idx, row in enumerate(chats):
        chat_key: Any = row.get("id", idx)
        for w in set(row.get("keywords") or []):
            kw_chats.setdefault(w, set()).add(chat_key)
    recurring = sorted(((w, len(ids)) for w, ids in kw_chats.items() if len(ids) >= 3), key=lambda x: (-x[1], x[0]))[:15]
    if recurring:
        lines = [f"- `{word}` appears in **{n}** conversations" for word, n in recurring]
        insights.append(
            {
                "title": "Ideas you return to",
                "description": "These are ideas or terms that appear across many conversations, suggesting recurring interests or ongoing projects.\n\n"
                + "\n".join(lines),
            }
        )
    else:
        insights.append(
            {
                "title": "Ideas you return to",
                "description": "No extracted keyword appears across three or more conversations in the current result set.",
            }
        )

    return insights


def find_similar_conversations(target: dict, pool: list[dict], top_k: int = 3) -> list[dict]:
    """
    Deterministic similarity search (no embeddings):
    - shared topics (strong signal)
    - shared keywords
    - optional title word overlap

    Expects dicts in `pool` to include: id, title, created_at, topics, keywords.
    """
    target_id = target.get("id")
    t_topics = set(target.get("topics") or [])
    t_kws = set(target.get("keywords") or [])
    t_title = _title_tokens(str(target.get("title") or ""))

    scored: list[tuple[int, dict]] = []
    for row in pool:
        if not row:
            continue
        if target_id is not None and row.get("id") == target_id:
            continue
        r_topics = set(row.get("topics") or [])
        r_kws = set(row.get("keywords") or [])
        r_title = _title_tokens(str(row.get("title") or ""))

        overlap_topics = sorted(t_topics & r_topics)
        overlap_kws = sorted(t_kws & r_kws)
        overlap_title = sorted(t_title & r_title)

        score = 0
        score += 3 * len(overlap_topics)
        score += 1 * len(overlap_kws)
        score += 1 * min(2, len(overlap_title))

        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "id": row.get("id"),
                    "title": row.get("title") or "(untitled)",
                    "created_at": row.get("created_at"),
                    "overlap_topics": overlap_topics,
                    "overlap_keywords": overlap_kws[:8],
                },
            )
        )

    scored.sort(key=lambda x: (-x[0], str(x[1].get("title") or "")))
    return [r for _, r in scored[:top_k]]
