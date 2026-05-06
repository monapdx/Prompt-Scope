# 📁 Prompt Scope

**Turn your ChatGPT history into something you can actually explore.**

Prompt Scope is a local-first Streamlit app that lets you **import, analyze, categorize, and extract insights** from your ChatGPT conversations — without sending your data anywhere.

---

## ✨ What It Does

Prompt Scope transforms raw ChatGPT exports into a structured, searchable, analyzable dataset.

* 📥 Import your ChatGPT `conversations.json`
* 🗂️ Organize chats with categories
* 🔍 Search across titles + full content
* 📊 Generate insights about how you use ChatGPT
* 🧠 Detect topics, keywords, and patterns
* 🔎 Analyze message-level behavior (words, phrases, trends)
* 📤 Export everything back to JSON

All processing happens **locally on your machine**.

---

## 🧠 Why This Exists

ChatGPT stores your conversations, but it doesn’t help you understand them.

Prompt Scope answers questions like:

* *What do I actually use ChatGPT for?*
* *What topics come up over and over?*
* *Which conversations go deep vs. quick hits?*
* *What patterns exist in how I write or prompt?*

It turns your chat history into something closer to:

* a dataset
* a personal archive
* a reflection tool

---

## 🏗️ How It Works

### 1. Import

Upload your ChatGPT export JSON.

Prompt Scope:

* Normalizes different export formats
* Extracts messages (even messy nested structures)
* Stores everything in a local SQLite database

---

### 2. Explore

Browse your conversations with:

* Search (title + content)
* Category filters
* Sorting (newest, oldest, A→Z)
* Inline previews

---

### 3. Analyze

#### 🔹 Insights Tab

* Topic detection (rule-based, no AI calls)
* Keyword extraction
* Conversation depth analysis
* Global summaries of your usage

Powered by deterministic logic (no external APIs)



---

#### 🔹 Patterns Tab

Message-level analysis:

* Most common words
* Bigrams / trigrams
* Conversation frequency
* Monthly usage trends

Includes:

* Code filtering
* Speaker filtering (user vs assistant)
* Token cleaning

<img src="https://raw.githubusercontent.com/monapdx/Prompt-Scope/refs/heads/main/patterns.png">

---

### 4. Organize

* Create categories
* Assign chats to multiple categories
* Filter by combinations

Everything is stored locally in SQLite.

---

### 5. Export

Export your enriched dataset:

```json
{
  "schema_version": 1,
  "exported_at": "...",
  "categories": [...],
  "chats": [...]
}
```

---

## 🔐 Privacy First

* No API calls
* No tracking
* No cloud sync
* No external dependencies

Your data never leaves your machine.

---

## ⚡ Performance Design

Prompt Scope is optimized for large chat histories:

* Content truncation for fast UI rendering
* Cached analysis results
* Lazy loading (analysis only runs when requested)

Key limits:

* `MAX_ANALYSIS_CHARS = 20,000`
* `MAX_PREVIEW_CHARS = 4,000`

→ see 

---

## 🧱 Tech Stack

* **Frontend/UI:** Streamlit
* **Storage:** SQLite (local file)
* **Data processing:** Pandas
* **Visualization:** Plotly / Altair
* **Language:** Python 3

Dependencies:
→ see 

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/prompt-scope.git
cd prompt-scope
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
prompt-scope/
├── app.py                      # Main Streamlit app
├── promptscope.db              # Local SQLite database
├── conversation_insights.py    # Topic + insight engine
├── chat_patterns.py            # Message-level analysis
├── requirements.txt
```

---

## 🧩 Key Features (Deep Dive)

### 🧠 Deterministic Topic Detection

No AI, no embeddings — just rule-based scoring.

* Keyword pattern matching
* Multi-hit thresholding
* Topic confidence scoring

---

### 🔍 Robust Chat Parsing

Handles messy real-world exports:

* Nested JSON structures
* `mapping`-based exports
* Transcript-style `[user]` / `[assistant]` logs
* Mixed formats

---

### 🧼 Noise Reduction

Removes:

* JSON structural tokens (`content`, `parts`, etc.)
* Code blocks (optional)
* Stopwords

So your analysis reflects **actual human language**.

---

### ⚡ Local-First Architecture

* SQLite for persistence
* No server required
* Works offline
* Instant startup after import

---

## 🧪 Use Cases

* Personal knowledge mining
* Prompt engineering analysis
* Writing pattern discovery
* AI usage reflection
* Research / journaling analysis
* Dataset generation for future tools

---

## 🔮 Future Ideas

* Timeline visualizations
* Conversation clustering
* Prompt quality scoring
* Export to CSV / embeddings-ready formats
* GitHub-style diff for conversations

---

## 🤝 Contributing

Contributions are welcome — especially:

* Better topic rules
* New analysis modules
* UI improvements
* Export formats

---

## 📜 License

MIT

---

## 💭 Final Thought

ChatGPT remembers everything.

Prompt Scope helps **you understand it.**

