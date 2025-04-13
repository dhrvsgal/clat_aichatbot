# 🧠 CLAT Query Chatbot

An intelligent chatbot designed to answer user queries related to the **Common Law Admission Test (CLAT)** using a hybrid search model and a Streamlit interface.

---

## 🚀 Features

- ✅ Answers questions about CLAT syllabus, eligibility, exam pattern, preparation tips, and more.
- 🧠 Uses both **TF-IDF** and **Semantic Search (Sentence Transformers)** for accurate information retrieval.
- 💬 Interactive **chat interface** using **Streamlit**.
- 📚 Easy-to-update knowledge base stored in JSON format.
- 🧹 NLP-powered text preprocessing for cleaner inputs.

---

## 🛠️ Tech Stack

- Python
- NLTK (for text preprocessing)
- Scikit-learn (TF-IDF & cosine similarity)
- SentenceTransformers (semantic embeddings)
- Streamlit (frontend UI)
- JSON (data storage)

---

## 🧩 Architecture Overview

### 1. Knowledge Base
- A list of question-answer pairs about CLAT, defined in `clat_knowledge_base`.
- Also saved/loaded from a `clat_knowledge_base.json` file.

### 2. NLP Preprocessing
- Converts user input to lowercase.
- Removes punctuation and numbers.
- Tokenizes using NLTK.
- Removes stopwords.
- Lemmatizes words.

### 3. Information Retrieval Methods

#### 🔍 TF-IDF Search
- Vectorizes combined questions and answers using `TfidfVectorizer`.
- Uses cosine similarity to find top relevant matches.

#### 🧠 Semantic Search
- Embeds questions using a pretrained SentenceTransformer model (`all-MiniLM-L6-v2`).
- Computes semantic similarity between the user query and stored question embeddings.

### 4. Response Generation
- Combines semantic and TF-IDF results (semantic preferred).
- Filters out low-confidence results (similarity score < 0.3).
- Fallback strategies:
  - Keyword-based match
  - Generic intent-based reply (e.g., syllabus, cutoff, preparation tips)

### 5. Streamlit Web Interface
- Clean chat-style UI with:
  - Chat history display
  - Sample questions in the sidebar
  - Realtime Q&A interaction

---

## 🧪 Sample Questions

Try asking:
- "What is the syllabus for CLAT 2025?"
- "Give me last year's cut-off for NLSIU Bangalore."
- "How should I prepare for the Legal Reasoning section?"

---

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clat-chatbot.git
cd clat-chatbot
```
## 2. Install Dependencies
```
pip install -r requirements.txt
```
## 3. Download NLTK Data
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
## 4. Run the Streamlit App
```
streamlit run assignment_.py
```

# 📌 Future Improvements
Add more Q&A pairs for broader coverage.

Integrate with a database or external source for dynamic updates.

Implement voice-based interaction or API access.

