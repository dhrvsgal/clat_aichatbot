# 🧠 CLAT Query Chatbot

An intelligent chatbot designed to answer user queries related to the **Common Law Admission Test (CLAT)** using a hybrid search model and a Streamlit interface.

---

---    [🚀 Scaling the CLAT Query Chatbot to a GPT-Based Model](#-scaling-the-clat-query-chatbot-to-a-gpt-based-model-fine-tuned-on-nlti-content)


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

# 🚀 Scaling the CLAT Query Chatbot to a GPT-Based Model (Fine-Tuned on NLTI Content)

## 📄 Overview

This guide describes how to evolve the current CLAT chatbot—which uses TF-IDF and Sentence Transformers—into a powerful GPT-based model, fine-tuned on NLTI’s proprietary educational content. This will provide users with highly contextual, natural, and dynamic answers to their queries related to CLAT and other law-related content.

---

## 🔁 Comparison: Current Model vs GPT-Based Model

| Feature              | Current Chatbot                     | GPT-Based Model (Fine-Tuned)                          |
|----------------------|-------------------------------------|--------------------------------------------------------|
| Retrieval Method     | TF-IDF + Sentence Transformers      | Fine-tuned GPT LLM (e.g., GPT-3.5 / LLaMA2)            |
| Answer Source        | Predefined answers in JSON format   | Dynamically generated answers from trained content     |
| Context Handling     | One-shot answers                    | Multi-turn conversations with memory/context           |
| Adaptability         | Limited to trained questions        | Generalizes to new, unseen questions                   |
| Personalization      | No                                  | Possible with session tracking and user-specific data  |
| Deployment           | Local / Streamlit                   | OpenAI API or locally hosted LLM                       |

---

## 🧱 Steps to Scale to a GPT-Based Model

### 1. 🗂️ Data Collection

Prepare a structured dataset based on NLTI’s content:
- CLAT Q&A material
- Definitions, concepts, tips & tricks
- Legal case summaries
- Past year paper explanations

Format it for fine-tuning as `prompt-completion` pairs in `.jsonl`:

```json
{"prompt": "What is CLAT?", "completion": "CLAT stands for Common Law Admission Test. It is a centralized national-level entrance test for admissions to National Law Universities in India."}

### 2. 🧠 Fine-Tuning the Model
#### Option A: Using OpenAI
Use OpenAI CLI to upload and fine-tune GPT-3.5 Turbo.

Documentation: https://platform.openai.com/docs/guides/fine-tuning

#### Option B: Using Open-Source LLMs
Use Hugging Face Transformers with a base model like LLaMA2, Mistral, Falcon, or Phi-2.

Apply parameter-efficient fine-tuning (LoRA/QLoRA).

Tools: PEFT, Transformers, bitsandbytes.

## 🖥️ Deployment Options
Option	                                              Description
OpenAI	                                              Easiest to integrate; fully managed
HuggingFace              	                            Flexible and free if hosted locally
Modal/Replicate	                                      For hosted open-source LLMs
NVIDIA Jetson / On-Prem GPU	                          For edge or secure private use

## 🔐 Data Privacy and Ethics
Ensure compliance with NLTI’s data usage policies.

Mask any sensitive information during fine-tuning.

Prefer local fine-tuning and inference for confidential data.

## 📈 Future Roadmap
✅ Add RAG for factual accuracy

🎤 Add Whisper (Speech-to-Text) for voice queries

🗣️ Integrate TTS for spoken answers

👥 Track user sessions for personalized learning

📚 Expand dataset to cover other law entrance exams

🧪 Add mock test generation via GPT





