# 🤖 CodeAlpha_FAQChatbot

**CodeAlpha Artificial Intelligence Internship – Task 2: Chatbot for FAQs**

A chatbot for an online store (**ShopEasy**) that answers customer questions by finding the most similar FAQ using **NLP pre-processing (NLTK)** and **TF-IDF + cosine similarity**.

## How it works

```
User question
   │
   ▼
Pre-processing (NLTK):  lower-case → tokenize → typo correction → synonyms
                        → stop-word removal → Porter stemming
   │
   ▼
TF-IDF vectors  (word n-grams + character n-grams)
   │
   ▼
Cosine similarity with every FAQ question / alternative phrasing
   │
   ▼
Best match ≥ threshold ?  ──yes──►  show the FAQ answer
                          └─no───►  polite fallback + "Did you mean…" suggestions
```

### Features
- 24 FAQs about orders, payments, delivery, returns, refunds, accounts, support (`faqs.json`)
- Each FAQ can have extra phrasings (`variants`) to understand different ways of asking
- NLTK pre-processing + small typo corrector (`trak my oder` → `track my order`)
- Handles greetings / thanks / goodbye (small-talk intents)
- Off-topic questions are rejected with a helpful fallback and suggestions
- **Chat UI** (Streamlit) and a **terminal chat** version
- `test_chatbot.py` – measures accuracy on 22 re-worded questions + 3 off-topic questions
  (≈ 68 % exact-match on this deliberately tricky set; add more `variants` to improve it)

## Project structure

```
CodeAlpha_FAQChatbot/
├── faqs.json          # the FAQ data (questions, variants, answers) – edit to change the topic
├── chatbot.py         # NLP pipeline + matching logic (also a terminal chat)
├── app.py             # Streamlit chat interface
├── test_chatbot.py    # accuracy test
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# chat interface in the browser
streamlit run app.py

# or chat in the terminal
python chatbot.py

# run the accuracy test
python test_chatbot.py
```

NLTK downloads its English stop-word list automatically on the first run (internet needed once).
If it cannot be downloaded, the bot silently uses scikit-learn's stop-word list instead.

## Use your own FAQs

Open `faqs.json`, change `topic` and replace the items:

```json
{"question": "Your question?", "variants": ["other way to ask it"], "answer": "Your answer."}
```

## Technologies

Python · NLTK · scikit-learn (TF-IDF, cosine similarity) · Streamlit
