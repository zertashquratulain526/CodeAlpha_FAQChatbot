"""
chatbot.py
----------
Core logic of the CodeAlpha FAQ Chatbot.

Pipeline
--------
1. Load FAQs (question + answer pairs) from faqs.json
2. Pre-process text with NLTK  (lower-case -> tokenize -> remove stop-words -> stem)
3. Convert the FAQ questions to TF-IDF vectors
4. For every user message compute the COSINE SIMILARITY with all FAQs
5. Return the answer of the most similar FAQ (or a polite fallback + suggestions)

Run in the terminal:
    python chatbot.py
"""

import difflib
import json
import os
import re
from dataclasses import dataclass, field

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# NLP helpers
# --------------------------------------------------------------------------

_tokenizer = RegexpTokenizer(r"[a-z0-9]+")  # keeps only words / numbers (removes punctuation)
_stemmer = PorterStemmer()


def _load_stopwords() -> set:
    """Use NLTK's stop-word list (downloaded once automatically). If the NLTK
    data cannot be downloaded (e.g. no internet) fall back to scikit-learn's list."""
    try:
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:
        return set(ENGLISH_STOP_WORDS)


# Words that look like stop-words but carry meaning in an FAQ
STOP_WORDS = _load_stopwords() - {"not", "no", "after", "before", "free"}


# Domain synonyms: different words that mean the same thing for this shop
SYNONYMS = {
    "parcel": "order", "package": "order", "shipment": "order", "purchase": "order",
    "broken": "damaged", "defective": "damaged", "faulty": "damaged", "scratched": "damaged",
    "abroad": "international", "overseas": "international",
    "helpline": "support", "helpdesk": "support",
    "voucher": "coupon", "register": "signup",
    "didnt": "not", "wont": "not", "cant": "not",
}

# Filled by FAQChatbot with every word that appears in the FAQ file; used to fix typos
_VOCAB: set = set()


def _correct_typos(tokens: list) -> list:
    """Replace unknown words by the closest known word ('trak' -> 'track')."""
    fixed = []
    for tok in tokens:
        if _VOCAB and tok not in _VOCAB and tok not in STOP_WORDS and len(tok) >= 4:
            close = difflib.get_close_matches(tok, _VOCAB, n=1, cutoff=0.75)
            tok = close[0] if close else tok
        fixed.append(tok)
    return fixed


def preprocess(text: str) -> str:
    """lower-case -> tokenize -> fix typos -> synonyms -> remove stop-words -> stem."""
    tokens = _tokenizer.tokenize(text.lower())
    tokens = _correct_typos(tokens)
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Small-talk (intent matching)
# --------------------------------------------------------------------------

SMALL_TALK = [
    (re.compile(r"\b(hi|hello|hey|salam|assalam\w*|good (morning|afternoon|evening))\b"),
     "Hello! 👋 I'm the ShopEasy assistant. Ask me anything about orders, delivery, payments or returns."),
    (re.compile(r"\b(thanks|thank you|thx|shukriya)\b"),
     "You're welcome! Is there anything else I can help you with?"),
    (re.compile(r"\b(bye|goodbye|see you|allah hafiz|khuda hafiz)\b"),
     "Goodbye! Have a great day 😊"),
]

FALLBACK = ("Sorry, I couldn't find an answer to that. "
            "Please try rephrasing your question or contact our support team at support@shopeasy.example.")


# --------------------------------------------------------------------------
# Chatbot
# --------------------------------------------------------------------------

@dataclass
class BotReply:
    answer: str
    matched_question: str = ""
    score: float = 0.0
    suggestions: list = field(default_factory=list)


class FAQChatbot:
    """
    Every FAQ has one main question and (optionally) a few alternative
    phrasings ("variants"). All of them are turned into TF-IDF vectors and the
    user's message is compared to each with COSINE SIMILARITY; the FAQ whose
    best phrasing is most similar wins.

    Two kinds of TF-IDF vectors are combined:
      * word n-grams  -> understands meaning / keywords
      * character n-grams -> tolerant to typos and word forms (pay / payment)
    """

    def __init__(self, faq_path: str = None, threshold: float = 0.25):
        faq_path = faq_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "faqs.json")
        with open(faq_path, encoding="utf-8") as f:
            data = json.load(f)

        self.topic = data.get("topic", "FAQ")
        self.faqs = data["faqs"]
        self.threshold = threshold

        self.questions = [item["question"] for item in self.faqs]

        # vocabulary of the FAQ file (for typo correction)
        _VOCAB.clear()
        for item in self.faqs:
            for text in [item["question"], item["answer"]] + item.get("variants", []):
                _VOCAB.update(w for w in _tokenizer.tokenize(text.lower()) if len(w) >= 4)
        self.answers = [item["answer"] for item in self.faqs]

        # every phrasing of every FAQ -> (text, index of the FAQ it belongs to)
        self.patterns, self.pattern_to_faq = [], []
        for i, item in enumerate(self.faqs):
            for text in [item["question"]] + item.get("variants", []):
                self.patterns.append(preprocess(text))
                self.pattern_to_faq.append(i)
        self.pattern_to_faq = np.array(self.pattern_to_faq)

        # question + answer text (smaller weight): helps with words that only
        # appear in the answer, e.g. "wallet" or "invoice"
        self.qa_docs = [preprocess(q + " " + a) for q, a in zip(self.questions, self.answers)]

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2)).fit(self.patterns + self.qa_docs)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(self.patterns + self.qa_docs)

        self.word_pat = self.word_vec.transform(self.patterns)
        self.char_pat = self.char_vec.transform(self.patterns)
        self.word_qa = self.word_vec.transform(self.qa_docs)

    # ------------------------------------------------------------------
    def _best_per_faq(self, sims: np.ndarray) -> np.ndarray:
        """Highest similarity among all phrasings that belong to the same FAQ."""
        best = np.zeros(len(self.faqs))
        np.maximum.at(best, self.pattern_to_faq, sims)
        return best

    def _similarities(self, user_text: str) -> np.ndarray:
        cleaned = preprocess(user_text)
        word_sim = self._best_per_faq(cosine_similarity(self.word_vec.transform([cleaned]), self.word_pat)[0])
        char_sim = self._best_per_faq(cosine_similarity(self.char_vec.transform([cleaned]), self.char_pat)[0])
        qa_sim = cosine_similarity(self.word_vec.transform([cleaned]), self.word_qa)[0]
        return 0.6 * word_sim + 0.25 * char_sim + 0.15 * qa_sim

    def get_reply(self, user_text: str) -> BotReply:
        user_text = (user_text or "").strip()
        if not user_text:
            return BotReply("Please type a question 🙂")

        lowered = user_text.lower()
        # Only treat very short messages as small-talk, so real questions
        # like "hello, how do I track my order?" still go to the FAQ matcher.
        if len(lowered.split()) <= 4:
            for pattern, response in SMALL_TALK:
                if pattern.search(lowered):
                    return BotReply(response)

        scores = self._similarities(user_text)
        ranked = scores.argsort()[::-1]
        best = int(ranked[0])

        if scores[best] < self.threshold:
            suggestions = [self.questions[i] for i in ranked[:3] if scores[i] > 0]
            return BotReply(FALLBACK, score=float(scores[best]), suggestions=suggestions)

        return BotReply(self.answers[best], self.questions[best], float(scores[best]))


# --------------------------------------------------------------------------
# Terminal chat
# --------------------------------------------------------------------------

def main():
    bot = FAQChatbot()
    print(f"🤖 {bot.topic} FAQ Chatbot  (type 'quit' to exit)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit", "q"}:
            break
        reply = bot.get_reply(user)
        print(f"Bot: {reply.answer}")
        if reply.matched_question:
            print(f"     (matched: \"{reply.matched_question}\" | similarity {reply.score:.2f})")
        if reply.suggestions:
            print("     Did you mean: " + " | ".join(reply.suggestions))
        print()
    print("Bot: Goodbye!")


if __name__ == "__main__":
    main()
