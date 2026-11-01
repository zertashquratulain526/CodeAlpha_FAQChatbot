"""
app.py
------
CodeAlpha AI Internship - Task 2: Chatbot for FAQs  (chat user interface)

Run with:
    streamlit run app.py
"""

import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")


@st.cache_resource
def load_bot() -> FAQChatbot:
    return FAQChatbot()


bot = load_bot()

st.title(f"🤖 {bot.topic} – FAQ Chatbot")
st.caption("CodeAlpha AI Internship · Task 2 — ask me about orders, delivery, payments, returns and more.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Try asking")
    examples = [
        "Where is my parcel?",
        "Can I pay with JazzCash?",
        "How do I return a product?",
        "My order arrived broken",
        "Is shipping free?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------- history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! 👋 I'm the ShopEasy assistant. How can I help you today?"}
    ]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------- new message
user_input = st.chat_input("Type your question here...") or st.session_state.pop("pending", None)

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = bot.get_reply(user_input)
    text = reply.answer
    if reply.suggestions:
        text += "\n\n**Did you mean:**\n" + "\n".join(f"- {s}" for s in reply.suggestions)

    st.session_state["messages"].append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)
        if reply.matched_question:
            st.caption(f"Matched FAQ: “{reply.matched_question}” · similarity {reply.score:.2f}")
