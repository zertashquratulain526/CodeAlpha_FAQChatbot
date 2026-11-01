"""
test_chatbot.py
---------------
Quick accuracy check: differently-worded questions must map to the right FAQ.
(The sentences below are NOT copies of the FAQ phrasings in faqs.json.)

Run with:  python test_chatbot.py
"""

from chatbot import FAQChatbot

# (user message, expected FAQ question)
CASES = [
    ("Where can I see the progress of my purchase?", "How do I track my order?"),
    ("I would like to follow my shipment", "How do I track my order?"),
    ("Can I use my Visa card to pay?", "What payment methods do you accept?"),
    ("Is paying through Easypaisa possible", "What payment methods do you accept?"),
    ("How many days until it reaches me?", "How long does delivery take?"),
    ("do i have to pay for delivery", "How much are the shipping charges?"),
    ("What is the procedure to send back an item?", "What is your return policy?"),
    ("how soon will the refund hit my account", "How do I get a refund?"),
    ("The phone I got is damaged", "What should I do if I received a damaged or wrong product?"),
    ("They sent me the wrong colour shoes", "What should I do if I received a damaged or wrong product?"),
    ("I cannot remember my password", "I forgot my password. How can I reset it?"),
    ("how do I open a new account", "How can I create an account?"),
    ("will you send parcels to Dubai", "Do you deliver internationally?"),
    ("until what time can I call your helpline", "What are your customer support working hours?"),
    ("I have a voucher, where do I type it", "How do I use a discount coupon or promo code?"),
    ("are these branded items real", "Are the products original and authentic?"),
    ("can I give the money to the courier", "Is Cash on Delivery available?"),
    ("please cancel the order I made yesterday", "Can I cancel my order?"),
    ("I typed the wrong street, can I fix it", "How can I change my delivery address?"),
    ("do you keep my card number on your servers", "Is my payment information safe?"),
    ("my order didnt arrive on time", "What should I do if my order is delayed?"),
    ("hw do i trak my oder", "How do I track my order?"),  # typos
]

OFF_TOPIC = ["who won the cricket world cup", "what is the capital of France", "tell me a joke about cats"]


def main():
    bot = FAQChatbot()
    correct = 0
    for message, expected in CASES:
        reply = bot.get_reply(message)
        ok = reply.matched_question == expected
        correct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r:55} -> {reply.matched_question or '(no match)'}  ({reply.score:.2f})")

    print(f"\nAccuracy: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")

    rejected = sum(1 for q in OFF_TOPIC if not bot.get_reply(q).matched_question)
    print(f"Off-topic questions correctly rejected: {rejected}/{len(OFF_TOPIC)}")


if __name__ == "__main__":
    main()
