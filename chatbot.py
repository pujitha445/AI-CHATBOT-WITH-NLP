"""
ADVANCED NLP CHATBOT SYSTEM (FIXED & IMPROVED)
Task 3 – AI Chatbot with NLP
"""

# ===============================
# IMPORTS
# ===============================
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ===============================
# INITIALIZATION
# ===============================
lemmatizer = WordNetLemmatizer()

# ===============================
# GREETING HANDLER (IMPORTANT)
# ===============================
greetings = ["hi", "hello", "hey", "good morning", "good evening"]

def is_greeting(text):
    return text.lower().strip() in greetings

# ===============================
# KNOWLEDGE BASE (EXPLICIT)
# ===============================
knowledge_base = {
    "what is python": "Python is a high-level programming language used in AI, ML, web development, and data science.",
    "what is java": "Java is an object-oriented programming language widely used for enterprise and Android development.",
    "what is machine learning": "Machine learning allows systems to learn patterns from data without explicit programming.",
    "what is artificial intelligence": "Artificial Intelligence focuses on building intelligent systems that mimic human intelligence.",
    "what is data science": "Data science involves analyzing data to extract meaningful insights.",
    "what is internship": "An internship provides hands-on industry experience for students."
}

questions = list(knowledge_base.keys())
answers = list(knowledge_base.values())

# ===============================
# TEXT PREPROCESSING
# ===============================
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    return " ".join(lemmas)

processed_questions = [preprocess(q) for q in questions]

# ===============================
# TF-IDF MODEL
# ===============================
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# ===============================
# CHATBOT RESPONSE ENGINE
# ===============================
def chatbot_response(user_input):
    if is_greeting(user_input):
        return "Hello! 😊 How can I help you?"

    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])

    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_index = similarity_scores.argmax()
    confidence = similarity_scores[0][best_index]

    # STRICT CONFIDENCE CHECK (NO WRONG ANSWERS)
    if confidence < 0.35:
        return "Sorry, I don't have information on that topic."

    return answers[best_index]

# ===============================
# CHAT LOGGING
# ===============================
def log_chat(user, bot):
    with open("chat_log.txt", "a") as file:
        file.write(f"{datetime.now()} | User: {user} | Bot: {bot}\n")

# ===============================
# MAIN LOOP
# ===============================
def start_chatbot():
    print("🤖 Advanced NLP Chatbot is Online!")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        reply = chatbot_response(user)
        print("Bot:", reply)

        log_chat(user, reply)

    print("\n[SESSION SAVED] chat_log.txt created")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    start_chatbot()
