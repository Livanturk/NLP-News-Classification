import streamlit as st
import joblib
import re
import spacy
from rapidfuzz import process
import numpy as np

# Load assets
nlp = spacy.load("en_core_web_sm")
model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

vocab = set(vectorizer.get_feature_names_out())

def preserve_entities(text):
    doc = nlp(text)
    preserved = text
    for ent in doc.ents:
        tag = f"<{ent.label_}:{ent.text.replace(' ', '_')}>"
        preserved = preserved.replace(ent.text, tag)
    return preserved

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s<>_:]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def correct_typos(text, vocab, threshold=90):
    corrected_words = []
    for word in text.split():
        if word in vocab or word.startswith("<") or word.endswith(">"):
            corrected_words.append(word)
        else:
            match, score, _ = process.extractOne(word, vocab)
            if score >= threshold:
                corrected_words.append(match)
            else:
                corrected_words.append(word)
    return ' '.join(corrected_words)

st.title("News Article Classifier")
st.write("Enter a news article below. The system will predict its category:")

user_input = st.text_area("Paste your news text here...", height=300)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please paste some news text.")
    else:
        text = preserve_entities(user_input)
        text = clean_text(text)
        text = lemmatize_text(text)
        text = correct_typos(text, vocab)
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]

        st.success(f"**Predicted Category:** `{prediction}`")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            sorted_idx = np.argsort(probs)[::-1]
            st.write("🔢 Top Predictions:")
            for i in sorted_idx[:3]:
                label = model.classes_[i]
                st.write(f"{label}: {probs[i]*100:.2f}%")
