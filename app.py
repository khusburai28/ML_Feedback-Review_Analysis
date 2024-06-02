import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Ensure stopwords are available
stop_words = set(stopwords.words('english'))


# Function to process text and perform sentiment analysis
def process_text(text):
    text = text.lower()
    text = ''.join(c for c in text if not c.isdigit())
    processed_text = ' '.join([word for word in text.split() if word not in stop_words])
    return processed_text

# Function to perform sentiment analysis
def sentiment_analysis(text):
    sa = SentimentIntensityAnalyzer()
    scores = sa.polarity_scores(text)
    compound = round((1 + scores['compound']) / 2, 2)
    return scores, compound

def generate_summary(scores, compound):
    summary = (
        f"The sentiment analysis of the provided text reveals a positive sentiment score of {scores['pos']*100}%, "
        f"a negative sentiment score of {scores['neg']*100}%, and a neutral sentiment score of {scores['neu']*100}%. "
        f"The compound score, which represents the overall sentiment, is {compound*100}%. This indicates a "
        f"{'positive' if compound > 50 else 'neutral' if compound == 50 else 'negative'} sentiment overall."
    )
    return summary

# Streamlit UI
st.title("Feedback & Review Analysis")

text_input = st.text_area("Enter text for analysis:", height=200)

if st.button("Analyze"):
    processed_text = process_text(text_input)
    scores, compound = sentiment_analysis(processed_text)

    st.subheader("Summary")
    st.write(generate_summary(scores, compound))

    # st.write(processed_text)
    st.subheader("Scores")
    st.write(f"Positive: {scores['pos']*100}%")
    st.write(f"Negative: {scores['neg']*100}%")
    st.write(f"Neutral: {scores['neu']*100}%")
    st.write(f"Compound: {compound*100}%")




