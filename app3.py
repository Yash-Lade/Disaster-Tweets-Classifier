import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib
import tweepy
import streamlit as st
from datetime import datetime, timedelta, timezone

# Load pre-trained model and vectorizers
model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
bow = joblib.load('bow_vectorizer.pkl')

# Twitter API credentials
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAO9puQEAAAAAibBDDQEa%2BOYP%2BI%2BkQNdGzbfgOQQ%3DSryMUwRRPymuUcj0pA44K7TkHdRKXPE8LQrBeXkrQsIKgUHWDd'  # Replace with your Bearer Token

# Initialize tweepy client
client = tweepy.Client(bearer_token=bearer_token)

def fetch_tweet_text(tweet_url):
    tweet_id = tweet_url.split('/')[-1]
    try:
        response = client.get_tweet(tweet_id, tweet_fields=["text"])
        tweet_text = response.data["text"]
        return tweet_text
    except Exception as e:
        st.error(f"Error fetching tweet: {e}")
        return None

def fetch_recent_tweets(query, time_range):
    end_time = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=5)
    if time_range == "Last 5 hours":
        start_time = end_time - timedelta(hours=5)
    elif time_range == "Last 24 hours":
        start_time = end_time - timedelta(hours=24)
    else:
        start_time = end_time - timedelta(hours=5)  # Default to last 5 hours if unspecified

    start_time_str = start_time.isoformat().replace("+00:00", "Z")
    end_time_str = end_time.isoformat().replace("+00:00", "Z")

    try:
        tweets = client.search_recent_tweets(query=query, start_time=start_time_str, end_time=end_time_str, max_results=10, tweet_fields=["created_at", "text", "geo"])
        return tweets.data if tweets.data else []
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return []

# Function to predict a single tweet
def predict_single_tweet(tweet):
    tweet_tfidf = tfidf.transform([tweet])
    tweet_bow = bow.transform([tweet])
    tweet_combined = hstack([tweet_tfidf, tweet_bow])
    prediction = model.predict(tweet_combined)
    return "Disaster" if prediction[0] == 1 else "Not a Disaster"

# Set page configuration
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")

page_bg="""
    <style>
        [data-testid="stAppViewContainer"]{
            background-image:url("https://news.northeastern.edu/wp-content/uploads/2023/02/TWEET1400.jpg?w=1024");
            background-size:cover;
        }
    </style>
"""
# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to bottom, rgb(25, 78, 82), rgb(0, 0, 0));
        color: white;
    }
    .title {
        font-family: 'Helvetica', sans-serif;
        color: rgb(0, 227, 235);
        text-align: center;
        padding: 20px;
        font-size: 35px;
        font-weight:bolder
    }
    .header {
        font-family: 'Helvetica', sans-serif;
        color: #000000;
        padding: 10px;
        font-size: 24px;
        text-align: center;
    }
    .header1{
        font-family: 'Helvetica', sans-serif;
        color: #f9ff3d;
        font-weight:bolder;
    }
    .subheader {
        font-family: 'Helvetica', sans-serif;
        color: #f7ab33;
        padding: 10px;
        font-size: 20px;
    }
    .result {
        font-family: 'Helvetica', sans-serif;
        text-align: center;
        padding: 10px;
        font-size: 20px;
    }
    .disaster {
        color: red;
    }
    .not-disaster {
        color: #00ff15;
    }
    .stTextArea textarea {
        background-color: #333333;
        color: white;
    }
    .stSelectbox div[role="listbox"] {
        font-size: 25px;
        background-color: #333333;
        color: white;
    }
    .stTextInput input {
        background-color: #333333;
        color: white;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        background-color: #4c4c4c;
        color: white;
    }
    .stButton button {
        background-color: #4c4c4c;
        color: white;
    }
    .stButton button:hover {
        background-color: #0075ac;
        color: white;
    }
    .tweet-separator {
        border-bottom: 1px solid #4c4c4c;
        margin: 20px 0;
    }
    .input-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .etr-txt{
        padding-bottom:0;
        margin-bottom:0;
    }
    .input-container img {
        margin-right: 20px;
        width: 50px;
        height: auto;
    }
    .blurred-background {
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.5);
        margin: 10px 0;
    }
    .blur-bg-hdr{
        backdrop-filter: blur(10px);
        font-weight:bolder;
    }
    .slctbx{
        font-size:15px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and header
st.markdown("<div class='title'>Disaster Tweet Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='header1'>Classify a Tweet</div>", unsafe_allow_html=True)
st.markdown("<div class='etr-txt'>Enter the Twitter post URL or tweet text: </div>", unsafe_allow_html=True)

# Container for logo and input
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.markdown(page_bg, unsafe_allow_html=True)
tweet_input = st.text_area("", height=100)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button
if st.button("Fetch and Classify"):
    if tweet_input:
        if 'twitter.com' in tweet_input or 'x.com' in tweet_input:
            tweet_text = fetch_tweet_text(tweet_input)
            if tweet_text:
                result = predict_single_tweet(tweet_text)
                result_color = "disaster" if result == "Disaster" else "not-disaster"
                st.markdown(f"<div class='blurred-background'><div class='result'>Tweet: {tweet_text}</div><div class='result {result_color}'>Prediction: {result}</div></div>", unsafe_allow_html=True)
        else:
            result = predict_single_tweet(tweet_input)
            result_color = "disaster" if result == "Disaster" else "not-disaster"
            st.markdown(f"<div class='blurred-background'><div class='result {result_color}'>Prediction: {result}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result'>Please enter a tweet.</div>", unsafe_allow_html=True)

# Fetch recent tweets
st.markdown("<div class='header blur-bg-hdr'>Fetch Recent Disaster Related Tweets</div>", unsafe_allow_html=True)
time_range = st.selectbox("Select time range:", ["Last 5 hours", "Last 24 hours"])

if st.button("Fetch Disaster Related Tweets"):
    query = "disaster"  # Example query, you can customize it
    tweets = fetch_recent_tweets(query, time_range)
    for tweet in tweets:
        tweet_text = tweet.text
        created_at = tweet.created_at
        user_location = tweet.geo if tweet.geo else "Unknown location"
        result = predict_single_tweet(tweet_text)
        result_color = "disaster" if result == "Disaster" else "not-disaster"
        st.markdown(f"<div class='result {result_color} blurred-background'>Tweet: {tweet_text} \n\n Created At: {created_at} \n\n User Location: {user_location}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='result {result_color} blurred-background'>Prediction: {result}</div>", unsafe_allow_html=True)
        st.markdown("<div class='tweet-separator'></div>", unsafe_allow_html=True)
