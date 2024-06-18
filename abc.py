import streamlit as st
from googleapiclient.discovery import build
import re
from langdetect import detect
import demoji
import matplotlib.pyplot as plt
import numpy as np
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

# Streamlit page configuration
st.set_page_config(layout="wide")

st.write(
    """
    <style>
    .title { text-align: center; color: #FF5733; }
    .link { color: #0000ff; }
    .info { color: #ffbf00; }
    .title2 { color: #ff8000; }
    .author { color: #bf00ff; }
    .title5 { color: #006600; }
    .ans { color: #006600; }
    .red { color: #990000; }
    .grey { color: #e6e6e6; }
    out { color: #4dff4d; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a centered heading in the wider Streamlit window
st.markdown('<h1 class="title">SENTIMENT ANALYSIS FOR YOUTUBE EDUCATIONAL VIDEOS USING NLP AND MACHINE LEARNING APPROACH</h1>', unsafe_allow_html=True)
st.write("")
st.write("")
st.markdown('<h4 class="info">This machine learning model can prove highly valuable in assessing user sentiment and video quality by considering various factors, including comments, views, likes, subscriber counts, and the video\'s publication date.</h4>', unsafe_allow_html=True)
st.markdown('<h2 class="link">Enter Youtube Video Link:</h2>', unsafe_allow_html=True)
link = st.text_input("")
API_KEY = 'AIzaSyA95t-mAtHEdIVEzg0EUkA1PxUWP5v5168'  # Replace with your YouTube Data API key

def extract_video_id(link):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.findall(pattern, link)
    if match:
        return match[0]
    else:
        st.warning("Invalid YouTube link.")
        return None

def get_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    nextPageToken = None
    while True:
        try:
            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=nextPageToken,
                maxResults=100
            ).execute()

            for comment in comment_response['items']:
                text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(text)

            nextPageToken = comment_response.get('nextPageToken')
            if not nextPageToken:
                break
        except Exception as e:
            st.warning(f"An error occurred: {e}")
            break
    return comments

def remove_emojis(lst):
    demoji.download_codes()
    cleaned_sentence = [demoji.replace(text, '') for text in lst if len(demoji.replace(text, '')) > 1]
    return cleaned_sentence

def detect_languages(lst):
    ta, en, other = 0, 0, 0
    dict1 = {}
    tamil_lst, english_lst, other_languages = [], [], []
    for i in lst:
        if len(i) >= 15:
            try:
                language = detect(i)
            except Exception:
                continue
            if language == 'ta':
                ta += 1
                tamil_lst.append(i)
            elif language == 'en':
                en += 1
                english_lst.append(i)
            else:
                other += 1
                other_languages.append(i)
            dict1[language] = dict1.get(language, 0) + 1
    return en, ta, other, english_lst, tamil_lst

def chart(lst):
    st.markdown('<h4 class="title3">This Chart represents the visual representation of comments in the video: </h4>', unsafe_allow_html=True)
    labels = ['English', 'Tamil', 'Other']
    plt.figure(figsize=(8, 4))
    plt.pie(lst, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Language Distribution of Comments')
    plt.axis('equal')
    plt.savefig('pie_chart.png')
    st.image('pie_chart.png', caption='Pie Chart')

def sentimental(lst):
    positive, negative, neutral = 0, 0, 0
    pos_, neg_ = [], []
    for i in lst:
        blob = TextBlob(i)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            pos_.append(polarity)
            positive += 1
        elif polarity == 0:
            neutral += 1
        else:
            neg_.append(polarity)
            negative += 1
    st.markdown('<h1 class="ans">Positive comments:</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="grey">{positive}</h2>', unsafe_allow_html=True)
    st.markdown('<h1 class="red">Negative comments:</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="grey">{negative}</h2>', unsafe_allow_html=True)
    st.markdown('<h1 class="link">Neutral comments:</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="grey">{neutral}</h2>', unsafe_allow_html=True)
    return positive, negative, neutral, pos_, neg_

def bar_chart(pos, neg, neut):
    st.markdown('<h4 class="title3">This Bar represents the visual representation of sentiment in comments in the video: </h4>', unsafe_allow_html=True)
    lst = [pos, neg, neut]
    categories = ['Positive', 'Negative', 'Neutral']
    plt.figure(figsize=(8, 4))
    plt.bar(categories, lst)
    plt.xlabel('Sentiments')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')
    plt.savefig('bar_chart.png')
    st.image('bar_chart.png', caption='Bar Chart')

def get_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        if video_response['items']:
            video = video_response['items'][0]
            snippet = video['snippet']
            statistics = video['statistics']
            return {
                'Video Title': snippet['title'],
                'Content Creator Name': snippet['channelTitle'],
                'Published Date': snippet['publishedAt'],
                'Views': statistics['viewCount'],
                'Likes': statistics.get('likeCount', 0),
            }
        else:
            st.warning("No video found with the given ID.")
            return None
    except Exception as e:
        st.warning(f"An error occurred: {e}")
        return None

def get_subs_count(channel_url):
    try:
        response = requests.get(channel_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        pattern = r"subscribers\"}},\"simpleText\":\"(\d.+) subscribers\"},\"trackingParams\""
        match = re.findall(pattern, str(soup))
        if match:
            return match[0]
        else:
            return "Subscriber count not found"
    except Exception as e:
        st.warning(f"An error occurred while fetching subscriber count: {e}")
        return "Error"

def calci(pos, neut, total_comments):
    return (pos + neut) * 100 / total_comments if total_comments > 0 else 0

# Main program
if link:
    video_id = extract_video_id(link)
    if video_id:
        video_info = get_video_info(video_id)
        if video_info:
            st.markdown('<h3 class="author">Video Title:</h3>', unsafe_allow_html=True)
            st.write(video_info["Video Title"])
            st.markdown('<h3 class="author">Content Creator Name:</h3>', unsafe_allow_html=True)
            st.write(video_info["Content Creator Name"])
            st.markdown('<h3 class="author">Published Date:</h3>', unsafe_allow_html=True)
            st.write(video_info["Published Date"])
            st.markdown('<h3 class="author">Views:</h3>', unsafe_allow_html=True)
            st.write(video_info["Views"])
            st.markdown('<h3 class="author">Likes:</h3>', unsafe_allow_html=True)
            st.write(video_info["Likes"])

            # Get channel URL for subscriber count
            channel_url = f"https://www.youtube.com/channel/{video_info['Content Creator Name']}"
            subscriber_count = get_subs_count(channel_url)
            st.markdown('<h3 class="author">Subscriber count:</h3>', unsafe_allow_html=True)
            st.write(subscriber_count)

            comments = get_comments(video_id)
            st.markdown('<h4 class="title2">The Total Number Of Comments Extracted From This Video IS:</h4>', unsafe_allow_html=True)
            st.write(len(comments))

            cleaned_comments = remove_emojis(comments)
            st.markdown('<h4 class="title5">The Total Number Of Cleaned Comments Extracted From This Video IS:</h4>', unsafe_allow_html=True)
            st.write(len(cleaned_comments))

            english, tamil, other, english_lst, tamil_lst = detect_languages(cleaned_comments)
            chart([english, tamil, other])
            positive, negative, neutral, pos_, neg_ = sentimental(cleaned_comments)
            bar_chart(positive, negative, neutral)
            result = calci(positive, neutral, len(cleaned_comments))
            st.markdown('<h4 class="title3">The Analysis Result of This Video Comments is:</h4>', unsafe_allow_html=True)
            st.markdown('<h2 class="grey">The video is highly recommendable!!!</h2>', unsafe_allow_html=True)
            st.write(f"Percentage of positive and neutral comments: {result:.2f}%")
        else:
            st.warning("Failed to retrieve video information.")
    else:
        st.warning("No video ID found in the link.")
else:
    st.warning("Please enter a valid YouTube video link.")
