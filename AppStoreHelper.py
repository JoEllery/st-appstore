# Imports for general data manipulation
import pandas as pd # For making data into tables we can manipulate (and many other functions besides!)
import numpy as np # For handling matrix operations, plus several great utilities
import torch # A package for working with arrays and optimization, great in an ML context
import json # Allows python to work easily with json files.
from datetime import datetime # Allows us to handle dates
from dateutil.relativedelta import relativedelta
import math as m # Adds useful basic math, like rounding

# Import the scraping library we installed in the last cell.
from app_store_scraper import AppStore

# Imports for sentiment analysis
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification # For transformer-based analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # For rule/dictionary based analysis
from tqdm import tqdm # For showing progress on running models

# Import seaborn for graphics
import seaborn as sns
import matplotlib.pyplot as plt

# Import streamlit
import streamlit as st


def transformer_sentiment(text, tokenizer, model):

  tokens = tokenizer(text, return_tensors="pt")
  ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']

  raw_score = model(ids, attention_mask).logits.detach().numpy().argmax()

  if raw_score == 0:
    return -1
  else:
    return 1


def wordcount_sentiment(text, vader_sia):

  raw_score = vader_sia.polarity_scores(text)['compound']

  if raw_score > 0.05:
    return 1
  elif raw_score < -0.05:
    return -1
  else:
    return 0


def word_count(text):
  return len(text.split())


def appanalytics(app_name, earliest_review_date, review_number):

  try:
    datetime_obj = datetime.strptime(earliest_review_date, "%Y-%m-%d")
  except ValueError as e:
    st.write("Incorrect date format. Using default.")
    datetime_obj = datetime.strptime("2022-01-01", "%Y-%m-%d")

  try:
    review_number = int(review_number)
  except ValueError as e:
    st.write("Incorrect review number format. Using default.")
    review_number = 200

  try:

    appstore_scraper = AppStore(country='us', app_name=app_name)
    appstore_scraper.review(how_many=review_number, after=datetime_obj)
    reviews = pd.DataFrame(appstore_scraper.reviews)

    if len(reviews) < review_number:
      raise ValueError

    use_emergency_data = False
    st.write("Scraping successful.")

  except ValueError as e:

    use_emergency_data = True
    st.write("Scraping failed.")

  tokenizer = DistilBertTokenizer.from_pretrained("AdamCodd/distilbert-base-uncased-finetuned-sentiment-amazon")
  model = DistilBertForSequenceClassification.from_pretrained("AdamCodd/distilbert-base-uncased-finetuned-sentiment-amazon")

  vader_sia = SentimentIntensityAnalyzer()

  if use_emergency_data:
    reviews = pd.read_csv("ubereats_data.csv")
    st.write("Using emergency data.")

  reviews_text = reviews['review']

  progress_text = "Running analysis."
  bar = st.progress(0, text=progress_text)
  for r in range(len(reviews)):

    text = reviews_text[r]

    reviews.loc[r,'transformer_sentiment'] = transformer_sentiment(text, tokenizer, model)
    reviews.loc[r,'wordcount_sentiment'] = wordcount_sentiment(text, vader_sia)
    reviews.loc[r,'review_length'] = word_count(text)

    bar.progress(r/len(reviews), text=progress_text)
  
  bar.progress(1.0, text="Analysis done!")

  possible_stars = list(range(1,6))
  count = np.zeros(5)

  average_transformer_sentiment = np.zeros(5)
  average_wordcount_sentiment = np.zeros(5)
  average_length = np.zeros(5)

  for star in possible_stars:

    mask = reviews['rating'] == star

    star_subset = reviews.loc[mask]
    count[star-1] = len(star_subset)

    average_transformer_sentiment[star-1] = np.average(star_subset['transformer_sentiment'])
    average_wordcount_sentiment[star-1] = np.average(star_subset['wordcount_sentiment'])
    average_length[star-1] = np.average(star_subset['review_length'])

  data = {
      'Stars': possible_stars,
      'Transformer': average_transformer_sentiment,
      'Wordcount': average_wordcount_sentiment,
  }

  data = pd.DataFrame(data)

  st.line_chart(data=data, x='Stars', y=['Transformer', 'Wordcount'])


  




















