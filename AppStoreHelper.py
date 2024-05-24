# Imports for general data manipulation
import pandas as pd # For making data into tables we can manipulate (and many other functions besides!)
import numpy as np # For handling matrix operations, plus several great utilities
import torch # A package for working with arrays and optimization, great in an ML context
import json # Allows python to work easily with json files.
from datetime import datetime # Allows us to handle dates
import math as m # Adds useful basic math, like rounding

# Import the scraping library we installed in the last cell.
from app_store_scraper import AppStore

# Imports for sentiment analysis
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification # For transformer-based analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # For rule/dictionary based analysis
from tqdm import tqdm # For showing progress on running models

# Import seaborn for graphics
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

  ### ANALYTICS ###

  possible_stars = list(range(1,6))

  # Make four empty vectors of length five to hold average sentiment, length and total count for each star value.
  count = np.zeros(5)
  average_transformer_sentiment = np.zeros(5)
  average_wordcount_sentiment = np.zeros(5)
  average_length = np.zeros(5)
  
  # For each possible star value...
  for star in possible_stars:
  
    # ... Mark which reviews have that star value...
    mask = reviews['rating'] == star
    
    # ... Subset the data with only reviews with that star value, and get the count of that star value...
    star_subset = reviews.loc[mask]
    count[star-1] = len(star_subset)
    
    # ... And average the statistics of interest over the subset.
    average_transformer_sentiment[star-1] = np.average(star_subset['transformer_sentiment'])
    average_wordcount_sentiment[star-1] = np.average(star_subset['wordcount_sentiment'])
    average_length[star-1] = np.average(star_subset['review_length'])  

  ### Handle maybe not all stars being in sample ###

  average_transformer_sentiment = np.nan_to_num(average_transformer_sentiment, nan=0.0)
  average_wordcount_sentiment = np.nan_to_num(average_wordcount_sentiment, nan=0.0)
  average_length = np.nan_to_num(average_length, nan=0.0)
  
  ### Now, plot the data we just gathered. ###
  
  # Collect star value, transformer sentiment and wordcount sentiment in a data frame.
  data = {
    'Star': possible_stars,
    'Tr_Sent': average_transformer_sentiment,
    'Wc_Sent': average_wordcount_sentiment,
  }
  
  # Set the size of the figure we want.
  fig, ax = plt.subplots(figsize=(6, 6))
  
  # Plot the two sentiment analysis averages against the star values.
  ax.plot(data['Star'], data['Tr_Sent'], label='Transformer Sentiment')
  ax.plot(data['Star'], data['Wc_Sent'], label='Wordcount Sentiment')
  
  ax.set_xlabel('Star Rating')  # Label for the x-axis
  ax.set_ylabel('Average Sentiment')  # Label for the y-axis
  ax.set_xticks(ticks=range(1,6), labels=data['Star']) # Ensure the x-ticks don't have unneeded decimals
  ax.set_title('Review Sentiment: Method Comparison')  # Title of the chart
  ax.legend()  # Add a legend
  
  st.pyplot(fig) # Finally, show the final plot
  
  ### Plot the rest of the gathered data ###
  
  # First, gather the data we'll use for this set of charts.
  data = {
    'Star': possible_stars,
    'Len': average_length,
    'Count': count,
  }
  
  # Create a figure to show two charts side by side. The figure has two subplots.
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  
  # First subplot
  ax1.bar(data['Star'], data['Len']) # Graph average length against stars
  ax1.set_xlabel('Star Rating')  # Label for the x-axis of the first subplot
  ax1.set_ylabel('Average Length')  # Label for the y-axis of the first subplot
  ax1.set_title('Review Length by Star Rating')  # Title of the first subplot
  ax1.set_ylim(min(data['Len'])-20, max(data['Len']) + 20) # Set the y axis in a band around the bars
  
  # Second subplot
  ax2.bar(data['Star'], data['Count']) # Graph review count against stars
  ax2.set_xlabel('Star Rating')  # Label for the x-axis of the second subplot
  ax2.set_ylabel('Frequency')  # Label for the y-axis of the second subplot
  ax2.set_title('Frequency of Star Ratings')  # Title of the second subplot
  
  plt.tight_layout()  # Adjust subplots to fit into the figure area.
  st.pyplot(fig)  # Display the plots
  
  ### Now Consider Sentiment Through Time. ###
  
  # Set the number of months in each bucket.
  fineness = 2
  
  # Get the dates of each reviews as a datetime object (a computable date) rather than text.
  dates = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in analyzed_reviews['date']]
  analyzed_reviews['datetime'] = dates
  
  # Get the start and end dates of the reviews.
  start_date = min(dates)
  end_date = max(dates)
  
  # Get the number of months between the start and end dates: the month difference, plus twelve times the year difference.
  number_of_months = (end_date.month - start_date.month) + 12*(end_date.year - start_date.year)
  
  # Make buckets for the dates by adding the month increment to the start date.
  date_buckets = [datetime(start_date.year + x//12, start_date.month + x%12, 1) for x in range(m.ceil(number_of_months/fineness))]
  bucket_num = len(date_buckets)-1
  
  # Make empty vectors as holders for averages of the variables of interest.
  average_transformer_sentiment_d = np.zeros(bucket_num)
  average_wordcount_sentiment_d = np.zeros(bucket_num)
  average_stars_d = np.zeros(bucket_num)
  mids = []
  
  # Now iterate over the buckets.
  for b in range(bucket_num):
  
    # Get the start and end dates of the bucket from the list made above...
    min_d = date_buckets[b]
    max_d = date_buckets[b+1]
    
    # ... Create a string for the midpoint of the bucket, to label our graph...
    mids.append((min_d + (max_d - min_d) / 2).strftime("%B %Y"))
    
    # ... Mark which reviews fall into the bucket of dates ...
    mask = (analyzed_reviews['datetime'] >= min_d) & (analyzed_reviews['datetime'] < max_d)
    
    # ... Subset the data based on the relevant dates ...
    date_subset = analyzed_reviews.loc[mask]
    
    # ... And average the statistics of interest over the subset.
    average_transformer_sentiment_d[b] = np.average(date_subset['transformer_sentiment'])
    average_wordcount_sentiment_d[b] = np.average(date_subset['wordcount_sentiment'])
    average_stars_d[b] = np.average(date_subset['rating'])

  average_transformer_sentiment_d = np.nan_to_num(average_transformer_sentiment_d, nan=0.0)
  average_wordcount_sentiment_d = np.nan_to_num(average_wordcount_sentiment_d, nan=0.0)
  average_stars_d = np.nan_to_num(average_stars_d, nan=0.0)
  
  ### Once more we plot! ###
  
  # Gather the data together for easy plotting.
  # Normalize each line to center on zero.
  data = {
    'dates': mids, # Note that these are the strings we made that represent bucket midpoints.
    'Stars': (average_stars_d-np.average(average_stars_d)),
    'Transformer': (average_transformer_sentiment_d - np.average(average_transformer_sentiment_d)),
    'Wordcount': (average_wordcount_sentiment_d - np.average(average_wordcount_sentiment_d))
  }
  
  # Set the size of the figure. This one is long and short, as befitting a time series.
  fig, ax = plt.subplots(figsize=(10, 5))
  
  ax.plot(data['dates'], data['Transformer'], label='Transformer Sentiment')  # Plot transformer sentiment
  ax.plot(data['dates'], data['Wordcount'], label='Wordcount Sentiment')  # Plot word count sentiment
  ax.plot(data['dates'], data['Stars'], label='Star Rating')  # Plot star rating
  
  ax.set_ylabel('Average Sentiment')  # Label for the y-axis
  ax.set_title('Review Sentiment Through Time')  # Title of the chart
  ax.legend()  # Show legend
  ax.set_xticklabels(data['dates'], rotation=90) # Rotate the labels so they don't overlap
  
  # Display the plot!
  st.pyplot(fig)


  




















