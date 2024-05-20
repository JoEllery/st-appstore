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

def appanalytics(app_name, earliest_review_date, review_number):

  try:
    datetime_obj = datetime.strptime(earliest_review_date, "%Y-%m-%d")
  except ValueError as e:
    st.write("Incorrect date format. Using default.")
    datetime_obj = datetime.strptime("2022-01-01", "%Y-%m-%d")

  if not type(review_number) == int:
    st.write("Incorrect review number format. Using default.")
    review_number = 200
