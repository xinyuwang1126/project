import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

def article_to_words(raw_article):
    article_text = BeautifulSoup(raw_article).get_text()
    letters_only = re.sub("[^a-zA-Z0-9]", " ", article_text)
    words = letters_only.lower().split()
    return( " ".join(words))