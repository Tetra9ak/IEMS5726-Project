import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelBinarizer


df = pd.read_csv("IMDB Dataset.csv")
# show basic info
print(df.head())
print(f'shape of dataset is {df.shape}')
print(df.describe())

review, label = df['review'].values, df['sentiment'].values
raw_label = pd.Series(label).value_counts()
sns.barplot(x=np.array(['negative', 'positive']), y=raw_label.values)
plt.title("Raw Label Distribution")
plt.ylabel("Numbers of reviews")
plt.savefig('Raw Dataset Distribution')

# divide attribute and label
X, y = df['review'], df['sentiment']


# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# Define function for removing special characters
def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


# preprocessing data
X = X.apply(denoise_text)
print("denoise completed")
reviews = X.apply(remove_special_characters)
print("special characters removed")
print("stopwords removed")
print(reviews.head)

# convert label from pos & neg to 1 & 0
lb = LabelBinarizer()
# transformed sentiment data
sentiment_data = lb.fit_transform(y)
print('Label set:', sentiment_data.shape)
