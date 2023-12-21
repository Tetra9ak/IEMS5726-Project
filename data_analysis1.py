import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

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
# print(reviews.head)

# convert label from pos & neg to 1 & 0
lb = LabelBinarizer()
# transformed sentiment data
sentiment_data = lb.fit_transform(y)
print('Label set:', sentiment_data.shape)

# Splitting data for SVM
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(X, sentiment_data,
                                                                                  stratify=sentiment_data,
                                                                                  test_size=0.2)

# split data visualization
training_label = pd.Series(train_sentiments.squeeze()).value_counts()
test_label = pd.Series(test_sentiments.squeeze()).value_counts()
height = [training_label[0], training_label[1], test_label[0], test_label[1]]
sns.barplot(x=np.array(['train_negative', 'train_positive', 'test_negative', 'test_positive']), y=height)
plt.title("Split Data Distribution")
plt.ylabel("Numbers of reviews")
plt.savefig('Split Dataset Distribution')

# BoW method
baseline_model = make_pipeline(CountVectorizer(ngram_range=(1, 3)),
                               TfidfTransformer(),
                               LinearSVC()).fit(train_reviews, train_sentiments)

predicted = baseline_model.predict(test_reviews)

val_acc = accuracy_score(test_sentiments, predicted)
val_f1_score = f1_score(test_sentiments, predicted, average='micro')

print(f'val_acc: {val_acc:.5f} f1_score: {val_f1_score:.5f}')
print(classification_report(test_sentiments, predicted, digits=4))

sns.heatmap(confusion_matrix(test_sentiments, predicted),
            annot=True, fmt='.0f',
            xticklabels=['Predicted negative', 'Predicted positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix of SVM model")
plt.savefig("Confusion Matrix of SVM model")
