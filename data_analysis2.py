import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

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

# preprocessing LSTM data
df['review_500'] = reviews.apply(lambda x:x[:500])

# tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['review_500'])
seq = tokenizer.texts_to_sequences(df['review_500'])
X = pad_sequences(seq, padding='post')
y = sentiment_data

print(f'X_shape: {X.shape}, X_min: {np.min(X)}, X_max: {np.max(X)}, y: {y.shape}')

# split data
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=42)
print(f'X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}')

# LSTM v1 structure
embed_size = 64
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=embed_size, input_shape=[None], mask_zero=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

optimizer =keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_valid, y_valid))
# Training history of LSTM v1
pd.DataFrame(history.history).plot()
plt.savefig("LSTMv1")
# LSTM v1 result
y_pred = model.predict(X_test)
y_pred = np.int64(y_pred>0.5)

print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='.0f',
            xticklabels=['Predicted negative', 'Predicted positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix of LSTM v1")
plt.savefig('Confusion Matrix of LSTM v1')


# updated LSTM to reduce overfitting
embed_size = 64
model_v2 = keras.models.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=embed_size, input_shape=[None], mask_zero=True),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])
print(model_v2.summary())

early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model_v2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model_v2.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb])
pd.DataFrame(history.history).plot()
plt.savefig('LSTMv2')

y_pred = model_v2.predict(X_test)
y_pred = np.int64(y_pred>0.5)

print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='.0f',
            xticklabels=['Predicted negative', 'Predicted positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix of LSTM v2")
