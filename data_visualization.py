import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# FOR SVM MODEL
# sns.heatmap(confusion_matrix(test_sentiments, predicted),
#             annot=True, fmt='.0f',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title("Confusion Matrix of SVM model")
# plt.savefig("Confusion Matrix of SVM model")


# FOR LSTMv1
# Training history of LSTM v1
# history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_valid, y_valid))?
# pd.DataFrame(history.history).plot()
# plt.savefig("LSTMv1")
# sns.heatmap(confusion_matrix(y_test, y_pred),
#             annot=True, fmt='.0f',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title("Confusion Matrix of LSTM v1")
# plt.savefig('Confusion Matrix of LSTM v1')


# FOR LSTMv2
# history = model_v2.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_valid, y_valid),
# callbacks=[early_stopping_cb])
# pd.DataFrame(history.history).plot()
# plt.savefig('LSTMv2')
# sns.heatmap(confusion_matrix(y_test, y_pred),
#             annot=True, fmt='.0f',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title("Confusion Matrix of LSTM v2")
