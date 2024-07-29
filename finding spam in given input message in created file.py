import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from wordcloud import WordCloud

spam_data = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')

def clean_data(df):
    df.drop(columns=['un1'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    return df

cleaned_data = clean_data(spam_data)

X_train, X_test, y_train, y_test = train_test_split(cleaned_data['text'], cleaned_data['target'], test_size=0.2)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_cv, y_train)

predictions = mnb.predict(X_test_cv)

print("Accuracy:", metrics.accuracy_score(y_test, predictions))

def visualise(label):
    words = ''
    for msg in spam_data[spam_data['target'] == label]['text']:
        msg = msg.lower()
        wordcloud=WordCloud(width=1000,height=500).generate(msg)
        plt.figure(figsize=(10,8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
visualise('spam')
visualise('ham')

# Create a new file and write the message to it
with open('test.txt', 'w') as f:
    f.write('you won a lucky price on login reward for login rummy circle.com')

# Read the file and transform it using the CountVectorizer
with open('test.txt', 'r') as f:
    your_message = f.read()
your_message_cv = cv.transform([your_message])

# Make a prediction using the trained model
prediction = mnb.predict(your_message_cv)

# Print the prediction
if prediction[0] == 1:
    print('spam')
else:
    print('ham')
