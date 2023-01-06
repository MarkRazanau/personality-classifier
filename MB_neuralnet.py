import sklearn
import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

preprocessed_mb_data = pd.read_csv('preprocessed_mbti.csv', sep='\t')

# Encode each different Myers-Briggs personality type
encoder = LabelEncoder()
preprocessed_mb_data['encoded_type'] = encoder.fit_transform(preprocessed_mb_data['type'])

# Tokenize each entry post
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(preprocessed_mb_data['posts'])

tokenized_posts = tokenizer.texts_to_sequences(preprocessed_mb_data['posts'])

train = pad_sequences(tokenized_posts, padding='post', maxlen=50)
encoded_vals = preprocessed_mb_data['encoded_type']


# Split data into 65/35 split, stratifying the data to make sure proportionate entries of less numerous classes
x_train, x_test, y_train, y_test = train_test_split(train, encoded_vals, test_size=0.35, stratify=encoded_vals, random_state=42)


# Find accuracy
logisticReg = LogisticRegression(solver = 'saga', max_iter = 200)
logisticReg.fit(x_train,y_train)

y_pred = logisticReg.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

