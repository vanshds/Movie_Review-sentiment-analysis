import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load data
df = pd.read_csv('IMDB Dataset.csv')

X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
total_words = len(tokenizer.word_index) + 1

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Sequence Preparation
sequences = tokenizer.texts_to_sequences(X)
max_sequence_len = max(len(seq) for seq in sequences)

X_padded = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

# Build model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_padded, y, epochs=5, batch_size=64)

# Save model
model.save('sentiment_model.h5')

# Save sequence length
with open('max_sequence_len.pkl', 'wb') as f:
    pickle.dump(max_sequence_len, f)
