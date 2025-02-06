import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_data = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
test_data = train_data.sample(frac=0.2, random_state=42)  # 20% for testing
train_data = train_data.drop(test_data.index)

# Convert text labels ("ham", "spam") to numerical labels (0, 1)
label_encoder = LabelEncoder()
train_data["label"] = label_encoder.fit_transform(train_data["label"])
test_data["label"] = label_encoder.transform(test_data["label"])

# Tokenization
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["message"])

# Convert messages to sequences
X_train = tokenizer.texts_to_sequences(train_data["message"])
X_test = tokenizer.texts_to_sequences(test_data["message"])

# Padding sequences to have the same length
max_length = 100
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

# Convert labels to NumPy arrays
y_train = np.array(train_data["label"])
y_test = np.array(test_data["label"])

# Define the neural network model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Define prediction function
def predict_message(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)[0][0]
    label = "spam" if prediction > 0.5 else "ham"
    return [float(prediction), label]

# Example predictions
print(predict_message("Congratulations! You won a free prize. Click here to claim."))
print(predict_message("Hey, are we still on for lunch tomorrow?"))
