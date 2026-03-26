import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
TRAIN_CSV = os.path.join(BASE_DIR, 'CSV', 'prompt_engineering_dataset_train.csv')

# --- CUSTOM CALLBACK FOR 94% CAP ---
class AccuracyCapCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        if acc and acc >= 0.94:
            print(f"\n[INFO] Accuracy reached {acc*100:.2f}% (Limit: 94%). Stopping training to prevent overfitting.")
            self.model.stop_training = True

def train():
    if not os.path.exists(TRAIN_CSV):
        print("Dataset not found. Run dataset_gen.py first.")
        return

    print(">>> Loading Data...")
    df = pd.read_csv(TRAIN_CSV)
    df['text'] = df['Prompt'].astype(str)
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['HCI_Application'])
    
    # Tokenization
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=50)
    
    # Model Architecture
    print(">>> Building BiLSTM...")
    model = Sequential([
        Embedding(5000, 64, input_length=50),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    
    # Train with Cap
    print(">>> Training (Capped at 94%)...")
    model.fit(
        X, y, 
        epochs=15, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1,
        callbacks=[AccuracyCapCallback()] # Enforce Limit
    )
    
    # Save Artifacts
    print(">>> Saving Artifacts...")
    model.save(os.path.join(MODEL_DIR, 'berthci_model.h5'))
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f: pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f: pickle.dump(label_encoder, f)
    print("Model Saved.")

if __name__ == '__main__':
    train()