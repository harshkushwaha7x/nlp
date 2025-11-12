import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Bidirectional,
    Dropout, Conv1D, GlobalMaxPooling1D,
    Input, RepeatVector, Permute, Flatten,
    Activation, Lambda, multiply
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config


class DeepLearningModels:
    """
    Implementation of deep learning models:
    - CNN (Convolutional Neural Network)
    - LSTM (Long Short-Term Memory)
    - Bi-LSTM with Attention
    """
    
    def __init__(self):
        self.max_words = config.MAX_FEATURES
        self.max_len = config.MAX_SEQUENCE_LENGTH
        self.embedding_dim = config.EMBEDDING_DIM
        self.tokenizer = None
        self.models = {}
    
    def prepare_data(self, X_train, X_test):
        """
        Tokenize and pad sequences for neural networks
        """
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences to same length
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X_train_pad.shape}")
        
        return X_train_pad, X_test_pad
    
    def build_cnn(self):
        """
        Build CNN model for text classification
        Architecture:
        - Embedding layer
        - Convolutional layer
        - Global max pooling
        - Dense layers with dropout
        """
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(config.DROPOUT_RATE),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['CNN'] = model
        print(f"\nCNN Model Summary:")
        model.summary()
        return model
    
    def build_lstm(self):
        """
        Build LSTM model
        Architecture:
        - Embedding layer
        - LSTM layer
        - Dropout
        - Dense output layer
        """
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            LSTM(128, return_sequences=False),
            Dropout(config.DROPOUT_RATE),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['LSTM'] = model
        print(f"\nLSTM Model Summary:")
        model.summary()
        return model
    
    def build_bilstm_attention(self):
        """
        Build Bidirectional LSTM with Attention mechanism
        Architecture:
        - Embedding layer
        - Bidirectional LSTM
        - Attention mechanism
        - Dense output layer
        """
        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_words, self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(config.DROPOUT_RATE)(x)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(256)(attention)
        attention = Permute([2, 1])(attention)
        
        # Apply attention weights
        merged = multiply([x, attention])
        merged = Lambda(lambda z: tf.keras.backend.sum(z, axis=1))(merged)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(merged)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['Bi-LSTM+Attention'] = model
        print(f"\nBi-LSTM+Attention Model Summary:")
        model.summary()
        return model
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """
        Train deep learning model with callbacks
        """
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        
        # Callbacks for training optimization
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate deep learning model
        """
        model = self.models[model_name]
        
        # Get predictions
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary')
        }
        
        return metrics