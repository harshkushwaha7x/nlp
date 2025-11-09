"""
Complete Sentiment Analysis Implementation
Based on the research paper by Harsh Prakash Kushwaha
"""

import numpy as np
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ============================================================================
# PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Complete preprocessing pipeline from Section III.B"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}
        self.stop_words -= self.negation_words
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Expand contractions
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'ve", " have")
        text = text.replace("'ll", " will")
        text = text.replace("'d", " would")
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize and stem text"""
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token.isalpha() and token not in self.stop_words]
        return tokens
    
    def handle_negation(self, tokens):
        """Handle negation scope - Section III.B.6"""
        result = []
        negation_scope = False
        
        for token in tokens:
            if token in self.negation_words:
                negation_scope = True
                result.append(token)
            elif token in ['.', '!', '?', ',', ';']:
                negation_scope = False
            elif negation_scope:
                result.append('NOT_' + token)
            else:
                result.append(token)
        
        return result
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize_and_stem(text)
        tokens = self.handle_negation(tokens)
        return ' '.join(tokens)

# ============================================================================
# MODELS
# ============================================================================

class NaiveBayesModel:
    """Naive Bayes Classifier - Section III.D.1"""
    
    def __init__(self):
        self.model = MultinomialNB(alpha=1.0)
        self.name = "Naive Bayes"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

class SVMModel:
    """SVM Classifier - Section III.D.2"""
    
    def __init__(self):
        self.model = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        self.name = "SVM (Linear)"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        scores = self.model.decision_function(X_test)
        # Normalize to probabilities
        return (scores - scores.min()) / (scores.max() - scores.min())

class CNNModel:
    """CNN Model - Section III.D.5"""
    
    def __init__(self, vocab_size=10000, embedding_dim=100, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.name = "CNN"
        self.build_model()
    
    def build_model(self):
        inputs = layers.Input(shape=(self.max_length,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Multiple filter sizes (3, 4, 5)
        conv_blocks = []
        for filter_size in [3, 4, 5]:
            conv = layers.Conv1D(128, filter_size, activation='relu')(x)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        x = layers.Concatenate()(conv_blocks)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', 
                          metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=batch_size, 
                      callbacks=[early_stopping], verbose=0)
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X_test):
        return self.model.predict(X_test, verbose=0).flatten()

class LSTMModel:
    """Bi-LSTM with Attention - Section III.D.6"""
    
    def __init__(self, vocab_size=10000, embedding_dim=100, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.name = "Bi-LSTM+Attention"
        self.build_model()
    
    def build_model(self):
        inputs = layers.Input(shape=(self.max_length,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Bidirectional LSTM
        lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(256)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        sent_representation = layers.Multiply()([lstm_out, attention])
        sent_representation = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sent_representation)
        
        x = layers.Dropout(0.5)(sent_representation)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                          loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=batch_size,
                      callbacks=[early_stopping], verbose=0)
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X_test):
        return self.model.predict(X_test, verbose=0).flatten()

class EnsembleModel:
    """Ensemble with weighted voting - Section IV.K"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.name = "Ensemble"
        if weights is None:
            self.weights = [0.824, 0.857, 0.938]  # From Table I
        else:
            self.weights = weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def predict(self, X_features, X_sequences):
        predictions = []
        
        # Traditional models (NB, SVM)
        for i in range(2):
            prob = self.models[i].predict_proba(X_features)
            predictions.append(prob)
        
        # Neural model (LSTM)
        prob = self.models[2].predict_proba(X_sequences)
        predictions.append(prob)
        
        # Weighted voting
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        return (weighted_pred > 0.5).astype(int)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model with metrics from Section III.F"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall:    {rec:.1%}")
    print(f"F1-Score:  {f1:.1%}")
    print(f"{'='*60}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS: COMPARATIVE STUDY")
    print("Traditional ML vs Deep Learning Approaches")
    print("="*70)
    
    # Sample dataset (replace with your data)
    print("\n[1/6] Loading dataset...")
    # Example data structure - replace with: df = pd.read_csv('your_data.csv')
    texts = [
        "This product is absolutely amazing! I love it so much!",
        "Terrible quality, complete waste of money. Very disappointed.",
        "Good value for money, works exactly as expected.",
        "Worst purchase ever. Do not buy this product!",
        "Excellent service and outstanding product quality!",
        "Not satisfied at all. Poor quality and bad service.",
        "Great features and very user-friendly interface!",
        "Horrible experience. Would not recommend to anyone.",
    ] * 20  # Multiply to have more samples
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 20)
    
    print(f"Dataset size: {len(texts)} samples")
    
    # Preprocessing
    print("\n[2/6] Preprocessing texts...")
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    # Split data (70% train, 15% val, 15% test)
    print("\n[3/6] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Feature extraction for traditional models
    print("\n[4/6] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Prepare sequences for neural models
    print("\n[5/6] Preparing sequences...")
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), 
                                maxlen=200, padding='post')
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val),
                             maxlen=200, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test),
                              maxlen=200, padding='post')
    
    # Train and evaluate models
    print("\n[6/6] Training and evaluating models...")
    results = {}
    
    # 1. Naive Bayes
    print("\n--- Training Naive Bayes ---")
    start = time.time()
    nb_model = NaiveBayesModel()
    nb_model.train(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_time = time.time() - start
    results['Naive Bayes'] = evaluate_model(y_test, nb_pred, "Naive Bayes")
    results['Naive Bayes']['time'] = nb_time
    
    # 2. SVM
    print("\n--- Training SVM ---")
    start = time.time()
    svm_model = SVMModel()
    svm_model.train(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_time = time.time() - start
    results['SVM'] = evaluate_model(y_test, svm_pred, "SVM (Linear)")
    results['SVM']['time'] = svm_time
    
    # 3. CNN
    print("\n--- Training CNN ---")
    start = time.time()
    cnn_model = CNNModel()
    cnn_model.train(X_train_seq, y_train, X_val_seq, y_val, epochs=10)
    cnn_pred = cnn_model.predict(X_test_seq)
    cnn_time = time.time() - start
    results['CNN'] = evaluate_model(y_test, cnn_pred, "CNN")
    results['CNN']['time'] = cnn_time
    
    # 4. LSTM
    print("\n--- Training Bi-LSTM with Attention ---")
    start = time.time()
    lstm_model = LSTMModel()
    lstm_model.train(X_train_seq, y_train, X_val_seq, y_val, epochs=10)
    lstm_pred = lstm_model.predict(X_test_seq)
    lstm_time = time.time() - start
    results['LSTM'] = evaluate_model(y_test, lstm_pred, "Bi-LSTM+Attention")
    results['LSTM']['time'] = lstm_time
    
    # 5. Ensemble
    print("\n--- Creating Ensemble ---")
    ensemble = EnsembleModel([nb_model, svm_model, lstm_model])
    ensemble_pred = ensemble.predict(X_test_tfidf, X_test_seq)
    results['Ensemble'] = evaluate_model(y_test, ensemble_pred, "Ensemble")
    
    # Final comparison table
    print("\n" + "="*70)
    print("FINAL RESULTS - TABLE I COMPARISON")
    print("="*70)
    print(f"{'Model':<20} {'Acc.':<8} {'Prec.':<8} {'Rec.':<8} {'F1':<8} {'Time(s)':<10}")
    print("-"*70)
    for name, metrics in results.items():
        time_str = f"{metrics.get('time', 0):.1f}" if 'time' in metrics else "N/A"
        print(f"{name:<20} {metrics['accuracy']:.1%}  "
              f"{metrics['precision']:.1%}  "
              f"{metrics['recall']:.1%}  "
              f"{metrics['f1_score']:.1%}  "
              f"{time_str:<10}")
    print("="*70)
    
    # Expected results from paper
    print("\nExpected Results from Paper (Table I):")
    print("-"*70)
    print("Naive Bayes          82.4%    81.9%    82.8%    82.3%")
    print("SVM (Linear)         85.7%    85.2%    86.1%    85.6%")
    print("CNN                  89.6%    89.3%    89.9%    89.6%")
    print("Bi-LSTM+Attn         93.8%    93.6%    94.0%    93.8%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()