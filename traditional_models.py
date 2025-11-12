from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config


class TraditionalMLModels:
    """
    Implementation of traditional machine learning models:
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Random Forest
    - Logistic Regression
    """
    
    def __init__(self):
        # Initialize all models
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'SVM': LinearSVC(
                C=1.0,
                random_state=config.RANDOM_SEED,
                max_iter=2000,
                dual=False
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=50,
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            )
        }
        
        self.vectorizer = None
        self.trained_models = {}
    
    def create_features(self, X_train, X_test):
        """
        Create TF-IDF features from text
        TF-IDF = Term Frequency - Inverse Document Frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=config.NGRAM_RANGE,
            min_df=2,
            sublinear_tf=True,
            norm='l2'
        )
        
        # Fit on training data and transform both train and test
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_features.shape}")
        
        return X_train_features, X_test_features
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        """
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate model performance
        Returns accuracy, precision, recall, and F1-score
        """
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary')
        }
        
        return metrics
    
    def train_all(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all traditional models
        """
        results = {}
        
        for model_name in self.models.keys():
            # Train
            self.train_model(model_name, X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results[model_name] = metrics
            
            # Print results
            print(f"\n{model_name} Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        return results
