"""
Simple prediction script for new text
"""

import pickle
from sentiment_analysis import TextPreprocessor

def predict_sentiment(text, model, vectorizer, preprocessor):
    """Predict sentiment for new text"""
    # Preprocess
    processed = preprocessor.preprocess(text)
    
    # Transform
    features = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability if prediction == 1 else 1 - probability
    
    return sentiment, confidence * 100

# Example usage
if __name__ == "__main__":
    # Load your trained model (save it first from sentiment_analysis.py)
    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    
    # Example predictions
    test_texts = [
        "This product is amazing and works perfectly!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special.",
    ]
    
    preprocessor = TextPreprocessor()
    
    print("\nSample Predictions:")
    print("="*60)
    for text in test_texts:
        print(f"\nText: {text}")
        # sentiment, confidence = predict_sentiment(text, model, vectorizer, preprocessor)
        # print(f"Sentiment: {sentiment} ({confidence:.1f}% confidence)")