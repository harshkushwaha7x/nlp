import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class TextPreprocessor:
    """
    Text preprocessing pipeline for sentiment analysis
    Implements cleaning, tokenization, negation handling, and lemmatization
    """
    
    def __init__(self):
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Keep negation words as they're crucial for sentiment
        self.stop_words -= {
            'not', 'no', 'nor', 'never', 'none', 'neither',
            "n't", "don't", "doesn't", "didn't", "won't",
            "wouldn't", "can't", "couldn't", "shouldn't"
        }
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Negation words for scope detection
        self.negation_words = {
            'not', 'no', 'nor', 'never', 'none', 'neither', "n't"
        }
    
    def clean_text(self, text):
        """
        Remove HTML tags, URLs, special characters
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def handle_negation(self, tokens):
        """
        Handle negation by marking words in negation scope with NOT_ prefix
        Example: "not good" becomes "not NOT_good"
        """
        result = []
        negate = False
        
        for token in tokens:
            # Check if token is a negation word
            if token in self.negation_words or token.endswith("n't"):
                negate = True
                result.append(token)
            # Punctuation ends negation scope
            elif token in {'.', '!', '?', ',', 'but', 'however', 'although'}:
                negate = False
                result.append(token)
            # Apply NOT_ prefix to words in negation scope
            elif negate:
                result.append(f"NOT_{token}")
            else:
                result.append(token)
        
        return result
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline:
        1. Clean text
        2. Tokenize
        3. Remove stopwords
        4. Handle negation
        5. Lemmatize
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Tokenization
        tokens = word_tokenize(text)
        
        # Step 3: Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Step 4: Handle negation
        tokens = self.handle_negation(tokens)
        
        # Step 5: Lemmatization
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts):
        """
        Preprocess a list of texts
        """
        return [self.preprocess(text) for text in texts]