import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences  # ADD THIS LINE
from preprocessor import TextPreprocessor
from traditional_models import TraditionalMLModels
from deep_learning_models import DeepLearningModels
import config
import warnings
warnings.filterwarnings('ignore')


def generate_sample_data(n_samples=2000):
    """
    Generate sample sentiment data for demonstration
    """
    print("Generating sample sentiment data...")
    
    positive_samples = [
        "This product is absolutely amazing! I love it so much.",
        "Excellent quality and fast shipping. Highly recommended!",
        "Best purchase I've made this year. Very satisfied.",
        "Outstanding performance. Exceeded my expectations.",
        "Great value for money. Will definitely buy again.",
        "Superb quality! Very happy with this purchase.",
        "Fantastic product! Works perfectly as described.",
        "Extremely satisfied. Worth every penny!",
        "Wonderful experience! Highly recommend to everyone.",
        "Perfect! Exactly what I was looking for.",
    ] * (n_samples // 20)
    
    negative_samples = [
        "Terrible product. Waste of money. Very disappointed.",
        "Poor quality and bad customer service. Not recommended.",
        "Worst purchase ever. Broke after one day.",
        "Completely useless. Don't buy this product.",
        "Very unhappy with this purchase. Total disaster.",
        "Awful quality! Regret buying this product.",
        "Horrible experience. Product not as described.",
        "Pathetic quality and worst customer support ever.",
        "Disappointing purchase. Complete waste of money.",
        "Terrible! Stopped working within a week.",
    ] * (n_samples // 20)
    
    # Combine and create labels
    texts = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    # Shuffle data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels


def visualize_results(results_traditional, results_dl):
    """
    Create visualization comparing all models
    """
    # Combine all results
    all_results = {**results_traditional, **results_dl}
    df = pd.DataFrame(all_results).T
    
    # Print results table
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(df.round(4))
    print("\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        bars = df[metric].plot(kind='bar', ax=ax, color=colors[:len(df)], edgecolor='black', width=0.7)
        
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylim([0.7, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.suptitle('Sentiment Analysis: Model Performance Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    print("Results visualization saved as 'results.png'")
    plt.show()


def main():
    """
    Main execution function for sentiment analysis experiment
    """
    # Print header
    print("="*70)
    print("SENTIMENT ANALYSIS: TRADITIONAL ML vs DEEP LEARNING")
    print("="*70)
    print("Comparative Study of Sentiment Classification Methods")
    print("Natural Language Processing Research")
    print("="*70)
    
    # Step 1: Generate data
    texts, labels = generate_sample_data(2000)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(texts)}")
    print(f"  Positive samples: {sum(labels)}")
    print(f"  Negative samples: {len(labels) - sum(labels)}")
    
    # Step 2: Split data
    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
        stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Step 3: Preprocessing
    print("\n" + "="*70)
    print("TEXT PREPROCESSING")
    print("="*70)
    preprocessor = TextPreprocessor()
    
    # Preprocess for traditional models
    X_train_processed = preprocessor.preprocess_corpus(X_train)
    X_test_processed = preprocessor.preprocess_corpus(X_test)
    
    print("Sample preprocessing:")
    print(f"  Original: {X_train[0]}")
    print(f"  Processed: {X_train_processed[0]}")
    
    # Step 4: Traditional ML Models
    print("\n" + "="*70)
    print("TRADITIONAL MACHINE LEARNING MODELS")
    print("="*70)
    
    traditional = TraditionalMLModels()
    X_train_features, X_test_features = traditional.create_features(
        X_train_processed, X_test_processed
    )
    results_traditional = traditional.train_all(
        X_train_features, X_test_features, y_train, y_test
    )
    
    # Step 5: Deep Learning Models
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS")
    print("="*70)
    
    # Minimal preprocessing for DL
    X_train_dl = [preprocessor.clean_text(t) for t in X_train]
    X_val_dl = [preprocessor.clean_text(t) for t in X_val]
    X_test_dl = [preprocessor.clean_text(t) for t in X_test]
    
    dl_models = DeepLearningModels()
    X_train_pad, X_test_pad = dl_models.prepare_data(X_train_dl, X_test_dl)
    
    # Prepare validation data
    X_val_seq = dl_models.tokenizer.texts_to_sequences(X_val_dl)
    X_val_pad = pad_sequences(X_val_seq, maxlen=dl_models.max_len, padding='post')
    
    results_dl = {}
    
    # Train CNN
    print("\n" + "-"*70)
    print("TRAINING CNN MODEL")
    print("-"*70)
    dl_models.build_cnn()
    dl_models.train_model('CNN', X_train_pad, np.array(y_train),
                         X_val_pad, np.array(y_val))
    metrics = dl_models.evaluate_model('CNN', X_test_pad, np.array(y_test))
    results_dl['CNN'] = metrics
    print(f"\nCNN Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1']:.4f}")
    
    # Train LSTM
    print("\n" + "-"*70)
    print("TRAINING LSTM MODEL")
    print("-"*70)
    dl_models.build_lstm()
    dl_models.train_model('LSTM', X_train_pad, np.array(y_train),
                         X_val_pad, np.array(y_val))
    metrics = dl_models.evaluate_model('LSTM', X_test_pad, np.array(y_test))
    results_dl['LSTM'] = metrics
    print(f"\nLSTM Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1']:.4f}")
    
    # Train Bi-LSTM with Attention
    print("\n" + "-"*70)
    print("TRAINING BI-LSTM WITH ATTENTION")
    print("-"*70)
    dl_models.build_bilstm_attention()
    dl_models.train_model('Bi-LSTM+Attention', X_train_pad, np.array(y_train),
                         X_val_pad, np.array(y_val))
    metrics = dl_models.evaluate_model('Bi-LSTM+Attention', X_test_pad, np.array(y_test))
    results_dl['Bi-LSTM+Attention'] = metrics
    print(f"\nBi-LSTM+Attention Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1']:.4f}")
    
    # Step 6: Visualize Results
    visualize_results(results_traditional, results_dl)
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nKey Findings:")
    all_results = {**results_traditional, **results_dl}
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"  Best Model: {best_model[0]}")
    print(f"  Best Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"  Best F1-Score: {best_model[1]['f1']:.4f}")
    print("\nOutput files generated:")
    print("  - results.png (Visualization)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()