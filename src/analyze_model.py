import numpy as np
from tensorflow.keras import models
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ModelAnalyzer:
    def __init__(self, model_path='models/asl_model_v2.h5', 
                 processed_data_dir='data/processed_v2'):
        
        # Load model
        self.model = models.load_model(model_path)
        
        # Load test data
        self.X_test = np.load(os.path.join(processed_data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(processed_data_dir, 'y_test.npy'))
        
        # Load label encoder
        with open(os.path.join(processed_data_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def get_predictions(self):
        """Get model predictions on test set"""
        predictions = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        
        y_pred = self.get_predictions()
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - ASL Letters', fontsize=16)
        plt.ylabel('True Letter', fontsize=12)
        plt.xlabel('Predicted Letter', fontsize=12)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix_v2.png', dpi=150)
        print("✓ Confusion matrix saved to models/confusion_matrix_v2.png")
        plt.show()
    
    def find_most_confused_pairs(self, top_n=10):
        """Find which letter pairs are most confused"""
        
        y_pred = self.get_predictions()
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Find off-diagonal elements (misclassifications)
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    true_letter = self.label_encoder.classes_[i]
                    pred_letter = self.label_encoder.classes_[j]
                    count = cm[i, j]
                    confused_pairs.append((true_letter, pred_letter, count))
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\n" + "="*60)
        print(f"TOP {top_n} MOST CONFUSED LETTER PAIRS")
        print("="*60)
        print(f"{'True Letter':<12} {'Predicted As':<12} {'Count':<8}")
        print("-"*60)
        
        for true_letter, pred_letter, count in confused_pairs[:top_n]:
            print(f"{true_letter:<12} {pred_letter:<12} {count:<8}")
        
        return confused_pairs[:top_n]
    
    def get_per_letter_accuracy(self):
        """Get accuracy for each letter"""
        
        y_pred = self.get_predictions()
        
        print("\n" + "="*60)
        print("PER-LETTER ACCURACY")
        print("="*60)
        print(f"{'Letter':<8} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-"*60)
        
        accuracies = []
        
        for i, letter in enumerate(self.label_encoder.classes_):
            mask = self.y_test == i
            if mask.sum() > 0:
                correct = (y_pred[mask] == self.y_test[mask]).sum()
                total = mask.sum()
                accuracy = correct / total
                accuracies.append((letter, accuracy, correct, total))
                
                # Color code output
                if accuracy >= 0.9:
                    status = "✓ GOOD"
                elif accuracy >= 0.7:
                    status = "→ OK"
                else:
                    status = "✗ POOR"
                
                print(f"{letter:<8} {accuracy:>6.1%} {status:<8} {correct}/{total}")
        
        # Sort by accuracy (worst first)
        accuracies.sort(key=lambda x: x[1])
        
        print("\n" + "="*60)
        print("LETTERS THAT NEED MORE TRAINING DATA:")
        print("="*60)
        
        for letter, accuracy, correct, total in accuracies[:5]:
            if accuracy < 0.8:
                print(f"  • {letter} - Only {accuracy:.1%} accurate ({correct}/{total})")
        
        return accuracies
    
    def show_classification_report(self):
        """Show detailed classification report"""
        
        y_pred = self.get_predictions()
        
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            digits=3
        )
        print(report)

if __name__ == "__main__":
    print("="*60)
    print("ASL MODEL ANALYSIS")
    print("="*60)
    
    analyzer = ModelAnalyzer()
    
    # Per-letter accuracy
    accuracies = analyzer.get_per_letter_accuracy()
    
    # Most confused pairs
    confused_pairs = analyzer.find_most_confused_pairs(top_n=15)
    
    # Detailed report
    analyzer.show_classification_report()
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    analyzer.plot_confusion_matrix()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)