import numpy as np
from tensorflow.keras import models, layers, callbacks, optimizers
import matplotlib.pyplot as plt
import os
import pickle

class ModelTrainer:
    def __init__(self, processed_data_dir='data/processed_v2', models_dir='models'):
        self.processed_data_dir = processed_data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.model = None
        self.history = None
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        X_train = np.load(os.path.join(self.processed_data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(self.processed_data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(self.processed_data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(self.processed_data_dir, 'y_test.npy'))
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """Build an improved neural network classifier"""
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # Deeper architecture with more capacity
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer (26 classes for A-Z)
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Using a lower learning rate for better convergence
        optimizer = optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nImproved Model Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the model with improved callbacks"""
        
        print("\nStarting training...")
        
        # Callbacks for better training
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        # Save best model during training
        checkpoint = callbacks.ModelCheckpoint(
            os.path.join(self.models_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        
        print("\nEvaluating model...")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """Plot training curves"""
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'training_history_v2.png'), dpi=150)
        print(f"\nTraining curves saved to {self.models_dir}/training_history_v2.png")
        plt.show()
    
    def save_model(self, model_name='asl_model_v2.h5'):
        """Save the trained model"""
        
        model_path = os.path.join(self.models_dir, model_name)
        self.model.save(model_path)
        print(f"\n✓ Model saved to {model_path}")
    
    def get_predictions(self, X_test, y_test):
        """Get detailed predictions"""
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Load label encoder to get letter names
        with open(os.path.join(self.processed_data_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Calculate per-class accuracy
        print("\nPer-letter accuracy:")
        accuracies = []
        for i, letter in enumerate(label_encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                accuracy = (y_pred_classes[mask] == y_test[mask]).mean()
                accuracies.append((letter, accuracy))
                
                # Color indicators
                if accuracy >= 0.95:
                    indicator = "✓✓"
                elif accuracy >= 0.85:
                    indicator = "✓"
                elif accuracy >= 0.70:
                    indicator = "→"
                else:
                    indicator = "✗"
                
                print(f"{letter}: {accuracy:.2%} {indicator}")
        
        # Summary
        avg_accuracy = np.mean([acc for _, acc in accuracies])
        print(f"\n{'='*50}")
        print(f"Average per-letter accuracy: {avg_accuracy:.2%}")
        
        # Count by performance
        excellent = sum(1 for _, acc in accuracies if acc >= 0.95)
        good = sum(1 for _, acc in accuracies if 0.85 <= acc < 0.95)
        okay = sum(1 for _, acc in accuracies if 0.70 <= acc < 0.85)
        poor = sum(1 for _, acc in accuracies if acc < 0.70)
        
        print(f"\nPerformance breakdown:")
        print(f"  Excellent (≥95%): {excellent} letters")
        print(f"  Good (85-95%):    {good} letters")
        print(f"  Okay (70-85%):    {okay} letters")
        print(f"  Poor (<70%):      {poor} letters")
        print(f"{'='*50}")
        
        return y_pred_classes

if __name__ == "__main__":
    print("="*60)
    print("IMPROVED ASL FINGERSPELLING MODEL TRAINING")
    print("="*60)
    print("\nEnhancements:")
    print("  • Using 86 engineered features (vs 63)")
    print("  • Deeper network architecture")
    print("  • Batch normalization for stability")
    print("  • More training data for weak letters")
    print("="*60)
    
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Build model
    input_shape = X_train.shape[1]  # 86 features
    num_classes = len(np.unique(y_train))  # 26 letters
    
    model = trainer.build_model(input_shape, num_classes)
    
    # Train with more epochs
    history = trainer.train(X_train, y_train, X_test, y_test, epochs=100)
    
    # Evaluate
    test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
    
    # Show predictions breakdown
    trainer.get_predictions(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    # Plot training curves
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal test accuracy: {test_accuracy*100:.2f}%")
    
    if test_accuracy >= 0.95:
        print("🎉 EXCELLENT! Near-perfect accuracy achieved!")
    elif test_accuracy >= 0.90:
        print("✓ GREAT! Very high accuracy!")
    elif test_accuracy >= 0.85:
        print("→ GOOD! Solid improvement!")
    else:
        print("→ Better, but more work needed")