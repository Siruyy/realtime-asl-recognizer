import numpy as np
from tensorflow.keras import models, layers, callbacks, optimizers
import matplotlib.pyplot as plt
import os
import pickle

class StaticModelTrainer:
    """Train model on static letters only (exclude J and Z)"""
    
    def __init__(self, processed_data_dir='data/processed_v2', models_dir='models'):
        self.processed_data_dir = processed_data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Letters to EXCLUDE (motion-based)
        self.excluded_letters = ['J', 'Z']
        
        self.model = None
        self.history = None
        self.label_encoder = None
    
    def load_and_filter_data(self):
        """Load data and exclude motion-based letters"""
        print("Loading preprocessed data...")
        
        X_train = np.load(os.path.join(self.processed_data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(self.processed_data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(self.processed_data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(self.processed_data_dir, 'y_test.npy'))
        
        # Load label encoder
        with open(os.path.join(self.processed_data_dir, 'label_encoder.pkl'), 'rb') as f:
            full_label_encoder = pickle.load(f)
        
        print(f"\nOriginal data: Train={len(X_train)}, Test={len(X_test)}")
        
        # Find indices to exclude
        excluded_indices = [i for i, letter in enumerate(full_label_encoder.classes_) 
                           if letter in self.excluded_letters]
        
        print(f"Excluding letters: {self.excluded_letters}")
        print(f"Excluded class indices: {excluded_indices}")
        
        # Filter training data
        train_mask = ~np.isin(y_train, excluded_indices)
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        
        # Filter test data
        test_mask = ~np.isin(y_test, excluded_indices)
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        print(f"Filtered data: Train={len(X_train_filtered)}, Test={len(X_test_filtered)}")
        
        # Create new label encoder for remaining letters
        remaining_letters = [letter for letter in full_label_encoder.classes_ 
                            if letter not in self.excluded_letters]
        
        # Remap labels to be continuous (0-23 instead of having gaps)
        label_mapping = {old_idx: new_idx for new_idx, old_idx in 
                        enumerate([i for i in range(len(full_label_encoder.classes_)) 
                                  if i not in excluded_indices])}
        
        y_train_remapped = np.array([label_mapping[y] for y in y_train_filtered])
        y_test_remapped = np.array([label_mapping[y] for y in y_test_filtered])
        
        # Create new label encoder
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(remaining_letters)
        
        print(f"\nStatic letters (24 total): {', '.join(remaining_letters)}")
        
        return X_train_filtered, X_test_filtered, y_train_remapped, y_test_remapped
    
    def build_model(self, input_shape, num_classes):
        """Build model for static letters only"""
        
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            
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
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nStatic Letters Model Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the model"""
        
        print("\nStarting training...")
        
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
        
        checkpoint = callbacks.ModelCheckpoint(
            os.path.join(self.models_dir, 'best_static_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
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
        
        print("\nEvaluating static model...")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy
    
    def get_predictions(self, X_test, y_test):
        """Get detailed predictions"""
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nPer-letter accuracy (24 static letters):")
        accuracies = []
        
        for i, letter in enumerate(self.label_encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                accuracy = (y_pred_classes[mask] == y_test[mask]).mean()
                accuracies.append((letter, accuracy))
                
                if accuracy >= 0.95:
                    indicator = "✓✓"
                elif accuracy >= 0.85:
                    indicator = "✓"
                elif accuracy >= 0.70:
                    indicator = "→"
                else:
                    indicator = "✗"
                
                print(f"{letter}: {accuracy:.2%} {indicator}")
        
        avg_accuracy = np.mean([acc for _, acc in accuracies])
        print(f"\n{'='*50}")
        print(f"Average accuracy (24 static letters): {avg_accuracy:.2%}")
        
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
    
    def save_model(self):
        """Save model and label encoder"""
        
        model_path = os.path.join(self.models_dir, 'asl_static_model.h5')
        self.model.save(model_path)
        print(f"\n✓ Model saved to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.models_dir, 'static_label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved to {encoder_path}")
    
    def plot_training_history(self):
        """Plot training curves"""
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy (24 Static Letters)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss (24 Static Letters)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'training_history_static.png'), dpi=150)
        print(f"\nTraining curves saved to {self.models_dir}/training_history_static.png")
        plt.show()

if __name__ == "__main__":
    print("="*60)
    print("ASL STATIC LETTERS MODEL (24 LETTERS - EXCLUDING J & Z)")
    print("="*60)
    print("\nFocusing on letters that don't require motion:")
    print("  • Excluding: J, Z (motion-based)")
    print("  • Training: 24 static letters")
    print("  • Expected accuracy: 95%+")
    print("="*60)
    
    trainer = StaticModelTrainer()
    
    # Load and filter data
    X_train, X_test, y_train, y_test = trainer.load_and_filter_data()
    
    # Build model
    input_shape = X_train.shape[1]
    num_classes = len(trainer.label_encoder.classes_)
    
    print(f"\nTraining on {num_classes} static letters")
    
    model = trainer.build_model(input_shape, num_classes)
    
    # Train
    history = trainer.train(X_train, y_train, X_test, y_test, epochs=100)
    
    # Evaluate
    test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
    
    # Show predictions
    trainer.get_predictions(X_test, y_test)
    
    # Save
    trainer.save_model()
    
    # Plot
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("STATIC MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal test accuracy: {test_accuracy*100:.2f}%")
    
    if test_accuracy >= 0.95:
        print("🎉 EXCELLENT! Near-perfect accuracy on static letters!")
    elif test_accuracy >= 0.90:
        print("✓ GREAT! Very high accuracy!")