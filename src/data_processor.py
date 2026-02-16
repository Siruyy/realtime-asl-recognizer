import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class DataProcessor:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """Load all landmark data and labels"""
        X = []  # Features (landmarks)
        y = []  # Labels (letters)
        
        print("Loading data...")
        
        # Get all letter folders
        letters = sorted([d for d in os.listdir(self.raw_data_dir) 
                         if os.path.isdir(os.path.join(self.raw_data_dir, d))])
        
        for letter in letters:
            letter_dir = os.path.join(self.raw_data_dir, letter)
            files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
            
            print(f"Loading {len(files)} samples for letter {letter}")
            
            for file in files:
                filepath = os.path.join(letter_dir, file)
                landmarks = np.load(filepath)
                
                X.append(landmarks)
                y.append(letter)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Classes: {len(np.unique(y))}")
        
        return X, y
    
    def normalize_data(self, X):
        """Normalize landmark coordinates"""
        # Landmarks are already 0-1 normalized by MediaPipe
        # But we can standardize them further
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        X_normalized = (X - mean) / std
        
        return X_normalized, mean, std
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """Prepare train/test split and save"""
        
        # Load data
        X, y = self.load_data()
        
        # Encode labels (A-Z -> 0-25)
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nLabel mapping:")
        for i, letter in enumerate(self.label_encoder.classes_):
            print(f"{letter} -> {i}")
        
        # Normalize features
        X_normalized, mean, std = self.normalize_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded  # Ensure balanced split
        )
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Save processed data
        np.save(os.path.join(self.processed_data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_data_dir, 'y_test.npy'), y_test)
        
        # Save normalization parameters
        np.save(os.path.join(self.processed_data_dir, 'mean.npy'), mean)
        np.save(os.path.join(self.processed_data_dir, 'std.npy'), std)
        
        # Save label encoder
        with open(os.path.join(self.processed_data_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n✓ Data saved to {self.processed_data_dir}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_dataset()
    
    print("\n" + "="*50)
    print("Data preprocessing complete!")
    print("="*50)
    print("\nReady for model training!")