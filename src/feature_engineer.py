import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class FeatureEngineer:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed_v2'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        
        # MediaPipe hand landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17
    
    def load_raw_data(self):
        """Load all landmark data"""
        X = []
        y = []
        
        print("Loading raw data...")
        
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
        print(f"Raw feature shape: {X.shape}")
        
        return X, y
    
    def extract_point(self, landmarks, index):
        """Extract a 3D point from flattened landmarks"""
        return landmarks[index*3:(index+1)*3]
    
    def euclidean_distance(self, p1, p2):
        """Calculate distance between two 3D points"""
        return np.linalg.norm(p1 - p2)
    
    def angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    
    def engineer_features(self, landmarks):
        """Extract engineered features from raw landmarks"""
        
        features = []
        
        # 1. Original landmarks (63 features)
        features.extend(landmarks)
        
        # Get key points
        wrist = self.extract_point(landmarks, self.WRIST)
        thumb_tip = self.extract_point(landmarks, self.THUMB_TIP)
        index_tip = self.extract_point(landmarks, self.INDEX_TIP)
        middle_tip = self.extract_point(landmarks, self.MIDDLE_TIP)
        ring_tip = self.extract_point(landmarks, self.RING_TIP)
        pinky_tip = self.extract_point(landmarks, self.PINKY_TIP)
        
        thumb_mcp = self.extract_point(landmarks, self.THUMB_MCP)
        index_mcp = self.extract_point(landmarks, self.INDEX_MCP)
        middle_mcp = self.extract_point(landmarks, self.MIDDLE_MCP)
        ring_mcp = self.extract_point(landmarks, self.RING_MCP)
        pinky_mcp = self.extract_point(landmarks, self.PINKY_MCP)
        
        # 2. Finger tip distances from wrist (5 features)
        features.append(self.euclidean_distance(thumb_tip, wrist))
        features.append(self.euclidean_distance(index_tip, wrist))
        features.append(self.euclidean_distance(middle_tip, wrist))
        features.append(self.euclidean_distance(ring_tip, wrist))
        features.append(self.euclidean_distance(pinky_tip, wrist))
        
        # 3. Finger spread - distances between adjacent fingertips (4 features)
        features.append(self.euclidean_distance(thumb_tip, index_tip))
        features.append(self.euclidean_distance(index_tip, middle_tip))
        features.append(self.euclidean_distance(middle_tip, ring_tip))
        features.append(self.euclidean_distance(ring_tip, pinky_tip))
        
        # 4. Finger curl - tip to MCP distances (5 features)
        features.append(self.euclidean_distance(thumb_tip, thumb_mcp))
        features.append(self.euclidean_distance(index_tip, index_mcp))
        features.append(self.euclidean_distance(middle_tip, middle_mcp))
        features.append(self.euclidean_distance(ring_tip, ring_mcp))
        features.append(self.euclidean_distance(pinky_tip, pinky_mcp))
        
        # 5. Palm orientation angles (3 features)
        palm_vector = middle_mcp - wrist
        features.append(palm_vector[0])  # X component
        features.append(palm_vector[1])  # Y component
        features.append(palm_vector[2])  # Z component
        
        # 6. Finger angles relative to palm (5 features)
        for tip, mcp in [(thumb_tip, thumb_mcp), (index_tip, index_mcp),
                          (middle_tip, middle_mcp), (ring_tip, ring_mcp),
                          (pinky_tip, pinky_mcp)]:
            finger_vec = tip - mcp
            angle = self.angle_between_vectors(finger_vec, palm_vector)
            features.append(angle)
        
        # 7. Hand span - max distance between any two fingertips (1 feature)
        tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        max_span = 0
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                span = self.euclidean_distance(tips[i], tips[j])
                max_span = max(max_span, span)
        features.append(max_span)
        
        # Total: 63 + 5 + 4 + 5 + 3 + 5 + 1 = 86 features
        
        return np.array(features)
    
    def process_all_data(self):
        """Process all data with feature engineering"""
        
        # Load raw landmarks
        X_raw, y = self.load_raw_data()
        
        print("\nEngineering features...")
        X_engineered = []
        
        for i, landmarks in enumerate(X_raw):
            features = self.engineer_features(landmarks)
            X_engineered.append(features)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(X_raw)} samples")
        
        X_engineered = np.array(X_engineered)
        
        print(f"\nEngineered feature shape: {X_engineered.shape}")
        print(f"Features increased from {X_raw.shape[1]} to {X_engineered.shape[1]}")
        
        return X_engineered, y
    
    def normalize_data(self, X):
        """Normalize features"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        
        X_normalized = (X - mean) / std
        
        return X_normalized, mean, std
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """Prepare train/test split with engineered features"""
        
        # Engineer features
        X, y = self.process_all_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nLabel mapping:")
        for i, letter in enumerate(self.label_encoder.classes_):
            print(f"{letter} -> {i}")
        
        # Normalize
        X_normalized, mean, std = self.normalize_data(X)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
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
        
        print(f"\n✓ Enhanced data saved to {self.processed_data_dir}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("="*60)
    print("FEATURE ENGINEERING FOR IMPROVED ACCURACY")
    print("="*60)
    
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.prepare_dataset()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print("\nNew features added:")
    print("  • Finger tip distances from wrist")
    print("  • Finger spread measurements")
    print("  • Finger curl indicators")
    print("  • Palm orientation vectors")
    print("  • Finger angles relative to palm")
    print("  • Hand span measurements")
    print("\nTotal features: 63 → 86")
    print("\nReady for improved model training!")