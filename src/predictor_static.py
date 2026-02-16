import cv2
import numpy as np
from tensorflow.keras import models
import pickle
import os
from hand_detector import HandDetector
from feature_engineer import FeatureEngineer

class ASLStaticPredictor:
    def __init__(self, model_path='models/asl_static_model.h5', 
                 models_dir='models',
                 processed_data_dir='data/processed_v2'):
        
        print("Loading ASL Static Model (24 letters)...")
        
        # Load trained model
        self.model = models.load_model(model_path)
        
        # Load normalization parameters
        self.mean = np.load(os.path.join(processed_data_dir, 'mean.npy'))
        self.std = np.load(os.path.join(processed_data_dir, 'std.npy'))
        
        # Load label encoder
        encoder_path = os.path.join(models_dir, 'static_label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Feature engineer for extracting enhanced features
        self.feature_engineer = FeatureEngineer()
        
        # Hand detector
        self.detector = HandDetector()
        
        print(f"✓ Model loaded!")
        print(f"✓ Recognizes {len(self.label_encoder.classes_)} letters: {', '.join(self.label_encoder.classes_)}")
        print("✓ Note: J and Z excluded (motion-based)")
    
    def extract_landmarks(self, results):
        """Extract hand landmarks as numpy array"""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def normalize_features(self, features):
        """Normalize features using saved parameters"""
        return (features - self.mean) / self.std
    
    def predict(self, landmarks):
        """Predict ASL letter from landmarks with feature engineering"""
        
        # Engineer features (63 → 86 features)
        features = self.feature_engineer.engineer_features(landmarks)
        
        # Normalize
        features_normalized = self.normalize_features(features)
        
        # Reshape for model input
        features_input = features_normalized.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(features_input, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Convert to letter
        predicted_letter = self.label_encoder.classes_[predicted_class]
        
        return predicted_letter, confidence, predictions[0]
    
    def run_live_prediction(self, confidence_threshold=0.7):
        """Run real-time ASL recognition"""
        
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("REAL-TIME ASL STATIC LETTERS RECOGNITION (24 LETTERS)")
        print("="*60)
        print("Controls:")
        print("  • Press 'q' to quit")
        print("  • Press '+' to increase confidence threshold")
        print("  • Press '-' to decrease confidence threshold")
        print(f"\nCurrent confidence threshold: {confidence_threshold:.1%}")
        print("\nExcluded letters (motion-based): J, Z")
        print("="*60)
        
        # For smoothing predictions
        prediction_history = []
        history_size = 5
        
        # FPS calculation
        import time
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            results = self.detector.detect_hands(frame)
            frame = self.detector.draw_hands(frame, results)
            
            # Extract landmarks and predict
            landmarks = self.extract_landmarks(results)
            
            if landmarks is not None:
                try:
                    predicted_letter, confidence, all_probs = self.predict(landmarks)
                    
                    # Add to history for smoothing
                    prediction_history.append(predicted_letter)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)
                    
                    # Get most common prediction from history
                    if len(prediction_history) >= 3:
                        from collections import Counter
                        smoothed_letter = Counter(prediction_history).most_common(1)[0][0]
                    else:
                        smoothed_letter = predicted_letter
                    
                    # Display prediction
                    if confidence >= confidence_threshold:
                        # High confidence - green
                        color = (0, 255, 0)
                        display_text = f"{smoothed_letter}"
                        conf_text = f"Confidence: {confidence:.1%}"
                    else:
                        # Low confidence - yellow
                        color = (0, 255, 255)
                        display_text = f"{predicted_letter}?"
                        conf_text = f"Confidence: {confidence:.1%} (Low)"
                    
                    # Draw black rectangle for text background
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 200), (0, 0, 0), -1)
                    
                    # Draw large prediction
                    cv2.putText(frame, display_text, (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
                    cv2.putText(frame, conf_text, (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show top 3 predictions
                    top_3_idx = np.argsort(all_probs)[-3:][::-1]
                    y_pos = 170
                    cv2.putText(frame, "Top 3:", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    for idx in top_3_idx:
                        letter = self.label_encoder.classes_[idx]
                        prob = all_probs[idx]
                        text = f"{letter}: {prob:.0%}"
                        cv2.putText(frame, text, (80 + (top_3_idx.tolist().index(idx) * 120), y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                except Exception as e:
                    cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # No hand detected
                prediction_history.clear()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (40, 40, 40), -1)
                cv2.putText(frame, "No hand detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0
            
            # Display FPS and threshold
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {confidence_threshold:.0%}", (frame.shape[1] - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display instructions at bottom
            instructions = "Q: Quit  |  +/-: Adjust threshold"
            cv2.putText(frame, instructions, (10, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            
            # Display frame
            cv2.imshow("ASL Static Recognition (24 Letters)", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Confidence threshold: {confidence_threshold:.0%}")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.3, confidence_threshold - 0.05)
                print(f"Confidence threshold: {confidence_threshold:.0%}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Session ended")

if __name__ == "__main__":
    print("="*60)
    print("ASL STATIC LETTERS RECOGNIZER")
    print("="*60)
    print("\nModel: 24 static letters (excluding J & Z)")
    print("Accuracy: 94.28% average per-letter")
    print("="*60)
    
    predictor = ASLStaticPredictor()
    predictor.run_live_prediction(confidence_threshold=0.7)