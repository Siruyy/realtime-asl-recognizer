import cv2
import numpy as np
from tensorflow.keras import models
import pickle
import os
from hand_detector import HandDetector

class ASLPredictor:
    def __init__(self, model_path='models/asl_model.h5', 
                 processed_data_dir='data/processed'):
        
        # Load trained model
        print("Loading model...")
        self.model = models.load_model(model_path)
        
        # Load normalization parameters
        self.mean = np.load(os.path.join(processed_data_dir, 'mean.npy'))
        self.std = np.load(os.path.join(processed_data_dir, 'std.npy'))
        
        # Load label encoder
        with open(os.path.join(processed_data_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Hand detector
        self.detector = HandDetector()
        
        print("✓ Model loaded and ready!")
    
    def extract_landmarks(self, results):
        """Extract hand landmarks as numpy array"""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks using saved parameters"""
        return (landmarks - self.mean) / self.std
    
    def predict(self, landmarks):
        """Predict ASL letter from landmarks"""
        
        # Normalize
        landmarks_normalized = self.normalize_landmarks(landmarks)
        
        # Reshape for model input (batch_size, features)
        landmarks_input = landmarks_normalized.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(landmarks_input, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Convert to letter
        predicted_letter = self.label_encoder.classes_[predicted_class]
        
        return predicted_letter, confidence, predictions[0]
    
    def run_live_prediction(self, confidence_threshold=0.6):
        """Run real-time ASL recognition"""
        
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*50)
        print("REAL-TIME ASL FINGERSPELLING RECOGNITION")
        print("="*50)
        print("Press 'q' to quit")
        print(f"Confidence threshold: {confidence_threshold}")
        print("\nShow ASL letters to the camera!")
        
        # For smoothing predictions
        prediction_history = []
        history_size = 5
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Flip frame for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            results = self.detector.detect_hands(frame)
            frame = self.detector.draw_hands(frame, results)
            
            # Extract landmarks and predict
            landmarks = self.extract_landmarks(results)
            
            if landmarks is not None:
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
                    display_text = f"{smoothed_letter} ({confidence:.2%})"
                else:
                    # Low confidence - yellow
                    color = (0, 255, 255)
                    display_text = f"{predicted_letter}? ({confidence:.2%})"
                
                # Draw prediction on frame
                cv2.putText(frame, display_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                
                # Show top 3 predictions
                top_3_idx = np.argsort(all_probs)[-3:][::-1]
                y_pos = 100
                for idx in top_3_idx:
                    letter = self.label_encoder.classes_[idx]
                    prob = all_probs[idx]
                    text = f"{letter}: {prob:.1%}"
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 30
                
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prediction_history.clear()
            
            # Display frame
            cv2.imshow("ASL Recognition", frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Session ended")

if __name__ == "__main__":
    predictor = ASLPredictor()
    predictor.run_live_prediction(confidence_threshold=0.6)