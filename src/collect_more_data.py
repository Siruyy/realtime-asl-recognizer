import cv2
import numpy as np
import os
from hand_detector import HandDetector

class TargetedDataCollector:
    def __init__(self, data_dir='data/raw'):
        self.detector = HandDetector()
        self.data_dir = data_dir
    
    def extract_landmarks(self, results):
        """Extract hand landmarks as a numpy array"""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def collect_for_letter(self, letter, additional_samples=200):
        """Collect additional samples for a specific letter"""
        cap = cv2.VideoCapture(0)
        
        letter_dir = os.path.join(self.data_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)
        
        # Count existing samples
        existing_files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
        existing_count = len(existing_files)
        
        # Start counting from existing
        count = 0
        total_count = existing_count
        
        print(f"\n{'='*60}")
        print(f"COLLECTING MORE DATA FOR LETTER: {letter}")
        print(f"{'='*60}")
        print(f"Existing samples: {existing_count}")
        print(f"Target additional samples: {additional_samples}")
        print(f"\nTIPS FOR BETTER DATA:")
        print(f"  • Rotate your hand in different angles")
        print(f"  • Move closer and farther from camera")
        print(f"  • Try different positions in frame")
        print(f"  • Vary lighting by moving around")
        print(f"\nPress SPACE to capture, 'q' to quit")
        
        while count < additional_samples:
            success, frame = cap.read()
            if not success:
                break
            
            # Detect hand
            results = self.detector.detect_hands(frame)
            frame = self.detector.draw_hands(frame, results)
            
            # Progress display
            progress = (count / additional_samples) * 100
            remaining = additional_samples - count
            
            # Add text overlays
            cv2.putText(frame, f"Letter: {letter}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"New samples: {count}/{additional_samples}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Progress bar
            bar_width = 600
            bar_height = 30
            filled_width = int(bar_width * (count / additional_samples))
            cv2.rectangle(frame, (10, 140), (10 + bar_width, 140 + bar_height), 
                         (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 140), (10 + filled_width, 140 + bar_height), 
                         (0, 255, 0), -1)
            
            cv2.imshow("Targeted Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                landmarks = self.extract_landmarks(results)
                if landmarks is not None:
                    # Use total_count to avoid filename conflicts
                    filename = os.path.join(letter_dir, f"{letter}_{total_count}.npy")
                    np.save(filename, landmarks)
                    count += 1
                    total_count += 1
                    print(f"✓ Captured {count}/{additional_samples} (Total: {total_count})")
                else:
                    print("✗ No hand detected! Show your hand clearly.")
            
            elif key == ord('q'):
                print("\nQuitting early...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✓ Collected {count} additional samples")
        print(f"✓ Total samples for {letter}: {total_count}")
        print(f"{'='*60}")
        
        return count

if __name__ == "__main__":
    collector = TargetedDataCollector()
    
    # Letters that need improvement (from analysis)
    weak_letters = ['A', 'S', 'U', 'Z', 'Y']
    
    print("="*60)
    print("TARGETED DATA COLLECTION FOR WEAK LETTERS")
    print("="*60)
    print(f"\nWe'll collect extra data for: {', '.join(weak_letters)}")
    print("This will significantly improve model accuracy!")
    
    input("\nPress ENTER to start...")
    
    for letter in weak_letters:
        print(f"\n\nLook up ASL letter '{letter}' if needed!")
        input(f"Ready to collect data for '{letter}'? Press ENTER...")
        
        collected = collector.collect_for_letter(letter, additional_samples=200)
        
        if collected < 200:
            response = input(f"\nOnly collected {collected} samples. Continue to next letter? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "="*60)
    print("TARGETED DATA COLLECTION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python3 src/feature_engineer.py (we'll create this)")
    print("  2. Run: python3 src/model_trainer.py (retrain with more data)")