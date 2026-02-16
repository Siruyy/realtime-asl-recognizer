import cv2
import numpy as np
import os
from hand_detector import HandDetector

class DataCollector:
    def __init__(self, data_dir='data/raw'):
        self.detector = HandDetector()
        self.data_dir = data_dir
        
        # ASL alphabet (26 letters)
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Create directories for each letter
        for label in self.labels:
            label_dir = os.path.join(self.data_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
    def extract_landmarks(self, results):
        """Extract hand landmarks as a numpy array"""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x, y, z coords for each of 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            return np.array(landmarks)
        
        return None
    
    def collect_data(self, letter, samples_per_letter=100):
        """Collect data for a specific letter"""
        cap = cv2.VideoCapture(0)
        
        letter_dir = os.path.join(self.data_dir, letter)
        existing_samples = len(os.listdir(letter_dir))
        count = existing_samples
        
        print(f"\nCollecting data for letter: {letter}")
        print(f"Existing samples: {existing_samples}")
        print(f"Target: {samples_per_letter} samples")
        print(f"\nPress SPACE to capture, 'q' to quit, 'n' for next letter")
        
        while count < samples_per_letter:
            success, frame = cap.read()
            if not success:
                break
            
            # Detect hand
            results = self.detector.detect_hands(frame)
            frame = self.detector.draw_hands(frame, results)
            
            # Add instructions on frame
            cv2.putText(frame, f"Letter: {letter}" , (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{samples_per_letter}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture on SPACE
            if key == ord(' '):
                landmarks = self.extract_landmarks(results)
                if landmarks is not None:
                    # Save landmarks as .npy file
                    filename = os.path.join(letter_dir, f"{letter}_{count}.npy")
                    np.save(filename, landmarks)
                    count += 1
                    print(f"Captured sample {count}/{samples_per_letter}")
                else:
                    print("No hand detected! Please show your hand clearly to the camera.")
            
            # Quit
            elif key == ord('q'):
                break
            
            # Next letter
            elif key == ord('n'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCompleted! Collected {count} samples for letter {letter}")
        return count >= samples_per_letter

if __name__ == "__main__":
    collector = DataCollector()
    
    print("ASL Fingerspelling Data Collector")
    print("=" * 50)
    print("\nWe'll collect data for each letter A-Z")
    print("You'll need to make the ASL hand sign for each letter")
    print("\nTip: Look up ASL alphabet if you're not familiar!")
    
    input("\nPress ENTER to start...")
    
    # Collect data for each letter
    for letter in collector.labels:
        completed = collector.collect_data(letter, samples_per_letter=100)
        
        if not completed:
            print(f"\nStopped at letter {letter}")
            break
        
        print(f"\n✓ Letter {letter} complete!")
        
        if letter != 'Z':
            response = input("\nContinue to next letter? (y/n): ")
            if response.lower() != 'y':
                break
        
    print("\n" + "=" * 50)
    print("Data collection complete!")