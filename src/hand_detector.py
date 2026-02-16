import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hands(self, frame):
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        return results
    
    def draw_hands(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame

if __name__ == "__main__":
    # Test the detector
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Detect hands
        results = detector.detect_hands(frame)
        
        # Draw landmarks
        frame = detector.draw_hands(frame, results)
        
        # Display
        cv2.imshow("Hand Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()