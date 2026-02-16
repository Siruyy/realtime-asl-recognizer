import sys
import cv2
import numpy as np
import os
from hand_detector import HandDetector

def collect_letter(letter, target_samples):
    detector = HandDetector()
    data_dir = 'data/raw'
    
    letter_dir = os.path.join(data_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
    existing_count = len(existing_files)
    
    count = 0
    total_count = existing_count
    
    cap = cv2.VideoCapture(0)
    
    print(f"\n{'='*60}")
    print(f"COLLECTING DATA FOR LETTER: {letter}")
    print(f"{'='*60}")
    print(f"Existing: {existing_count} | Target: {target_samples} more")
    
    # Letter-specific tips
    tips = {
        'R': [
            "R has INDEX and MIDDLE fingers CROSSED",
            "Show the cross clearly from different angles",
            "Distinguish from U (parallel fingers)",
            "Rotate hand to show side views"
        ],
        'U': [
            "Index and middle fingers UP and PARALLEL (together)",
            "NOT crossed like R! Keep them side-by-side",
            "Other fingers tucked down, thumb tucked",
            "Show from multiple angles - emphasize parallel position"
        ],
        'Q': [
            "Thumb and index finger pointing DOWN",
            "Forms an upside-down 'OK' sign",
            "Other fingers closed",
            "Rotate to show the downward point clearly"
        ],
        'I': [
            "ONLY pinky finger extended UP",
            "All other fingers closed in fist",
            "Thumb can be tucked or out",
            "Different from J (which requires motion)"
        ],
        'L': [
            "Index finger UP, thumb OUT at 90° angle",
            "Forms clear 'L' shape",
            "Other fingers closed",
            "Show the right angle from different views"
        ],
        'O': [
            "Fingertips touching thumb forming a circle",
            "Keep circle shape consistent",
            "Show from different angles (front, side)",
            "Distinguish from flat hand shapes"
        ],
        'V': [
            "Index and middle fingers UP and APART (peace sign)",
            "Keep other fingers tucked down",
            "Show clear separation between fingers",
            "Different angles showing the V shape"
        ],
        'T': [
            "Thumb BETWEEN index and middle finger",
            "Fist closed around thumb",
            "Show thumb poking through clearly",
            "Rotate to show thumb position"
        ]
    }
    
    if letter in tips:
        print(f"\nTIPS FOR LETTER {letter}:")
        for tip in tips[letter]:
            print(f"  • {tip}")
    else:
        print(f"\nCollecting data for letter: {letter}")
        print(f"  • Show diverse angles and positions")
    print(f"\nSPACE = Capture | Q = Quit")
    print(f"{'='*60}\n")
    
    while count < target_samples:
        success, frame = cap.read()
        if not success:
            break
        
        results = detector.detect_hands(frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_array = np.array(landmarks)
        else:
            landmarks_array = None
        
        frame = detector.draw_hands(frame, results)
        
        progress = (count / target_samples) * 100
        
        cv2.rectangle(frame, (0, 0), (650, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"Letter: {letter}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(frame, f"Progress: {count}/{target_samples} ({progress:.1f}%)", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE = Capture | Q = Quit", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        bar_width = 600
        bar_height = 25
        bar_y = frame.shape[0] - 40
        filled = int(bar_width * progress / 100)
        cv2.rectangle(frame, (10, bar_y), (610, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + filled, bar_y + bar_height), (0, 255, 0), -1)
        
        cv2.imshow(f"Collect {letter}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if landmarks_array is not None:
                filename = os.path.join(letter_dir, f"{letter}_{total_count}.npy")
                np.save(filename, landmarks_array)
                count += 1
                total_count += 1
                print(f"✓ {count}/{target_samples} (Total: {total_count})")
            else:
                print("✗ No hand detected!")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"✓ Collected {count} new samples")
    print(f"✓ Total for {letter}: {total_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Balance dataset - collect for letters that dropped
    target_letters = {
        'U': 150,  # Dropped from 98% to 75% - heavily confused with R
        'Q': 120,  # Dropped from 100% to 80%
        'I': 80,   # Dropped from 95% to 90%
        'L': 80,   # Dropped from 95% to 85%
    }
    
    print("="*60)
    print("BALANCING DATASET: RECOVER DROPPED LETTERS")
    print("="*60)
    print(f"\nLetters to balance: {', '.join(target_letters.keys())}")
    print("\nCurrent → Target:")
    print("  U: 75% → 95%+  (150 samples) - Distinguish from R!")
    print("  Q: 80% → 95%+  (120 samples)")
    print("  I: 90% → 95%+  (80 samples)")
    print("  L: 85% → 95%+  (80 samples)")
    print("\nTotal: 430 samples (~25-30 minutes)")
    print("Expected: Balanced dataset with 95%+ accuracy!")
    print("="*60)
    
    input("\nPress ENTER to start...")
    
    total_collected = 0
    
    for letter, samples in target_letters.items():
        print(f"\n\n{'#'*60}")
        print(f"   LETTER '{letter}' ({samples} samples)")
        print(f"{'#'*60}")
        
        response = input(f"\nReady to collect {letter}? (y/n/skip): ")
        
        if response.lower() in ['skip', 's']:
            print(f"Skipping {letter}...")
            continue
        elif response.lower() != 'y':
            print("Stopping collection.")
            break
        
        collect_letter(letter, samples)
        print(f"\n✓ {letter} complete!")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. python3 src/feature_engineer.py")
    print("  2. python3 src/train_static_only.py")
    print("\nExpected: 95%+ accuracy on 24 static letters!")