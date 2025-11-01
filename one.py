#!/usr/bin/env python3
"""
Face Filter Application
Real-time webcam filters: glasses and mustache for multiple faces
"""

import cv2
import mediapipe as mp
import numpy as np
import math


# ============================================================================
# Initialize MediaPipe Face Mesh
# ============================================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,  # Support 2 faces simultaneously
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================================
# Create Filter Images
# ============================================================================

def create_glasses_filter():
    """Create stylish sunglasses with gradient and highlights"""
    img = np.zeros((150, 400, 4), dtype=np.uint8)
    
    # Left lens - dark with gradient
    for y in range(35, 95):
        darkness = int(180 - (y - 35) * 1.5)
        cv2.rectangle(img, (30, y), (160, y+1), (darkness, darkness//2, 50, 220), -1)
    
    # Right lens
    for y in range(35, 95):
        darkness = int(180 - (y - 35) * 1.5)
        cv2.rectangle(img, (240, y), (370, y+1), (darkness, darkness//2, 50, 220), -1)
    
    # Lens highlights (shine effect)
    cv2.ellipse(img, (70, 50), (25, 15), -20, 0, 360, (255, 255, 255, 80), -1)
    cv2.ellipse(img, (280, 50), (25, 15), -20, 0, 360, (255, 255, 255, 80), -1)
    
    # Frame around lenses (thick black outline)
    cv2.ellipse(img, (95, 65), (70, 35), 0, 0, 360, (20, 20, 20, 255), 8)
    cv2.ellipse(img, (305, 65), (70, 35), 0, 0, 360, (20, 20, 20, 255), 8)
    
    # Bridge (thicker, stylish)
    cv2.rectangle(img, (160, 58), (240, 72), (30, 30, 30, 240), -1)
    cv2.rectangle(img, (160, 58), (240, 72), (10, 10, 10, 255), 3)
    
    # Nose pads
    cv2.circle(img, (160, 75), 8, (40, 40, 40, 200), -1)
    cv2.circle(img, (240, 75), 8, (40, 40, 40, 200), -1)
    
    # Temple arms
    cv2.rectangle(img, (20, 60), (30, 70), (30, 30, 30, 240), -1)
    cv2.rectangle(img, (370, 60), (380, 70), (30, 30, 30, 240), -1)
    
    return img

def create_mustache_filter():
    """Create realistic handlebar mustache"""
    img = np.zeros((120, 300, 4), dtype=np.uint8)
    
    # Main mustache body - gradient for depth
    for y in range(30, 75):
        darkness = int(60 - (y - 30) * 0.5)
        cv2.ellipse(img, (150, 60), (130, 40), 0, 0, 360, (darkness, darkness-10, darkness-20, 240), -1)
    
    # Center dip (philtrum area)
    cv2.ellipse(img, (150, 35), (15, 20), 0, 0, 360, (0, 0, 0, 0), -1)
    
    # Left curl
    cv2.ellipse(img, (50, 55), (35, 30), -30, 0, 360, (45, 35, 25, 230), -1)
    cv2.ellipse(img, (35, 48), (25, 20), -40, 0, 360, (50, 40, 30, 220), -1)
    
    # Right curl
    cv2.ellipse(img, (250, 55), (35, 30), 30, 0, 360, (45, 35, 25, 230), -1)
    cv2.ellipse(img, (265, 48), (25, 20), 40, 0, 360, (50, 40, 30, 220), -1)
    
    # Highlights for realism
    cv2.ellipse(img, (110, 50), (20, 8), -10, 0, 360, (80, 70, 60, 100), -1)
    cv2.ellipse(img, (190, 50), (20, 8), 10, 0, 360, (80, 70, 60, 100), -1)
    
    # Lower shadow for depth
    cv2.ellipse(img, (150, 75), (100, 15), 0, 0, 180, (20, 15, 10, 120), -1)
    
    return img

glasses = create_glasses_filter()
mustache = create_mustache_filter()


# ============================================================================
# Helper Functions
# ============================================================================

def overlay_transparent(background, overlay, x, y):
    """Overlay a transparent image on background"""
    h, w = overlay.shape[:2]
    
    if y + h > background.shape[0] or x + w > background.shape[1] or x < 0 or y < 0:
        return background
    
    alpha = overlay[:, :, 3] / 255.0
    
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    
    return background


def apply_filter(frame, landmarks, filter_type):
    """Apply selected filter to frame using facial landmarks"""
    h, w = frame.shape[:2]
    
    if filter_type == 'glasses':
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        
        left_x = int(left_eye_outer.x * w)
        left_y = int(left_eye_outer.y * h)
        right_x = int(right_eye_outer.x * w)
        right_y = int(right_eye_outer.y * h)
        
        angle = math.atan2(right_y - left_y, right_x - left_x)
        angle_degrees = math.degrees(angle)
        
        eye_width = abs(right_x - left_x)
        filter_width = int(eye_width * 2.2)
        filter_height = int(filter_width * glasses.shape[0] / glasses.shape[1])
        
        resized_filter = cv2.resize(glasses, (filter_width, filter_height))
        
        center = (filter_width // 2, filter_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated_filter = cv2.warpAffine(
            resized_filter, rotation_matrix, (filter_width, filter_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        center_x = (left_x + right_x) // 2
        center_y = (left_y + right_y) // 2
        x_pos = center_x - filter_width // 2
        y_pos = center_y - filter_height // 2 - int(filter_height * 0.1)
        
        frame = overlay_transparent(frame, rotated_filter, x_pos, y_pos)
    
    elif filter_type == 'mustache':
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        nose_tip = landmarks[0]
        nose_bridge = landmarks[2]
        
        left_x = int(left_mouth.x * w)
        left_y = int(left_mouth.y * h)
        right_x = int(right_mouth.x * w)
        right_y = int(right_mouth.y * h)
        nose_tip_y = int(nose_tip.y * h)
        nose_bridge_y = int(nose_bridge.y * h)
        
        angle = math.atan2(right_y - left_y, right_x - left_x)
        angle_degrees = math.degrees(angle)
        
        mouth_width = abs(right_x - left_x)
        filter_width = int(mouth_width * 1.8)
        filter_height = int(filter_width * mustache.shape[0] / mustache.shape[1])
        
        resized_filter = cv2.resize(mustache, (filter_width, filter_height))
        
        center = (filter_width // 2, filter_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated_filter = cv2.warpAffine(
            resized_filter, rotation_matrix, (filter_width, filter_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        center_x = (left_x + right_x) // 2
        nose_length = abs(nose_tip_y - nose_bridge_y)
        y_pos = nose_tip_y + int(nose_length * 0.3)
        x_pos = center_x - filter_width // 2
        
        frame = overlay_transparent(frame, rotated_filter, x_pos, y_pos)
    
    return frame


# ============================================================================
# Main Application
# ============================================================================

def run_face_filter():
    """Main function running the face filter application"""
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        print("Troubleshooting:")
        print("  - Make sure no other app is using the webcam")
        print("  - Try closing Zoom, Teams, Skype, etc.")
        print("  - Check if webcam is connected")
        input("\nPress Enter to exit...")
        return
    
    print("✅ Webcam opened successfully!")
    current_filter = 'glasses'
    
    print("\n🎥 Webcam started!")
    print("=" * 50)
    print("Controls:")
    print("  1 - Glasses 🕶️")
    print("  2 - Mustache 🧔")
    print("  3 - No Filter ✨")
    print("  ESC - Exit")
    print("\n👥 Supports up to 2 faces simultaneously!")
    print("=" * 50 + "\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame from webcam")
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        num_faces = 0
        
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            for face_landmarks in results.multi_face_landmarks:
                if current_filter != 'none':
                    frame = apply_filter(frame, face_landmarks.landmark, current_filter)
        
        # Draw UI overlay
        filter_names = {'glasses': 'Glasses 🕶️', 'mustache': 'Mustache 🧔', 'none': 'No Filter ✨'}
        status_line = f"Filter: {filter_names[current_filter]} | Faces: {num_faces}"
        
        text_bg_height = 80
        cv2.rectangle(frame, (0, 0), (frame.shape[1], text_bg_height), (0, 0, 0), -1)
        
        cv2.putText(frame, status_line, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, status_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        
        instruction = "1:Glasses  2:Mustache  3:None  |  ESC:Exit"
        cv2.putText(frame, instruction, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, instruction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Filter App', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            current_filter = 'glasses'
            print("✅ Glasses filter")
        elif key == ord('2'):
            current_filter = 'mustache'
            print("✅ Mustache filter")
        elif key == ord('3'):
            current_filter = 'none'
            print("✅ No filter")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ App closed")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║              Face Filter App - Simple Edition             ║
╚═══════════════════════════════════════════════════════════╝

Requirements: 
  pip install opencv-python mediapipe numpy

""")
    
    # Check OpenCV
    print("Checking dependencies...")
    try:
        print(f"  ✓ OpenCV version: {cv2.__version__}")
    except:
        print("  ❌ OpenCV not found! Run: pip install opencv-python")
        exit(1)
    
    try:
        print(f"  ✓ MediaPipe installed")
    except:
        print("  ❌ MediaPipe not found! Run: pip install mediapipe")
        exit(1)
    
    print("\nStarting application...\n")
    
    try:
        run_face_filter()
    except KeyboardInterrupt:
        print("\n\n✅ Exited by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n")
        input("Press Enter to exit...")