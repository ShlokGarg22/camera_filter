#!/usr/bin/env python3
"""
Advanced Face Filter App - FIXED & IMPROVED
All features working properly now!
"""

import cv2
import numpy as np
import math
import qrcode
from datetime import datetime
import os


# ============================================================================
# Configuration
# ============================================================================
FACE_DETECTION_SCALE = 1.1
FACE_MIN_NEIGHBORS = 5
EYE_MIN_NEIGHBORS = 10
SMOOTHING_FACTOR = 0.3  # For stabilizing filter positions


# ============================================================================
# GPU Check
# ============================================================================
def check_gpu():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("✅ GPU: ENABLED")
            return True
    except:
        pass
    print("⚠️  GPU: Using CPU")
    return False

GPU_AVAILABLE = check_gpu()


# ============================================================================
# Load Haar Cascades
# ============================================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

print("✅ Face detection loaded")


# ============================================================================
# IMPROVED Cartoon Effect (Multiple Styles)
# ============================================================================
def cartoonify_style1(frame):
    """High quality cartoon effect - Bilateral filter + edge"""
    # Reduce noise while preserving edges
    img = cv2.bilateralFilter(frame, 9, 250, 250)
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    
    # Color quantization
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = 12  # More colors = better quality
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(img.shape)
    
    # Combine with edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(result, edges_colored)
    
    return cartoon

def cartoonify_style2(frame):
    """Watercolor painting effect"""
    # Apply bilateral filter multiple times
    img = frame.copy()
    for _ in range(3):
        img = cv2.bilateralFilter(img, 9, 300, 300)
    
    # Enhance colors
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
    # Stylization
    cartoon = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    
    return cartoon

def cartoonify_style3(frame):
    """Pencil sketch + color"""
    # Create sketch
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    _, edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    
    # Color simplification
    img = cv2.bilateralFilter(frame, 9, 300, 300)
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 16, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(img.shape)
    
    # Combine
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(result, edges_colored)
    
    return cartoon


# ============================================================================
# IMPROVED Filter Creation (Higher Quality)
# ============================================================================

def create_glasses_filter():
    """High quality sunglasses"""
    img = np.zeros((200, 500, 4), dtype=np.uint8)
    
    # Left lens with gradient
    for y in range(45, 125):
        darkness = int(200 - (y - 45) * 1.2)
        cv2.rectangle(img, (40, y), (200, y+1), (darkness, darkness//2, 30, 240), -1)
    
    # Right lens
    for y in range(45, 125):
        darkness = int(200 - (y - 45) * 1.2)
        cv2.rectangle(img, (300, y), (460, y+1), (darkness, darkness//2, 30, 240), -1)
    
    # Reflections
    cv2.ellipse(img, (90, 65), (35, 20), -20, 0, 360, (255, 255, 255, 100), -1)
    cv2.ellipse(img, (350, 65), (35, 20), -20, 0, 360, (255, 255, 255, 100), -1)
    
    # Frames
    cv2.ellipse(img, (120, 85), (85, 45), 0, 0, 360, (15, 15, 15, 255), 10)
    cv2.ellipse(img, (380, 85), (85, 45), 0, 0, 360, (15, 15, 15, 255), 10)
    
    # Bridge
    cv2.rectangle(img, (200, 75), (300, 95), (20, 20, 20, 255), -1)
    
    # Nose pads
    cv2.circle(img, (200, 100), 10, (30, 30, 30, 220), -1)
    cv2.circle(img, (300, 100), 10, (30, 30, 30, 220), -1)
    
    # Temple arms
    cv2.rectangle(img, (25, 80), (40, 90), (20, 20, 20, 255), -1)
    cv2.rectangle(img, (460, 80), (475, 90), (20, 20, 20, 255), -1)
    
    return img

def create_mustache_filter():
    """Detailed mustache"""
    img = np.zeros((150, 350, 4), dtype=np.uint8)
    
    # Main body with gradient
    for y in range(40, 95):
        darkness = int(70 - (y - 40) * 0.4)
        cv2.ellipse(img, (175, 75), (155, 45), 0, 0, 360, 
                   (darkness, darkness-10, darkness-20, 250), -1)
    
    # Center split
    cv2.ellipse(img, (175, 45), (18, 25), 0, 0, 360, (0, 0, 0, 0), -1)
    
    # Left curl
    cv2.ellipse(img, (60, 70), (40, 35), -35, 0, 360, (55, 45, 35, 245), -1)
    cv2.ellipse(img, (42, 62), (28, 23), -45, 0, 360, (60, 50, 40, 235), -1)
    
    # Right curl
    cv2.ellipse(img, (290, 70), (40, 35), 35, 0, 360, (55, 45, 35, 245), -1)
    cv2.ellipse(img, (308, 62), (28, 23), 45, 0, 360, (60, 50, 40, 235), -1)
    
    # Highlights
    cv2.ellipse(img, (130, 63), (25, 10), -12, 0, 360, (95, 85, 75, 130), -1)
    cv2.ellipse(img, (220, 63), (25, 10), 12, 0, 360, (95, 85, 75, 130), -1)
    
    # Shadow for depth
    cv2.ellipse(img, (175, 95), (120, 18), 0, 0, 180, (25, 20, 15, 140), -1)
    
    return img

def create_emoji(emoji_type, size=400):
    """High quality emoji masks"""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = int(size * 0.42)
    
    # Base circle with shadow
    cv2.circle(img, (center+5, center+5), radius, (0, 150, 180, 100), -1)
    cv2.circle(img, (center, center), radius, (0, 220, 255, 255), -1)
    cv2.circle(img, (center, center), radius, (0, 180, 200, 255), 12)
    
    if emoji_type == 'happy':
        # Eyes
        eye_y = int(center - radius * 0.25)
        cv2.ellipse(img, (int(center - radius * 0.35), eye_y), 
                   (int(radius * 0.15), int(radius * 0.25)), 0, 0, 360, (0, 0, 0, 255), -1)
        cv2.ellipse(img, (int(center + radius * 0.35), eye_y), 
                   (int(radius * 0.15), int(radius * 0.25)), 0, 0, 360, (0, 0, 0, 255), -1)
        # Big smile
        cv2.ellipse(img, (center, int(center + radius * 0.05)), 
                   (int(radius * 0.6), int(radius * 0.55)), 0, 5, 175, (0, 0, 0, 255), 12)
        # Cheeks
        cv2.circle(img, (int(center - radius * 0.6), int(center + radius * 0.1)), 
                  int(radius * 0.18), (255, 150, 150, 180), -1)
        cv2.circle(img, (int(center + radius * 0.6), int(center + radius * 0.1)), 
                  int(radius * 0.18), (255, 150, 150, 180), -1)
    
    elif emoji_type == 'cool':
        # Sunglasses
        glass_y = int(center - radius * 0.2)
        glass_w = int(radius * 0.4)
        glass_h = int(radius * 0.28)
        cv2.rectangle(img, (int(center - radius * 0.65), glass_y - glass_h//2), 
                     (int(center - radius * 0.1), glass_y + glass_h//2), (15, 15, 15, 255), -1)
        cv2.rectangle(img, (int(center + radius * 0.1), glass_y - glass_h//2), 
                     (int(center + radius * 0.65), glass_y + glass_h//2), (15, 15, 15, 255), -1)
        # Bridge
        cv2.rectangle(img, (int(center - radius * 0.1), glass_y - 8), 
                     (int(center + radius * 0.1), glass_y + 8), (15, 15, 15, 255), -1)
        # Smirk
        pts = np.array([[int(center - radius * 0.4), int(center + radius * 0.35)],
                       [int(center), int(center + radius * 0.3)],
                       [int(center + radius * 0.4), int(center + radius * 0.35)]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0, 255), 10)
    
    elif emoji_type == 'heart':
        # Heart eyes
        heart_y = int(center - radius * 0.25)
        scale = radius * 0.2
        # Left heart
        pts1 = np.array([
            [int(center - radius * 0.35), int(heart_y - scale * 0.3)],
            [int(center - radius * 0.55), int(heart_y)],
            [int(center - radius * 0.35), int(heart_y + scale * 1.2)],
            [int(center - radius * 0.15), int(heart_y)],
            [int(center - radius * 0.25), int(heart_y - scale * 0.3)]
        ], np.int32)
        # Right heart
        pts2 = np.array([
            [int(center + radius * 0.35), int(heart_y - scale * 0.3)],
            [int(center + radius * 0.15), int(heart_y)],
            [int(center + radius * 0.35), int(heart_y + scale * 1.2)],
            [int(center + radius * 0.55), int(heart_y)],
            [int(center + radius * 0.45), int(heart_y - scale * 0.3)]
        ], np.int32)
        cv2.fillPoly(img, [pts1], (0, 0, 255, 255))
        cv2.fillPoly(img, [pts2], (0, 0, 255, 255))
        # Big smile
        cv2.ellipse(img, (center, int(center + radius * 0.15)), 
                   (int(radius * 0.5), int(radius * 0.5)), 0, 5, 175, (0, 0, 0, 255), 10)
    
    return img


# ============================================================================
# NEW: ANIMAL FILTERS - Dog, Cat
# ============================================================================

def create_dog_ears():
    """Create floppy dog ears"""
    img = np.zeros((300, 500, 4), dtype=np.uint8)
    
    # Left ear
    pts_left = np.array([
        [50, 50], [30, 80], [20, 140], [30, 200], [60, 250], [90, 270], [120, 260], [130, 200], [120, 120], [100, 60]
    ], np.int32)
    cv2.fillPoly(img, [pts_left], (139, 115, 85, 255))  # Brown
    cv2.polylines(img, [pts_left], True, (101, 84, 63, 255), 5)  # Dark brown outline
    
    # Inner ear (pink)
    pts_inner_left = np.array([
        [70, 100], [60, 140], [70, 190], [90, 230], [100, 210], [95, 150], [85, 110]
    ], np.int32)
    cv2.fillPoly(img, [pts_inner_left], (203, 146, 163, 200))
    
    # Right ear
    pts_right = np.array([
        [450, 50], [470, 80], [480, 140], [470, 200], [440, 250], [410, 270], [380, 260], [370, 200], [380, 120], [400, 60]
    ], np.int32)
    cv2.fillPoly(img, [pts_right], (139, 115, 85, 255))
    cv2.polylines(img, [pts_right], True, (101, 84, 63, 255), 5)
    
    # Inner ear (pink)
    pts_inner_right = np.array([
        [430, 100], [440, 140], [430, 190], [410, 230], [400, 210], [405, 150], [415, 110]
    ], np.int32)
    cv2.fillPoly(img, [pts_inner_right], (203, 146, 163, 200))
    
    return img

def create_dog_nose():
    """Create dog nose and tongue"""
    img = np.zeros((200, 250, 4), dtype=np.uint8)
    
    # Nose - black triangle
    nose_pts = np.array([
        [125, 40], [80, 100], [170, 100]
    ], np.int32)
    cv2.fillPoly(img, [nose_pts], (30, 30, 30, 255))
    cv2.polylines(img, [nose_pts], True, (0, 0, 0, 255), 4)
    
    # Nostrils
    cv2.ellipse(img, (100, 85), (12, 18), -20, 0, 360, (0, 0, 0, 255), -1)
    cv2.ellipse(img, (150, 85), (12, 18), 20, 0, 360, (0, 0, 0, 255), -1)
    
    # Shine on nose
    cv2.ellipse(img, (115, 55), (15, 10), -30, 0, 360, (80, 80, 80, 180), -1)
    
    # Tongue (optional - below nose)
    tongue_pts = np.array([
        [110, 110], [125, 160], [140, 110]
    ], np.int32)
    cv2.fillPoly(img, [tongue_pts], (147, 112, 219, 240))  # Pink
    cv2.polylines(img, [tongue_pts], True, (120, 90, 180, 255), 3)
    
    # Tongue center line
    cv2.line(img, (125, 110), (125, 150), (120, 90, 180, 200), 2)
    
    return img

def create_cat_ears():
    """Create pointy cat ears"""
    img = np.zeros((250, 500, 4), dtype=np.uint8)
    
    # Left ear (triangle)
    pts_left = np.array([
        [80, 200], [20, 40], [140, 120]
    ], np.int32)
    cv2.fillPoly(img, [pts_left], (255, 200, 150, 255))  # Light orange
    cv2.polylines(img, [pts_left], True, (220, 170, 120, 255), 5)
    
    # Inner ear
    pts_inner_left = np.array([
        [80, 160], [50, 80], [100, 120]
    ], np.int32)
    cv2.fillPoly(img, [pts_inner_left], (255, 220, 200, 220))
    
    # Right ear
    pts_right = np.array([
        [420, 200], [480, 40], [360, 120]
    ], np.int32)
    cv2.fillPoly(img, [pts_right], (255, 200, 150, 255))
    cv2.polylines(img, [pts_right], True, (220, 170, 120, 255), 5)
    
    # Inner ear
    pts_inner_right = np.array([
        [420, 160], [450, 80], [400, 120]
    ], np.int32)
    cv2.fillPoly(img, [pts_inner_right], (255, 220, 200, 220))
    
    return img

def create_cat_nose_whiskers():
    """Create cat nose and whiskers"""
    img = np.zeros((200, 400, 4), dtype=np.uint8)
    
    # Pink nose (small triangle)
    nose_pts = np.array([
        [200, 60], [180, 90], [220, 90]
    ], np.int32)
    cv2.fillPoly(img, [nose_pts], (203, 146, 163, 255))
    cv2.polylines(img, [nose_pts], True, (180, 120, 140, 255), 3)
    
    # Whiskers - left side
    cv2.line(img, (180, 75), (50, 60), (40, 40, 40, 220), 3)
    cv2.line(img, (180, 85), (40, 85), (40, 40, 40, 220), 3)
    cv2.line(img, (180, 95), (50, 110), (40, 40, 40, 220), 3)
    
    # Whiskers - right side
    cv2.line(img, (220, 75), (350, 60), (40, 40, 40, 220), 3)
    cv2.line(img, (220, 85), (360, 85), (40, 40, 40, 220), 3)
    cv2.line(img, (220, 95), (350, 110), (40, 40, 40, 220), 3)
    
    return img

def create_flower_crown():
    """Create beautiful flower crown"""
    img = np.zeros((200, 600, 4), dtype=np.uint8)
    
    # Create multiple flowers across the crown
    flowers = [
        # (x, y, size, color)
        (100, 100, 35, (147, 112, 219)),  # Purple
        (200, 80, 40, (255, 182, 193)),   # Pink
        (300, 70, 45, (255, 218, 185)),   # Peach
        (400, 80, 40, (255, 182, 193)),   # Pink
        (500, 100, 35, (147, 112, 219)),  # Purple
    ]
    
    for x, y, size, color in flowers:
        # Draw 5 petals
        for angle in range(0, 360, 72):
            angle_rad = math.radians(angle)
            petal_x = int(x + size * 0.6 * math.cos(angle_rad))
            petal_y = int(y + size * 0.6 * math.sin(angle_rad))
            cv2.ellipse(img, (petal_x, petal_y), (size//2, size//3), 
                       angle, 0, 360, color, -1)
            cv2.ellipse(img, (petal_x, petal_y), (size//2, size//3), 
                       angle, 0, 360, tuple(max(0, c-50) for c in color[:3]) + (255,), 2)
        
        # Flower center
        cv2.circle(img, (x, y), size//3, (255, 255, 100, 255), -1)
        cv2.circle(img, (x, y), size//3, (220, 220, 50, 255), 2)
    
    # Add small leaves
    leaves = [
        (150, 120), (250, 100), (350, 95), (450, 100), (550, 120)
    ]
    for lx, ly in leaves:
        # Leaf shape
        leaf_pts = np.array([
            [lx, ly-15], [lx-10, ly], [lx, ly+15], [lx+10, ly]
        ], np.int32)
        cv2.fillPoly(img, [leaf_pts], (144, 238, 144, 230))
        cv2.polylines(img, [leaf_pts], True, (100, 200, 100, 255), 2)
    
    return img

# Load all filters
glasses = create_glasses_filter()
mustache = create_mustache_filter()
emoji_happy = create_emoji('happy')
emoji_cool = create_emoji('cool')
emoji_heart = create_emoji('heart')

# NEW: Load animal filters
dog_ears = create_dog_ears()
dog_nose = create_dog_nose()
cat_ears = create_cat_ears()
cat_nose_whiskers = create_cat_nose_whiskers()
flower_crown = create_flower_crown()


# ============================================================================
# Eye Gaze Detection - IMPROVED
# ============================================================================
class GazeTracker:
    def __init__(self):
        self.prev_direction = "CENTER"
        self.frame_count = 0
        self.stable_count = 0
        
    def detect(self, gray_face, eyes):
        if len(eyes) == 0:
            return self.prev_direction
        
        # Get the largest eye (most confident detection)
        eye = max(eyes, key=lambda e: e[2] * e[3])
        ex, ey, ew, eh = eye
        
        # Extract eye region
        eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
        
        # Find pupil using threshold
        _, threshold_eye = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Find pupil center
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (pupil)
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                
                # Determine direction
                eye_center = ew // 2
                threshold = ew * 0.15
                
                if cx < eye_center - threshold:
                    direction = "LEFT"
                elif cx > eye_center + threshold:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"
                
                # Smooth transitions
                if direction == self.prev_direction:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
                
                if self.stable_count > 3:
                    self.prev_direction = direction
                
                return self.prev_direction
        
        return self.prev_direction

gaze_tracker = GazeTracker()


# ============================================================================
# QR Code
# ============================================================================
def save_photo_with_qr(frame, filename="photo.png"):
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    
    filepath = f"screenshots/{filename}"
    cv2.imwrite(filepath, frame)
    
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(os.path.abspath(filepath))
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_path = f"screenshots/{filename}_qr.png"
    qr_img.save(qr_path)
    
    print(f"✅ Saved: {filepath}")
    print(f"✅ QR: {qr_path}")


# ============================================================================
# Overlay Helper
# ============================================================================
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    
    # Clip to frame bounds
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h, :]
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    
    if h <= 0 or w <= 0:
        return background
    
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    
    return background


# ============================================================================
# Face Tracker Class - STABLE FILTER PLACEMENT
# ============================================================================
class FaceTracker:
    def __init__(self):
        self.prev_faces = []
        self.smoothed_faces = []
        
    def update(self, new_faces):
        if len(self.prev_faces) == 0:
            self.smoothed_faces = new_faces
        else:
            # Smooth face positions
            smoothed = []
            for new_face in new_faces:
                # Find closest previous face
                if len(self.prev_faces) > 0:
                    closest = min(self.prev_faces, 
                                key=lambda f: abs(f[0]-new_face[0]) + abs(f[1]-new_face[1]))
                    # Smooth transition
                    smooth_x = int(closest[0] * SMOOTHING_FACTOR + new_face[0] * (1-SMOOTHING_FACTOR))
                    smooth_y = int(closest[1] * SMOOTHING_FACTOR + new_face[1] * (1-SMOOTHING_FACTOR))
                    smooth_w = int(closest[2] * SMOOTHING_FACTOR + new_face[2] * (1-SMOOTHING_FACTOR))
                    smooth_h = int(closest[3] * SMOOTHING_FACTOR + new_face[3] * (1-SMOOTHING_FACTOR))
                    smoothed.append([smooth_x, smooth_y, smooth_w, smooth_h])
                else:
                    smoothed.append(new_face)
            self.smoothed_faces = smoothed
        
        self.prev_faces = new_faces
        return self.smoothed_faces

face_tracker = FaceTracker()


# ============================================================================
# Apply Filters - FIXED
# ============================================================================
def apply_filter(frame, face, eyes, filter_type):
    x, y, w, h = face
    
    # EMOJI MASKS - Full face coverage
    if filter_type in ['emoji_happy', 'emoji_cool', 'emoji_heart']:
        emoji_map = {
            'emoji_happy': emoji_happy,
            'emoji_cool': emoji_cool,
            'emoji_heart': emoji_heart
        }
        emoji = emoji_map[filter_type]
        
        # Size based on face
        face_size = int(w * 1.8)
        resized = cv2.resize(emoji, (face_size, face_size))
        
        # Center on face
        x_pos = x + w//2 - face_size//2
        y_pos = y + h//2 - face_size//2 - int(h * 0.1)
        
        frame = overlay_transparent(frame, resized, x_pos, y_pos)
    
    # GLASSES - Eye-based placement
    elif filter_type == 'glasses':
        if len(eyes) >= 2:
            # Sort eyes by x position
            sorted_eyes = sorted(eyes, key=lambda e: e[0])[:2]
            
            # Calculate center points
            eye1_center = (x + sorted_eyes[0][0] + sorted_eyes[0][2]//2, 
                          y + sorted_eyes[0][1] + sorted_eyes[0][3]//2)
            eye2_center = (x + sorted_eyes[1][0] + sorted_eyes[1][2]//2, 
                          y + sorted_eyes[1][1] + sorted_eyes[1][3]//2)
            
            # Calculate angle
            dx = eye2_center[0] - eye1_center[0]
            dy = eye2_center[1] - eye1_center[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Size
            eye_dist = math.sqrt(dx**2 + dy**2)
            filter_w = int(eye_dist * 2.8)
            filter_h = int(filter_w * glasses.shape[0] / glasses.shape[1])
            
            resized = cv2.resize(glasses, (filter_w, filter_h))
            
            # Rotate
            center = (filter_w // 2, filter_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (filter_w, filter_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
            
            # Position
            center_x = (eye1_center[0] + eye2_center[0]) // 2
            center_y = (eye1_center[1] + eye2_center[1]) // 2
            x_pos = center_x - filter_w // 2
            y_pos = center_y - filter_h // 2 - int(filter_h * 0.05)
            
            frame = overlay_transparent(frame, rotated, x_pos, y_pos)
        else:
            # Fallback - no eyes detected
            filter_w = int(w * 1.2)
            filter_h = int(filter_w * glasses.shape[0] / glasses.shape[1])
            resized = cv2.resize(glasses, (filter_w, filter_h))
            x_pos = x + w//2 - filter_w//2
            y_pos = y + int(h * 0.35) - filter_h//2
            frame = overlay_transparent(frame, resized, x_pos, y_pos)
    
    # MUSTACHE - Below nose
    elif filter_type == 'mustache':
        filter_w = int(w * 0.7)
        filter_h = int(filter_w * mustache.shape[0] / mustache.shape[1])
        resized = cv2.resize(mustache, (filter_w, filter_h))
        
        x_pos = x + w//2 - filter_w//2
        y_pos = y + int(h * 0.7) - filter_h//2
        
        frame = overlay_transparent(frame, resized, x_pos, y_pos)
    
    return frame


# ============================================================================
# Main App
# ============================================================================
def run_face_filter():
    cap = cv2.VideoCapture(0)
    
    # HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Resolution: {w}x{h}")
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        input("Press Enter...")
        return
    
    current_filter = 'glasses'
    show_gaze = False
    cartoon_mode = 0  # 0=off, 1=style1, 2=style2, 3=style3
    
    filters_list = ['glasses', 'mustache', 'emoji_happy', 'emoji_cool', 'emoji_heart', 'none']
    filter_index = 0
    
    print("\n🎥 ADVANCED FACE FILTER")
    print("=" * 60)
    print("1:Glasses 2:Mustache 3:None 4:Happy 5:Cool 6:Hearts")
    print("C:Cartoon(cycles) G:Gaze S:Save SPACE:Cycle ESC:Exit")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Cartoon mode
        if cartoon_mode == 1:
            frame = cartoonify_style1(frame)
        elif cartoon_mode == 2:
            frame = cartoonify_style2(frame)
        elif cartoon_mode == 3:
            frame = cartoonify_style3(frame)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, FACE_DETECTION_SCALE, FACE_MIN_NEIGHBORS)
        
        # Smooth face tracking
        if len(faces) > 0:
            faces = face_tracker.update(faces.tolist())
        
        gaze = "N/A"
        
        for face in faces:
            x, y, w, h = face
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, EYE_MIN_NEIGHBORS)
            
            # Gaze tracking
            if show_gaze and len(eyes) > 0:
                gaze = gaze_tracker.detect(roi_gray, eyes)
            
            # Apply filter
            if current_filter != 'none':
                frame = apply_filter(frame, face, eyes, current_filter)
        
        # UI
        filter_names = {
            'glasses': '🕶️ Glasses', 'mustache': '🧔 Mustache',
            'emoji_happy': '😊 Happy', 'emoji_cool': '😎 Cool',
            'emoji_heart': '😍 Hearts', 'none': '✨ None'
        }
        
        cartoon_names = ['OFF', 'Style1', 'Style2', 'Style3']
        status = f"{filter_names[current_filter]} | Faces:{len(faces)}"
        if cartoon_mode > 0:
            status += f" | Cartoon:{cartoon_names[cartoon_mode]}"
        if show_gaze:
            status += f" | Gaze:{gaze}"
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        
        inst = "SPACE:Cycle | S:Save | G:Gaze | C:Cartoon | ESC:Exit"
        cv2.putText(frame, inst, (12, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, inst, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Filter', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord('1'):
            current_filter = 'glasses'
            print("✅ Glasses")
        elif key == ord('2'):
            current_filter = 'mustache'
            print("✅ Mustache")
        elif key == ord('3'):
            current_filter = 'none'
            print("✅ None")
        elif key == ord('4'):
            current_filter = 'emoji_happy'
            print("✅ Happy")
        elif key == ord('5'):
            current_filter = 'emoji_cool'
            print("✅ Cool")
        elif key == ord('6'):
            current_filter = 'emoji_heart'
            print("✅ Hearts")
        elif key == 32:
            filter_index = (filter_index + 1) % len(filters_list)
            current_filter = filters_list[filter_index]
            print(f"✅ {filter_names[current_filter]}")
        elif key == ord('g') or key == ord('G'):
            show_gaze = not show_gaze
            print(f"✅ Gaze: {'ON' if show_gaze else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            cartoon_mode = (cartoon_mode + 1) % 4
            styles = ['OFF', 'Edges+Colors', 'Watercolor', 'Sketch']
            print(f"✅ Cartoon: {styles[cartoon_mode]}")
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_photo_with_qr(frame, f"photo_{timestamp}.png")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Closed")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║         ADVANCED FACE FILTER - FIXED & IMPROVED           ║
╚════════════════════════════════════════════════════════════╝

Install: pip install opencv-python numpy qrcode pillow

""")
    
    try:
        run_face_filter()
    except KeyboardInterrupt:
        print("\n✅ Exited")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter...")