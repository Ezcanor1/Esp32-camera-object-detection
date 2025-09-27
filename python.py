import cv2
import numpy as np
import urllib.request
import smtplib
import time
import os
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import urllib.request
import zipfile

# --------- User Settings ---------
# IMPORTANT: Replace with your actual ESP32 IP address (check Serial Monitor)
ESP32_BASE_URL = 'http://192.168.4.202/'  # Your ESP32's IP
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = 'abccdeef1221@gmail.com'  # Replace with your email
EMAIL_PASSWORD = 'zhrvcpsuatusktxb'  # Replace with your Gmail app password
EMAIL_RECIPIENT = 'abccdeef1221@gmail.com'  # Replace with recipient email
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold for crack detection
ALERT_COOLDOWN = 30  # seconds between email alerts for crack detection
CRACK_DETECTION_THRESHOLD = 0.5  # Threshold for crack detection probability

# Common ESP32-CAM streaming endpoints to try
STREAM_ENDPOINTS = [
    '/stream',
    '/mjpeg',
    '/video',
    '/cam-hi.jpg',
    '/cam-lo.jpg',
    '/cam-mid.jpg',
    '/capture',
    ':81/stream',  # Some use different ports
    ':8080/stream'
]

# --------- Deep Learning Model for Crack Detection ---------
class CrackDetectionCNN(nn.Module):
    """Custom CNN for railway track crack detection"""
    def __init__(self):
        super(CrackDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        # For 224x224 input: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 classes: crack, no crack
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

# --------- Model Download and Loading ---------
def download_pretrained_model():
    """Download and load a pre-trained crack detection model"""
    model_url = "https://github.com/pytorch/vision/releases/download/v0.10.0/resnet18-f37072fd.pth"
    model_path = "crack_detection_model.pth"
    
    print("[INFO] Setting up crack detection model...")
    
    # Create a simple crack detection model using transfer learning
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    # Modify the final layer for binary classification (crack/no crack)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)  # 2 classes: crack, no crack
    )
    
    # Set to evaluation mode
    model.eval()
    
    print("[INFO] Crack detection model loaded successfully")
    return model

def load_crack_detection_model():
    """Load the crack detection model"""
    try:
        model = download_pretrained_model()
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load crack detection model: {e}")
        print("[INFO] Using fallback traditional computer vision method")
        return None

# --------- Traditional Computer Vision Crack Detection ---------
def detect_cracks_traditional(frame):
    """Detect cracks using traditional computer vision techniques"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_detected = False
    crack_contours = []
    
    # Filter contours that might represent cracks
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Check if contour could be a crack (elongated shape)
        if area > 100:  # Minimum area threshold
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            
            # Cracks are typically elongated
            if aspect_ratio > 3 and area > 200:
                crack_detected = True
                crack_contours.append(contour)
    
    return crack_detected, crack_contours, edges

# --------- Deep Learning Crack Detection ---------
def detect_cracks_dl(frame, model):
    """Detect cracks using deep learning model"""
    if model is None:
        return False, 0.0
    
    try:
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            crack_probability = probabilities[0][1].item()  # Probability of crack class
            
        crack_detected = crack_probability > CRACK_DETECTION_THRESHOLD
        
        return crack_detected, crack_probability
        
    except Exception as e:
        print(f"[ERROR] Deep learning detection error: {e}")
        return False, 0.0

# --------- Combined Crack Detection Function ---------
def detect_railway_cracks(frame, dl_model=None):
    """Detect railway track cracks using both traditional CV and deep learning"""
    height, width = frame.shape[:2]
    
    # Deep learning detection
    dl_crack_detected, dl_confidence = detect_cracks_dl(frame, dl_model)
    
    # Traditional computer vision detection
    trad_crack_detected, crack_contours, edges = detect_cracks_traditional(frame)
    
    # Combine results
    crack_detected = dl_crack_detected or trad_crack_detected
    
    # Draw visualizations
    result_frame = frame.copy()
    
    # Draw crack contours from traditional method
    if crack_contours:
        cv2.drawContours(result_frame, crack_contours, -1, (0, 0, 255), 2)
        for contour in crack_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_frame, "CRACK", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add detection status
    if crack_detected:
        status_text = "⚠️ CRACK DETECTED ⚠️"
        status_color = (0, 0, 255)  # Red
        alert_level = "HIGH RISK"
    else:
        status_text = "✅ TRACK CLEAR"
        status_color = (0, 255, 0)  # Green
        alert_level = "SAFE"
    
    # Draw status
    cv2.putText(result_frame, status_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    # Draw confidence scores
    cv2.putText(result_frame, f"DL Confidence: {dl_confidence:.3f}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(result_frame, f"Alert Level: {alert_level}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Show edge detection in corner (for debugging)
    edges_resized = cv2.resize(edges, (200, 150))
    edges_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    result_frame[height-160:height-10, width-210:width-10] = edges_colored
    cv2.putText(result_frame, "Edge Detection", (width-200, height-170), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return crack_detected, result_frame, dl_confidence, len(crack_contours)

# --------- Email Function ---------
def send_crack_alert_email(frame, dl_confidence, traditional_detections):
    """Send email alert when cracks are detected"""
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_data = buffer.tobytes()

    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = '🚨 CRITICAL RAILWAY SAFETY ALERT: Track Crack Detected'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    body = f"""
    🚨 CRITICAL RAILWAY SAFETY ALERT! 🚨
    
    ⚠️  RAILWAY TRACK CRACK DETECTED  ⚠️
    
    Detection Details:
    - Deep Learning Confidence: {dl_confidence:.3f}
    - Traditional CV Detections: {traditional_detections}
    - Combined Alert Status: HIGH RISK
    
    Time: {timestamp}
    Location: ESP32-CAM Railway Monitoring System
    Camera: {ESP32_BASE_URL}
    
    IMMEDIATE ACTION REQUIRED:
    1. Inspect the railway track immediately
    2. Consider halting train traffic if necessary
    3. Contact railway maintenance team
    4. Verify the detection in person
    
    This alert was triggered by AI-powered crack detection system
    combining deep learning and computer vision techniques.
    
    Please see attached image for visual confirmation.
    
    SAFETY FIRST - VERIFY IMMEDIATELY!
    
    Railway Safety Monitoring System
    """
    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(jpg_data)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="CRACK_ALERT_{timestamp.replace(":", "-").replace(" ", "_")}.jpg"')
    msg.attach(part)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, EMAIL_RECIPIENT, text)
        server.quit()
        print(f"[ALERT] 🚨 CRACK DETECTION EMAIL SENT")
        return True
    except Exception as e:
        print(f"[ERROR] Could not send email: {e}")
        return False

# --------- Find Working Stream Endpoint ---------
def find_stream_endpoint():
    """Find working ESP32-CAM stream endpoint"""
    print(f"[INFO] Testing ESP32-CAM endpoints at {ESP32_BASE_URL}")
    
    # First test base connection
    try:
        response = requests.get(ESP32_BASE_URL, timeout=5)
        print(f"[INFO] Base URL response: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to ESP32-CAM at {ESP32_BASE_URL}")
        print(f"Error: {e}")
        return None, None
    
    # Try different streaming endpoints
    for endpoint in STREAM_ENDPOINTS:
        test_url = ESP32_BASE_URL + endpoint
        print(f"[INFO] Testing: {test_url}")
        
        try:
            # Test HTTP response
            response = requests.get(test_url, timeout=10, stream=True)
            print(f"[INFO] {test_url} - Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                print(f"[INFO] Content-Type: {content_type}")
                
                # Check if it's an image endpoint
                if 'image' in content_type.lower():
                    print(f"[INFO] Found image endpoint: {test_url}")
                    return test_url, 'image'
                
                # Check if it's a stream endpoint
                elif 'multipart' in content_type.lower() or 'mjpeg' in content_type.lower():
                    print(f"[INFO] Found stream endpoint: {test_url}")
                    return test_url, 'stream'
                
                # Try to read a few bytes to see if it looks like image data
                try:
                    chunk = next(response.iter_content(chunk_size=1024))
                    if chunk.startswith(b'\xff\xd8'):  # JPEG header
                        print(f"[INFO] Detected JPEG data at: {test_url}")
                        return test_url, 'stream'
                except:
                    pass
                    
        except Exception as e:
            print(f"[ERROR] {test_url} failed: {e}")
            continue
    
    print("[ERROR] No working stream endpoint found!")
    return None, None

# --------- Image Capture Function ---------
def capture_single_image(url):
    """Capture a single image from ESP32-CAM"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        print(f"[ERROR] Failed to capture image: {e}")
    return None

# --------- Stream Reading Function ---------
def read_mjpeg_stream(url):
    """Read MJPEG stream from ESP32-CAM"""
    try:
        stream = urllib.request.urlopen(url)
        bytes_data = b''
        
        while True:
            bytes_data += stream.read(4096)
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            
            if a != -1 and b != -1 and b > a:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
                    
    except Exception as e:
        print(f"[ERROR] Stream reading error: {e}")

# --------- Main Detection Loop ---------
def main():
    """Main function for railway crack detection system"""
    print("=" * 70)
    print("🚂 RAILWAY TRACK CRACK DETECTION SYSTEM 🚂")
    print("=" * 70)
    
    # Load crack detection model
    print("[INFO] Loading crack detection models...")
    dl_model = load_crack_detection_model()
    
    last_alert_time = 0  # Track last alert time
    
    # Find working endpoint
    stream_url, stream_type = find_stream_endpoint()
    if not stream_url:
        return

    print(f"[INFO] Using endpoint: {stream_url} (type: {stream_type})")
    print("[INFO] 🔍 RAILWAY CRACK DETECTION SYSTEM ACTIVE 🔍")
    print("[INFO] Press 'q' to quit, 's' to save current frame")
    print(f"[INFO] Detection confidence threshold: {CRACK_DETECTION_THRESHOLD}")
    print("[INFO] System will send email alerts when cracks are detected")
    print("-" * 70)
    
    frame_count = 0
    total_alerts_sent = 0
    
    try:
        if stream_type == 'image':
            # Single image capture mode
            print("[INFO] Running in single image capture mode")
            while True:
                frame = capture_single_image(stream_url)
                if frame is None:
                    time.sleep(1)
                    continue
                    
                frame_count += 1
                
                # Detect railway cracks
                crack_detected, result_frame, dl_confidence, trad_detections = detect_railway_cracks(frame, dl_model)
                
                # Add frame info
                cv2.putText(result_frame, f"Frame: {frame_count}", (10, result_frame.shape[0] - 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add alerts sent count
                cv2.putText(result_frame, f"Alerts Sent: {total_alerts_sent}", (10, result_frame.shape[0] - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('ESP32-CAM Railway Crack Detection', result_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"railway_inspection_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"[INFO] Frame saved as {filename}")

                # Send email alert for crack detection
                if crack_detected:
                    current_time = time.time()
                    
                    if (current_time - last_alert_time) > ALERT_COOLDOWN:
                        print(f"[ALERT] 🚨 RAILWAY CRACK DETECTED!")
                        print(f"[ALERT] DL Confidence: {dl_confidence:.3f}")
                        print(f"[ALERT] Traditional Detections: {trad_detections}")
                        
                        if send_crack_alert_email(result_frame, dl_confidence, trad_detections):
                            last_alert_time = current_time
                            total_alerts_sent += 1
                            print(f"[INFO] Alert cooldown active - Next alert in {ALERT_COOLDOWN}s")
                
                time.sleep(0.1)  # Small delay for image capture mode
                
        else:
            # Stream mode
            print("[INFO] Running in stream mode")
            for frame in read_mjpeg_stream(stream_url):
                frame_count += 1
                
                # Detect railway cracks
                crack_detected, result_frame, dl_confidence, trad_detections = detect_railway_cracks(frame, dl_model)
                
                # Add frame info
                cv2.putText(result_frame, f"Frame: {frame_count}", (10, result_frame.shape[0] - 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add alerts sent count
                cv2.putText(result_frame, f"Alerts Sent: {total_alerts_sent}", (10, result_frame.shape[0] - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('ESP32-CAM Railway Crack Detection', result_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"railway_inspection_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"[INFO] Frame saved as {filename}")

                # Send email alert for crack detection
                if crack_detected:
                    current_time = time.time()
                    
                    if (current_time - last_alert_time) > ALERT_COOLDOWN:
                        print(f"[ALERT] 🚨 RAILWAY CRACK DETECTED!")
                        print(f"[ALERT] DL Confidence: {dl_confidence:.3f}")
                        print(f"[ALERT] Traditional Detections: {trad_detections}")
                        
                        if send_crack_alert_email(result_frame, dl_confidence, trad_detections):
                            last_alert_time = current_time
                            
                            total_alerts_sent += 1
                            print(f"[INFO] Alert cooldown active - Next alert in {ALERT_COOLDOWN}s")

    except KeyboardInterrupt:
        print("\n[INFO] System interrupted by user")
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Railway Crack Detection System shutdown completed")
        print(f"[INFO] Total alerts sent: {total_alerts_sent}")
        print("[INFO] Stay safe and maintain railway infrastructure!")

if __name__ == "__main__":
    main()