#Esp32-camera-object-detection

Real-Time Object Detection with ESP32-CAM and YOLOv8
This project demonstrates a powerful and cost-effective system for real-time object detection. It uses an ESP32-CAM to capture and stream video to a Python server, which then leverages the high-performance YOLOv8n model to identify objects in the video feed.

(Replace the link above with a GIF of your project in action!)

🚀 Features
Live Video Streaming: Captures video from the ESP32-CAM and streams it over a local Wi-Fi network.

High-Performance Inference: Utilizes the lightweight and fast YOLOv8n (nano) model for real-time object detection.

Client-Server Architecture: The ESP32-CAM acts as a client, while a powerful Python script running on a host machine serves as the processing server.

Visual Feedback: Displays the video stream with bounding boxes and class labels drawn around detected objects using OpenCV.

Easy to Replicate: Minimal hardware and clearly structured code make the project simple to set up.

⚙️ How It Works
The system operates on a simple client-server model. The workflow is as follows:

Capture: The ESP32-CAM board is programmed using an Arduino .ino script. It connects to your local Wi-Fi and starts a video streaming server.

Stream: The Python script, running on your computer, connects to the ESP32-CAM's IP address to receive the video stream frame by frame.

Detect: Each frame is passed to the YOLOv8n model, which is loaded using the ultralytics library. The model processes the image and returns the coordinates and class of any detected objects.

Display: The Python script uses OpenCV to draw bounding boxes and labels on the original frame based on the model's output. The final annotated video is displayed on your screen in real-time.

📋 Components & Requirements
Hardware
ESP32-CAM Board: An AI-Thinker module or equivalent.

FTDI Programmer: To upload the Arduino code to the ESP32-CAM.

5V Power Supply: A stable power source for the ESP32-CAM.

Software
Arduino IDE: With the ESP32 board manager installed.

Python 3.8+

Required Python Libraries:

ultralytics (for YOLOv8)

opencv-python

numpy

You can install all Python dependencies at once by creating a requirements.txt file with the content below and running pip install -r requirements.txt.

requirements.txt:

ultralytics
opencv-python
numpy
🛠️ Setup and Installation
1. ESP32-CAM Setup (Client)
Open the .ino file in the Arduino IDE.

In the code, update the following lines with your Wi-Fi credentials:

C++

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
Go to Tools > Board and select "AI Thinker ESP32-CAM".

Connect the FTDI programmer to your ESP32-CAM and upload the code.

After uploading, open the Serial Monitor at a baud rate of 115200. Press the ESP32-CAM's reset button. The monitor will display "Camera Ready!" and the IP address of the stream (e.g., http://192.168.1.10). Copy this IP address.

2. Python Server Setup (Host Machine)
Clone this repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
It is recommended to create a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required Python packages:

Bash

pip install -r requirements.txt
Open the Python script (your_python_script.py) and paste the IP address you copied from the Arduino Serial Monitor into the stream_url variable:

Python

stream_url = 'http://192.168.1.10/stream' # <-- PASTE YOUR IP HERE
▶️ Usage
Ensure your ESP32-CAM is powered on and connected to your Wi-Fi network.

Run the Python script from your terminal:

Bash

python your_python_script.py
A new window should open, displaying the live video feed from your ESP32-CAM with object detection overlays!
