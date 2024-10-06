from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import math
import time
import sounddevice as sd
import numpy as np
import whisper
import warnings
import threading
import random  # Import to randomize colors
import textwrap  # Import to wrap text
import subprocess  # To run Streamlit app as a subprocess

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Open webcam
cap = cv2.VideoCapture(0)

# To store saved bounding boxes and their corresponding texts and colors
saved_boxes = []
last_saved_time = 0
saved_lines = []  # To store saved lines

# Load Whisper model
model = whisper.load_model("base")
warnings.filterwarnings("ignore")

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to record and transcribe audio asynchronously
def record_and_transcribe(box_id):
    print("Recording in background for 5 seconds...")
    
    # Record audio for 5 seconds
    samplerate = 16000
    audio = sd.rec(int(samplerate * 5), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for the recording to finish

    # Transcribe audio using Whisper
    result = model.transcribe(np.squeeze(audio))
    transcription = result['text']
    print(f"Transcription: {transcription}")
    
    # Update the text for the specific bounding box
    saved_boxes[box_id]['text'] = transcription

# Function to generate a random color
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to wrap text to fit inside the box width
def wrap_text(text, width, font_scale, thickness):
    # Adjust the number of characters per line to better fit within the box width
    max_chars_per_line = int(width / (12 * font_scale))  # Fine-tuned for better wrapping
    return textwrap.wrap(text, max_chars_per_line)

# Function to find the closest two boxes to a given point
def find_closest_boxes(point, boxes):
    distances = [(idx, calculate_distance(point, ((box['coords'][0] + box['coords'][2]) // 2, (box['coords'][1] + box['coords'][3]) // 2))) for idx, box in enumerate(boxes)]
    distances.sort(key=lambda x: x[1])
    return distances[:2]

# Function to generate the webcam feed
def generate():
    global saved_boxes, last_saved_time, saved_lines

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create an overlay for drawing translucent boxes
        overlay = frame.copy()

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw landmarks if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark coordinates
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # Thumb and pinky landmarks
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

                # Detect thumb and pinky touching
                touch_distance = calculate_distance(thumb_tip, pinky_tip)
                touch_threshold = 20  # Adjust this threshold as needed for sensitivity

                if touch_distance < touch_threshold:
                    # Thumb and pinky are touching - clear saved boxes and lines
                    saved_boxes.clear()
                    saved_lines.clear()
                    print("Thumb and pinky touch detected! Cleared all saved boxes and lines.")

                # Get thumb tip and index finger tip landmarks for pinch detection
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = calculate_distance(thumb_tip, index_tip)

                # Detect pinch gesture
                pinch_threshold = 40  # Increased threshold to make pinching easier
                current_time = time.time()
                if distance < pinch_threshold and (current_time - last_saved_time) > 10:
                    # Save the current bounding box location and initialize with empty text
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                    
                    # Save the box with an empty text and random color
                    box_id = len(saved_boxes)  # Unique ID for the new box
                    random_color = generate_random_color()  # Assign random color to each box
                    saved_boxes.append({'coords': (x_min, y_min, x_max, y_max), 'text': "Listening...", 'color': random_color})
                    
                    last_saved_time = current_time
                    print(f"Bounding box saved: {(x_min, y_min, x_max, y_max)} with color {random_color}")

                    # Start recording and transcribing in a separate thread for this specific box
                    whisper_thread = threading.Thread(target=record_and_transcribe, args=(box_id,))
                    whisper_thread.start()

                # Detect middle finger and thumb touching to draw line between two closest boxes
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_thumb_distance = calculate_distance(thumb_tip, middle_tip)
                if middle_thumb_distance < touch_threshold and len(saved_boxes) >= 2:
                    closest_boxes = find_closest_boxes(middle_tip, saved_boxes)
                    if len(closest_boxes) == 2:
                        box1 = saved_boxes[closest_boxes[0][0]]['coords']
                        box2 = saved_boxes[closest_boxes[1][0]]['coords']
                        # Calculate the center points of the two boxes
                        center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
                        center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                        # Save the line between the centers of the two closest boxes
                        saved_lines.append((center1, center2))

        # Draw saved bounding boxes with translucency, their specific text, and colors
        for box in saved_boxes:
            # Draw the translucent box with its random color
            cv2.rectangle(overlay, (box['coords'][0], box['coords'][1]), (box['coords'][2], box['coords'][3]), box['color'], -1)

            # Fit the text within the bounding box and wrap it
            box_width = box['coords'][2] - box['coords'][0]
            box_height = box['coords'][3] - box['coords'][1]

            # Set the original font style and size
            font_scale = 0.7
            thickness = 2

            # Wrap text based on the box width
            text_lines = wrap_text(box['text'], box_width, font_scale, thickness)

            # Calculate the line height based on the font size
            line_height = int(30 * font_scale)

            # Draw each line of text within the box
            for i, line in enumerate(text_lines):
                # Calculate text size
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = box['coords'][0] + 10  # Some padding from the left edge
                text_y = box['coords'][1] + int(line_height * (i + 1))  # Vertical position

                # Ensure text does not overflow the box height
                if text_y + text_size[1] <= box['coords'][3]:
                    # Add text inside the box
                    cv2.putText(overlay, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw saved lines
        for line in saved_lines:
            cv2.line(overlay, line[0], line[1], (0, 140, 255), 5)

        # Apply the overlay with more opacity (less translucent)
        alpha = 0.7  # Increased opacity for more solid boxes
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/camerathingy')
def camerathingy():
    return render_template('camerathingy.html')

# Route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to run the Streamlit app
@app.route('/streamlit')
def run_streamlit():
    # Run the Streamlit app as a subprocess
    subprocess.Popen(['streamlit', 'run', 'app2.py'])  # Adjust to your Streamlit app file path
    return redirect("http://localhost:8501")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')



# from flask import Flask, render_template, Response
# import cv2
# import mediapipe as mp
# import math
# import time
# import sounddevice as sd
# import numpy as np
# import whisper
# import warnings
# import threading
# import random  # Import to randomize colors
# import textwrap  # Import to wrap text

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize Mediapipe hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)

# # Open webcam
# cap = cv2.VideoCapture(0)

# # To store saved bounding boxes and their corresponding texts and colors
# saved_boxes = []
# last_saved_time = 0
# saved_lines = []  # To store saved lines

# # Load Whisper model
# model = whisper.load_model("base")
# warnings.filterwarnings("ignore")

# # Function to calculate the distance between two points
# def calculate_distance(point1, point2):
#     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# # Function to record and transcribe audio asynchronously
# def record_and_transcribe(box_id):
#     print("Recording in background for 5 seconds...")
    
#     # Record audio for 5 seconds
#     samplerate = 16000
#     audio = sd.rec(int(samplerate * 5), samplerate=samplerate, channels=1, dtype=np.float32)
#     sd.wait()  # Wait for the recording to finish

#     # Transcribe audio using Whisper
#     result = model.transcribe(np.squeeze(audio))
#     transcription = result['text']
#     print(f"Transcription: {transcription}")
    
#     # Update the text for the specific bounding box
#     saved_boxes[box_id]['text'] = transcription

# # Function to generate a random color
# def generate_random_color():
#     return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# # Function to wrap text to fit inside the box width
# def wrap_text(text, width, font_scale, thickness):
#     # Adjust the number of characters per line to better fit within the box width
#     max_chars_per_line = int(width / (12 * font_scale))  # Fine-tuned for better wrapping
#     return textwrap.wrap(text, max_chars_per_line)

# # Function to find the closest two boxes to a given point
# def find_closest_boxes(point, boxes):
#     distances = [(idx, calculate_distance(point, ((box['coords'][0] + box['coords'][2]) // 2, (box['coords'][1] + box['coords'][3]) // 2))) for idx, box in enumerate(boxes)]
#     distances.sort(key=lambda x: x[1])
#     return distances[:2]

# # Function to generate the webcam feed
# def generate():
#     global saved_boxes, last_saved_time, saved_lines

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Create an overlay for drawing translucent boxes
#         overlay = frame.copy()

#         # Convert frame to RGB for Mediapipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb_frame)

#         # Draw landmarks if hand is detected
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Get landmark coordinates
#                 h, w, _ = frame.shape
#                 landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

#                 # Thumb and pinky landmarks
#                 thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
#                 pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

#                 # Detect thumb and pinky touching
#                 touch_distance = calculate_distance(thumb_tip, pinky_tip)
#                 touch_threshold = 20  # Adjust this threshold as needed for sensitivity

#                 if touch_distance < touch_threshold:
#                     # Thumb and pinky are touching - clear saved boxes and lines
#                     saved_boxes.clear()
#                     saved_lines.clear()
#                     print("Thumb and pinky touch detected! Cleared all saved boxes and lines.")

#                 # Get thumb tip and index finger tip landmarks for pinch detection
#                 index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 distance = calculate_distance(thumb_tip, index_tip)

#                 # Detect pinch gesture
#                 pinch_threshold = 40  # Increased threshold to make pinching easier
#                 current_time = time.time()
#                 if distance < pinch_threshold and (current_time - last_saved_time) > 10:
#                     # Save the current bounding box location and initialize with empty text
#                     x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
#                     y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
#                     x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
#                     y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                    
#                     # Save the box with an empty text and random color
#                     box_id = len(saved_boxes)  # Unique ID for the new box
#                     random_color = generate_random_color()  # Assign random color to each box
#                     saved_boxes.append({'coords': (x_min, y_min, x_max, y_max), 'text': "Listening...", 'color': random_color})
                    
#                     last_saved_time = current_time
#                     print(f"Bounding box saved: {(x_min, y_min, x_max, y_max)} with color {random_color}")

#                     # Start recording and transcribing in a separate thread for this specific box
#                     whisper_thread = threading.Thread(target=record_and_transcribe, args=(box_id,))
#                     whisper_thread.start()

#                 # Detect middle finger and thumb touching to draw line between two closest boxes
#                 middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                 middle_thumb_distance = calculate_distance(thumb_tip, middle_tip)
#                 if middle_thumb_distance < touch_threshold and len(saved_boxes) >= 2:
#                     closest_boxes = find_closest_boxes(middle_tip, saved_boxes)
#                     if len(closest_boxes) == 2:
#                         box1 = saved_boxes[closest_boxes[0][0]]['coords']
#                         box2 = saved_boxes[closest_boxes[1][0]]['coords']
#                         # Calculate the center points of the two boxes
#                         center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
#                         center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
#                         # Save the line between the centers of the two closest boxes
#                         saved_lines.append((center1, center2))

#         # Draw saved bounding boxes with translucency, their specific text, and colors
#         for box in saved_boxes:
#             # Draw the translucent box with its random color
#             cv2.rectangle(overlay, (box['coords'][0], box['coords'][1]), (box['coords'][2], box['coords'][3]), box['color'], -1)

#             # Fit the text within the bounding box and wrap it
#             box_width = box['coords'][2] - box['coords'][0]
#             box_height = box['coords'][3] - box['coords'][1]

#             # Set the original font style and size
#             font_scale = 0.7
#             thickness = 2

#             # Wrap text based on the box width
#             text_lines = wrap_text(box['text'], box_width, font_scale, thickness)

#             # Calculate the line height based on the font size
#             line_height = int(30 * font_scale)

#             # Draw each line of text within the box
#             for i, line in enumerate(text_lines):
#                 # Calculate text size
#                 text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#                 text_x = box['coords'][0] + 10  # Some padding from the left edge
#                 text_y = box['coords'][1] + int(line_height * (i + 1))  # Vertical position

#                 # Ensure text does not overflow the box height
#                 if text_y + text_size[1] <= box['coords'][3]:
#                     # Add text inside the box
#                     cv2.putText(overlay, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

#         # Draw saved lines
#         for line in saved_lines:
#             cv2.line(overlay, line[0], line[1], (0, 140, 255), 5)

#         # Apply the overlay with more opacity (less translucent)
#         alpha = 0.7  # Increased opacity for more solid boxes
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         # Encode the frame to JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route to render the index page
# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/api/camerathingy')
# def camerathingy():
#     return render_template('camerathingy.html')

# # Route to serve video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0')