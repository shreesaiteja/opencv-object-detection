import cv2
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Function to set the desired voice pack
def set_voice(gender='female', language='en'):
    voices = engine.getProperty('voices')
    for voice in voices:
        if gender in voice.name and language in voice.languages:
            engine.setProperty('voice', voice.id)
            return

# Set voice to female English
set_voice(gender='female', language='en')

def speak(text):
    engine.say(text)
    engine.runAndWait()

# URL of the IP webcam (front camera)
url = 'http://192.168.6.148:8080/video'  # Replace with your actual IP and port

# Load YOLO
net = cv2.dnn.readNet("C:/Users/SHREE SAI TEJA/OneDrive/Documents/TEJA/vscode/project/yolov3.weights", "C:/Users/SHREE SAI TEJA/OneDrive/Documents/TEJA/vscode/project/yolov3.cfg")
with open("C:/Users/SHREE SAI TEJA/OneDrive/Documents/TEJA/vscode/project/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(url)

plt.ion()  # Turn interactive mode on
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    directions = {"left": False, "right": False, "forward": False}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check positions to provide directions
                if center_x < width // 3:
                    directions["left"] = True
                elif center_x > 2 * (width // 3):
                    directions["right"] = True
                else:
                    directions["forward"] = True

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = (1 / (h / height)) * 2  # Approximate distance based on bounding box height

            # Announce the detected object
            speak(f"{label} ahead")

            # Check if distance is less than 1 meter
            if distance < 1.0:
                speak("Obstacle ahead. Please move right or left.")
                color = (0, 0, 255)  # Red for close obstacle
            else:
                color = (0, 255, 0)  # Green for safe distance
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {round(distance, 2)}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Provide directional guidance
    if directions["left"]:
        speak("Move left.")
    elif directions["right"]:
        speak("Move right.")
    elif directions["forward"]:
        speak("Move forward.")

    # Convert BGR image to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    plt.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.001)  # Pause to allow display update

cap.release()
plt.close()  # Close the plot
