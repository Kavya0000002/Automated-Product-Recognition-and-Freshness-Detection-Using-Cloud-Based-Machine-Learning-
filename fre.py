import cv2
import numpy as np
from google.cloud import vision
import requests
import os

# Initialize Google Vision API client
client = vision.ImageAnnotatorClient()

def detect_objects(frame):
    # Convert the image to the format required by Google Vision API
    _, encoded_image = cv2.imencode('.jpg', frame)
    image = vision.Image(content=encoded_image.tobytes())
    
    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    return objects

def get_average_color(frame, points):
    # Create a mask for the detected object
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)

    # Extract the region of interest (ROI)
    mean_val = cv2.mean(frame, mask=mask)[:3]  # Get mean color in BGR format
    return mean_val  # Returns (B, G, R)

def analyze_freshness_with_openai(color):
    # Use OpenAI API to determine freshness
    api_url = "https://api.openai.com/v1/chat/completions"
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Assuming you set your API key in environment variables
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the prompt with color information
    prompt = f"Based on the color B:{color[0]}, G:{color[1]}, R:{color[2]}, determine the freshness of the object and give fresh or not in one word fresh or rotton."
    
    payload = {
        "model": "gpt-4",  # Change to the model you are using
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50  # Limit the response length
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        # Extract and return the freshness assessment
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return "Error in API request"

def main():
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = detect_objects(frame)

        # Draw bounding boxes and labels
        for obj in objects:
            # Get the object's bounding box
            box = obj.bounding_poly.normalized_vertices
            h, w, _ = frame.shape
            points = [(int(v.x * w), int(v.y * h)) for v in box]

            # Get the average color from the detected object area
            avg_color = get_average_color(frame, points)
            freshness = analyze_freshness_with_openai(avg_color)

            # Draw bounding box in yellow
            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=2)

            # Get label
            label = obj.name

            # Display label and freshness
            cv2.putText(frame, f'{label} - {freshness}', (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Print freshness in console
            print(f'Detected {label} - Freshness: {freshness}')

        # Show the frame with detections
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
