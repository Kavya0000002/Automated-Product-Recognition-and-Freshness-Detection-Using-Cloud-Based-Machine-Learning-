import os 
import cv2
import numpy as np
import serial
from google.cloud import vision
import openai
import time
from datetime import datetime
import re
from google.cloud import bigquery  # Import BigQuery client library

# Initialize the Vision API client
vision_client = vision.ImageAnnotatorClient()

# Initialize the BigQuery client
bq_client = bigquery.Client()

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up the Arduino connection (make sure COM3 is the correct port)
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Give some time for the connection to initialize

# Function to draw bounding boxes and labels for the central object
def draw_bounding_boxes(frame, obj, label):
    vertices = obj.bounding_poly.normalized_vertices
    h, w, _ = frame.shape
    pts = [(int(v.x * w), int(v.y * h)) for v in vertices]
    
    cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 255), thickness=2)

    label_pos = (pts[0][0], pts[0][1] - 10 if pts[0][1] - 10 > 10 else pts[0][1] + 10)
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Function to detect objects in a frame using the Google Cloud Vision API
def detect_objects_from_frame(frame):
    try:
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            print("Error encoding image")
            return []
        
        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        response = vision_client.object_localization(image=image)

        if response.error.message:
            raise Exception(f'Error in Vision API: {response.error.message}')

        return response.localized_object_annotations
    except Exception as e:
        print(f"Object detection failed: {e}")
        return []

# Function to extract text from objects using the Google Vision API
def extract_text_from_objects(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        print("Error encoding image")
        return []
    
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    
    if response.error.message:
        raise Exception(f'Error in Vision API: {response.error.message}')
    
    return response.text_annotations

# Function to get the current time in UTC
def get_current_utc_time():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# Function to classify text using ChatGPT and segregate classified information
def classify_text_with_chatgpt(extracted_text):
    messages = [
        {
            "role": "user",
            "content": (
                f"Extract and classify the following text into specific fields including Name, Brand, MRP, "
                f"Manufacturing Day, Manufacturing Month, Manufacturing Year, Expiration Day, Expiration Month, "
                f"Expiration Year, and QR Link. Return the result in the following format:\n"
                f"Name: <name>\n"
                f"Brand: <brand>\n"
                f"MRP: <mrp>\n"
                f"Manufacturing Day: <day>\n"
                f"Manufacturing Month: <month>\n"
                f"Manufacturing Year: <year>\n"
                f"Expiration Day: <day>\n"
                f"Expiration Month: <month>\n"
                f"Expiration Year: <year>\n"
                f"QR Link: <qr_link>\n\n"
                f"Extracted Text: {extracted_text}"
            )
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.5
    )
    
    classification = response.choices[0].message['content'].strip()
    
    # Print the raw response for debugging
    print("Raw API Response:", classification)

    # Initialize the classified_info dictionary
    classified_info = {
        "Name": "N/A",
        "Brand": "N/A",
        "MRP": "N/A",
        "Manufacturing Day": "N/A",
        "Manufacturing Month": "N/A",
        "Manufacturing Year": "N/A",
        "Expiration Day": "N/A",
        "Expiration Month": "N/A",
        "Expiration Year": "N/A",
        "QR Link": "N/A",
        "Quantity": 1,
        "Current Time (UTC)": get_current_utc_time()
    }

    # Use regex to find relevant information
    name_match = re.search(r'Name:\s*(.*)', classification)
    brand_match = re.search(r'Brand:\s*(.*)', classification)
    mrp_match = re.search(r'MRP:\s*(.*)', classification)
    manufacturing_date_match = re.search(r'Manufacturing Day:\s*(\d+)\s*\nManufacturing Month:\s*(\d+)\s*\nManufacturing Year:\s*(\d+)', classification)
    expiration_date_match = re.search(r'Expiration Day:\s*(\d+)\s*\nExpiration Month:\s*(\d+)\s*\nExpiration Year:\s*(\d+)', classification)

    if name_match:
        classified_info["Name"] = name_match.group(1).strip()
    if brand_match:
        classified_info["Brand"] = brand_match.group(1).strip()
    if mrp_match:
        classified_info["MRP"] = mrp_match.group(1).strip()
    if manufacturing_date_match:
        classified_info["Manufacturing Day"] = manufacturing_date_match.group(1).strip()
        classified_info["Manufacturing Month"] = manufacturing_date_match.group(2).strip()
        classified_info["Manufacturing Year"] = manufacturing_date_match.group(3).strip()
    if expiration_date_match:
        classified_info["Expiration Day"] = expiration_date_match.group(1).strip()
        classified_info["Expiration Month"] = expiration_date_match.group(2).strip()
        classified_info["Expiration Year"] = expiration_date_match.group(3).strip()

    return classified_info

# Function to insert data into BigQuery
def is_valid_date(year, month, day):
    """Returns True if the given year, month, and day form a valid date, False otherwise."""
    try:
        # Convert to integer if possible, and check if valid date
        datetime(int(year), int(month), int(day))
        return True
    except (ValueError, TypeError):
        return False

# Function to convert 2-digit year to 4-digit year
def convert_year(year):
    if len(year) == 2:  # If year is in 2-digit format
        return "20" + year  # Assuming years are in the 2000s
    return year

# Function to insert data into BigQuery
def insert_data_to_bigquery(classified_info, weight):
    # Check if manufacturing and expiration dates are valid
    manu_date = None
    exp_date = None

    # Convert 2-digit year to 4-digit year before validating
    classified_info['Manufacturing Year'] = convert_year(classified_info['Manufacturing Year'])
    classified_info['Expiration Year'] = convert_year(classified_info['Expiration Year'])

    # Validate the manufacturing and expiration dates
    if is_valid_date(classified_info['Manufacturing Year'], classified_info['Manufacturing Month'], classified_info['Manufacturing Day']):
        manu_date = f"{classified_info['Manufacturing Year']}-{classified_info['Manufacturing Month']}-{classified_info['Manufacturing Day']}"
    
    if is_valid_date(classified_info['Expiration Year'], classified_info['Expiration Month'], classified_info['Expiration Day']):
        exp_date = f"{classified_info['Expiration Year']}-{classified_info['Expiration Month']}-{classified_info['Expiration Day']}"

    # Safely handle MRP conversion
    price = None
    try:
        if classified_info["MRP"] not in ["N/A", "Not Available", "Not provided", "Not Provided"]:
            price = float(classified_info["MRP"].replace(' PER g', '').replace(',', '').strip())
    except ValueError:
        print(f"Warning: MRP '{classified_info['MRP']}' could not be converted to a float. Setting it to None.")

    # Prepare the row to insert
    row_to_insert = {
        "name": classified_info["Name"] if classified_info["Name"] != "Not Available" else None,
        "brand": classified_info["Brand"] if classified_info["Brand"] != "Not Available" else None,
        "in_time": classified_info["Current Time (UTC)"],
        "out_time": None,
        "manu_date": manu_date,
        "exp_date": exp_date,
        "price": price,  # Use the parsed price or None
        "quantity": classified_info["Quantity"],
        "total": None,  # You can calculate total if necessary
        "color": None,  # Add color if available
        "qr": classified_info["QR Link"] if classified_info["QR Link"] != "N/A" else None,
        "weight": weight  # Add weight from Arduino
    }

    # Insert the row into BigQuery
    errors = bq_client.insert_rows_json('flipkart-438204.inventory.rack', [row_to_insert])
    if errors:
        print(f"Failed to insert rows: {errors}")

# Function to find the object closest to the center
def find_central_object(objects, frame):
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    closest_obj = None
    min_distance = float('inf')

    for obj in objects:
        vertices = obj.bounding_poly.normalized_vertices
        obj_x = int((vertices[0].x + vertices[2].x) / 2 * w)
        obj_y = int((vertices[0].y + vertices[2].y) / 2 * h)
        distance = np.sqrt((obj_x - center_x) ** 2 + (obj_y - center_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_obj = obj

    return closest_obj

# Function to read weight from Arduino
def read_weight_from_arduino():
    weight = None
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8').strip()
            if "Weight:" in line:
                weight = line.split("Weight:")[1].strip()  # Assuming the format is "Weight: <value>"
                print(f"Received Weight: 0.022")
                break
    return weight

# Main function
# Main function
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    display_time = 5
    start_time = time.time()
    frame_pair = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        objects = detect_objects_from_frame(frame)
        central_object = find_central_object(objects, frame)

        if central_object and (time.time() - start_time) < display_time:
            object_name = central_object.name
            confidence = int(central_object.score * 100)
            label = f"{object_name} {confidence}%"
            draw_bounding_boxes(frame, central_object, label)

            texts = extract_text_from_objects(frame)

            if texts:
                detected_text = texts[0].description.strip()
                
                # Wait until a valid weight is read from the Arduino
                weight = read_weight_from_arduino()
                if weight:  # Only proceed if a weight has been successfully read
                    # Clean the weight string and ensure it's valid
                    classified_info = classify_text_with_chatgpt(detected_text)                
                    frame_pair.append(classified_info)
                    
                    if len(frame_pair) == 2:  # Once two consecutive frames are processed
                        # Combine info from two consecutive frames
                        combined_info = {
                            "Name": frame_pair[0]["Name"],  # From the first image
                            "Brand": frame_pair[0]["Brand"],  # From the first image
                            "MRP": frame_pair[1]["MRP"],  # From the second image
                            "Manufacturing Day": frame_pair[1]["Manufacturing Day"],  # From the second image
                            "Manufacturing Month": frame_pair[1]["Manufacturing Month"],  # From the second image
                            "Manufacturing Year": frame_pair[1]["Manufacturing Year"],  # From the second image
                            "Expiration Day": frame_pair[1]["Expiration Day"],  # From the second image
                            "Expiration Month": frame_pair[1]["Expiration Month"],  # From the second image
                            "Expiration Year": frame_pair[1]["Expiration Year"],  # From the second image
                            "QR Link": frame_pair[1]["QR Link"],  # From the second image
                            "Quantity": 1,
                            "Current Time (UTC)": get_current_utc_time()
                        }
                        
                        # Insert the combined information into BigQuery, including weight
                        insert_data_to_bigquery(combined_info, weight)
                        
                        # Reset the frame pair
                        frame_pair = []
                        
                        print("Combined Information Inserted:")
                        for key, value in combined_info.items():
                            print(f"{key}: {value}")
                        print("------------------------------")

        # Display the frame live
        cv2.imshow('Live Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= display_time:
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
