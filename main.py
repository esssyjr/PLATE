from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import uvicorn
from ultralytics import YOLO
import easyocr
import io
import json

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
segmentation_model = YOLO('plate_segment_best (3).pt')

# Initialize EasyOCR
ocr = easyocr.Reader(['en'], gpu=False)

# Example FRSC database
vehicle_info_db = {
    "ABC123KJ": {"owner": "John Doe", "vehicle": "Toyota Corolla, 2015, White"},
    "XYZ789FG": {"owner": "Jane Smith", "vehicle": "Honda Accord, 2018, Black"},
}

# Extract plate number
def extract_plate_number(ocr_results):
    plate_number = ""
    for result in ocr_results:
        text = result[1]
        plate_number += text + " "
    return plate_number.strip().replace(" ", "")

# Process image with YOLO + OCR
def process_image(image):
    image_np = np.array(image.convert("RGB"))
    results = segmentation_model(image_np)

    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0]
    segmented_plate = image_np[int(y1):int(y2), int(x1):int(x2)]

    ocr_results = ocr.readtext(segmented_plate)
    return extract_plate_number(ocr_results)

# API endpoint
@app.post("/detect")
async def detect_plate_api(
    file: UploadFile = File(...),
    mode: str = Form(...),  # "Police" or "FRSC"
    suspected_numbers: str = Form(""),
    frsc_data: str = Form(""),
):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        plate_number = process_image(image)

        if not plate_number:
            return JSONResponse({"status": "error", "message": "License Plate Number could not be extracted."}, status_code=400)

        if mode == "Police":
            suspected_list = [num.strip().replace(" ", "") for num in suspected_numbers.split('\n') if num.strip()]
            if plate_number in suspected_list:
                return {
                    "status": "success",
                    "plate_number": plate_number,
                    "message": "🚨 WANTED VEHICLE DETECTED!"
                }
            else:
                return {
                    "status": "success",
                    "plate_number": plate_number,
                    "message": "✅ Vehicle not in the wanted list."
                }

        elif mode == "FRSC":
            try:
                plate_db = json.loads(frsc_data) if frsc_data else vehicle_info_db
            except:
                return JSONResponse({"status": "error", "message": "Invalid FRSC DB format. Provide a valid JSON dictionary."}, status_code=400)

            info = plate_db.get(plate_number)
            if info:
                return {
                    "status": "success",
                    "plate_number": plate_number,
                    "owner": info["owner"],
                    "vehicle": info["vehicle"]
                }
            else:
                return {
                    "status": "success",
                    "plate_number": plate_number,
                    "message": "❌ No matching vehicle record found."
                }

        return JSONResponse({"status": "error", "message": "Unknown mode selected."}, status_code=400)

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Run with: uvicorn script_name:app --reload
