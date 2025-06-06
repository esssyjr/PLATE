from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from paddleocr import PaddleOCR
import uvicorn
import json

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
segmentation_model = YOLO('plate_segment_best.pt')

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# Example FRSC database
vehicle_info_db = {
    "ABC123KJ": {"owner": "John Doe", "vehicle": "Toyota Corolla, 2015, White"},
    "XYZ789FG": {"owner": "Jane Smith", "vehicle": "Honda Accord, 2018, Black"},
}

# Extract plate number from OCR result
def extract_plate_number(ocr_results):
    plate_number = ""
    for line in ocr_results:
        for word in line:
            plate_number += word[1][0] + " "
    return plate_number.strip().replace(" ", "")

# Segment and OCR
def process_image(image: Image.Image):
    image_np = np.array(image.convert("RGB"))
    results = segmentation_model(image_np)

    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0]
    segmented_plate = image_np[int(y1):int(y2), int(x1):int(x2)]

    ocr_results = ocr.ocr(segmented_plate)
    return extract_plate_number(ocr_results)

@app.post("/detect")
async def detect_plate_api(
    file: UploadFile = File(...),
    mode: str = Form(...),
    suspected_numbers: str = Form(""),
    frsc_data: str = Form("")
):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        plate_number = process_image(image)

        if not plate_number:
            return JSONResponse(status_code=200, content={"status": "error", "message": "License Plate Number could not be extracted."})

        # Police mode
        if mode == "Police":
            suspected_list = [num.strip().replace(" ", "") for num in suspected_numbers.split('\n') if num.strip()]
            if plate_number in suspected_list:
                return {"status": "alert", "plate": plate_number, "message": "🚨 WANTED VEHICLE DETECTED! 🚨"}
            else:
                return {"status": "ok", "plate": plate_number, "message": "✅ Vehicle not in the wanted list."}

        # FRSC mode
        elif mode == "FRSC":
            try:
                plate_db = json.loads(frsc_data or "{}")
            except:
                return {"status": "error", "message": "Invalid FRSC DB format. Please provide a valid dictionary."}

            info = plate_db.get(plate_number)
            if info:
                return {
                    "status": "found",
                    "plate": plate_number,
                    "owner": info['owner'],
                    "vehicle": info['vehicle']
                }
            else:
                return {"status": "not_found", "plate": plate_number, "message": "No matching vehicle record found."}

        return {"status": "error", "message": "Unknown mode selected."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

