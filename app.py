import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load YOLOv8 model for license plate segmentation
segmentation_model = YOLO('plate_segment_best.pt')

# Initialize PaddleOCR
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

# Segment plate and extract text
def process_image(image):
    image_np = np.array(image.convert("RGB"))  # Ensure image is in RGB
    results = segmentation_model(image_np)

    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    # Use the first detected plate
    x1, y1, x2, y2 = boxes[0]
    segmented_plate = image_np[int(y1):int(y2), int(x1):int(x2)]

    # Run OCR (cls=False since we initialized with use_angle_cls=False)
    ocr_results = ocr.ocr(segmented_plate)
    return extract_plate_number(ocr_results)

# Detection logic based on mode
def detect_plate(image, mode, suspected_numbers, frsc_data):
    plate_number = process_image(image)

    if not plate_number:
        return "❌ License Plate Number could not be extracted."

    if mode == "Police":
        suspected_list = [num.strip().replace(" ", "") for num in suspected_numbers.split('\n') if num.strip()]
        if plate_number in suspected_list:
            return f"🔍 Detected Plate: {plate_number}\n🚨 **WANTED VEHICLE DETECTED!** 🚨"
        else:
            return f"🔍 Detected Plate: {plate_number}\n✅ Vehicle not in the wanted list."

    elif mode == "FRSC":
        try:
            plate_db = eval(frsc_data or "{}")
        except:
            return "⚠️ Invalid FRSC DB format. Please provide a valid dictionary."
        
        info = plate_db.get(plate_number)
        if info:
            return f"🔍 Detected Plate: {plate_number}\n👤 Owner: {info['owner']}\n🚗 Vehicle Info: {info['vehicle']}"
        else:
            return f"🔍 Detected Plate: {plate_number}\n❌ No matching vehicle record found."

    return "❌ Unknown mode selected."

# Gradio UI
def main():
    mode = gr.Radio(["Police", "FRSC"], label="Select Mode")
    suspected_numbers_input = gr.Textbox(lines=5, label="Wanted Plate Numbers (One per line)")
    frsc_data_input = gr.Textbox(lines=5, label="FRSC DB: Dictionary of Plate -> Info", value=str(vehicle_info_db))
    image_input = gr.Image(type="pil", label="Upload Plate Image")
    output_text = gr.Textbox(label="Result")

    interface = gr.Interface(
        fn=detect_plate,
        inputs=[image_input, mode, suspected_numbers_input, frsc_data_input],
        outputs=output_text,
        live=False
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()
