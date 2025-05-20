import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from rembg import remove
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import shutil
import uuid
import json
from pathlib import Path

# Create upload and output directories if they don't exist
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Store image metadata for the session
IMAGE_METADATA = {}

app = FastAPI(title="Person Detector and Cropper API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class DetectionResult(BaseModel):
    id: str
    box: List[float]
    score: float
    
class SegmentationResult(BaseModel):
    id: str
    success: bool
    filename: str
    
class ProcessedImage(BaseModel):
    image_id: str
    filename: str
    detections: List[DetectionResult]
    
class ExtractedObject(BaseModel):
    image_id: str
    object_id: str
    filename: str
    clean_filename: Optional[str] = None  
    full_image_filename: Optional[str] = None 

class PersonDetector:
    def __init__(self):
        # Load the TFLite model
        self.interpreter = self.load_model()
        # Colors for drawing boundaries
        self.boundary_color = (0, 255, 0)  # Green color for boundaries
        self.box_color = (0, 255, 0)  # Green color for bounding boxes
        self.text_color = (0, 0, 0)  # Black color for text

    def load_model(self):
        # Load TFLite model
        model_path = Path("model/detect.tflite")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
        
    def detect_persons(self, image):
        if image is None:
            raise ValueError("No image provided")
            
        # Get input and output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Prepare input image
        input_shape = input_details[0]['shape'][1:3]  # Height, width
        processed_img = cv2.resize(image, input_shape)
        processed_img = np.expand_dims(processed_img, axis=0).astype(np.uint8)
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], processed_img)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Filter for person class (usually class 0 in COCO dataset)
        person_indices = np.where((classes == 0) & (scores > 0.5))
        person_boxes = boxes[person_indices]
        person_scores = scores[person_indices]
        
        # Create detection results
        results = []
        for i, (box, score) in enumerate(zip(person_boxes, person_scores)):
            results.append(DetectionResult(
                id=str(i),
                box=box.tolist(),
                score=float(score)
            ))
        
        return results
        
    def draw_detection_boxes(self, image, detections):
        """
        Draw bounding boxes around detected objects
        """
        result_img = image.copy()
        h, w = image.shape[:2]
        
        for detection in detections:
            # Get box coordinates
            y1, x1, y2, x2 = detection.box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), self.box_color, 2)
            
            # Draw label with score
            label = f"Person: {detection.score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), self.box_color, -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2)
            
            # Draw ID for reference
            id_label = f"ID: {detection.id}"
            cv2.putText(result_img, id_label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2)
        
        return result_img
    
    def crop_and_remove_bg(self, image, box, draw_boundary=True):
        h, w = image.shape[:2]
        
        # If box is in normalized coordinates (0-1), convert to absolute coordinates
        if isinstance(box, list) and len(box) == 4 and all(0 <= x <= 1 for x in box):
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
        else:
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        
        # Ensure coordinates are valid (y2 > y1 and x2 > x1)
        if y1 > y2:
            y1, y2 = y2, y1
        if x1 > x2:
            x1, x2 = x2, x1
            
        # Ensure coordinates are within image bounds
        y1 = max(0, min(y1, h-1))
        x1 = max(0, min(x1, w-1))
        y2 = max(0, min(y2, h))
        x2 = max(0, min(x2, w))
        
        # Ensure we have a valid crop area
        if y2 <= y1 or x2 <= x1:
            raise ValueError("Invalid bounding box: height or width is zero or negative")
        
        # Crop image
        cropped_img = image[y1:y2, x1:x2]
        
        # Convert to PIL Image for background removal
        pil_img = Image.fromarray(cropped_img)
        
        # Remove background
        output = remove(pil_img)
        
        # Draw boundary on the output image if requested
        if draw_boundary:
            # Convert back to numpy array for contour detection
            output_array = np.array(output)
            
            # Extract alpha channel as mask
            if output_array.shape[2] == 4:  # Check if there's an alpha channel
                mask = output_array[:, :, 3]
                
                # Find contours in the mask
                mask_8bit = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create a copy of the output for drawing
                output_with_boundary = output_array.copy()
                
                # Draw contours on the output image
                for contour in contours:
                    # Convert to PIL drawing
                    pil_output = Image.fromarray(output_with_boundary)
                    draw = ImageDraw.Draw(pil_output)
                    
                    # Convert contour to list of points for PIL
                    points = []
                    for point in contour.reshape(-1, 2):
                        points.append(tuple(point))
                    
                    # Draw the boundary
                    if len(points) > 2:  # Need at least 3 points to draw a polygon
                        draw.line(points + [points[0]], fill=(0, 255, 0, 255), width=2)
                    
                    output = pil_output
        
        return output
        
    def remove_bg_full_image(self, image):
        """
        Remove background from the full image without cropping
        """
        # Convert to PIL Image for background removal
        pil_img = Image.fromarray(image)
        
        # Remove background
        output = remove(pil_img)
        
        return output
        
    def find_object_at_coordinates(self, detections, click_x, click_y, image_height, image_width):
        """
        Find which detected object contains the clicked coordinates
        """
        for detection in detections:
            y1, x1, y2, x2 = detection.box
            # Convert normalized coordinates to absolute
            y1, x1, y2, x2 = int(y1 * image_height), int(x1 * image_width), int(y2 * image_height), int(x2 * image_width)
            
            # Check if click is inside this bounding box
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                return detection
                
        return None

# Initialize the detector
detector = PersonDetector()

@app.get("/")
async def root():
    return {"message": "Welcome to Person Detector and Cropper API"}

@app.post("/detect/", response_model=ProcessedImage)
async def detect_and_visualize(file: UploadFile = File(...)):
    """
    Process an image to detect objects and return a visualized result with bounding boxes
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    image_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{image_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read image
    image = cv2.imread(str(file_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect persons
    try:
        detections = detector.detect_persons(image)
        
        # Draw bounding boxes on the image
        result_img = detector.draw_detection_boxes(image, detections)
        
        # Convert back to BGR for saving with OpenCV
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # Save the processed image
        output_filename = f"{image_id}_detected.jpg"
        output_path = PROCESSED_DIR / output_filename
        cv2.imwrite(str(output_path), result_img_bgr)
        
        # Store metadata for later use
        IMAGE_METADATA[image_id] = {
            "filename": file.filename,
            "path": str(file_path),
            "height": image.shape[0],
            "width": image.shape[1],
            "detections": [detection.dict() for detection in detections]
        }
        
        # Return the processed image info
        return ProcessedImage(
            image_id=image_id,
            filename=output_filename,
            detections=detections
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/extract-object/", response_model=ExtractedObject)
async def extract_object_at_click(
    image_id: str = Form(...),
    click_x: int = Form(...),
    click_y: int = Form(...),
    draw_boundary: bool = Form(True),  # Optional parameter to control boundary drawing
    full_image: bool = Form(False)  # Optional parameter to request full image with background removed
):
    """
    Extract an object from a previously processed image based on click coordinates
    """
    # Check if the image exists in our metadata
    if image_id not in IMAGE_METADATA:
        raise HTTPException(status_code=404, detail="Image not found. Please process an image first.")
    
    metadata = IMAGE_METADATA[image_id]
    
    # Read the original image
    image_path = metadata["path"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert stored detections back to DetectionResult objects
    detections = [DetectionResult(**det) for det in metadata["detections"]]
    
    # Find which object was clicked
    object_detection = detector.find_object_at_coordinates(
        detections, 
        click_x, 
        click_y, 
        metadata["height"], 
        metadata["width"]
    )
    
    if not object_detection:
        raise HTTPException(status_code=400, detail="No object found at the clicked coordinates")
    
    try:
        # Get object ID
        object_id = object_detection.id
        
        # Create response object
        response = ExtractedObject(
            image_id=image_id,
            object_id=object_id,
            filename=""
        )
        
        # Process with boundaries if requested
        if draw_boundary:
            output_with_boundary = detector.crop_and_remove_bg(image, object_detection.box, draw_boundary=True)
            boundary_filename = f"{image_id}_{object_id}_extracted.png"
            boundary_path = OUTPUT_DIR / boundary_filename
            output_with_boundary.save(boundary_path)
            response.filename = boundary_filename
        
        # Always create a clean version without boundaries
        clean_output = detector.crop_and_remove_bg(image, object_detection.box, draw_boundary=False)
        clean_filename = f"{image_id}_{object_id}_clean.png"
        clean_path = OUTPUT_DIR / clean_filename
        clean_output.save(clean_path)
        response.clean_filename = clean_filename
        
        # Create a full image with background removed
        full_output = detector.remove_bg_full_image(image)
        full_filename = f"{image_id}_full_removed_bg.png"
        full_path = OUTPUT_DIR / full_filename
        full_output.save(full_path)
        response.full_image_filename = full_filename
        
        # If no boundary was requested, use clean version as primary
        if not draw_boundary:
            response.filename = clean_filename
        
        # Return the file info
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    Get any image by filename - will check both processed and extracted directories
    """
    # First check in the processed directory
    processed_path = PROCESSED_DIR / filename
    if processed_path.exists():
        return FileResponse(path=processed_path)
    
    # Then check in the output directory
    output_path = OUTPUT_DIR / filename
    if output_path.exists():
        return FileResponse(path=output_path)
    
    # If not found in either directory
    raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in any directory")

# Keep the original endpoints for backward compatibility
@app.get("/processed/{filename}")
async def get_processed_image(filename: str):
    """
    Get a processed image by filename
    """
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        # Try in the output directory as a fallback
        output_path = OUTPUT_DIR / filename
        if output_path.exists():
            return FileResponse(path=output_path)
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found")
    
    return FileResponse(path=file_path)

@app.get("/extracted/{filename}")
async def get_extracted_object(filename: str):
    """
    Get an extracted object image by filename
    """
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        # Try in the processed directory as a fallback
        processed_path = PROCESSED_DIR / filename
        if processed_path.exists():
            return FileResponse(path=processed_path)
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found")
    
    return FileResponse(path=file_path)
