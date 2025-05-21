import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from rembg import remove
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import shutil
import uuid
import json
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("processed")
BACKGROUND_DIR = Path("backgrounds")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
BACKGROUND_DIR.mkdir(exist_ok=True)

aiml_client = OpenAI(
    api_key=aiml_api_key,
    base_url="https://api.aimlapi.com/v1"
)

DEFAULT_BACKGROUND_OPTIONS = [
    "beach", "mountain", "city", "forest", "space", "sunset", "office",
    "studio", "abstract", "gradient", "solid color", "vintage", "futuristic"
]

IMAGE_METADATA = {}

app = FastAPI(title="Person Detector and Cropper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class DetectionResult(BaseModel):
    id: str
    box: List[float]
    score: float
    class_id: int
    class_name: str
    
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

def generate_background_suggestions(image_path=None):
    """Generate background suggestions using OpenAI API based on the image"""
    try:
        fallback_suggestions = random.sample(DEFAULT_BACKGROUND_OPTIONS, min(3, len(DEFAULT_BACKGROUND_OPTIONS)))
        fallback_result = [
            BackgroundSuggestion(name=bg, description=f"A {bg} background") 
            for bg in fallback_suggestions
        ]
        
        if not openai.api_key:
            print("No OpenAI API key provided, using fallback background suggestions")
            return fallback_result
        
        image_description = "a person's photo"
        
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    import base64
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                try:
                    vision_response = aiml_client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image briefly, focusing on the subject, their attire, and the overall mood. Keep it under 50 words."},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=100
                    )
                    
                    image_description = vision_response.choices[0].message.content
                except Exception as e:
                    print(f"Error in GPT-4 Vision API call: {e}")
                    image_description = "a person in a photo"
                print(f"Image description: {image_description}")
            except Exception as e:
                print(f"Error analyzing image: {e}")
        
        try:
            response = aiml_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative assistant that suggests unique and visually appealing background ideas for photos."},
                    {"role": "user", "content": f"Based on this image description: '{image_description}', suggest three creative and visually appealing background ideas that would complement the subject. For each suggestion, provide a short name and a brief description that could be used to generate the background with DALL-E. Make sure each suggestion is distinct and would look good with the subject."},
                ],
                temperature=0.8,
                max_tokens=200,
            )
        except Exception as e:
            print(f"Error with AIML API chat completion: {e}")
            return fallback_result
        
        content = response.choices[0].message.content
        
        suggestions = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('1.', '2.', '3.', '•', '-', '*')) or line[0].isdigit() and line[1:3] in ('. ', ') '):
                line = line.split(' ', 1)[1].strip() if ' ' in line else line
                
                if ':' in line:
                    name, description = line.split(':', 1)
                    suggestions.append(BackgroundSuggestion(name=name.strip(), description=description.strip()))
                else:
                    suggestions.append(BackgroundSuggestion(name=line, description=line))
        
        if not suggestions or len(suggestions) < 3:
            words = content.replace('.', ' ').replace(',', ' ').split()
            potential_names = [w for w in words if len(w) > 4 and w.lower() not in ['background', 'suggest', 'creative', 'photo']]
            
            while len(potential_names) < 3:
                potential_names.append(random.choice(DEFAULT_BACKGROUND_OPTIONS))
                
            suggestions = [
                BackgroundSuggestion(name=name, description=f"A {name.lower()} background") 
                for name in random.sample(potential_names, min(3, len(potential_names)))
            ]
        
        return suggestions[:3]  # Return at most 3 suggestions
        
    except Exception as e:
        print(f"Error generating background suggestions: {e}")
        # Fallback to default options
        suggestions = random.sample(DEFAULT_BACKGROUND_OPTIONS, min(3, len(DEFAULT_BACKGROUND_OPTIONS)))
        return [
            BackgroundSuggestion(name=bg, description=f"A {bg} background") 
            for bg in suggestions
        ]

def combine_with_background(foreground_image, background_image, position="center", scale=1.0):
    """Combine a foreground image (with transparency) with a background image"""
    try:
        # Convert to PIL Image if they're not already
        if not isinstance(foreground_image, Image.Image):
            foreground_image = Image.open(foreground_image)
        if not isinstance(background_image, Image.Image):
            background_image = Image.open(background_image)
            
        # Ensure foreground has alpha channel
        if foreground_image.mode != 'RGBA':
            foreground_image = foreground_image.convert('RGBA')
            
        # Convert background to RGBA
        background = background_image.convert('RGBA')
        
        # Get dimensions
        fg_width, fg_height = foreground_image.size
        bg_width, bg_height = background.size
        
        # Scale foreground if needed
        if scale != 1.0:
            new_width = int(fg_width * scale)
            new_height = int(fg_height * scale)
            foreground_image = foreground_image.resize((new_width, new_height), Image.LANCZOS)
            fg_width, fg_height = foreground_image.size
        
        # Calculate position to place foreground
        if position == "center":
            position = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)
        elif position == "bottom_center":
            position = ((bg_width - fg_width) // 2, bg_height - fg_height - 10)
        elif position == "top_center":
            position = ((bg_width - fg_width) // 2, 10)
        elif position == "random":
            max_x = max(0, bg_width - fg_width)
            max_y = max(0, bg_height - fg_height)
            position = (random.randint(0, max_x), random.randint(0, max_y))
        elif isinstance(position, tuple) and len(position) == 2:
            # Use provided position tuple
            position = position
        else:
            # Default to center if position is not recognized
            position = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)
        
        # Create a new composite image
        composite = Image.new('RGBA', background.size, (0, 0, 0, 0))
        composite.paste(background, (0, 0))
        composite.paste(foreground_image, position, foreground_image)
        
        # Convert back to RGB for saving
        return composite.convert('RGB')
        
    except Exception as e:
        print(f"Error combining images: {e}")
        # Return the background image if there's an error
        if isinstance(background_image, Image.Image):
            return background_image.convert('RGB') if background_image.mode == 'RGBA' else background_image
        return None

def add_text_to_image(image, text, font_size=24, color="white", position="bottom", outline_color="black"):
    """Add text to an image with optional outline"""
    try:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
            
        # Create a copy of the image to avoid modifying the original
        img_with_text = image.copy()
        
        # Create a drawing context
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to load a font, fall back to default if not available
        try:
            # Try to find a system font
            import matplotlib.font_manager as fm
            system_fonts = fm.findSystemFonts()
            if system_fonts:
                # Use the first available font
                font = ImageFont.truetype(system_fonts[0], font_size)
            else:
                font = ImageFont.load_default()
                font_size = 16  # Default font size
        except Exception as e:
            print(f"Error loading font: {e}")
            font = ImageFont.load_default()
            font_size = 16  # Default font size
        
        # Get image dimensions
        width, height = img_with_text.size
        
        # Calculate text size
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except:
            # For newer Pillow versions
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Determine text position based on the specified position
        if position == "top":
            text_position = ((width - text_width) // 2, 10)
        elif position == "bottom":
            text_position = ((width - text_width) // 2, height - text_height - 10)
        elif position == "left":
            text_position = (10, (height - text_height) // 2)
        elif position == "right":
            text_position = (width - text_width - 10, (height - text_height) // 2)
        elif position == "center":
            text_position = ((width - text_width) // 2, (height - text_height) // 2)
        else:
            # Default to bottom if position is not recognized
            text_position = ((width - text_width) // 2, height - text_height - 10)
        
        # Draw text outline if outline color is provided
        if outline_color:
            # Draw text multiple times with small offsets to create outline effect
            for offset_x, offset_y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                draw.text(
                    (text_position[0] + offset_x, text_position[1] + offset_y),
                    text,
                    font=font,
                    fill=outline_color
                )
        
        # Draw the main text
        draw.text(
            text_position,
            text,
            font=font,
            fill=color
        )
        
        return img_with_text
        
    except Exception as e:
        print(f"Error adding text to image: {e}")
        # Return the original image if there's an error
        return image

def generate_background_image(description, size=(1024, 1024)):
    """Generate a background image using DALL-E 2 based on the description"""
    try:
        if not aiml_api_key:
            print("No AIML API key provided, cannot generate background image")
            return None
            
        # Enhance the description for better background generation
        enhanced_prompt = f"A person standing in front of {description}, realistic, high quality, full background"
        print(f"Generating DALL-E image with prompt: {enhanced_prompt}")
        
        # Generate image with DALL-E 2
        try:
            response = aiml_client.images.generate(
                model="dall-e-2",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except Exception as e:
            print(f"Error in DALL-E API call: {e}")
            # Try with a simpler prompt if the original fails
            simplified_prompt = f"A person standing in front of {description}, photo realistic"
            print(f"Trying with simplified prompt: {simplified_prompt}")
            try:
                response = aiml_client.images.generate(
                    model="dall-e-2",
                    prompt=simplified_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
            except Exception as e2:
                print(f"Error in second DALL-E API call: {e2}")
                return None
        
        # Get the image URL
        image_url = response.data[0].url
        
        # Download the image
        import requests
        from io import BytesIO
        
        response = requests.get(image_url)
        if response.status_code == 200:
            # Create a PIL Image from the response content
            image = Image.open(BytesIO(response.content))
            
            # Resize if needed
            if image.size != size:
                image = image.resize(size)
                
            # Create a unique filename for the background
            background_id = str(uuid.uuid4())
            filename = f"background_{background_id}.png"
            filepath = BACKGROUND_DIR / filename
            
            # Save the image
            image.save(filepath)
            print(f"Generated background saved to {filepath}")
            
            return filepath
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error generating background image: {e}")
        return None

class ObjectDetector:
    def __init__(self):
        # Load the TFLite model
        self.interpreter = self.load_model()
        # Colors for drawing boundaries
        self.boundary_color = (255, 255, 255)  # White color for boundaries
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
        
    def detect_objects(self, image):
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
        
        # Filter for objects with confidence > 0.5
        valid_indices = np.where(scores > 0.5)
        valid_boxes = boxes[valid_indices]
        valid_classes = classes[valid_indices].astype(np.int32)
        valid_scores = scores[valid_indices]
        
        # Create detection results
        results = []
        for i, (box, class_id, score) in enumerate(zip(valid_boxes, valid_classes, valid_scores)):
            # Get class name from COCO labels
            class_name = COCO_LABELS.get(class_id, f"Unknown-{class_id}")
            
            results.append(DetectionResult(
                id=str(i),
                box=box.tolist(),
                score=float(score),
                class_id=int(class_id),
                class_name=class_name
            ))
        
        return results
        
    def draw_detection_boxes(self, image, detections):
        """
        Draw bounding boxes around detected objects
        """
        result_img = image.copy()
        h, w = image.shape[:2]
        
        # Create a color map for different classes
        color_map = {
            'person': (0, 255, 0),  # Green for person
            'default': (0, 255, 0)  # Default green
        }
        
        for detection in detections:
            # Get box coordinates
            y1, x1, y2, x2 = detection.box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
            
            # Choose color based on class
            box_color = color_map.get(detection.class_name, color_map['default'])
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 2)
            
            # Add label with class name
            label = f"{detection.class_name}: {detection.score:.2f}"
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
            # Create a drawing context
            draw = ImageDraw.Draw(output)
            
            # Get dimensions of the cropped image
            img_width, img_height = output.size
            
            # Draw a rectangle around the boundary (3 pixels wide)
            for i in range(3):
                draw.rectangle(
                    [(i, i), (img_width - 1 - i, img_height - 1 - i)],
                    outline=self.boundary_color,
                    width=1
                )
        
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
        
    def apply_background(self, image_with_transparent_bg, background_name, position="center", scale=1.0):
        """
        Apply a background to an image with transparent background
        """
        try:
            # Check if the image has an alpha channel (transparency)
            if image_with_transparent_bg.mode != 'RGBA':
                image_with_transparent_bg = image_with_transparent_bg.convert('RGBA')
                
            # Get image dimensions
            width, height = image_with_transparent_bg.size
            
            # First try to generate a background using DALL-E 2
            background_path = None
            try:
                print(f"Generating background with DALL-E 2: {background_name}")
                background_path = generate_background_image(background_name, size=(width*2, height*2))  # Generate larger background for better composition
            except Exception as e:
                print(f"Error generating background with DALL-E 2: {e}")
            
            # If DALL-E 2 generation was successful, use that background
            if background_path and os.path.exists(background_path):
                background = Image.open(background_path).convert('RGBA')
                # Use the new combine function for better results
                return combine_with_background(image_with_transparent_bg, background, position=position, scale=scale)
            else:
                # Try to find a matching background image in the backgrounds directory
                background_files = list(BACKGROUND_DIR.glob(f"*{background_name}*.{{'jpg','jpeg','png'}}*"))
                
                if background_files:
                    # Use the first matching background file
                    background_path = background_files[0]
                    background = Image.open(background_path).convert('RGBA')
                    # Use the new combine function for better results
                    return combine_with_background(image_with_transparent_bg, background, position=position, scale=scale)
                else:
                    # Generate a simple background based on the name
                    background = None
                    
                    # Try to interpret the background name as a color
                    try:
                        # Check for common color names
                        color_map = {
                            'red': (255, 0, 0, 255),
                            'green': (0, 255, 0, 255),
                            'blue': (0, 0, 255, 255),
                            'yellow': (255, 255, 0, 255),
                            'purple': (128, 0, 128, 255),
                            'orange': (255, 165, 0, 255),
                            'pink': (255, 192, 203, 255),
                            'black': (0, 0, 0, 255),
                            'white': (255, 255, 255, 255),
                            'gray': (128, 128, 128, 255),
                            'brown': (165, 42, 42, 255),
                            'cyan': (0, 255, 255, 255),
                            'magenta': (255, 0, 255, 255),
                            'teal': (0, 128, 128, 255),
                            'navy': (0, 0, 128, 255),
                            'olive': (128, 128, 0, 255),
                            'maroon': (128, 0, 0, 255),
                        }
                        
                        # Check if the background name contains a color name
                        for color_name, color_value in color_map.items():
                            if color_name in background_name.lower():
                                background = Image.new('RGBA', (width*2, height*2), color_value)  # Create larger background
                                break
                                
                        # If no color match, create a gradient or pattern based on the name
                        if background is None and 'gradient' in background_name.lower():
                            # Create a simple gradient background
                            background = Image.new('RGBA', (width*2, height*2), (0, 0, 0, 255))  # Create larger background
                            draw = ImageDraw.Draw(background)
                            
                            # Choose colors based on the background name
                            if 'blue' in background_name.lower():
                                start_color = (0, 0, 255, 255)
                                end_color = (0, 255, 255, 255)
                            elif 'red' in background_name.lower():
                                start_color = (255, 0, 0, 255)
                                end_color = (255, 255, 0, 255)
                            elif 'green' in background_name.lower():
                                start_color = (0, 255, 0, 255)
                                end_color = (0, 255, 255, 255)
                            else:
                                # Random gradient colors
                                start_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                                end_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                                
                            # Draw gradient (horizontal)
                            bg_width, bg_height = background.size
                            for x in range(bg_width):
                                r = int(start_color[0] + (end_color[0] - start_color[0]) * x / bg_width)
                                g = int(start_color[1] + (end_color[1] - start_color[1]) * x / bg_width)
                                b = int(start_color[2] + (end_color[2] - start_color[2]) * x / bg_width)
                                draw.line([(x, 0), (x, bg_height)], fill=(r, g, b, 255))
                        
                        # If we have a background, use the combine function
                        if background is not None:
                            return combine_with_background(image_with_transparent_bg, background, position=position, scale=scale)
                                
                    except Exception as e:
                        print(f"Error creating background: {e}")
                    
                    # Fallback to a simple colored background if all else fails
                    background = Image.new('RGBA', (width*2, height*2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255))
                    return combine_with_background(image_with_transparent_bg, background, position=position, scale=scale)
            
        except Exception as e:
            print(f"Error applying background: {e}")
            # Return the original image if there's an error
            return image_with_transparent_bg.convert('RGB') if image_with_transparent_bg.mode == 'RGBA' else image_with_transparent_bg
        
    def highlight_with_background(self, image, box):
        """
        Crop the image and add a white boundary without removing the background
        """
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
        
        # Convert to PIL Image for processing
        pil_img = Image.fromarray(cropped_img)
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_img)
        
        # Get dimensions of the cropped image
        img_width, img_height = pil_img.size
        
        # Draw a rectangle around the boundary (3 pixels wide)
        for i in range(3):
            draw.rectangle(
                [(i, i), (img_width - 1 - i, img_height - 1 - i)],
                outline=self.boundary_color,
                width=1
            )
        
        return pil_img
    
    def highlight_with_segmentation(self, image, box):
        """
        Highlight the detected object with white boundaries without removing the background
        Uses segmentation to identify the object boundaries more precisely
        """
        h, w = image.shape[:2]
        
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
        
        # Convert to PIL Image for processing
        pil_img = Image.fromarray(cropped_img)
        
        # Create a version with background removed to get the object mask
        temp_no_bg = remove(pil_img)
        
        # Create a mask from the alpha channel of the background-removed image
        # This gives us the precise shape of the object
        if temp_no_bg.mode == 'RGBA':
            r, g, b, a = temp_no_bg.split()
            mask = a
        else:
            # If not RGBA, create a simple mask from the bounding box
            mask = Image.new('L', pil_img.size, 0)
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.rectangle([0, 0, pil_img.width, pil_img.height], fill=255)
        
        # Convert mask to numpy array for processing
        mask_np = np.array(mask)
        
        # Find the contour of the mask using OpenCV
        # First convert to binary image
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert PIL image to OpenCV format for drawing contours
        img_cv = np.array(pil_img)
        
        # Draw the contours on the image with white color (3 pixels wide)
        cv2.drawContours(img_cv, contours, -1, (255, 255, 255), 3)
        
        # Convert back to PIL image
        highlighted_img = Image.fromarray(img_cv)
        
        return highlighted_img
        
    def find_object_at_coordinates(self, detections, click_x, click_y, image_height, image_width):
        """
        Find which detected object contains the clicked coordinates
        """
        # Print debug info
        print(f"Finding object at coordinates: ({click_x}, {click_y}) in image of size {image_width}x{image_height}")
        print(f"Number of detections: {len(detections)}")
        
        # Sort detections by area (smallest first) to handle overlapping boxes
        # This ensures we select the smallest object that contains the click point
        sorted_detections = sorted(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        
        for detection in sorted_detections:
            # Get normalized coordinates
            y1, x1, y2, x2 = detection.box
            
            # Convert normalized coordinates to absolute pixels
            abs_x1 = max(0, int(x1 * image_width))
            abs_y1 = max(0, int(y1 * image_height))
            abs_x2 = min(image_width, int(x2 * image_width))
            abs_y2 = min(image_height, int(y2 * image_height))
            
            # Print debug info for each box
            print(f"Detection {detection.id} ({detection.class_name}): Box [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
            
            # Check if click is inside this bounding box
            if abs_x1 <= click_x <= abs_x2 and abs_y1 <= click_y <= abs_y2:
                print(f"Found match: {detection.class_name} (ID: {detection.id})")
                return detection
        
        print("No object found at the clicked coordinates")
        return None

    # Initialize the detector
detector = ObjectDetector()

@app.get("/")
async def root():
    return {"message": "Welcome to Person Detector and Cropper API"}

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

# Only keep the essential endpoints

class TextOverlay(BaseModel):
    text: str
    font_size: int = 24
    color: str = "white"  # Can be color name or hex code
    position: str = "bottom"  # top, bottom, left, right, center
    outline_color: Optional[str] = "black"  # Optional outline color

class CombinedRequest(BaseModel):
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    draw_boundary: bool = True
    background_name: Optional[str] = None
    generate_background_suggestions: bool = True
    add_text: bool = False
    text_overlay: Optional[TextOverlay] = None

class BackgroundSuggestion(BaseModel):
    name: str
    description: str

class CombinedResponse(BaseModel):
    image_id: str
    detected_image_url: str
    detected_image_filename: Optional[str] = None
    extracted_image_url: Optional[str] = None
    extracted_image_filename: Optional[str] = None
    clean_image_url: Optional[str] = None
    clean_image_filename: Optional[str] = None
    full_image_url: Optional[str] = None
    full_image_filename: Optional[str] = None
    highlighted_with_bg_url: Optional[str] = None
    highlighted_with_bg_filename: Optional[str] = None
    highlighted_segmentation_url: Optional[str] = None
    highlighted_segmentation_filename: Optional[str] = None
    background_suggestions: Optional[List[BackgroundSuggestion]] = None
    background_image_url: Optional[str] = None
    background_image_filename: Optional[str] = None
    text_image_url: Optional[str] = None
    text_image_filename: Optional[str] = None
    detections: List[DetectionResult]

@app.post("/process", response_model=CombinedResponse)
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    click_x: Optional[int] = Form(None),
    click_y: Optional[int] = Form(None),
    draw_boundary: bool = Form(True),
    background_name: Optional[str] = Form(None),
    generate_background_suggestions: bool = Form(False),
    background_position: str = Form("center"),
    background_scale: float = Form(1.0),
    add_text: bool = Form(False),
    text: Optional[str] = Form(None),
    font_size: int = Form(24),
    text_color: str = Form("white"),
    text_position: str = Form("bottom"),
    text_outline_color: Optional[str] = Form("black")
):
    image_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    upload_path = UPLOAD_DIR / f"{image_id}{file_extension}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(str(upload_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    detections = detector.detect_objects(image)
    vis_img = detector.draw_detection_boxes(image, detections)
    vis_filename = f"{image_id}_detected.png"
    vis_path = PROCESSED_DIR / vis_filename
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    IMAGE_METADATA[image_id] = {
        "path": str(upload_path),
        "detections": [det.dict() for det in detections],
        "height": height,
        "width": width
    }

    base_url = str(request.base_url).rstrip('/')
    if not base_url.endswith('/'):
        base_url += '/'
    if os.environ.get("RENDER", "") == "true" and os.environ.get("RENDER_EXTERNAL_URL"):
        base_url = os.environ.get("RENDER_EXTERNAL_URL").rstrip('/')
        if not base_url.endswith('/'):
            base_url += '/'

    response = CombinedResponse(
        image_id=image_id,
        detected_image_url=f"{base_url}image/{vis_filename}" if os.path.exists(vis_path) else None,
        detected_image_filename=vis_filename if os.path.exists(vis_path) else None,
        detections=detections
    )

    try:
        suggestions = generate_background_suggestions(image_path=str(upload_path))
        response.background_suggestions = suggestions
        print(f"Generated background suggestions: {[s.name for s in suggestions]}")
    except Exception as e:
        print(f"Error generating background suggestions in process_image: {e}")
        fallback_suggestions = random.sample(DEFAULT_BACKGROUND_OPTIONS, min(3, len(DEFAULT_BACKGROUND_OPTIONS)))
        response.background_suggestions = [
            BackgroundSuggestion(name=bg, description=f"A {bg} background") 
            for bg in fallback_suggestions
        ]

    # Determine which detection to use
    selected_detection = None
    if click_x is not None and click_y is not None:
        print(f"Received click coordinates: ({click_x}, {click_y})")
        selected_detection = detector.find_object_at_coordinates(detections, click_x, click_y, height, width)
        if selected_detection:
            print(f"Selected detection based on click: ID={selected_detection.id}, class={selected_detection.class_name}")
        else:
            print("No detection found at the clicked location.")

    if selected_detection is None and detections:
        selected_detection = detections[0]
        print(f"Falling back to first detection: ID={selected_detection.id}, class={selected_detection.class_name}")

    if not selected_detection:
        return response

    object_id = selected_detection.id

    highlighted_with_bg = detector.highlight_with_background(image, selected_detection.box)
    highlighted_with_bg_filename = f"{image_id}_{object_id}_highlighted_with_bg.png"
    highlighted_with_bg_path = OUTPUT_DIR / highlighted_with_bg_filename
    highlighted_with_bg.save(highlighted_with_bg_path)
    if os.path.exists(highlighted_with_bg_path):
        response.highlighted_with_bg_url = f"{base_url}image/{highlighted_with_bg_filename}"
        response.highlighted_with_bg_filename = highlighted_with_bg_filename

    highlighted_segmentation = detector.highlight_with_segmentation(image, selected_detection.box)
    highlighted_segmentation_filename = f"{image_id}_{object_id}_highlighted_segmentation.png"
    highlighted_segmentation_path = OUTPUT_DIR / highlighted_segmentation_filename
    highlighted_segmentation.save(highlighted_segmentation_path)
    if os.path.exists(highlighted_segmentation_path):
        response.highlighted_segmentation_url = f"{base_url}image/{highlighted_segmentation_filename}"
        response.highlighted_segmentation_filename = highlighted_segmentation_filename

    if click_x is not None and click_y is not None and selected_detection:
        if draw_boundary:
            output_with_boundary = detector.crop_and_remove_bg(image, selected_detection.box, draw_boundary=True)
            boundary_filename = f"{image_id}_{object_id}_extracted.png"
            boundary_path = OUTPUT_DIR / boundary_filename
            output_with_boundary.save(boundary_path)
            if os.path.exists(boundary_path):
                response.extracted_image_url = f"{base_url}image/{boundary_filename}"
                response.extracted_image_filename = boundary_filename

        clean_output = detector.crop_and_remove_bg(image, selected_detection.box, draw_boundary=False)
        clean_filename = f"{image_id}_{object_id}_clean.png"
        clean_path = OUTPUT_DIR / clean_filename
        clean_output.save(clean_path)
        if os.path.exists(clean_path):
            response.clean_image_url = f"{base_url}image/{clean_filename}"
            response.clean_image_filename = clean_filename

        full_output = detector.remove_bg_full_image(image)
        full_filename = f"{image_id}_full_removed_bg.png"
        full_path = OUTPUT_DIR / full_filename
        full_output.save(full_path)
        if os.path.exists(full_path):
            response.full_image_url = f"{base_url}image/{full_filename}"
            response.full_image_filename = full_filename

        if os.path.exists(clean_path):
            clean_img = Image.open(clean_path)
            bg_name = background_name
            if not bg_name and response.background_suggestions:
                bg_name = response.background_suggestions[0].name
            elif not bg_name:
                bg_name = random.choice(DEFAULT_BACKGROUND_OPTIONS)
            try:
                position = background_position
                scale = background_scale
                background_img = detector.apply_background(clean_img, bg_name, position=position, scale=scale)
                background_filename = f"{image_id}_{object_id}_with_background.png"
                background_path = OUTPUT_DIR / background_filename
                background_img.save(background_path)
                if os.path.exists(background_path):
                    response.background_image_url = f"{base_url}image/{background_filename}"
                    response.background_image_filename = background_filename
                    if add_text and text:
                        img_with_text = add_text_to_image(
                            background_path,
                            text,
                            font_size=font_size,
                            color=text_color,
                            position=text_position,
                            outline_color=text_outline_color
                        )
                        text_filename = f"{image_id}_{object_id}_with_text.png"
                        text_path = OUTPUT_DIR / text_filename
                        img_with_text.save(text_path)
                        if os.path.exists(text_path):
                            response.text_image_url = f"{base_url}image/{text_filename}"
                            response.text_image_filename = text_filename
            except Exception as e:
                print(f"Error applying background or text: {e}")

    return response