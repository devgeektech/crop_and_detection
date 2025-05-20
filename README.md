# Person Detector and Cropper API

This is a FastAPI-based web API for the Person Detector and Cropper application. It allows you to detect persons in images, crop them with background removal, and draw boundaries around detected objects.

## Features

- Person detection using TensorFlow Lite
- Background removal using rembg
- Segmentation boundary drawing
- Full image background removal
- CORS support for frontend integration

## Deployment Guide for API Providers

### Local Deployment (For Development)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the model file is in the correct location:
   - The TFLite model should be in `model/detect.tflite`

3. Run the API locally:

```bash
python run_api.py
```

This will start the API server at http://0.0.0.0:8000

### Production Deployment

To provide this API as a service to frontend developers without sharing the TFLite model:

1. Deploy the API on a cloud server (AWS, GCP, Azure, etc.) or a dedicated server
2. Set up proper authentication (API keys, JWT tokens, etc.)
3. Configure HTTPS for secure communication
4. Provide the API endpoint URL to frontend developers

#### Example Deployment with Docker

```bash
# Build Docker image
docker build -t person-detector-api .

# Run container
docker run -d -p 8000:8000 --name person-detector-api person-detector-api
```

#### Example Deployment with Kubernetes

```bash
# Apply Kubernetes deployment
kubectl apply -f deployment.yaml

# Expose service
kubectl apply -f service.yaml
```

## For Frontend Developers

As a frontend developer, you only need the API endpoint URL provided by the API service. You don't need to install any dependencies or have the TFLite model locally.

## Frontend Integration Guide

This API is designed to be easily integrated with frontend applications. As a frontend developer, you only need to make HTTP requests to the API endpoint URL provided by the API service. You don't need to install any dependencies or have the TFLite model locally.

Here's how to use the API in a typical frontend workflow:

### Step 1: Upload and Detect Objects

```javascript
// Example using fetch API
async function detectObjects(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/detect/', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result; // Contains image_id, filename, and detections
}
```

### Step 2: Display the Processed Image

```javascript
function displayProcessedImage(filename) {
  const imageUrl = `http://localhost:8000/image/${filename}`;
  // Use this URL in an <img> tag or set as background
  document.getElementById('processedImage').src = imageUrl;
}
```

### Step 3: Extract Object on Click

```javascript
async function extractObjectAtClick(imageId, clickX, clickY, drawBoundary = true) {
  const formData = new FormData();
  formData.append('image_id', imageId);
  formData.append('click_x', clickX);
  formData.append('click_y', clickY);
  formData.append('draw_boundary', drawBoundary);
  
  const response = await fetch('http://localhost:8000/extract-object/', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result; // Contains filenames for different versions
}
```

### Step 4: Display Extracted Object(s)

```javascript
function displayExtractedObject(result) {
  // Display object with boundaries
  if (result.filename) {
    document.getElementById('extractedObject').src = 
      `http://localhost:8000/image/${result.filename}`;
  }
  
  // Display clean object without boundaries
  if (result.clean_filename) {
    document.getElementById('cleanObject').src = 
      `http://localhost:8000/image/${result.clean_filename}`;
  }
  
  // Display full image with background removed
  if (result.full_image_filename) {
    document.getElementById('fullImage').src = 
      `http://localhost:8000/image/${result.full_image_filename}`;
  }
}
```

## Complete API Reference

### GET /

Returns a welcome message.

### POST /detect/

Detects persons in an uploaded image and returns a processed image with bounding boxes.

**Request:**
- Form data with an image file

**Response (JSON):**
```json
{
  "image_id": "unique-id",
  "filename": "unique-id_detected.jpg",
  "detections": [
    {
      "id": "0",
      "box": [y1, x1, y2, x2],
      "score": 0.95
    },
    ...
  ]
}
```

### POST /extract-object/

Extracts an object from a previously processed image based on click coordinates.

**Request:**
- Form data with:
  - `image_id`: ID from the detection step
  - `click_x`: X coordinate of the click
  - `click_y`: Y coordinate of the click
  - `draw_boundary`: (Optional) Whether to draw green boundaries (default: true)
  - `full_image`: (Optional) Whether to return full image (default: false)

**Response (JSON):**
```json
{
  "image_id": "unique-id",
  "object_id": "0",
  "filename": "unique-id_0_extracted.png",
  "clean_filename": "unique-id_0_clean.png",
  "full_image_filename": "unique-id_full_removed_bg.png"
}
```

### GET /image/{filename}

Retrieve any image by filename (works for both processed and extracted images).

### GET /processed/{filename}

Retrieve a processed image by filename.

### GET /extracted/{filename}

Retrieve an extracted object image by filename.

## Image Types

1. **Detected Image** (`{image_id}_detected.jpg`):
   - Original image with bounding boxes around detected objects
   - Each object has an ID label for reference

2. **Extracted Object with Boundaries** (`{image_id}_{object_id}_extracted.png`):
   - Cropped object with background removed
   - Green boundary drawn around the object

3. **Clean Extracted Object** (`{image_id}_{object_id}_clean.png`):
   - Cropped object with background removed
   - No boundaries drawn

4. **Full Image with Background Removed** (`{image_id}_full_removed_bg.png`):
   - Entire original image with background removed
   - Transparent background (PNG format)

## Interactive Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
