import os
import numpy as np
import warnings
import sys
import uuid
from datetime import datetime
import json
import cv2

# Try importing TensorFlow with a timeout safety mechanism
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Will create a mock model instead.")

class TFLiteModelHandler:
    def __init__(self, model_path="model/detect.tflite", image_save_dir="saved_images"):
        """
        Initialize the TFLite model handler
        
        Args:
            model_path: Path to the TFLite model
            image_save_dir: Directory to save images
        """
        # Create directory for saving images if it doesn't exist
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        # Load the TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_loaded = True
        except Exception as e:
            warnings.warn(f"Failed to load TFLite model: {e}")
            self.model_loaded = False
    
    def save_image(self, image, coordinates=None):
        """
        Save image received from frontend
        
        Args:
            image: numpy array or image data
            coordinates: optional bounding box or coordinates data
        
        Returns:
            dict: information about saved image including path and ID
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        filename = f"{img_id}.jpg"
        filepath = os.path.join(self.image_save_dir, filename)
        
        # Save the image
        if isinstance(image, np.ndarray):
            cv2.imwrite(filepath, image)
        else:
            # Handle case where image might be in different format
            with open(filepath, 'wb') as f:
                f.write(image)
        
        # Save coordinates if provided
        metadata = {"image_id": img_id, "path": filepath}
        if coordinates:
            metadata["coordinates"] = coordinates
            # Save coordinates to a JSON file
            coord_file = os.path.join(self.image_save_dir, f"{img_id}_coords.json")
            with open(coord_file, 'w') as f:
                json.dump(coordinates, f)
            metadata["coord_file"] = coord_file
            
        return metadata
    
    def process_image(self, image, save_image=True, coordinates=None):
        """
        Process image with TFLite model and optionally save it
        
        Args:
            image: input image
            save_image: whether to save the image
            coordinates: optional coordinates data
            
        Returns:
            dict: model output and saved image info if applicable
        """
        result = {}
        
        # Save image if requested
        if save_image:
            saved_info = self.save_image(image, coordinates)
            result["saved_image"] = saved_info
        
        # Process with model if available
        if self.model_loaded:
            # Preprocess image for model
            processed_image = self._preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            outputs = {}
            for output in self.output_details:
                tensor = self.interpreter.get_tensor(output['index'])
                outputs[output['name'] if 'name' in output else f"output_{output['index']}"] = tensor
            
            result["model_output"] = outputs
            
        return result
    
    def _preprocess_image(self, image):
        """
        Preprocess image for the model
        """
        # Handle different image formats
        if not isinstance(image, np.ndarray):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get input shape from model
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4:  # [batch, height, width, channels]
            target_height, target_width = input_shape[1], input_shape[2]
            
            # Resize image to match model input
            preprocessed = cv2.resize(image, (target_width, target_height))
            
            # Convert to RGB if needed (model expects RGB but OpenCV uses BGR)
            if preprocessed.shape[-1] == 3:
                preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values if needed
            if self.input_details[0]['dtype'] == np.float32:
                preprocessed = preprocessed.astype(np.float32) / 255.0
            
            # Add batch dimension if needed
            if len(preprocessed.shape) == 3:
                preprocessed = np.expand_dims(preprocessed, axis=0)
            
            return preprocessed
        
        return image

def create_mock_detection_model(output_path="model/detect.tflite"):
    """
    Creates a simple mock detection model when TensorFlow is not available
    or when network issues prevent downloading models.
    """
    print("Creating a mock detection model...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a very simple model file with some bytes
    # This is just a placeholder - it won't actually work for detection
    with open(output_path, 'wb') as f:
        # Write a dummy header to make it look like a TFLite file
        f.write(b'TFL3')
        # Add some random data to fill it out
        f.write(os.urandom(1024))  
    
    print(f"Mock model saved to {output_path}")
    print("NOTE: This is a placeholder model for development purposes.")
    print("When online, run this script again to download the real model.")
    return True

def download_and_convert_model():
    """
    Downloads a pre-trained model and converts it to TFLite format.
    Falls back to a mock model if there are issues.
    """
    # Check if TensorFlow is available
    if not TF_AVAILABLE:
        return create_mock_detection_model()
    
    model_dir = "model"
    output_model_path = f"{model_dir}/detect.tflite"
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # If the model already exists, don't download it again
    if os.path.exists(output_model_path):
        print(f"Model already exists at {output_model_path}")
        return True
    
    try:
        # Try to import tensorflow_hub with a timeout
        try:
            import tensorflow_hub as hub
        except ImportError:
            print("TensorFlow Hub not installed. Trying to install it...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-hub", "--no-cache-dir"])
            import tensorflow_hub as hub
        
        print("Downloading model from TensorFlow Hub...")
        
        # Set a timeout for the download
        import socket
        # Set a 30-second timeout for socket operations
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        
        try:
            # Use a smaller, faster model
            detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")            
            # Create a concrete function from the model
            input_shape = [1, 320, 320, 3]
            input_tensor = tf.ones(input_shape, dtype=tf.uint8)
            
            @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.uint8)])
            def detect_fn(input_tensor):
                # Get model predictions
                model_output = detector(input_tensor)
                return {
                    'detection_boxes': model_output['detection_boxes'],
                    'detection_classes': model_output['detection_classes'],
                    'detection_scores': model_output['detection_scores'],
                    'num_detections': model_output['num_detections']
                }
            
            concrete_func = detect_fn.get_concrete_function()
            
            # Convert the model to TFLite
            print("Converting model to TFLite format...")
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            
            # Set optimization flags for a smaller model
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            # Try to make the model smaller
            converter.target_spec.supported_types = [tf.float16]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(output_model_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Model saved to {output_model_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading or converting model: {e}")
            return create_mock_detection_model()
            
        finally:
            # Restore original socket timeout
            socket.setdefaulttimeout(original_timeout)
            
    except Exception as e:
        print(f"Error in model generation: {e}")
        return create_mock_detection_model()

def download_pretrained_model():
    """
    Alternative approach: Download a pre-trained TFLite model directly
    """
    import urllib.request
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    output_path = f"{model_dir}/detect.tflite"
    
    # If the model already exists, don't download it again
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return True
        
    print("Downloading pre-trained TFLite model...")
    
    try:
        # URL to a small pre-trained TFLite model (SSD MobileNet)
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"        
        # Set a timeout for the download
        import socket
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        
        # Download the zip file to a temporary location
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
            
        urllib.request.urlretrieve(model_url, temp_path)
        
        # Extract the model file
        import zipfile
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # Extract the detect.tflite file
            for file in zip_ref.namelist():
                if file.endswith('.tflite'):
                    # Extract and rename
                    with zip_ref.open(file) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
                    break
        
        # Remove the temporary zip file
        os.unlink(temp_path)
        
        print(f"Pre-trained model saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading pre-trained model: {e}")
        return create_mock_detection_model()
        
    finally:
        # Restore original socket timeout
        socket.setdefaulttimeout(original_timeout)

# Function to handle images from the frontend
def process_frontend_image(image_data, save_image=True, coordinates=None):
    """
    Process an image received from the frontend
    
    Args:
        image_data: The image data from the frontend
        save_image: Whether to save the image
        coordinates: Optional coordinates data
        
    Returns:
        dict: Result of model processing and image saving
    """
    model_handler = TFLiteModelHandler()
    return model_handler.process_image(image_data, save_image, coordinates)


if __name__ == "__main__":
    # First try downloading a pre-trained model directly
    if not download_pretrained_model():
        # If that fails, try converting from TF Hub
        if not download_and_convert_model():
            # If both methods fail, create a mock model
            create_mock_detection_model()