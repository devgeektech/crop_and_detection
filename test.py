import os
import cv2
import numpy as np
import uuid
from datetime import datetime

class TFLiteModelHandler:
    def __init__(self, model_path, image_save_dir="saved_images"):
        # Initialize TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Create directory for saving images if it doesn't exist
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
    
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
            # You could save coordinates to a JSON file or database here
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
        # Preprocess image for model (resize, normalize, etc.)
        processed_image = self._preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        result = {"model_output": output_data}
        
        # Save image if requested
        if save_image:
            saved_info = self.save_image(image, coordinates)
            result["saved_image"] = saved_info
            
        return result
    
    def _preprocess_image(self, image):
        """
        Preprocess image for the model
        """
        # Add your preprocessing steps here
        # Example: resize to input dimensions
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4:  # [batch, height, width, channels]
            target_height, target_width = input_shape[1], input_shape[2]
            preprocessed = cv2.resize(image, (target_width, target_height))
            preprocessed = preprocessed.astype(np.float32)
            preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
            return preprocessed
        return image

# Example usage in API endpoint or frontend handler
def handle_frontend_image(image_data, coordinates=None):
    model_handler = TFLiteModelHandler("path/to/your/model.tflite")
    result = model_handler.process_image(image_data, save_image=True, coordinates=coordinates)
    return result