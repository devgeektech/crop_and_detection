from functools import lru_cache
import tensorflow as tf
import numpy as np

class ObjectDetector:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectDetector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        if self._model is None:
            self._model = self.load_model()
        self.boundary_color = (255, 255, 255)
        self.box_color = (0, 255, 0)
        self.text_color = (0, 0, 0)
        self._cache = {}
    
    @lru_cache(maxsize=100)
    def get_cached_detection(self, image_hash):
        return self._cache.get(image_hash)
    
    def load_model(self):
        # Load TFLite model
        try:
            interpreter = tf.lite.Interpreter(model_path="model/detect.tflite")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    async def detect_objects(self, image):
        # Generate image hash for caching
        if isinstance(image, np.ndarray):
            image_hash = hash(image.tobytes())
        else:
            image_hash = hash(image)
            
        # Check cache first
        cached_result = self.get_cached_detection(image_hash)
        if cached_result:
            return cached_result
            
        # Process image if not in cache
        results = []
        try:
            # Get input and output tensors
            input_details = self._model.get_input_details()
            output_details = self._model.get_output_details()
            
            # Prepare input image
            input_shape = input_details[0]['shape']
            input_data = np.expand_dims(image, axis=0)
            if input_data.shape != tuple(input_shape):
                input_data = tf.image.resize(input_data, (input_shape[1], input_shape[2]))
                input_data = tf.cast(input_data, tf.uint8)
                input_data = input_data.numpy()
            
            # Set input tensor
            self._model.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self._model.invoke()
            
            # Get results
            boxes = self._model.get_tensor(output_details[0]['index'])[0]
            classes = self._model.get_tensor(output_details[1]['index'])[0]
            scores = self._model.get_tensor(output_details[2]['index'])[0]
            
            # Filter results
            for i in range(len(scores)):
                if scores[i] >= 0.5:  # Confidence threshold
                    results.append({
                        'box': boxes[i].tolist(),
                        'class': int(classes[i]),
                        'score': float(scores[i])
                    })
            
            # Cache results
            self._cache[image_hash] = results
            return results
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return []

# Global detector instance
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector
