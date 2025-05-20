import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from rembg import remove
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PersonDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Detector and Cropper")
        
        # Load the TFLite model
        self.interpreter = self.load_model()
        
        # UI setup
        self.setup_ui()
        
        # Variables
        self.image = None
        self.detection_results = None
        self.selected_box = None
        
    def load_model(self):
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="model/detect.tflite")
        interpreter.allocate_tensors()
        return interpreter
        
    def setup_ui(self):
        # Frame for buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Load Image Button
        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Detect Persons Button
        detect_btn = tk.Button(btn_frame, text="Detect Persons", command=self.detect_persons)
        detect_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Crop and Remove BG Button
        crop_btn = tk.Button(btn_frame, text="Crop & Remove BG", command=self.crop_and_remove_bg)
        crop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create a matplotlib figure for display
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image()
            self.detection_results = None
            self.selected_box = None
            
    def detect_persons(self):
        if self.image is None:
            return
            
        # Get input and output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Prepare input image
        input_shape = input_details[0]['shape'][1:3]  # Height, width
        processed_img = cv2.resize(self.image, input_shape)
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
        
        # Store detection results
        self.detection_results = {
            'boxes': person_boxes,
            'scores': person_scores
        }
        
        # Display results
        self.display_detections()
        
    def display_image(self):
        if self.image is None:
            return
            
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.axis('off')
        self.canvas.draw()
        
    def display_detections(self):
        if self.image is None or self.detection_results is None:
            return
            
        self.ax.clear()
        self.ax.imshow(self.image)
        
        h, w = self.image.shape[:2]
        
        # Draw detection boxes
        for i, box in enumerate(self.detection_results['boxes']):
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
            
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='red', linewidth=2)
            self.ax.add_patch(rect)
            self.ax.text(x1, y1, f"Person {i+1}: {self.detection_results['scores'][i]:.2f}", 
                        color='white', backgroundcolor='red', fontsize=8)
        
        # Highlight selected box if any
        if self.selected_box is not None:
            y1, x1, y2, x2 = self.selected_box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
            
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='green', linewidth=3)
            self.ax.add_patch(rect)
        
        self.ax.axis('off')
        self.canvas.draw()
        
    def on_click(self, event):
        if self.detection_results is None or event.xdata is None or event.ydata is None:
            return
            
        click_x, click_y = event.xdata, event.ydata
        h, w = self.image.shape[:2]
        
        # Check if click is inside any box
        for box in self.detection_results['boxes']:
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
            
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.selected_box = box
                self.display_detections()
                return
                
        # If click is outside any box, deselect
        self.selected_box = None
        self.display_detections()
        
    def crop_and_remove_bg(self):
        if self.image is None or self.selected_box is None:
            return
            
        h, w = self.image.shape[:2]
        y1, x1, y2, x2 = self.selected_box
        y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)
        
        # Crop image
        cropped_img = self.image[y1:y2, x1:x2]
        
        # Convert to PIL Image for background removal
        pil_img = Image.fromarray(cropped_img)
        
        # Remove background
        output = remove(pil_img)
        
        # Save the result
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG files", "*.png")])
        if save_path:
            output.save(save_path)
            
if __name__ == "__main__":
    root = tk.Tk()
    app = PersonDetectorApp(root)
    root.mainloop()