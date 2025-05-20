import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/detect.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Input Details ===")
for input in input_details:
    print(input)

print("\n=== Output Details ===")
for output in output_details:
    print(output)

# Optional: list all tensor details
print("\n=== All Tensor Details ===")
all_tensors = interpreter.get_tensor_details()
for tensor in all_tensors:
    print(tensor['name'])
