A quantized TensorFlow Lite (.tflite) model has been shared in place of the original .h5 model to reduce file size and ensure efficient deployment. The model was converted using post-training quantization techniques to optimize it for inference on resource-constrained environments. While the original training and evaluation were performed using a full-precision Keras model, the final .tflite format was chosen for its portability and compatibility with web and mobile applications.
We sincerely apologize for not including the original .h5 model due to its large size. However, all training steps, model architecture, and conversion details have been thoroughly documented in the accompanying notebook. In addition, the Flask application and prediction logic have been updated to work seamlessly with the provided .tflite model.
app.py for tflite model is:
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model/model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of class labels
labels = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis',
    'mabea', 'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus',
    'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa'
]

# Home route
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('predict.html', prediction=None, image_path=None, confidence=None)

        # Save uploaded image
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        upload_path = os.path.join(upload_folder, file.filename)
        file.save(upload_path)

        # Preprocess image
        img = image.load_img(upload_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0).astype(np.float32)  # Make sure dtype matches input tensor

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Interpret results
        pred = np.argmax(output_data)
        predicted_label = labels[pred]
        confidence = float(output_data[pred] * 100)

        return render_template('predict.html',
                               prediction=predicted_label,
                               image_path=file.filename,
                               confidence=confidence)

    return render_template('predict.html', prediction=None, image_path=None, confidence=None)

# Logout route
@app.route('/logout.html')
def logout():
    return render_template('logout.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
