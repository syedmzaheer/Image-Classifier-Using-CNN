from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model('model.py')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        img_path = 'path/to/save/uploaded/image' + img.filename
        img.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))  # Adjust the input size according to your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        predictions = model.predict(img_array)
        class_names = ['class1', 'class2', 'class3']  # Replace with your actual class names
        predicted_class = class_names[np.argmax(predictions)]

        return redirect(url_for('result', predicted_class=predicted_class))

    return render_template('index.html')

@app.route('/result/<predicted_class>')
def result(predicted_class):
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)