import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, render_template

batch_size = 32 
num_classes = 10
#epochs = 1600
data_augmentation = True
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Normalize the input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model with different hyperparameters
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
optimizers = [Adam, RMSprop, SGD]

for lr in learning_rates:
    for bs in batch_sizes:
        for opt in optimizers:
            model.compile(loss='categorical_crossentropy',
              optimizer="RMSProp",
              metrics=['accuracy'])
            m1=model.fit(x_train, y_train,
              batch_size=bs,
              epochs=50,
              validation_data=(x_test, y_test),
              shuffle=True)

            # Evaluate the model
            loss, accuracy = model.evaluate(x_test, y_test)
            print(f'Learning Rate: {lr}, Batch Size: {bs}, Optimizer: {opt.__name__}, Accuracy: {accuracy:.3f}')

            # Calculate precision, recall, and F1 score
            y_pred = model.predict(x_test)
            y_pred_class = np.argmax(y_pred, axis=1)
            precision = precision_score(y_test, y_pred_class, average='macro')
            recall = recall_score(y_test, y_pred_class, average='macro')
            f1 = f1_score(y_test, y_pred_class, average='macro')
            print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')

# Fine-tune the model using data augmentation and regularization
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(x_train)

model.add(Dropout(0.2))
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))

# Build a user-friendly interface using Flask
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        img = tf.io.read_file(image)
        img = tf.image.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        return render_template('result.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)