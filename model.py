import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
import cv2

# Load and preprocess your dataset
def load_and_preprocess_data():
    # Load images and labels from disk
    X = []
    y = []
    for i in range(num_samples):
        img = cv2.imread('marble_' + str(i) + '.jpg')
        label = int(open('marble_' + str(i) + '.txt').read())
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    # Resize images to a uniform size
    X = [cv2.resize(img, (128, 128)) for img in X]
    X = np.array(X)

    # Normalize pixel values
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, y_train, X_test, y_test


# Load and preprocess your dataset
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Define your CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile your model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit your model to the training data
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate your model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
