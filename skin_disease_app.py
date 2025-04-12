import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR =r"C:\Users\Manjesh R\OneDrive\Desktop\sk\testdata\dataset\archive (2)\skin-disease-datasaet\train_set"
IMG_SIZE = 64

def load_images(data_dir, img_size=IMG_SIZE):
    images, labels = [], []
    classes = os.listdir(data_dir)
    for label in classes:
        class_path = os.path.join(data_dir, label)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            file_path = os.path.join(class_path, file)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f" Skipped (not readable): {file_path}")
                    continue
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f" Could not process image: {file_path} - {e}")
    return np.array(images), np.array(labels)

print("📥 Loading images...")
X, y = load_images(DATA_DIR)
X = X / 255.0  

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(" Training the model...")
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test),
          epochs=10)

model.save('skin_disease_model.h5')
print(" Model trained and saved as skin_disease_model.h5")

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_label = le.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    except Exception as e:
        return f" Error: {e}"
print(predict_image(r"C:\Users\Manjesh R\AppData\Local\Microsoft\Windows\INetCache\IE\7Z9TUE4F\IMG-20250410-WA0058[1].jpg"))
