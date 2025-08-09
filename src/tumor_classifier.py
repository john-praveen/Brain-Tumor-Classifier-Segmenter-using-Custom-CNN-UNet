from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

#image pre-processing for Training data
image_size=(128,128)
data_dir_classification="/kaggle/input/brain-tumor-dataset-segmentation-and-classification/DATASET/classification/Training"
X=[]
Y=[]
class_names=sorted(os.listdir(data_dir_classification))
print(class_names)

for label, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir_classification, class_name)
    for img_file in os.listdir(class_path):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path).resize(image_size).convert('RGB')
            img_array = np.array(img) / 255.0  
            X.append(img_array)
            Y.append(label)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)

#checking the loaded image has correct lables
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X[i+1500])
    plt.title(class_names[Y[i+1500]])
    plt.axis("off")

#splitting data 
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#CNN for classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])


#model compiling 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'best_model_val_loss.h5',       
    monitor='val_loss',             
    mode='min',                     
    save_best_only=True,            
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[checkpoint]
)

#plotting Tranning accuracy Vs Validation accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#image pre-processing for Testing data
image_size = (128, 128)
data_dir = "/kaggle/input/brain-tumor-dataset-segmentation-and-classification/DATASET/classification/Testing"

X_test = []
y_test = []
class_names_test = sorted(os.listdir(data_dir))
print(class_names_test)

for label, class_name in enumerate(class_names_test):
    class_path = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_path):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path).resize(image_size).convert('RGB')
            img_array = np.array(img) / 255.0  
            X_test.append(img_array)
            y_test.append(label)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

#checking accuracy of model predictions 
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
acc = accuracy_score(y_test, predicted_classes)
print(f"Test Accuracy: {acc*100:.2f}%")
correct = (predicted_classes == y_test)
num_correct = np.sum(correct)
num_incorrect = len(y_test) - num_correct

plt.figure(figsize=(6,4))
plt.bar(['Correct', 'Incorrect'], [num_correct, num_incorrect], color=['green', 'red'])
plt.title(f'Model Prediction Accuracy on Test Set\nAccuracy = {acc*100:.2f}%')
plt.ylabel('Number of Samples')
plt.show()