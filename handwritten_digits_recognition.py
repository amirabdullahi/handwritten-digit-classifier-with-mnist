# MNIST Digit Recognition with CNN Only

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========== STEP 1: Load & Visualize Data ==========
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.tight_layout()
plt.show()

# ========== STEP 2: Preprocess Data ==========
# Normalize to [0, 1] and reshape for CNN input
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# ========== STEP 3: Build CNN ==========
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== STEP 4: Train CNN ==========
history = model.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=128, validation_split=0.1, verbose=2)

# ========== STEP 5: Evaluate CNN ==========
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ========== STEP 6: Visualize Training ==========
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ========== STEP 7: Classification Report & Confusion Matrix ==========
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========== STEP 8: Sample Predictions ==========
print("\nSample Predictions:")
plt.figure(figsize=(15, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    conf = np.max(y_pred_probs[i]) * 100
    plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]} ({conf:.1f}%)")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ========== STEP 9: Show Misclassified ==========
misclassified = np.where(y_pred != y_test)[0]
print(f"\nTotal misclassified samples: {len(misclassified)}")

plt.figure(figsize=(15, 6))
for i in range(min(10, len(misclassified))):
    idx = misclassified[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    conf = np.max(y_pred_probs[idx]) * 100
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]} ({conf:.1f}%)")
    plt.axis('off')
plt.suptitle("Misclassified Samples")
plt.tight_layout()
plt.show()

# ========== STEP 10: Save & Reload Model ==========
model.save("cnn_mnist_model.keras")
print("\nModel saved as cnn_mnist_model.keras")

# Test reloading
reloaded_model = keras.models.load_model("cnn_mnist_model.keras")
_, reloaded_accuracy = reloaded_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"Reloaded model accuracy: {reloaded_accuracy:.4f}")
