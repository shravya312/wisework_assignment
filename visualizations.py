import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


base_dir = r"D:\Website Codes\dog-breed-identification"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")
labels_path = os.path.join(base_dir, "labels.csv")


labels = pd.read_csv(labels_path)


image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, 
                                        target_size=image_size,
                                        batch_size=batch_size, 
                                        class_mode='categorical')

valid_gen = datagen.flow_from_directory(valid_dir, 
                                        target_size=image_size, 
                                        batch_size=batch_size, 
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(test_dir, 
                                       target_size=image_size, 
                                       batch_size=batch_size, 
                                       class_mode='categorical')

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

epochs = 5

history = model.fit(train_gen,
                    validation_data=valid_gen,
                    epochs=epochs)

# Save the model
model.save(os.path.join(base_dir, "dog_breed_model.h5"))


def get_img_array(img_path, size):
    """Load and preprocess image"""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.show()

# Example usage
img_path = os.path.join(test_dir, "class1/sample.jpg")  # Replace with a valid image path
img_array = get_img_array(img_path, size=(224, 224))
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv')
display_gradcam(img_path, heatmap)


def prune_model(model):
    """Prune model with TensorFlow Model Optimization"""
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=0,
            end_step=1000)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', 
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])
    
    return model_for_pruning

# Apply pruning
pruned_model = prune_model(model)

# Retrain pruned model
pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
]

history = pruned_model.fit(train_gen,
                           validation_data=valid_gen,
                           batch_size=32,
                           epochs=5,
                           callbacks=pruning_callbacks)

# Strip pruning for export
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
final_model.save(os.path.join(base_dir, "pruned_dog_breed_model.h5"))


# Load the pruned model
pruned_model = load_model(os.path.join(base_dir, "pruned_dog_breed_model.h5"))

# Evaluate on test set
loss, accuracy = pruned_model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy*100:.2f}%")


img_array = get_img_array(img_path, size=(224, 224))
heatmap = make_gradcam_heatmap(img_array, pruned_model, last_conv_layer_name='top_conv')
display_gradcam(img_path, heatmap)

export_dir = os.path.join(base_dir, "dog_breed_final_model")
final_model.save(export_dir)

print(" Production-ready model saved successfully!")
