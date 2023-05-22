import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Preprocess the image
resized_image = cv2.resize(image, (224, 224))
normalized_image = resized_image / 255.0
expanded_image = np.expand_dims(normalized_image, axis=0)

# Make predictions
predictions = model.predict(expanded_image)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

# Print the top predictions
for (index, (class_id, class_name, probability)) in enumerate(decoded_predictions):
    print(f'{index + 1}. {class_name}: {probability * 100:.2f}%')
