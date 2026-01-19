from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import os

# Model
base = MobileNetV2(weights="imagenet", include_top=False)

# Freeze all layers
for layer in base.layers:
    layer.trainable = False

# Unfreeze last layers for fine-tuning
for layer in base.layers[-30:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base.output)
output = Dense(5, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Data
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "data/frames",
    target_size=(224, 224),
    subset="training",
    batch_size=32,
    shuffle=True
)

val = datagen.flow_from_directory(
    "data/frames",
    target_size=(224, 224),
    subset="validation",
    batch_size=32,
    shuffle=False
)

# Class weights (IMPORTANT)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)
print("Class indices:", train.class_indices)

# Save class mapping
os.makedirs("models", exist_ok=True)
with open("models/class_indices.json", "w") as f:
    json.dump(train.class_indices, f)

# Training
model.fit(
    train,
    validation_data=val,
    epochs=10,
    class_weight=class_weights
)

model.save("models/cnn_action.h5")
print("✅ Model saved to models/cnn_action.h5")
