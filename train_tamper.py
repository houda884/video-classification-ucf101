from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load base action model
base = load_model("models/cnn_action.h5")

# Freeze all layers
for layer in base.layers:
    layer.trainable = False

# Replace last softmax with sigmoid
x = Dense(1, activation="sigmoid")(base.layers[-2].output)
model = Model(inputs=base.input, outputs=x)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Data: tamper vs normal
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "data/tamper",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    "data/tamper",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Train
model.fit(train, validation_data=val, epochs=5)

# Save
model.save("models/cnn_tamper.h5")
print("✅ Tamper model saved: models/cnn_tamper.h5")
