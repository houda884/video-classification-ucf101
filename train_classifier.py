"""
Train CNN classifier for normal vs tampered video frames
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

def load_dataset(data_dir='data/tamper', img_size=(224, 224)):
    """
    Load normal and tampered images
    Returns: (X, y) where:
        X: array of images
        y: labels (0=normal, 1=tampered)
    """
    print("📂 Loading dataset...")
    
    normal_dir = Path(data_dir) / 'normal'
    tampered_dir = Path(data_dir) / 'tampered'
    
    # Load normal images
    normal_images = []
    normal_files = list(normal_dir.glob('*.jpg'))
    print(f"Found {len(normal_files)} normal images")
    
    for img_path in normal_files:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        normal_images.append(img)
    
    # Load tampered images
    tampered_images = []
    tampered_files = list(tampered_dir.glob('*.jpg'))
    print(f"Found {len(tampered_files)} tampered images")
    
    for img_path in tampered_files:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        tampered_images.append(img)
    
    # Create labels
    X = np.array(normal_images + tampered_images)
    y = np.array([0] * len(normal_images) + [1] * len(tampered_images))
    
    print(f"✅ Dataset loaded: {X.shape[0]} total images")
    print(f"   - Normal: {len(normal_images)}")
    print(f"   - Tampered: {len(tampered_images)}")
    
    return X, y

def create_model(input_shape=(224, 224, 3)):
    """
    Create a CNN model for binary classification
    """
    print("🤖 Creating CNN model...")
    
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    model.summary()
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate model and show detailed metrics"""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Tampered']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX:")
    print(f"True Normal: {cm[0,0]} | False Tampered: {cm[0,1]}")
    print(f"False Normal: {cm[1,0]} | True Tampered: {cm[1,1]}")
    
    # ROC-AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
    
    return y_pred

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """Visualize predictions vs ground truth"""
    print("\n👁️ Visualizing predictions...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            i = indices[idx]
            img = X_test[i]
            true_label = "Tampered" if y_test[i] == 1 else "Normal"
            pred_label = "Tampered" if y_pred[i] == 1 else "Normal"
            confidence = y_pred_proba[i][0] if y_pred[i] == 1 else 1 - y_pred_proba[i][0]
            
            # Show image
            ax.imshow(img.astype('uint8'))
            
            # Color code: green=correct, red=wrong
            color = 'green' if y_test[i] == y_pred[i] else 'red'
            
            ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("VIDEO TAMPERING CLASSIFIER TRAINING")
    print("="*60)
    
    # 1. Load dataset
    X, y = load_dataset()
    
    # 2. Normalize pixel values to [0, 1]
    X = X.astype('float32') / 255.0
    
    # 3. Split dataset
    print("\n📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"   Training set: {X_train.shape[0]} images")
    print(f"   Validation set: {X_val.shape[0]} images")
    print(f"   Test set: {X_test.shape[0]} images")
    
    # 4. Check class distribution
    print(f"\n📈 Class distribution:")
    print(f"   Training - Normal: {np.sum(y_train == 0)}, Tampered: {np.sum(y_train == 1)}")
    print(f"   Validation - Normal: {np.sum(y_val == 0)}, Tampered: {np.sum(y_val == 1)}")
    print(f"   Test - Normal: {np.sum(y_test == 0)}, Tampered: {np.sum(y_test == 1)}")
    
    # 5. Create model
    model = create_model()
    
    # 6. Data augmentation
    print("\n🔄 Setting up data augmentation...")
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # 7. Train model
    print("\n🚀 Training model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 8. Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # 9. Save model
    print("\n💾 Saving model...")
    model.save('tampering_classifier.h5')
    print("✅ Model saved as 'tampering_classifier.h5'")
    
    # 10. Visualize results
    plot_training_history(history)
    visualize_predictions(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Outputs created:")
    print("   - tampering_classifier.h5 (model)")
    print("   - training_history.png (training curves)")
    print("   - predictions_visualization.png (sample predictions)")

if __name__ == "__main__":
    # Check GPU availability
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Set memory growth to avoid OOM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()