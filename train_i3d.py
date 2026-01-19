"""
Train I3D model on your video dataset - CORRECTED VERSION
"""
import sys
import os
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Ajoutez le chemin pour importer votre module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importez votre modèle I3D
try:
    from src.i3d_3d_cnn import I3DModel
except ImportError:
    print("⚠️  Could not import I3DModel from src.i3d_3d_cnn")
    print("   Creating simplified I3D model instead...")
    
    # Fallback si l'import échoue
    class I3DModel:
        def __init__(self, input_shape, num_classes):
            from tensorflow.keras import layers, Model
            
            inputs = keras.Input(shape=input_shape)
            
            # Simplified I3D-like architecture
            x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
            x = layers.MaxPool3D((2, 2, 2))(x)
            
            x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPool3D((2, 2, 2))(x)
            
            x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPool3D((2, 2, 2))(x)
            
            x = layers.GlobalAveragePooling3D()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            
            if num_classes == 1:
                outputs = layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = layers.Dense(num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            
        def compile(self, learning_rate=0.001):
            if self.model.output.shape[-1] == 1:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=metrics
            )
        
        def summary(self):
            self.model.summary()

class FixedVideoDataGenerator:
    """Fixed version of VideoDataGenerator with label encoding"""
    def __init__(self, clip_length: int = 16, frame_size: Tuple[int, int] = (224, 224)):
        self.clip_length = clip_length
        self.frame_size = frame_size
    
    def load_video_clips(self, video_paths: list, labels: list, batch_size: int = 8, one_hot: bool = True):
        """Load video clips in batches with proper label encoding"""
        num_videos = len(video_paths)
        num_classes = len(set(labels))
        
        while True:
            indices = np.random.permutation(num_videos)
            
            for start_idx in range(0, num_videos, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_paths = [video_paths[i] for i in batch_indices]
                batch_labels = [labels[i] for i in batch_indices]
                
                clips = []
                batch_labels_processed = []
                
                for video_path, label in zip(batch_paths, batch_labels):
                    try:
                        clip = self.extract_random_clip(video_path)
                        if clip is not None:
                            clips.append(clip)
                            batch_labels_processed.append(label)
                    except Exception as e:
                        print(f"  Error loading {video_path}: {e}")
                        continue
                
                if clips:
                    X_batch = np.array(clips)
                    
                    # Convert labels to one-hot if needed
                    if one_hot:
                        y_batch = keras.utils.to_categorical(batch_labels_processed, 
                                                           num_classes=num_classes)
                    else:
                        y_batch = np.array(batch_labels_processed)
                    
                    yield X_batch, y_batch
    
    def extract_random_clip(self, video_path):
        """Extract random clip from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.clip_length:
            cap.release()
            return None
        
        # Random starting point
        start_frame = np.random.randint(0, total_frames - self.clip_length)
        
        clip = []
        for i in range(self.clip_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize and normalize
            frame = cv2.resize(frame, self.frame_size)
            frame = frame / 255.0
            clip.append(frame)
        
        cap.release()
        
        if len(clip) == self.clip_length:
            return np.array(clip)
        return None

def load_video_paths(data_dir="data/raw_videos"):
    """Load video paths and labels with proper encoding"""
    video_paths = []
    labels = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return [], [], 0
    
    classes = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"❌ No class directories found in {data_dir}")
        return [], [], 0
    
    print(f"Found {len(classes)} classes: {[c.name for c in classes]}")
    
    for label, class_dir in enumerate(classes):
        videos = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
        
        if not videos:
            print(f"⚠️  No videos found in {class_dir.name}")
            continue
            
        video_paths.extend(videos)
        labels.extend([label] * len(videos))
        print(f"  {class_dir.name}: {len(videos)} videos")
    
    return video_paths, labels, len(classes)

def main():
    print("="*60)
    print("STARTING I3D MODEL TRAINING - CORRECTED")
    print("="*60)
    
    # Setup MLflow
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    except:
        mlflow.set_tracking_uri("file:./mlruns")
    
    mlflow.set_experiment("i3d_training")
    
    # 1. Load data
    print("\n📂 Loading video data...")
    video_paths, labels, num_classes = load_video_paths()
    
    if not video_paths:
        print("❌ No videos found. Exiting.")
        return
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total videos: {len(video_paths)}")
    print(f"   Number of classes: {num_classes}")
    
    # 2. Split data
    print("\n📊 Splitting dataset...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   Training videos: {len(train_paths)}")
    print(f"   Validation videos: {len(val_paths)}")
    
    # 3. Create generators
    print("\n🎬 Creating data generators...")
    clip_length = 16
    frame_size = (224, 224)
    
    train_gen = FixedVideoDataGenerator(clip_length=clip_length, frame_size=frame_size)
    val_gen = FixedVideoDataGenerator(clip_length=clip_length, frame_size=frame_size)
    
    # 4. Create model - FIXED: Use multi-class classification
    print("\n🤖 Creating I3D model...")
    input_shape = (clip_length, *frame_size, 3)
    
    model = I3DModel(input_shape=input_shape, num_classes=num_classes)
    
    # FIXED: Use categorical crossentropy for multi-class
    if num_classes == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', 'top_k_categorical_accuracy']
    
    model.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=metrics
    )
    
    model.model.summary()
    
    # 5. Start MLflow run
    print("\n📈 Starting MLflow tracking...")
    
    with mlflow.start_run(run_name="i3d_action_recognition"):
        # Log parameters
        mlflow.log_params({
            "model": "I3D",
            "clip_length": clip_length,
            "frame_size": f"{frame_size[0]}x{frame_size[1]}",
            "num_classes": num_classes,
            "learning_rate": 0.0001,
            "batch_size": 4,
            "total_videos": len(video_paths),
            "loss_function": loss
        })
        
        # 6. Training
        print("\n🚀 Starting training...")
        print("   This may take a while...")
        
        try:
            # Calculate steps (small for testing)
            train_steps = min(10, len(train_paths) // 4)
            val_steps = min(5, len(val_paths) // 4)
            
            print(f"   Train steps per epoch: {train_steps}")
            print(f"   Validation steps: {val_steps}")
            
            history = model.model.fit(
                train_gen.load_video_clips(train_paths, train_labels, batch_size=4, one_hot=True),
                steps_per_epoch=train_steps,
                validation_data=val_gen.load_video_clips(val_paths, val_labels, batch_size=4, one_hot=True),
                validation_steps=val_steps,
                epochs=5,  # Small for testing
                verbose=1
            )
            
            # Log metrics
            print("\n📊 Logging metrics to MLflow...")
            if 'accuracy' in history.history:
                for epoch in range(len(history.history['accuracy'])):
                    mlflow.log_metric("train_accuracy", 
                                     history.history['accuracy'][epoch], 
                                     step=epoch)
                    
                if 'val_accuracy' in history.history:
                    for epoch in range(len(history.history['val_accuracy'])):
                        mlflow.log_metric("val_accuracy", 
                                         history.history['val_accuracy'][epoch], 
                                         step=epoch)
                
                # Log final accuracy
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                mlflow.log_metric("final_val_accuracy", final_val_acc)
                
                print(f"   Final training accuracy: {final_train_acc:.4f}")
                print(f"   Final validation accuracy: {final_val_acc:.4f}")
            
            # 7. Save model
            print("\n💾 Saving model...")
            model_path = 'trained_i3d_model.keras'
            model.model.save(model_path)
            mlflow.keras.log_model(model.model, "i3d_model")
            mlflow.log_artifact(model_path)
            
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            
            print("\n📁 Outputs:")
            print(f"   - Model saved: {model_path}")
            print(f"   - MLflow UI: Run 'mlflow ui' to view results")
            print(f"   - Training history available in MLflow")
            
            # Display training summary
            if history:
                print("\n📈 Training Summary:")
                print(f"   Epochs completed: {len(history.history.get('accuracy', []))}")
                if 'accuracy' in history.history:
                    print(f"   Best train accuracy: {max(history.history['accuracy']):.4f}")
                if 'val_accuracy' in history.history:
                    print(f"   Best val accuracy: {max(history.history['val_accuracy']):.4f}")
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print("\n💡 TROUBLESHOOTING DETAILS:")
            print(f"   Error type: {type(e).__name__}")
            
            # Try a simpler approach
            print("\n🔧 Trying simpler approach...")
            try:
                # Create a very simple test
                print("   Creating simple test data...")
                dummy_X = np.random.randn(8, clip_length, *frame_size, 3).astype(np.float32)
                dummy_y = keras.utils.to_categorical(np.random.randint(0, num_classes, 8), 
                                                    num_classes=num_classes)
                
                print("   Training on dummy data...")
                simple_history = model.model.fit(
                    dummy_X, dummy_y,
                    validation_split=0.2,
                    epochs=2,
                    verbose=1
                )
                
                print("   ✅ Simple training successful!")
                
            except Exception as e2:
                print(f"   ❌ Simple test also failed: {e2}")

if __name__ == "__main__":
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth error: {e}")
    
    main()