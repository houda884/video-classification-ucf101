"""
3D CNN (I3D) for Video Classification
Inception-based 3D CNN for spatio-temporal feature extraction
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional

class Inception3DModule(layers.Layer):
    """Inception module for 3D CNN"""
    def __init__(self, filters: Tuple[int, int, int, int], **kwargs):
        super().__init__(**kwargs)
        f1, f2_in, f2_out, f3_out = filters
        
        # Branch 1: 1x1x1 convolution
        self.branch1 = layers.Conv3D(f1, (1, 1, 1), padding='same', activation='relu')
        
        # Branch 2: 1x1x1 -> 3x3x3
        self.branch2_conv1 = layers.Conv3D(f2_in, (1, 1, 1), padding='same', activation='relu')
        self.branch2_conv2 = layers.Conv3D(f2_out, (3, 3, 3), padding='same', activation='relu')
        
        # Branch 3: 1x1x1 -> 3x3x3 -> 3x3x3
        self.branch3_conv1 = layers.Conv3D(f2_in, (1, 1, 1), padding='same', activation='relu')
        self.branch3_conv2 = layers.Conv3D(f3_out, (3, 3, 3), padding='same', activation='relu')
        self.branch3_conv3 = layers.Conv3D(f3_out, (3, 3, 3), padding='same', activation='relu')
        
        # Branch 4: 3x3x3 max pooling -> 1x1x1
        self.branch4_pool = layers.MaxPool3D((3, 3, 3), strides=(1, 1, 1), padding='same')
        self.branch4_conv = layers.Conv3D(f1, (1, 1, 1), padding='same', activation='relu')
        
    def call(self, inputs):
        b1 = self.branch1(inputs)
        
        b2 = self.branch2_conv1(inputs)
        b2 = self.branch2_conv2(b2)
        
        b3 = self.branch3_conv1(inputs)
        b3 = self.branch3_conv2(b3)
        b3 = self.branch3_conv3(b3)
        
        b4 = self.branch4_pool(inputs)
        b4 = self.branch4_conv(b4)
        
        return layers.concatenate([b1, b2, b3, b4])

class I3DModel:
    """Inception 3D Model for video classification"""
    def __init__(self, input_shape: Tuple[int, int, int, int], num_classes: int):
        self.input_shape = input_shape  # (frames, height, width, channels)
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """Build I3D model architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Stem layers
        x = layers.Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same', activation='relu')(inputs)
        x = layers.MaxPool3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        
        x = layers.Conv3D(64, (1, 1, 1), padding='same', activation='relu')(x)
        x = layers.Conv3D(192, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPool3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        
        # Inception modules
        # Inception 3a
        x = Inception3DModule(filters=(64, 96, 128, 16))(x)
        
        # Inception 3b
        x = Inception3DModule(filters=(128, 128, 192, 32))(x)
        x = layers.MaxPool3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        
        # Inception 4a
        x = Inception3DModule(filters=(192, 96, 208, 16))(x)
        
        # Inception 4b
        x = Inception3DModule(filters=(160, 112, 224, 24))(x)
        
        # Inception 4c
        x = Inception3DModule(filters=(128, 128, 256, 24))(x)
        
        # Inception 4d
        x = Inception3DModule(filters=(112, 144, 288, 32))(x)
        
        # Inception 4e
        x = Inception3DModule(filters=(256, 160, 320, 32))(x)
        x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
        
        # Inception 5a
        x = Inception3DModule(filters=(256, 160, 320, 32))(x)
        
        # Inception 5b
        x = Inception3DModule(filters=(384, 192, 384, 48))(x)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile model with optimizer and loss"""
        if self.num_classes == 1:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'auc']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'top_k_categorical_accuracy']
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary"""
        self.model.summary()

class VideoDataGenerator:
    """Generate 3D video clips for training"""
    def __init__(self, clip_length: int = 16, frame_size: Tuple[int, int] = (224, 224)):
        self.clip_length = clip_length
        self.frame_size = frame_size
    
    def load_video_clips(self, video_paths: list, labels: list, batch_size: int = 8):
        """Load video clips in batches"""
        num_videos = len(video_paths)
        
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
                        print(f"Error loading {video_path}: {e}")
                        continue
                
                if clips:
                    yield np.array(clips), np.array(batch_labels_processed)
    
    def extract_random_clip(self, video_path: str) -> Optional[np.ndarray]:
        """Extract random clip from video"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
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

def train_i3d_model():
    """Train I3D model for video classification"""
    import mlflow
    from sklearn.model_selection import train_test_split
    
    # Setup MLflow
    mlflow.set_experiment("i3d_video_classification")
    
    # Parameters
    CLIP_LENGTH = 16
    FRAME_SIZE = (224, 224)
    NUM_CLASSES = 6  # For action recognition
    BATCH_SIZE = 8
    EPOCHS = 50
    
    # Create model