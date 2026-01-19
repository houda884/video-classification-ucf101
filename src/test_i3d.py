"""
Test the 3D CNN (I3D) implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.i3d_3d_cnn import I3DModel, VideoDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path

def test_model_creation():
    """Test creating and compiling the I3D model"""
    print("🧪 Testing I3D Model Creation...")
    
    # Define input shape: (frames, height, width, channels)
    input_shape = (16, 224, 224, 3)  # 16 frames, 224x224 RGB
    
    # Test binary classification (tamper detection)
    print("\n1. Testing binary classification model...")
    binary_model = I3DModel(input_shape=input_shape, num_classes=1)
    binary_model.compile(learning_rate=0.001)
    binary_model.summary()
    
    # Test multi-class classification (action recognition)
    print("\n2. Testing multi-class classification model...")
    multi_model = I3DModel(input_shape=input_shape, num_classes=6)
    multi_model.compile(learning_rate=0.001)
    multi_model.summary()
    
    # Generate dummy data for testing
    print("\n3. Testing with dummy data...")
    dummy_input = np.random.randn(2, *input_shape).astype(np.float32)
    
    # Test binary model prediction
    binary_pred = binary_model.model.predict(dummy_input)
    print(f"   Binary model predictions shape: {binary_pred.shape}")
    print(f"   Sample prediction: {binary_pred[0][0]:.4f}")
    
    # Test multi-class model prediction
    multi_pred = multi_model.model.predict(dummy_input)
    print(f"   Multi-class predictions shape: {multi_pred.shape}")
    print(f"   Sample prediction sum (should be ~1.0): {np.sum(multi_pred[0]):.4f}")
    
    return binary_model, multi_model

def test_data_generator():
    """Test the video data generator"""
    print("\n🎬 Testing Video Data Generator...")
    
    # Create dummy video files list (replace with real paths)
    dummy_videos = ["dummy_video1.mp4", "dummy_video2.mp4", "dummy_video3.mp4"]
    dummy_labels = [0, 1, 0]  # Binary labels
    
    # Create generator
    generator = VideoDataGenerator(clip_length=16, frame_size=(224, 224))
    
    print(f"   Clip length: {generator.clip_length}")
    print(f"   Frame size: {generator.frame_size}")
    
    # Test extract_random_clip with real video if available
    test_video = "data/raw_videos/Basketball/v_Basketball_g01_c01.avi"
    if os.path.exists(test_video):
        print(f"\n   Testing with real video: {test_video}")
        clip = generator.extract_random_clip(test_video)
        if clip is not None:
            print(f"   Successfully extracted clip!")
            print(f"   Clip shape: {clip.shape}")
            print(f"   Clip dtype: {clip.dtype}")
            print(f"   Clip range: [{clip.min():.2f}, {clip.max():.2f}]")
        else:
            print(f"   Could not extract clip (video might be too short)")
    else:
        print(f"\n   Test video not found, skipping real video test")
        print(f"   Please add videos to: data/raw_videos/")
    
    return generator

def visualize_model_architecture():
    """Visualize the model architecture"""
    print("\n📊 Visualizing Model Architecture...")
    
    # Create a smaller model for visualization
    input_shape = (8, 112, 112, 3)  # Smaller for faster visualization
    model = I3DModel(input_shape=input_shape, num_classes=6)
    model.compile()
    
    # Plot model
    try:
        keras.utils.plot_model(
            model.model,
            to_file="i3d_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96
        )
        print("   Model architecture saved to: i3d_architecture.png")
    except ImportError:
        print("   Graphviz not installed. Install with: pip install graphviz")
        print("   Or view model summary above.")
    
    # Print layer information
    print("\n   Model Layer Information:")
    print("   " + "="*50)
    total_params = 0
    for i, layer in enumerate(model.model.layers):
        params = layer.count_params()
        total_params += params
        if params > 0:  # Only show layers with parameters
            print(f"   Layer {i:2d}: {layer.name:30} | Params: {params:,}")
    
    print("   " + "="*50)
    print(f"   Total trainable parameters: {total_params:,}")
    
    return model

def create_training_pipeline():
    """Create a complete training pipeline example"""
    print("\n🚀 Creating Training Pipeline Example...")
    
    # Example parameters
    config = {
        "clip_length": 16,
        "frame_size": (224, 224),
        "batch_size": 4,
        "learning_rate": 0.0001,
        "epochs": 50,
        "num_classes": 6
    }
    
    print("   Training Configuration:")
    for key, value in config.items():
        print(f"     {key}: {value}")
    
    # Create model
    input_shape = (config["clip_length"], *config["frame_size"], 3)
    model = I3DModel(input_shape=input_shape, num_classes=config["num_classes"])
    model.compile(learning_rate=config["learning_rate"])
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_i3d_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    print("\n   Callbacks configured:")
    print("     - EarlyStopping (patience=10)")
    print("     - ReduceLROnPlateau")
    print("     - ModelCheckpoint")
    
    return model, config, callbacks

def main():
    print("="*60)
    print("3D CNN (I3D) MODEL TEST SUITE")
    print("="*60)
    
    # Test 1: Model creation
    binary_model, multi_model = test_model_creation()
    
    # Test 2: Data generator
    generator = test_data_generator()
    
    # Test 3: Visualization
    viz_model = visualize_model_architecture()
    
    # Test 4: Training pipeline
    train_model, config, callbacks = create_training_pipeline()
    
    # Recommendations
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS FOR USE:")
    print("="*60)
    print("\n1. For Tamper Detection (binary):")
    print("   - Use I3DModel(input_shape=(16, 224, 224, 3), num_classes=1)")
    print("   - Output: Sigmoid activation (0=normal, 1=tampered)")
    
    print("\n2. For Action Recognition (multi-class):")
    print("   - Use I3DModel(input_shape=(16, 224, 224, 3), num_classes=6)")
    print("   - Output: Softmax activation (6 action classes)")
    
    print("\n3. Data Preparation:")
    print("   - Videos should be in data/raw_videos/")
    print("   - Use VideoDataGenerator to create clips")
    print("   - Recommended clip length: 16-32 frames")
    
    print("\n4. Training Tips:")
    print("   - Start with small learning rate (1e-4)")
    print("   - Use small batch size (2-4) due to memory")
    print("   - Consider transfer learning from pre-trained I3D")
    
    print("\n" + "="*60)
    print("✅ 3D CNN TEST COMPLETE!")
    print("="*60)
    
    # Save test models
    print("\n💾 Saving test models...")
    binary_model.model.save('test_binary_i3d.keras')
    multi_model.model.save('test_multi_i3d.keras')
    print("   Models saved as:")
    print("   - test_binary_i3d.keras")
    print("   - test_multi_i3d.keras")

if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()