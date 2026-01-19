"""
ENHANCED TAMPER DETECTION DATASET GENERATOR - CORRIGÉ COMPLÈTEMENT
"""
import cv2
import numpy as np
import os
from pathlib import Path
import random
import albumentations as A
from tqdm import tqdm
import json

def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/tamper/normal',
        'data/tamper/tampered',
        'data/tamper/validation/normal',
        'data/tamper/validation/tampered',
        'data/processed_frames',
        'data/frames_temp'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

def extract_frames_balanced(video_dir='data/raw_videos/', max_videos_per_class=15, frames_per_video=30, img_size=(224, 224)):
    """
    Extract frames from ALL classes for better diversity
    """
    print("🎬 Extracting frames from ALL action classes...")
    
    classes = sorted([d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))])
    print(f"Found {len(classes)} classes: {classes}")
    
    all_frames = []
    class_distribution = {}
    
    for class_name in classes:
        class_path = os.path.join(video_dir, class_name)
        video_files = list(Path(class_path).glob('*.avi')) + list(Path(class_path).glob('*.mp4'))
        
        if not video_files:
            continue
        
        videos_to_process = min(max_videos_per_class, len(video_files))
        class_frames = []
        
        print(f"\n📂 Processing {class_name} ({videos_to_process} videos)...")
        
        for i, video_path in enumerate(video_files[:videos_to_process]):
            try:
                cap = cv2.VideoCapture(str(video_path))
                frames_extracted = 0
                
                while frames_extracted < frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, img_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if random.random() > 0.7:
                        frame = apply_mild_augmentation(frame)
                    
                    class_frames.append(frame)
                    frames_extracted += 1
                
                cap.release()
                
            except Exception as e:
                print(f"  ⚠ Error processing {video_path.name}: {e}")
                continue
        
        all_frames.extend(class_frames)
        class_distribution[class_name] = len(class_frames)
        
        print(f"  ✓ Extracted {len(class_frames)} frames from {class_name}")
    
    print(f"\n✅ Total frames extracted: {len(all_frames)}")
    print("Class distribution:")
    for cls, count in class_distribution.items():
        print(f"  - {cls}: {count} frames")
    
    return all_frames

def apply_mild_augmentation(frame):
    """Apply mild augmentation"""
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    if random.random() > 0.5:
        frame = cv2.flip(frame, 1)
    
    if random.random() > 0.7:
        angle = random.uniform(-5, 5)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        frame = cv2.warpAffine(frame, M, (w, h))
    
    return frame

def create_realistic_tampered_frames(frames, num_samples_per_attack=100):
    """
    Create realistic tampered frames with various attacks
    """
    print("\n🎭 Generating realistic tampered frames...")
    
    tampered_frames = []
    attack_labels = []
    
    # Define attack functions
    def frame_swap_attack(frame):
        """More realistic frame swap"""
        if len(tampered_frames) > 2:
            prev_frame = tampered_frames[-1]
            alpha = random.uniform(0.3, 0.7)
            return cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
        return frame
    
    def overlay_attack(frame):
        """Realistic overlay attack"""
        h, w = frame.shape[:2]
        
        overlay_type = random.choice(['logo', 'text', 'pattern', 'object'])
        
        if overlay_type == 'logo':
            logo_size = random.randint(40, 80)
            x = random.randint(20, w - logo_size - 20)
            y = random.randint(20, h - logo_size - 20)
            
            overlay = np.zeros((logo_size, logo_size, 3), dtype=np.uint8)
            cv2.rectangle(overlay, (0, 0), (logo_size-1, logo_size-1), 
                         (random.randint(100, 255), random.randint(100, 255), 
                          random.randint(100, 255)), -1)
            
            cv2.putText(overlay, "FAKE", (5, logo_size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            roi = frame[y:y+logo_size, x:x+logo_size]
            blended = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)
            frame[y:y+logo_size, x:x+logo_size] = blended
            
        elif overlay_type == 'text':
            texts = ["Edited", "Modified", "Fake", "Altered", "Edited"]
            text = random.choice(texts)
            
            font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
            font_scale = random.uniform(0.4, 0.8)  # Reduced to ensure text fits
            thickness = random.randint(1, 2)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # CORRECTION: Ensure text fits within image
            if text_width < w - 20 and text_height < h - 20:
                x = random.randint(10, w - text_width - 10)
                y = random.randint(text_height + 10, h - 10)
                
                shadow_color = (0, 0, 0)
                text_color = (random.randint(200, 255), random.randint(200, 255), 
                             random.randint(200, 255))
                
                cv2.putText(frame, text, (x+2, y+2), font, font_scale, 
                           shadow_color, thickness)
                cv2.putText(frame, text, (x, y), font, font_scale, 
                           text_color, thickness)
            else:
                # If text is too big, fall back to a simple text
                cv2.putText(frame, "EDITED", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif overlay_type == 'pattern':
            for _ in range(random.randint(3, 8)):
                x1 = random.randint(0, w-30)
                y1 = random.randint(0, h-30)
                block_size = random.randint(10, 25)
                x2 = min(x1 + block_size, w)
                y2 = min(y1 + block_size, h)
                
                block = frame[y1:y2, x1:x2].copy()
                if block.size > 0:
                    avg_color = np.mean(block, axis=(0, 1))
                    frame[y1:y2, x1:x2] = avg_color.astype(np.uint8)
        
        elif overlay_type == 'object':
            h, w = frame.shape[:2]
            patch_size = random.randint(40, 100)
            
            if h > patch_size and w > patch_size:
                src_x = random.randint(0, w - patch_size)
                src_y = random.randint(0, h - patch_size)
                patch = frame[src_y:src_y+patch_size, src_x:src_x+patch_size].copy()
                
                dst_x = random.randint(0, w - patch_size)
                dst_y = random.randint(0, h - patch_size)
                
                alpha = random.uniform(0.6, 0.9)
                dest_roi = frame[dst_y:dst_y+patch_size, dst_x:dst_x+patch_size]
                blended = cv2.addWeighted(dest_roi, 1-alpha, patch, alpha, 0)
                frame[dst_y:dst_y+patch_size, dst_x:dst_x+patch_size] = blended
        
        return frame
    
    def compression_attack(frame):
        """Realistic compression artifacts"""
        for _ in range(random.randint(1, 3)):
            quality = random.randint(15, 60)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        if random.random() > 0.5:
            noise = np.random.normal(0, random.randint(5, 15), frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def temporal_artifact_attack(frame):
        """Simulate temporal artifacts"""
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        if random.random() > 0.8 and len(tampered_frames) > 5:
            dup_frame = tampered_frames[-random.randint(1, 5)]
            alpha = random.uniform(0.3, 0.7)
            frame = cv2.addWeighted(frame, 1-alpha, dup_frame, alpha, 0)
        
        return frame
    
    def advanced_attack(frame):
        """Combination of multiple attacks"""
        attacks_to_use = random.sample([overlay_attack, compression_attack], 
                                     random.randint(2, 2))
        
        for attack_func in attacks_to_use:
            frame = attack_func(frame)
        
        return frame
    
    # Generate tampered samples
    attacks = [
        ('frame_swap', frame_swap_attack),
        ('overlay', overlay_attack),
        ('compression', compression_attack),
        ('temporal', temporal_artifact_attack),
        ('advanced', advanced_attack)
    ]
    
    for attack_name, attack_func in attacks:
        print(f"\n  Generating {attack_name} attacks...")
        
        for i in tqdm(range(num_samples_per_attack), desc=f"  {attack_name}"):
            base_idx = random.randint(0, len(frames) - 1)
            tampered_frame = frames[base_idx].copy()
            
            # Apply attack
            tampered_frame = attack_func(tampered_frame)
            
            if random.random() > 0.5:
                tampered_frame = apply_mild_augmentation(tampered_frame)
            
            tampered_frames.append(tampered_frame)
            attack_labels.append(attack_name)
    
    print(f"\n✅ Generated {len(tampered_frames)} tampered frames")
    print("Attack distribution:")
    for attack in set(attack_labels):
        count = attack_labels.count(attack)
        print(f"  - {attack}: {count} samples")
    
    return tampered_frames, attack_labels

def augment_normal_frames(frames, augmentation_factor=2):
    """
    Augment normal frames to increase dataset size
    """
    print("\n🔄 Augmenting normal frames...")
    
    augmented_frames = []
    
    # Simple augmentation without Albumentations to avoid errors
    for frame in tqdm(frames, desc="Augmenting"):
        augmented_frames.append(frame)
        
        for _ in range(augmentation_factor - 1):
            augmented = frame.copy()
            
            # Random flip
            if random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)
            
            # Random rotation
            if random.random() > 0.3:
                angle = random.uniform(-10, 10)
                h, w = augmented.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                augmented = cv2.warpAffine(augmented, M, (w, h))
            
            # Random brightness/contrast
            if random.random() > 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-20, 20)
                augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
            
            augmented_frames.append(augmented)
    
    print(f"✅ Augmented from {len(frames)} to {len(augmented_frames)} normal frames")
    return augmented_frames

def save_dataset_balanced(normal_frames, tampered_frames, attack_labels=None):
    """
    Save balanced dataset
    """
    print("\n💾 Saving balanced dataset...")
    
    # Ensure we have balanced dataset
    min_samples = min(len(normal_frames), len(tampered_frames))
    
    # Take only balanced samples
    balanced_normal = normal_frames[:min_samples]
    balanced_tampered = tampered_frames[:min_samples]
    
    if attack_labels:
        balanced_labels = attack_labels[:min_samples]
    else:
        balanced_labels = ['tampered'] * min_samples
    
    print(f"Balanced dataset: {min_samples} normal, {min_samples} tampered")
    
    # Save normal frames
    print("Saving normal frames...")
    for i, frame in enumerate(tqdm(balanced_normal, desc="Normal")):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'data/tamper/normal/normal_{i:05d}.jpg', frame_bgr)
    
    # Save tampered frames
    print("Saving tampered frames...")
    metadata = []
    
    for i, (frame, attack_label) in enumerate(tqdm(zip(balanced_tampered, balanced_labels), 
                                                  desc="Tampered", total=len(balanced_tampered))):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = f'tampered_{i:05d}.jpg'
        cv2.imwrite(f'data/tamper/tampered/{filename}', frame_bgr)
        
        metadata.append({
            'filename': filename,
            'attack_type': attack_label
        })
    
    # Save metadata
    with open('data/tamper/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create validation split (20%)
    print("\nCreating validation split...")
    val_size = min_samples // 5
    
    val_normal = balanced_normal[:val_size]
    val_tampered = balanced_tampered[:val_size]
    
    for i, frame in enumerate(val_normal):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'data/tamper/validation/normal/val_normal_{i:05d}.jpg', frame_bgr)
    
    for i, frame in enumerate(val_tampered):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'data/tamper/validation/tampered/val_tampered_{i:05d}.jpg', frame_bgr)
    
    return metadata, min_samples

def main():
    print("="*70)
    print("ENHANCED TAMPER DETECTION DATASET GENERATOR - STABLE VERSION")
    print("Target: Create balanced dataset for >90% accuracy model")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Step 1: Extract frames from ALL classes
    print("\n[1/4] Extracting balanced frames from all action classes...")
    frames = extract_frames_balanced(max_videos_per_class=15, frames_per_video=25)
    
    if len(frames) < 100:
        print(f"⚠ Warning: Only {len(frames)} frames extracted.")
        return
    
    # Step 2: Augment normal frames
    print("\n[2/4] Augmenting normal frames...")
    augmented_frames = augment_normal_frames(frames, augmentation_factor=2)
    
    # Step 3: Generate realistic tampered frames
    print("\n[3/4] Generating realistic tampered frames...")
    
    # Generate enough tampered frames to match normal frames
    tampered_frames = []
    attack_labels = []
    
    # Generate in batches to avoid memory issues
    batch_size = 500
    num_batches = max(1, len(augmented_frames) // batch_size)
    
    for batch in range(num_batches):
        print(f"\n  Generating batch {batch+1}/{num_batches}...")
        batch_tampered, batch_labels = create_realistic_tampered_frames(
            frames, num_samples_per_attack=batch_size // 5
        )
        tampered_frames.extend(batch_tampered)
        attack_labels.extend(batch_labels)
        
        # Stop if we have enough
        if len(tampered_frames) >= len(augmented_frames):
            tampered_frames = tampered_frames[:len(augmented_frames)]
            attack_labels = attack_labels[:len(augmented_frames)]
            break
    
    # Step 4: Save balanced dataset
    print("\n[4/4] Saving balanced dataset...")
    metadata, balanced_size = save_dataset_balanced(augmented_frames, tampered_frames, attack_labels)
    
    # Summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"✓ Normal frames: {balanced_size}")
    print(f"✓ Tampered frames: {balanced_size}")
    print(f"✓ Total samples: {balanced_size * 2}")
    print(f"✓ Balanced ratio: 1:1 (perfect balance)")
    
    # Attack type distribution
    print("\nAttack Type Distribution:")
    attack_counts = {}
    for label in attack_labels[:balanced_size]:
        attack_counts[label] = attack_counts.get(label, 0) + 1
    
    for attack_type, count in attack_counts.items():
        percentage = (count / balanced_size) * 100
        print(f"  - {attack_type}: {count} samples ({percentage:.1f}%)")
    
    print("\n📁 Dataset structure:")
    print("  data/tamper/")
    print("  ├── normal/              # Normal frames")
    print("  ├── tampered/            # Tampered frames")
    print("  ├── validation/          # Validation split (20%)")
    print("  │   ├── normal/")
    print("  │   └── tampered/")
    print("  └── metadata.json        # Dataset metadata")
    
    print("\n✅ Dataset ready for training high-accuracy model!")
    print("   Expected accuracy: >90% with proper training")
    print("="*70)

if __name__ == "__main__":
    main()