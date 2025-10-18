#!/usr/bin/env python3
"""
Enhanced GPU-Accelerated Training Script for Solar Panel Dust Detection
Optimized for RTX 3050 and other CUDA-compatible GPUs
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Optimized for RTX 3050
EPOCHS = 50
LEARNING_RATE = 0.0001

# GPU Configuration
def setup_gpu():
    """Setup GPU configuration for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"✅ GPU detected: {gpus}")
            print(f"✅ Mixed precision enabled for better performance")
            print(f"✅ Memory growth enabled to prevent OOM errors")
            return True
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
            return False
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
        return False

def create_enhanced_data_generators(data_dir):
    """Create optimized data generators with advanced augmentation"""
    
    # Enhanced training data generator with advanced augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,           # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,             # Increased zoom
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2], # Brightness variation
        channel_shift_range=20,      # Color variation
        validation_split=0.2,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        seed=42,
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=42,
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_enhanced_mobilenet_model():
    """Create enhanced MobileNetV2 model with better architecture"""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Enhanced model architecture
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model

def compile_and_train_model(model, train_gen, val_gen):
    """Compile and train the model with enhanced callbacks"""
    
    # Compile model with mixed precision
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_solar_dust_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            verbose=1,
            min_lr=1e-7,
            cooldown=2
        ),
        CSVLogger(
            'models/training_log.csv',
            append=False
        )
    ]
    
    # Train the model
    print("🚀 Starting GPU-accelerated training...")
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=4,  # Parallel data loading
        use_multiprocessing=True
    )
    
    training_time = datetime.now() - start_time
    print(f"⏱️  Training completed in: {training_time}")
    
    return history

def fine_tune_model(model, train_gen, val_gen):
    """Enhanced fine-tuning with layer-wise unfreezing"""
    
    # Unfreeze the top layers of the base model
    model.layers[0].trainable = True
    
    # Fine-tune from this layer onwards (more layers for better results)
    fine_tune_at = 120
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[0].layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    print("🎯 Fine-tuning model with GPU acceleration...")
    start_time = datetime.now()
    
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,  # More epochs for fine-tuning
        callbacks=[
            ModelCheckpoint(
                'models/fine_tuned_solar_dust_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                verbose=1,
                min_lr=1e-8
            )
        ],
        verbose=1,
        workers=4,
        use_multiprocessing=True
    )
    
    fine_tune_time = datetime.now() - start_time
    print(f"⏱️  Fine-tuning completed in: {fine_tune_time}")
    
    return history_fine

def save_training_results(history, history_fine=None):
    """Save comprehensive training results"""
    
    # Save training history as JSON
    results = {
        'training': {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'precision': history.history['precision'],
            'recall': history.history['recall']
        }
    }
    
    if history_fine:
        results['fine_tuning'] = {
            'accuracy': history_fine.history['accuracy'],
            'val_accuracy': history_fine.history['val_accuracy'],
            'loss': history_fine.history['loss'],
            'val_loss': history_fine.history['val_loss'],
            'precision': history_fine.history['precision'],
            'recall': history_fine.history['recall']
        }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create enhanced plots
    plt.figure(figsize=(20, 12))
    
    # Plot accuracy
    plt.subplot(2, 4, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    if history_fine:
        plt.plot(range(len(history.history['accuracy']), 
                      len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
                history_fine.history['accuracy'], label='Fine-tuning Accuracy', linewidth=2)
        plt.plot(range(len(history.history['val_accuracy']), 
                      len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy'])),
                history_fine.history['val_accuracy'], label='Fine-tuning Val Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(2, 4, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    if history_fine:
        plt.plot(range(len(history.history['loss']), 
                      len(history.history['loss']) + len(history_fine.history['loss'])),
                history_fine.history['loss'], label='Fine-tuning Loss', linewidth=2)
        plt.plot(range(len(history.history['val_loss']), 
                      len(history.history['val_loss']) + len(history_fine.history['val_loss'])),
                history_fine.history['val_loss'], label='Fine-tuning Val Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precision
    plt.subplot(2, 4, 3)
    plt.plot(history.history['precision'], label='Precision', linewidth=2)
    plt.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    if history_fine:
        plt.plot(range(len(history.history['precision']), 
                      len(history.history['precision']) + len(history_fine.history['precision'])),
                history_fine.history['precision'], label='Fine-tuning Precision', linewidth=2)
    plt.title('Model Precision', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot recall
    plt.subplot(2, 4, 4)
    plt.plot(history.history['recall'], label='Recall', linewidth=2)
    plt.plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    if history_fine:
        plt.plot(range(len(history.history['recall']), 
                      len(history.history['recall']) + len(history_fine.history['recall'])),
                history_fine.history['recall'], label='Fine-tuning Recall', linewidth=2)
    plt.title('Model Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot AUC if available
    if 'auc' in history.history:
        plt.subplot(2, 4, 5)
        plt.plot(history.history['auc'], label='AUC', linewidth=2)
        plt.plot(history.history['val_auc'], label='Val AUC', linewidth=2)
        plt.title('Model AUC', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/enhanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main GPU-accelerated training pipeline"""
    
    print("=" * 70)
    print("🚀 GPU-ACCELERATED SOLAR PANEL DUST DETECTION TRAINING")
    print("=" * 70)
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Check if dataset exists
    data_dir = "Detect_solar_dust"
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory '{data_dir}' not found!")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print(f"📁 Dataset directory: {data_dir}")
    print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE} (optimized for RTX 3050)")
    print(f"🔄 Max epochs: {EPOCHS}")
    print(f"⚡ Learning rate: {LEARNING_RATE}")
    
    # Create data generators
    print("\n📊 Creating enhanced data generators...")
    train_gen, val_gen = create_enhanced_data_generators(data_dir)
    
    print(f"✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    print(f"✅ Classes: {train_gen.class_indices}")
    
    # Create model
    print("\n🏗️  Building enhanced MobileNetV2 model...")
    model = create_enhanced_mobilenet_model()
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Train model
    start_time = datetime.now()
    history = compile_and_train_model(model, train_gen, val_gen)
    
    # Fine-tune model
    print("\n🎯 Starting enhanced fine-tuning phase...")
    history_fine = fine_tune_model(model, train_gen, val_gen)
    
    # Save final model
    model.save('models/final_solar_dust_model.h5')
    print("✅ Final model saved as 'models/final_solar_dust_model.h5'")
    
    # Save results and plots
    print("\n📈 Generating enhanced training plots...")
    save_training_results(history, history_fine)
    
    # Training summary
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 GPU-ACCELERATED TRAINING COMPLETED!")
    print("=" * 70)
    print(f"⏱️  Total training duration: {training_duration}")
    print(f"🎯 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    if history_fine:
        print(f"🎯 Fine-tuned validation accuracy: {max(history_fine.history['val_accuracy']):.4f}")
    print(f"💾 Models saved in 'models/' directory")
    print(f"📊 Enhanced training plots saved as 'models/enhanced_training_history.png'")
    print(f"📋 Training results saved as 'models/training_results.json'")
    
    # GPU usage summary
    if gpu_available:
        print("✅ Training completed using GPU acceleration with mixed precision")
        print("🚀 Ready for real-time inference!")
    else:
        print("⚠️  Training completed using CPU (consider installing CUDA for GPU acceleration)")

if __name__ == "__main__":
    main()
