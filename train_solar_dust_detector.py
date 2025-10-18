#!/usr/bin/env python3
"""
Automated Solar Panel Dust Detection Training Script
Using MobileNetV2 with Transfer Learning - Optimized for RTX 3050 GPU
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# GPU Configuration - Force GPU if available
def setup_gpu():
    """Setup GPU configuration for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detected: {gpus}")
            print(f"✅ Using GPU acceleration with RTX 3050")
            return True
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
            return False
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
        return False

def create_data_generators(data_dir):
    """Create optimized data generators with augmentation"""
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,  # Use 20% for validation
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
        seed=42
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=42
    )
    
    return train_generator, validation_generator

def create_mobilenet_model():
    """Create MobileNetV2 model optimized for solar panel dust detection"""
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model

def compile_and_train_model(model, train_gen, val_gen):
    """Compile and train the model with callbacks"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_solar_dust_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train the model
    print("🚀 Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_gen, val_gen):
    """Fine-tune the model by unfreezing some layers"""
    
    # Unfreeze the top layers of the base model
    model.layers[0].trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[0].layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("🎯 Fine-tuning model...")
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,  # Fewer epochs for fine-tuning
        callbacks=[
            ModelCheckpoint(
                'models/fine_tuned_solar_dust_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    return history_fine

def plot_training_history(history, history_fine=None):
    """Plot training metrics"""
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if history_fine:
        plt.plot(range(len(history.history['accuracy']), 
                      len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
                history_fine.history['accuracy'], label='Fine-tuning Accuracy')
        plt.plot(range(len(history.history['val_accuracy']), 
                      len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy'])),
                history_fine.history['val_accuracy'], label='Fine-tuning Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if history_fine:
        plt.plot(range(len(history.history['loss']), 
                      len(history.history['loss']) + len(history_fine.history['loss'])),
                history_fine.history['loss'], label='Fine-tuning Loss')
        plt.plot(range(len(history.history['val_loss']), 
                      len(history.history['val_loss']) + len(history_fine.history['val_loss'])),
                history_fine.history['val_loss'], label='Fine-tuning Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['recall'], label='Recall')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Model Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("🌞 SOLAR PANEL DUST DETECTION TRAINING")
    print("=" * 60)
    
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
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Max epochs: {EPOCHS}")
    print(f"⚡ Learning rate: {LEARNING_RATE}")
    
    # Create data generators
    print("\n📊 Creating data generators...")
    train_gen, val_gen = create_data_generators(data_dir)
    
    print(f"✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    print(f"✅ Classes: {train_gen.class_indices}")
    
    # Create model
    print("\n🏗️  Building MobileNetV2 model...")
    model = create_mobilenet_model()
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Train model
    start_time = datetime.now()
    history = compile_and_train_model(model, train_gen, val_gen)
    
    # Fine-tune model
    print("\n🎯 Starting fine-tuning phase...")
    history_fine = fine_tune_model(model, train_gen, val_gen)
    
    # Save final model
    model.save('models/final_solar_dust_model.h5')
    print("✅ Final model saved as 'models/final_solar_dust_model.h5'")
    
    # Plot results
    print("\n📈 Generating training plots...")
    plot_training_history(history, history_fine)
    
    # Training summary
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Training duration: {training_duration}")
    print(f"🎯 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    if history_fine:
        print(f"🎯 Fine-tuned validation accuracy: {max(history_fine.history['val_accuracy']):.4f}")
    print(f"💾 Models saved in 'models/' directory")
    print(f"📊 Training plots saved as 'models/training_history.png'")
    
    # GPU usage summary
    if gpu_available:
        print("✅ Training completed using RTX 3050 GPU acceleration")
    else:
        print("⚠️  Training completed using CPU (consider installing CUDA for GPU acceleration)")

if __name__ == "__main__":
    main()