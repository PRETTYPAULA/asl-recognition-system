import os
import importlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths (relative to project root)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAMPLE_IMAGE_PATH = os.path.join(DATASET_DIR, "A", "10.jpg")

print("VGG16 DEMO: Proving VGG16 works on your system!")
print("=" * 60)
print("This script loads pre-trained ImageNet VGG16,")
print("processes a sample ASL image, runs inference,")
print("and shows top-5 predictions + architecture flow.")
print()

# Step 1: Dynamic imports (matching project style)
print("Step 1: Importing Keras/TensorFlow modules...")
keras_applications = importlib.import_module("keras.applications")
vgg16 = importlib.import_module("keras.applications.vgg16")
preprocess_input = vgg16.preprocess_input
decode_predictions = vgg16.decode_predictions
keras_layers = importlib.import_module("keras.layers")
keras_models = importlib.import_module("keras.models")

print("✓ Keras modules imported successfully")

# Step 2: Load VGG16 model (ImageNet weights)
print("\nStep 2: Loading VGG16 model (138M params, ~528MB download if first time)...")
model = keras_applications.VGG16(weights='imagenet')
print(f"✓ VGG16 loaded! Input shape: {model.input.shape}")
print(f"  Output shape: {model.output.shape}")
print("  Status: READY FOR INFERENCE")

# Step 3: Load and preprocess sample image
print("\nStep 3: Loading sample image...")
if not os.path.exists(SAMPLE_IMAGE_PATH):
    raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE_PATH}\nRun this from project root.")

image = Image.open(SAMPLE_IMAGE_PATH)
image = image.resize((224, 224))
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)  # VGG16-specific: RGB->BGR, mean subtract

print(f"✓ Image loaded: {image.size} -> resized to 224x224")
print("  Preprocessed: BGR scale, zero-centered")

# Step 4: Run inference
print("\nStep 4: Running prediction...")
predictions = model.predict(image_array, verbose=0)
decoded = decode_predictions(predictions, top=5)[0]

print("✓ Prediction complete!")
print("\nTop-5 ImageNet predictions:")
print("-" * 50)
for i, (imagenet_id, class_name, score) in enumerate(decoded):
    print(f"{i+1:2d}. {class_name:25s} ({score:.4f})")

# Step 5: VGG16 Architecture Flow (text diagram)
print("\n" + "="*60)
print("VGG16 ARCHITECTURE FLOW (on your system):")
print("="*60)
print("""
Input Image (224x224x3)
    ↓
[Block 1] 2x Conv3x3 (64) → MaxPool → 112x112x64
    ↓
[Block 2] 2x Conv3x3 (128) → MaxPool → 56x56x128
    ↓
[Block 3] 3x Conv3x3 (256) → MaxPool → 28x28x256
    ↓
[Block 4] 3x Conv3x3 (512) → MaxPool → 14x14x512
    ↓
[Block 5] 3x Conv3x3 (512) → MaxPool → 7x7x512
    ↓
Flatten (25,088 → 25088)
    ↓
FC 4096 → ReLU → Dropout
    ↓
FC 4096 → ReLU → Dropout
    ↓
FC 1000 (ImageNet classes) → Softmax

Total params: 138,357,544 | Your system: ✓ LOADED & RUNNING!
(This matches exactly how train_model.py uses it as base_model)
""")

print("\n🎉 SUCCESS! VGG16 is fully working on your system.")
print("Run time: Complete in seconds.")
print("Pro tip: Your train_model.py uses this exact VGG16 as transfer learning base.")

