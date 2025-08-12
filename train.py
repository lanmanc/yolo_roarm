#!/usr/bin/env python3
"""
YOLOv8 Training Script - Optimized for RTX 4070
"""

import torch
from ultralytics import YOLO
import os


def check_gpu():
    """Check if GPU is available and print info"""
    print("=" * 50)
    print("🖥️ GPU Information:")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"📦 CUDA Version: {torch.version.cuda}")
        print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"🔧 PyTorch Version: {torch.__version__}")
        return True
    else:
        print("❌ No GPU detected! Training will be very slow.")
        return False


def create_data_yaml():
    """Create data.yaml file"""
    content = """
train: ./train/images
val: ./valid/images  # or ./test/images if no valid folder
test: ./test/images

nc: 1
names: ['Zhuang Chris']
"""
    with open('data.yaml', 'w') as f:
        f.write(content.strip())
    print("✅ Created data.yaml")


def train_with_rtx4070():
    """Training optimized for RTX 4070 (12GB VRAM)"""

    # Check GPU first
    if not check_gpu():
        print("⚠️ Warning: No GPU detected, continue? (y/n)")
        if input().lower() != 'y':
            return

    # Create data.yaml if needed
    if not os.path.exists('data.yaml'):
        create_data_yaml()

    print("\n" + "=" * 50)
    print("🚀 Starting YOLOv8 Training on RTX 4070")
    print("=" * 50)

    # Initialize model
    model = YOLO('yolov8s.pt')  # Can use yolov8m.pt with RTX 4070

    # Train with RTX 4070 optimized settings
    results = model.train(
        # Dataset
        data='data.yaml',

        # Training parameters - Optimized for RTX 4070
        epochs=200,  # More epochs since training is fast
        imgsz=640,  # Can use 640 or even 1280 with 12GB VRAM
        batch=32,  # RTX 4070 can handle batch 32 easily

        # Performance settings for RTX 4070
        device=0,  # Use first GPU (RTX 4070)
        workers=8,  # Adjust based on CPU cores
        amp=True,  # Mixed precision training (faster on RTX 4070)

        # Model saving
        project='runs/detect',
        name='zhuang_chris_rtx4070',
        exist_ok=True,
        save=True,
        save_period=20,

        # Advanced optimization
        optimizer='AdamW',  # Better than SGD for most cases
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation (all enabled for better performance)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # Loss settings
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Other
        close_mosaic=10,
        resume=False,
        patience=50,
        val=True,
        plots=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # Since you only have one class
        rect=False,
        cos_lr=False,
        label_smoothing=0.0,
        dropout=0.0,
        overlap_mask=True,
        mask_ratio=4,
        nbs=64
    )

    print("\n" + "=" * 50)
    print("✅ Training Completed Successfully!")
    print("=" * 50)
    print(f"📊 Results saved to: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    print(f"📈 Metrics: {results.save_dir}/results.csv")

    # Print final metrics
    print("\n📊 Final Metrics:")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


def test_model(model_path='runs/detect/zhuang_chris_rtx4070/weights/best.pt'):
    """Test the trained model"""

    print("\n" + "=" * 50)
    print("🧪 Testing Model")
    print("=" * 50)

    model = YOLO(model_path)

    # Run validation
    metrics = model.val(data='data.yaml')

    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # Run prediction on test images
    results = model.predict(
        source='test/images',
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        iou=0.45,
        max_det=300,
        device=0,  # Use GPU
        project='runs/detect',
        name='predictions_rtx4070'
    )

    print(f"✅ Predictions saved to runs/detect/predictions_rtx4070")


def export_model(model_path='runs/detect/zhuang_chris_rtx4070/weights/best.pt'):
    """Export model to different formats"""

    print("\n" + "=" * 50)
    print("📦 Exporting Model")
    print("=" * 50)

    model = YOLO(model_path)

    # Export to ONNX (most compatible)
    model.export(format='onnx', simplify=True)

    # Export to TensorRT for max speed on RTX 4070
    # model.export(format='engine', device=0)  # Uncomment if needed

    print("✅ Model exported successfully")


if __name__ == "__main__":
    # Main training
    results = train_with_rtx4070()

    # Test the model
    # test_model()

    # Export for deployment
    export_model()