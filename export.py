"""
Export YOLO models to TensorRT
Requires 'tensorrt' package for TensorRT export.
"""

from ultralytics import YOLO
import torch
import os
import sys

def check_requirements():
    """Check if system is ready for TensorRT export."""
    print("="*70)
    print("System Check")
    print("="*70)
    
    # Check PyTorch
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        print("\nInstall CUDA-enabled PyTorch:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print(f"✓ CUDA available")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 5:
        print(f"  ⚠️  Limited memory - using workspace=2GB")
    
    # Check Ultralytics
    try:
        from ultralytics import __version__
        print(f"✓ Ultralytics: {__version__}")
    except:
        print("✗ Ultralytics not found")
        print("pip install ultralytics")
        return False
    
    print("="*70)
    print("✅ System ready for TensorRT export!\n")
    return True

def export_model(model_path, imgsz=640, workspace=2):
    """Export a single model to TensorRT."""
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Exporting: {model_path}")
    print(f"{'='*70}")
    
    try:
        # Load model
        print(f"Loading model...")
        model = YOLO(model_path)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Export to TensorRT
        print(f"Exporting to TensorRT...")
        print(f"  → Image size: {imgsz}x{imgsz}")
        print(f"  → Precision: FP32")
        print(f"  → Workspace: {workspace}GB")
        
        engine_path = model.export(
            format='engine',      # TensorRT format
            imgsz=imgsz,         # Image size
            batch=1,             # Batch size
            half=True,           # FP16 (faster!)
            workspace=workspace, # Workspace in GB
            simplify=True,       # Simplify ONNX
            dynamic=False,       # Static shapes
            verbose=True         # Show progress
        )
        
        # Success
        print(f"\n✅ Export successful!")
        print(f"   Output: {engine_path}")
        
        if os.path.exists(engine_path):
            size_mb = os.path.getsize(engine_path) / 1e6
            print(f"   Size: {size_mb:.1f} MB")
        
        return engine_path
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU Out of Memory!")
            print(f"\nTry:")
            print(f"  • Reduce workspace: --workspace 1")
            print(f"  • Reduce image size: --imgsz 640")
            print(f"  • Close other GPU programs")
        else:
            print(f"\n❌ Export failed: {e}")
        return None
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Export all YOLO models."""
    
    # Check system requirements
    if not check_requirements():
        sys.exit(1)
    
    # Determine workspace based on GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    workspace = 2 if gpu_memory < 5 else 4
    
    # Models to export
    models = [
        'yolo26s.pt',
        'yolo26s-pose.pt',
        'yolo26l.pt',
        'yolo26l-pose.pt',
    ]
    
    print(f"Found {sum(os.path.exists(m) for m in models)} model(s) to export")
    print(f"Using workspace: {workspace}GB\n")
    
    # Export each model
    exported = []
    failed = []
    
    for model_path in models:
        engine_path = export_model(model_path, imgsz=640, workspace=workspace)
        
        if engine_path:
            exported.append((model_path, engine_path))
        else:
            failed.append(model_path)
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print(f"✅ Successful: {len(exported)}")
    print(f"❌ Failed: {len(failed)}")
    
    if exported:
        print("\n✅ Exported models:")
        for pt, engine in exported:
            print(f"  {pt} → {engine}")
    
    if failed:
        print("\n❌ Failed models:")
        for model in failed:
            print(f"  {model}")
    
    print("\n" + "="*70)
    print("Next: Update your code to use .engine files")
    print("="*70)

if __name__ == "__main__":
    main()