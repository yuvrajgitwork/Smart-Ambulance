#!/usr/bin/env python3
"""
run.py
Simple script to start the Smart Ambulance API with checks.
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'tensorflow',
        'sklearn',
        'numpy',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n📦 Install with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed\n")
    return True


def check_models():
    """Check if model files exist."""
    print("🔍 Checking model files...")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    required_models = [
        "iso_forest.pkl",
        "ocsvm.pkl",
        "lstm_autoencoder.keras"
    ]
    
    missing = []
    for model_file in required_models:
        model_path = models_dir / model_file
        if not model_path.exists():
            missing.append(model_file)
    
    if missing:
        print(f"\n❌ Missing model files: {', '.join(missing)}")
        print(f"\n📁 Expected location: {models_dir}/")
        print("\n📝 Train models using notebooks:")
        print("   1. notebooks/01_classical_ml.ipynb → iso_forest.pkl, ocsvm.pkl")
        print("   2. notebooks/02_deep_learning.ipynb → lstm_autoencoder.keras")
        print("   3. Move trained models to models/ directory")
        return False
    
    print("✅ All model files found\n")
    return True


def check_src():
    """Check if source files exist."""
    print("🔍 Checking source files...")
    
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    required_files = [
        "api.py",
        "inference.py",
        "preprocessing.py"
    ]
    
    missing = []
    for src_file in required_files:
        src_path = src_dir / src_file
        if not src_path.exists():
            missing.append(src_file)
    
    if missing:
        print(f"\n❌ Missing source files: {', '.join(missing)}")
        print(f"\n📁 Expected location: {src_dir}/")
        return False
    
    print("✅ All source files found\n")
    return True


def start_api():
    """Start the FastAPI server."""
    print("=" * 70)
    print("  🚑 STARTING SMART AMBULANCE API SERVER")
    print("=" * 70)
    print()
    
    # Import and run
    try:
        import uvicorn
        
        # Add src to path
        project_root = Path(__file__).parent
        src_dir = str(project_root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        print("🚀 Starting server...")
        print()
        print("📍 API will be available at:")
        print("   - Local:    http://localhost:8000")
        print("   - Docs:     http://localhost:8000/docs")
        print("   - Redoc:    http://localhost:8000/redoc")
        print()
        print("🛑 Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        # Run server
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("  🚑 SMART AMBULANCE API - STARTUP CHECKS")
    print("=" * 70)
    print()
    
    # Run checks
    if not check_requirements():
        sys.exit(1)
    
    if not check_models():
        sys.exit(1)
    
    if not check_src():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print()
    
    # Start server
    start_api()


if __name__ == "__main__":
    main()