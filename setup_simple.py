#!/usr/bin/env python3
"""
Simplified Setup Script for Face Recognition System
Compatible with Python 3.13 without build tools
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {command}:")
        print(f"  {e.stderr}")
        return False

def main():
    print("🚀 Setting up Face Recognition System (Simplified Version)")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Error: Python 3.8 or higher is required")
        return False
    
    # Create virtual environment if it doesn't exist
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print(f"\n📦 Creating virtual environment: {venv_path}")
        if not run_command(f"{sys.executable} -m venv {venv_path}"):
            return False
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        pip_path = os.path.join(venv_path, "Scripts", "pip")
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:  # Unix/Linux/Mac
        activate_script = os.path.join(venv_path, "bin", "activate")
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Upgrade pip
    print(f"\n⬆️  Upgrading pip...")
    if not run_command(f'"{pip_path}" install --upgrade pip'):
        return False
    
    # Install requirements
    print(f"\n📚 Installing dependencies...")
    if not run_command(f'"{pip_path}" install -r requirements_simple.txt'):
        print("\n⚠️  Some packages failed to install. Trying individual packages...")
        
        # Try installing packages individually
        packages = [
            "opencv-python==4.8.1.78",
            "ultralytics==8.0.196", 
            "numpy==1.24.3",
            "Pillow==10.0.1",
            "imagehash==4.3.1",
            "scikit-learn==1.3.0",
            "matplotlib==3.7.2",
            "tqdm==4.65.0"
        ]
        
        for package in packages:
            if not run_command(f'"{pip_path}" install {package}'):
                print(f"⚠️  Failed to install {package}, continuing...")
    
    # Test installation
    print(f"\n🧪 Testing installation...")
    test_script = f'"{python_path}" -c "import cv2; import ultralytics; import numpy; import PIL; import imagehash; print(\"✓ All packages imported successfully!\")"'
    if run_command(test_script):
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Run the system:")
        print("   python main_simple.py")
        print("\n📖 For more information, see README.md")
        return True
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 