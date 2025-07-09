#!/usr/bin/env python3
"""
Setup script for Face Recognition Visitor Counter.
This script automates the installation and setup process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🎭 Face Recognition Visitor Counter - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_system():
    """Check system information."""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("   CUDA: Not available (will use CPU)")
    except (ImportError, ModuleNotFoundError):
        print("   CUDA: PyTorch not installed yet")

def create_virtual_environment():
    """Create virtual environment."""
    print("\n📦 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the appropriate activation command for the current OS."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install Python dependencies."""
    print("\n📚 Installing dependencies...")
    
    # Determine pip command
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print("✅ Pip upgraded")
        
        # Try to install requirements
        try:
            subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            # If that fails, try with Python 3.13 compatible versions
            print("⚠️ Standard requirements failed, trying Python 3.13 compatible versions...")
            
            # Install core dependencies individually
            core_deps = [
                "opencv-python>=4.8.0",
                "numpy>=1.24.0", 
                "Pillow>=10.0.0",
                "ultralytics>=8.0.0",
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "insightface>=0.7.3",
                "onnxruntime>=1.15.0",
                "tqdm>=4.65.0"
            ]
            
            for dep in core_deps:
                try:
                    subprocess.run([pip_cmd, "install", dep], check=True)
                    print(f"✅ Installed: {dep}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Failed to install {dep}: {e}")
                    # Continue with other dependencies
            
            print("✅ Core dependencies installed (some optional packages may have failed)")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["logs", "logs/entries", "logs/exits"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    # Determine python command
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    try:
        result = subprocess.run([python_cmd, "test_pipeline.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print(f"1. Activate virtual environment:")
    print(f"   {activation_cmd}")
    print("\n2. Run the demo:")
    print("   python demo.py")
    print("\n3. Start the main application:")
    print("   python main.py")
    print("\n4. For help:")
    print("   python main.py --help")
    print("\n📖 For more information, see README.md")
    print("\n" + "=" * 60)

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system
    check_system()
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️ Dependency installation failed.")
        print("You can try installing manually:")
        print(f"   {get_activation_command()}")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run tests
    print("\n🧪 Would you like to run tests? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        sys.exit(1) 