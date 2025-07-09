# Installation Guide for Python 3.13

Since Python 3.13 is very new, some packages may not be fully compatible yet. Here's how to install the face recognition system on Python 3.13:

## Option 1: Automated Setup (Recommended)

Run the setup script which will handle Python 3.13 compatibility:

```bash
python setup.py
```

The script will automatically try alternative installation methods if the standard requirements fail.

## Option 2: Manual Installation

If the automated setup fails, follow these steps:

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Core Dependencies
```bash
pip install --upgrade pip
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install ultralytics>=8.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install insightface>=0.7.3
pip install onnxruntime>=1.15.0
pip install tqdm>=4.65.0
```

### 4. Install Optional Dependencies
```bash
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### 5. Create Directories
```bash
mkdir logs
mkdir logs\entries
mkdir logs\exits
```

## Option 3: Use Python 3.11 or 3.12 (Alternative)

If you continue to have issues with Python 3.13, consider using Python 3.11 or 3.12 which have better package compatibility:

1. Install Python 3.11 or 3.12
2. Create a new virtual environment with the older Python version
3. Follow the standard installation process

## Troubleshooting

### Common Issues:

1. **Package not found for Python 3.13**
   - Try installing packages individually
   - Use `--no-deps` flag for problematic packages
   - Check if there are pre-release versions available

2. **Build errors**
   - Install Visual Studio Build Tools (Windows)
   - Install development headers (Linux)

3. **CUDA issues**
   - Install CPU-only versions: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### Test Installation

After installation, test the system:

```bash
python test_pipeline.py
```

If tests pass, you're ready to use the system:

```bash
python main.py
```

## Notes

- The system will work with CPU-only processing if CUDA is not available
- Some optional packages (matplotlib, seaborn) may not be essential for core functionality
- The face recognition models will be downloaded automatically on first use 