# Installation Guide for Face Recognition System

## Quick Start (Recommended for Python 3.13)

If you're using Python 3.13 and want to get started immediately without build tools:

### Option 1: Use the Simplified Version (Easiest)

1. **Run the simplified setup script:**
   ```bash
   python setup_simple.py
   ```

2. **Activate the virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Run the system:**
   ```bash
   python main_simple.py
   ```

### Option 2: Manual Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_simple.txt
   ```

3. **Run the system:**
   ```bash
   python main_simple.py
   ```

## Full Version with InsightFace (Advanced)

If you want the full version with InsightFace for better face recognition accuracy:

### Prerequisites

#### Windows Users:
1. **Install Microsoft Visual C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install with "C++ build tools" workload
   - Or install Visual Studio Community with C++ tools

2. **Alternative: Use Python 3.11:**
   - Download Python 3.11 from: https://www.python.org/downloads/
   - Many packages have pre-built wheels for Python 3.11

#### Linux Users:
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

#### macOS Users:
```bash
xcode-select --install
```

### Installation Steps

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install full requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full system:**
   ```bash
   python main.py
   ```

## Troubleshooting

### Common Issues

#### 1. "Microsoft Visual C++ 14.0 or greater is required"

**Solution:** Install Microsoft Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install with "C++ build tools" workload
- Restart your terminal/command prompt

**Alternative:** Use the simplified version (`main_simple.py`) which doesn't require build tools

#### 2. "ModuleNotFoundError: No module named 'insightface'"

**Solution:** Use the simplified version or install build tools as above

#### 3. "Permission denied" errors

**Solution:** Run as administrator (Windows) or use `sudo` (Linux/Mac)

#### 4. "pip not found"

**Solution:** Upgrade pip first:
```bash
python -m pip install --upgrade pip
```

### Python Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.8-3.10 | ✅ Full Support | All features work |
| 3.11 | ✅ Full Support | Best compatibility |
| 3.12 | ⚠️ Limited | Some packages may have issues |
| 3.13 | ⚠️ Simplified Only | Use `main_simple.py` |

### Package Compatibility Matrix

| Package | Python 3.8-3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
|---------|-----------------|-------------|-------------|-------------|
| opencv-python | ✅ | ✅ | ✅ | ✅ |
| ultralytics | ✅ | ✅ | ✅ | ✅ |
| insightface | ✅ | ✅ | ⚠️ | ❌ |
| imagehash | ✅ | ✅ | ✅ | ✅ |
| numpy | ✅ | ✅ | ✅ | ✅ |

## System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python:** 3.8 or higher
- **RAM:** 4GB
- **Storage:** 2GB free space
- **Camera:** USB webcam or IP camera

### Recommended Requirements
- **OS:** Windows 11, Linux (Ubuntu 20.04+), macOS 12+
- **Python:** 3.11
- **RAM:** 8GB or more
- **Storage:** 5GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional, for faster processing)
- **Camera:** HD webcam or IP camera

## Performance Optimization

### For Better Performance:

1. **Use GPU acceleration (if available):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Adjust configuration in `config.json`:**
   - Reduce `frame_skip` for faster processing
   - Increase `confidence_threshold` for fewer false positives
   - Adjust `max_faces` based on your needs

3. **Use SSD storage** for faster database operations

4. **Close unnecessary applications** to free up RAM

## Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Try the simplified version first**
3. **Ensure you have the latest Python version**
4. **Check that all dependencies are installed correctly**

For additional help, refer to the README.md file or check the logs in the `logs/` directory. 