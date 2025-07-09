# AI-Driven Unique Visitor Counter

A production-ready face recognition system that processes video streams to detect, track, and recognize faces in real-time. The system automatically registers new faces, recognizes them in subsequent frames, tracks them until they exit, and maintains comprehensive logs with timestamped images.

## 🎯 Features

- **Real-time Face Detection**: Using YOLOv8 for accurate face detection
- **Face Recognition**: InsightFace (ArcFace) for robust face embedding generation
- **Face Tracking**: OpenCV trackers for consistent face tracking across frames
- **Unique Visitor Counting**: Prevents double-counting with persistent face IDs
- **Comprehensive Logging**: Database storage and structured image logging
- **Configurable Pipeline**: Adjustable parameters for different use cases
- **RTSP Stream Support**: Works with local files, cameras, and network streams
- **Robust Error Handling**: Graceful handling of disconnections and errors

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Face Detector  │───▶│  Face Tracker   │
│  (Camera/File/  │    │   (YOLOv8)      │    │   (OpenCV)      │
│    RTSP)        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Event Logger  │◀───│ Face Recognizer │◀───│  Face Database  │
│  (Images/Logs)  │    │  (InsightFace)  │    │   (SQLite)      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
face_recognition_visitor_counter/
├── main.py                 # Main application entry point
├── detector.py             # Face detection using YOLOv8
├── recognizer.py           # Face recognition using InsightFace
├── tracker.py              # Face tracking using OpenCV
├── database.py             # SQLite database operations
├── logger.py               # Event logging and image storage
├── config.json             # Configuration parameters
├── requirements.txt        # Python dependencies
├── test_pipeline.py        # End-to-end test suite
├── README.md               # This file
├── logs/                   # Generated log files
│   ├── entries/           # Entry face images
│   ├── exits/             # Exit face images
│   └── events.log         # Event log file
└── visitor_counter.db      # SQLite database
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam or video file for testing
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face_recognition_visitor_counter
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```bash
   python test_pipeline.py
   ```

### Basic Usage

1. **Start with default camera**
   ```bash
   python main.py
   ```

2. **Use a video file**
   ```bash
   python main.py --source "path/to/video.mp4"
   ```

3. **Use RTSP stream**
   ```bash
   python main.py --source "rtsp://username:password@ip:port/stream"
   ```

4. **Run without display (headless mode)**
   ```bash
   python main.py --no-display
   ```

### Interactive Controls

- **`q`**: Quit the application
- **`p`**: Pause/Resume processing
- **`s`**: Save screenshot

## ⚙️ Configuration

The system is configured via `config.json`. Here's an example configuration:

```json
{
  "video": {
    "source": "0",
    "frame_skip": 2,
    "fps": 30,
    "width": 640,
    "height": 480
  },
  "detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "model_size": "n",
    "device": "cpu"
  },
  "recognition": {
    "similarity_threshold": 0.6,
    "embedding_size": 512,
    "model_name": "buffalo_l",
    "device": "cpu"
  },
  "tracking": {
    "max_disappeared": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
  },
  "logging": {
    "log_level": "INFO",
    "save_images": true,
    "flush_interval": 10,
    "max_log_size": 100
  },
  "database": {
    "path": "visitor_counter.db",
    "backup_interval": 3600
  }
}
```

### Configuration Parameters

#### Video Settings
- `source`: Video source (camera index, file path, or RTSP URL)
- `frame_skip`: Process every nth frame (higher = faster, less accurate)
- `fps`: Target frames per second
- `width/height`: Video resolution

#### Detection Settings
- `confidence_threshold`: Minimum confidence for face detection (0.0-1.0)
- `nms_threshold`: Non-maximum suppression threshold
- `model_size`: YOLOv8 model size (n, s, m, l, x)
- `device`: Processing device (cpu, cuda)

#### Recognition Settings
- `similarity_threshold`: Minimum similarity for face matching (0.0-1.0)
- `embedding_size`: Face embedding vector size
- `model_name`: InsightFace model name
- `device`: Processing device (cpu, cuda)

#### Tracking Settings
- `max_disappeared`: Frames before considering face lost
- `min_hits`: Minimum detections before tracking
- `iou_threshold`: IoU threshold for detection matching

#### Logging Settings
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `save_images`: Whether to save face images
- `flush_interval`: Seconds between log flushes
- `max_log_size`: Maximum events in memory queue

## 📊 Output and Logging

### Database Schema

The system uses SQLite with three main tables:

1. **visitors**: Stores face embeddings and visitor information
2. **events**: Logs entry/exit events with metadata
3. **daily_stats**: Aggregated daily statistics

### Log Structure

```
logs/
├── entries/
│   └── YYYY-MM-DD/
│       ├── face_1_entry_14-30-25-123.jpg
│       └── face_2_entry_14-35-10-456.jpg
├── exits/
│   └── YYYY-MM-DD/
│       ├── face_1_exit_14-45-30-789.jpg
│       └── face_2_exit_14-50-15-012.jpg
└── events.log
```

### Sample Log Entry

```json
{
  "type": "entry",
  "face_id": "face_1_abc12345",
  "timestamp": "2024-01-15T14:30:25.123456",
  "confidence": 0.95,
  "location": [100, 150],
  "image_path": "logs/entries/2024-01-15/face_1_entry_14-30-25-123.jpg"
}
```

## 🔧 Advanced Usage

### Custom Model Configuration

To use different models or adjust parameters:

1. **Change YOLOv8 model size**:
   ```json
   "model_size": "l"  // Use larger model for better accuracy
   ```

2. **Adjust recognition sensitivity**:
   ```json
   "similarity_threshold": 0.7  // Higher = more strict matching
   ```

3. **Optimize for performance**:
   ```json
   "frame_skip": 3,  // Process every 3rd frame
   "device": "cuda"  // Use GPU acceleration
   ```

### Database Queries

Query visitor statistics:

```python
from database import VisitorDatabase

db = VisitorDatabase("visitor_counter.db")

# Get total unique visitors
total_visitors = db.get_visitor_count()

# Get all visitors
visitors = db.get_all_visitors()

# Get daily statistics
stats = db.get_daily_stats("2024-01-15")
```

### Custom Video Sources

The system supports various video sources:

- **Camera**: `"0"`, `"1"`, etc.
- **File**: `"path/to/video.mp4"`
- **RTSP**: `"rtsp://ip:port/stream"`
- **HTTP**: `"http://ip:port/stream"`

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_pipeline.py
```

The test suite verifies:
- Database operations
- Face detection pipeline
- Face recognition accuracy
- Tracking functionality
- Logging system
- End-to-end integration

## 📈 Performance Optimization

### For CPU-only systems:
```json
{
  "detection": {"device": "cpu"},
  "recognition": {"device": "cpu"},
  "video": {"frame_skip": 3}
}
```

### For GPU-accelerated systems:
```json
{
  "detection": {"device": "cuda"},
  "recognition": {"device": "cuda"},
  "video": {"frame_skip": 1}
}
```

### For high-traffic scenarios:
```json
{
  "video": {"frame_skip": 5},
  "detection": {"confidence_threshold": 0.7},
  "tracking": {"max_disappeared": 15}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2)
   - Verify camera drivers

2. **Low performance**
   - Increase `frame_skip` in config
   - Use GPU acceleration if available
   - Reduce video resolution

3. **Memory issues**
   - Reduce `max_log_size` in config
   - Enable periodic database cleanup
   - Monitor system resources

4. **Model download issues**
   - Check internet connection
   - Verify model paths in config
   - Clear model cache if needed

### Debug Mode

Enable debug logging:

```json
{
  "logging": {
    "log_level": "DEBUG"
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for face detection
- **InsightFace**: DeepInsight for face recognition
- **OpenCV**: Computer vision library
- **SQLite**: Database engine

---

**This project is a part of a hackathon run by https://katomaran.com** 