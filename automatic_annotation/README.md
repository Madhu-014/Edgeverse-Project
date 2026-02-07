# Percieva™ Auto-Annotation Studio

A state-of-the-art web application for automated video annotation using YOLO-powered AI detection. Built with Streamlit, this modern tool streamlines the entire annotation workflow from video processing to dataset creation with an intuitive interface and professional design.

## ✨ Features

### 🖥️ **Modern Web Interface**
- Professional, responsive Streamlit UI with state-of-the-art design
- Real-time image preview and annotation visualization
- Interactive sidebar navigation with page indicators
- Settings panel for directory configuration

### 🎬 **Video & Image Processing**
- Extract frames from videos at custom intervals
- Support for multiple formats (MP4, AVI, MOV, etc.)
- Automatic video segmentation for large files (200MB chunks)
- Direct image upload with ZIP support

### 🔄 **Data Augmentation**
- 6+ intelligent augmentation techniques:
  - Gaussian noise injection
  - Gaussian blur
  - Motion blur
  - Brightness/Contrast adjustment
  - Small rotation
  - Light fog effect
- Configurable variants per image (0-6)
- Automatic output to augmented folder

### 🤖 **AI Auto-Annotation**
- YOLO-powered automatic object detection
- Custom class management
- Batch processing of frames
- YOLO format annotation output
- Visual annotation preview

### 📊 **Image Gallery & Preview**
- Image Gallery: Browse dataset before annotation
- Annotated Gallery: Review annotated images with bounding boxes
- Adjustable grid layout (2-5 columns)
- Customizable image display count

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.10)
- ffmpeg (for video segmentation)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd edgeverse_project/automatic_annotation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install ffmpeg** (required for video segmentation)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

### Running the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Key Scripts

| Script | Purpose |
|--------|---------|
| `streamlit_app.py` | Main web interface (1600+ lines) |
| `auto_annotation.py` | YOLO-based auto-annotation engine |
| `segment_video.py` | Video segmentation utility |
| `data_augmentation.py` | Image augmentation functions |
| `write_frames.py` | Frame extraction utility |
| `create_dataset.py` | Dataset creation with train/val split |
| `analyze_dataset.py` | Annotation statistics and analysis |
| `model.py` | YOLO model utilities |

---

## 🎨 Streamlit UI Features

- **Video Upload**: Drag & drop or browse for video files
- **Frame Extraction**: Configure interval (seconds) and preview
- **Augmentation Settings**: Interactive sliders and checkboxes
- **Auto-Annotation**: One-click YOLO processing
- **Preview Gallery**: Browse annotated frames with bounding boxes
- **Export Options**: Download annotations as ZIP
- **Progress Tracking**: Real-time feedback on all operations
- **Configuration Persistence**: Settings saved across sessions

---

## 🛠️ System Requirements

### Minimum
- CPU: Dual-core processor
- RAM: 8GB
- Storage: 5GB free space
- OS: Windows 10, macOS 10.15+, Ubuntu 18.04+

### Recommended
- CPU: Quad-core or better
- RAM: 16GB+
- GPU: NVIDIA GPU with CUDA support (for faster inference)
- Storage: 20GB+ SSD
- OS: Latest versions

---

## 📝 Notes & Best Practices

1. **Model Weights**: Ensure YOLO model weights (`.pt` files) are present before running auto-annotation
2. **Class Files**: Verify `class/new_classes.txt` contains your target classes
3. **Video Format**: MP4 with H.264 codec recommended for best compatibility
4. **Frame Rate**: Higher intervals (e.g., 2-5 seconds) reduce redundancy
5. **Augmentation**: Use appropriately for your dataset
6. **Disk Space**: Ensure sufficient space for frames and annotations
7. **GPU Acceleration**: Install PyTorch with CUDA for faster processing

---

## 🐛 Troubleshooting

### General Issues

**Issue**: YOLO model not found
- **Solution**: Place `.pt` file in `automatic_annotation/` directory

**Issue**: Video upload fails
- **Solution**: Check file size or use video segmentation tool

**Issue**: Frames not extracted
- **Solution**: Verify OpenCV installation: `pip install opencv-python --upgrade`

**Issue**: Augmentation errors
- **Solution**: Check image format and ensure frames are valid

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. Please refer to the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics YOLO**: For the exceptional object detection framework
- **Streamlit**: For the intuitive web interface framework
- **OpenCV**: For video and image processing capabilities

---

## 📧 Support

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for the computer vision community**
