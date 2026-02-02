# ARAS Auto-Annotation Studio 🎯

A comprehensive toolkit for automated video annotation using YOLO models, specifically designed for ARAS (Advanced Rider Assistance Systems) applications. This project streamlines the entire annotation workflow from video processing to dataset creation.

## 🌟 Features

### 🖥️ **Streamlit Web Interface**
- Modern, user-friendly web UI for the complete annotation pipeline
- Real-time preview of annotated frames
- Interactive configuration of all parameters
- One-click LabelImg integration for manual refinement

### 🎬 **Video Processing**
- Extract frames from videos at custom intervals
- Automatic video segmentation for large files (200MB chunks)
- Support for multiple video formats (MP4, AVI, MOV, etc.)

### 🔄 **Data Augmentation**
- Gaussian noise injection
- Salt & pepper noise
- Motion blur (simulates 2-wheeler movement)
- Brightness/contrast adjustments
- Horizontal flipping
- Random cropping
- Configurable augmentation strategies

### 🤖 **Auto-Annotation**
- YOLO-based automatic object detection and annotation
- Custom class mapping support
- Batch processing of frames
- YOLO format label generation

### 📊 **Dataset Management**
- Analyze annotation distributions
- Create train/validation splits
- Class count statistics
- YOLO dataset structure creation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.10)
- ffmpeg (for video segmentation)
- LabelImg (optional, for manual annotation refinement)

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

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install LabelImg** (optional, for manual refinement)
   ```bash
   # Option 1 (recommended): pipx
   pipx install labelimg
   
   # Option 2: pip
   pip install labelImg
   
   # Option 3: Homebrew (macOS)
   brew install labelimg
   ```

5. **Install ffmpeg** (required for video segmentation)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

### 📁 Directory Structure

The application automatically creates the following structure:

```
automatic_annotation/
├── videos/                    # Input videos
├── output_frames/             # Extracted frames
├── output_annotation/         # Generated annotations
│   ├── classes.txt           # Class names
│   └── frame*.txt            # YOLO format labels
├── class/
│   ├── old_classes.txt       # Original YOLO classes
│   └── new_classes.txt       # Your custom classes
├── *.pt                      # YOLO model weights
└── scripts...
```

### 🎛️ **Custom Directory Configuration**

The application allows you to use custom directories for all operations:

1. **Go to Project Configuration** tab in the web interface
2. **Set custom paths:**
   - Frames directory: Where extracted frames are stored
   - Annotations directory: Where annotations will be saved
3. **All operations respect these settings:**
   - Frame extraction
   - Data augmentation
   - Auto-annotation
   - Preview and analysis

**Example custom setup:**
```
my_project/
├── raw_videos/
├── processed_frames/
├── annotations_v1/
└── annotations_v2/
```

You can configure the app to use any of these paths for your workflow!

---

## 🎯 Usage

### 1. **Launch the Web Interface**

```bash
streamlit run streamlit_app.py
```

The interface will open at `http://localhost:8501`

### 2. **Command-Line Tools**

#### **Video Segmentation**
Split large videos into manageable chunks:
```bash
python segment_video.py <input_video> [output_directory] [chunk_size_mb]

# Examples
python segment_video.py large_video.mp4
python segment_video.py video.mp4 ./segmented 150
```

#### **Frame Extraction**
Extract frames from video programmatically:
```bash
python write_frames.py
```

#### **Auto-Annotation**
Run automatic annotation on extracted frames:
```bash
python auto_annotation.py
```

#### **Dataset Analysis**
Analyze annotation distribution:
```bash
python analyze_dataset.py
```

#### **Dataset Creation**
Create train/val split from annotated data:
```bash
python create_dataset.py
```

---

## 🔧 Configuration

### Custom Class Mapping

1. Edit `class/old_classes.txt` - Original YOLO class names
2. Edit `class/new_classes.txt` - Your custom class names
3. The system automatically maps matching classes

### YOLO Model Selection

Place your YOLO model weights in the `automatic_annotation/` directory:
- Supported: YOLOv8, YOLOv11, YOLOv12
- Formats: `.pt` files
- Default models: `yolo12s.pt`, `yolo11n.pt`, `best.pt`

Update the model reference in `auto_annotation.py`:
```python
model = YOLO("your_model.pt")
```

### Augmentation Settings

Customize augmentation in the Streamlit UI or modify `data_augmentation.py`:
- Noise levels (Gaussian, salt & pepper)
- Motion blur intensity and angle
- Brightness/contrast ranges
- Flip probability
- Crop dimensions

---

## 📊 Workflow

```
┌─────────────┐
│ Upload Video│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Extract Frames│ (every N seconds)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Augment     │ (optional)
│ Images      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   YOLO      │
│ Auto-Annotate│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Preview   │
│  & Refine   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Export     │
│  Dataset    │
└─────────────┘
```

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
- **LabelImg Integration**: Launch directly from UI
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
5. **Augmentation**: Use moderately for 2-wheeler ARAS scenarios
6. **Manual Review**: Always review auto-annotations with LabelImg
7. **GPU Acceleration**: Install PyTorch with CUDA for faster processing
8. **Disk Space**: Ensure sufficient space for frames and annotations
9. **Directory Paths**: Use the Project Configuration tab to set custom directories
10. **Path Consistency**: Ensure directories exist and are readable/writable

---

## 🐛 Troubleshooting

### General Issues

**Issue**: LabelImg button doesn't work
- **Solution**: Ensure LabelImg is installed and accessible from terminal

**Issue**: YOLO model not found
- **Solution**: Place `.pt` file in `automatic_annotation/` directory (checks: `yolo12s.pt`, `yolo11n.pt`, `best.pt`, `yolo12m.pt`)

**Issue**: Video upload fails
- **Solution**: Check file size (<200MB) or use video segmentation tool

**Issue**: Frames not extracted
- **Solution**: Verify OpenCV installation: `pip install opencv-python --upgrade`

**Issue**: Augmentation errors
- **Solution**: Check image format and ensure frames are valid

### Directory-Related Issues

**Issue**: "No frames found in directory"
- **Cause**: Frames directory path is incorrect or doesn't contain frames
- **Solution**: 
  1. Go to **Project Configuration** tab
  2. Verify the frames directory path
  3. Ensure frames are extracted to that directory first
  4. Check that directory permissions allow reading

**Issue**: Annotations not saving to output_annotation/
- **Cause**: Incorrect annotation directory path or permission issues
- **Solution**:
  1. Verify annotation directory path in Project Configuration
  2. Ensure directory exists and has write permissions
  3. Check subprocess logs (click "View Full Logs" button)
  4. If custom path is used, verify it's absolute or relative correctly

**Issue**: "Annotations directory not found" in Preview tab
- **Cause**: Session state lost or path changed
- **Solution**: 
  1. Re-enter the annotation directory path in Project Configuration
  2. Press Enter to confirm
  3. Return to Preview tab

**Issue**: Different directories not working properly
- **Cause**: Paths not being passed correctly to subprocess
- **Solution**:
  1. Check logs for actual paths used
  2. Verify directory names don't contain spaces (if possible)
  3. Use absolute paths for consistency
  4. Ensure all directories are created before operations

### Command-Line Usage

**To use custom directories via command line:**

```bash
# Auto-annotation with custom directories
python auto_annotation.py --frames-dir ./my_frames --annot-dir ./my_annotations

# Analyze dataset with custom directories
python analyze_dataset.py --annot-dir ./my_annotations --output ./stats.txt

# Extract frames with custom output
python -c "from data_augmentation import extract_frames_every; extract_frames_every('video.mp4', './custom_frames', 3)"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the ARAS research initiative. Please contact the project maintainers for licensing information.

---

## 🙏 Acknowledgments

- **Ultralytics YOLO**: For the exceptional object detection framework
- **Streamlit**: For the intuitive web interface framework
- **LabelImg**: For the annotation refinement tool
- **OpenCV**: For video and image processing capabilities

---

## 📧 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

---

**Built with ❤️ for ARAS Research**
- LabelImg is a desktop Qt app and cannot be embedded inside Streamlit; the button launches it on the host.
