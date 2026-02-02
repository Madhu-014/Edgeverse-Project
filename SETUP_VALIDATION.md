# Setup Validation Guide 🔍

This document helps you verify that all components are working correctly with your custom directory configuration.

## ✅ Pre-Launch Checklist

### 1. **Environment Setup**
- [ ] Python 3.8+ installed: `python --version`
- [ ] Virtual environment activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] ffmpeg available: `ffmpeg -version`

### 2. **Directory Structure**
- [ ] Project root exists: `/your/path/edgeverse_project/`
- [ ] `automatic_annotation/` folder present
- [ ] `automatic_annotation/class/` folder with class files:
  - [ ] `old_classes.txt` (YOLO base classes)
  - [ ] `new_classes.txt` (your custom classes)

### 3. **Model Files**
- [ ] At least one YOLO model exists in `automatic_annotation/`:
  - [ ] `yolo12s.pt` OR
  - [ ] `yolo11n.pt` OR
  - [ ] `best.pt` OR
  - [ ] `yolo12m.pt`

## 🚀 Launch & Configuration

### Step 1: Start Streamlit
```bash
cd automatic_annotation
streamlit run streamlit_app.py
```

### Step 2: Go to Project Configuration
- Click the **Project Configuration** tab
- Notice the default paths:
  - Frames directory: `output_frames`
  - Annotations directory: `output_annotation`

### Step 3: (Optional) Set Custom Directories
```
Example setup for organizing multiple projects:
- Frames: ./project_v1/frames
- Annotations: ./project_v1/annotations
```

1. Edit the text fields
2. Click elsewhere or press Enter to confirm
3. Check that paths are displayed correctly in "Current Project Paths"

## 🧪 Test Each Tab

### Tab 1: Upload & Extract
- [ ] Go to **Upload & Extract Frames** tab
- [ ] Upload a test video (or select images)
- [ ] Click "Extract Frames"
- [ ] **Verify**: Check that frames appear in your configured frames directory

### Tab 2: Augment
- [ ] Go to **Augment** tab
- [ ] Select the same directory where you extracted frames
- [ ] Enable at least one augmentation option
- [ ] Click "Run Augmentation"
- [ ] **Verify**: Check for augmented files with `_aug` suffix in the frames directory

### Tab 3: Auto-Annotate
- [ ] Go to **Auto-Annotate** tab
- [ ] Check that it finds your frames
- [ ] Click "Run Auto-Annotation"
- [ ] **Verify**: 
  - [ ] Check logs for "Auto-annotation completed successfully!"
  - [ ] Look for `.txt` and `.jpg` files in your annotation directory
  - [ ] Count should match frame count

### Tab 4: Preview
- [ ] Go to **Preview** tab
- [ ] **Verify**: It shows annotations from your configured directory
- [ ] View annotated images with bounding boxes
- [ ] Check that images display correctly with boxes overlaid

### Tab 5: LabelImg
- [ ] Go to **LabelImg** tab (optional)
- [ ] Click "Launch LabelImg" (if installed)
- [ ] Verify it opens with correct annotations

## 🔧 Common Issues & Fixes

### Issue: "No frames found"
```
✓ Verify frames exist in configured directory
✓ Check directory path has no typos
✓ Ensure permissions allow reading
```

### Issue: "Annotations not saving"
```
✓ Check annotation directory path
✓ Verify write permissions
✓ Check logs in "View Full Logs" section
```

### Issue: Preview shows nothing
```
✓ Confirm annotations were created
✓ Check annotation directory path matches preview
✓ Reload the page
```

## 📊 Test Data Generation

To quickly test without real video:

```bash
# Create test frames
python -c "
from data_augmentation import extract_frames_every
import cv2
import os

# Create a simple test image
img = cv2.imread('sample.jpeg')  # Use existing sample
if img is not None:
    os.makedirs('test_frames', exist_ok=True)
    for i in range(5):
        cv2.imwrite(f'test_frames/frame{i}.jpg', img)
    print('✓ Created 5 test frames in test_frames/')
"
```

Then configure app to use `test_frames/` and `test_annotations/`

## ✨ Validation Script

Run this Python script to validate your setup:

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

print("=" * 60)
print("ARAS Auto-Annotation Setup Validator")
print("=" * 60)

checks_passed = 0
checks_failed = 0

def check(name, condition):
    global checks_passed, checks_failed
    if condition:
        print(f"✓ {name}")
        checks_passed += 1
    else:
        print(f"✗ {name}")
        checks_failed += 1
    return condition

# Check Python version
check("Python 3.8+", sys.version_info >= (3, 8))

# Check directories
base_dir = Path(".")
check("automatic_annotation/ exists", (base_dir / "automatic_annotation").exists())
check("class/ folder exists", (base_dir / "automatic_annotation" / "class").exists())

# Check class files
check("old_classes.txt exists", (base_dir / "automatic_annotation" / "class" / "old_classes.txt").exists())
check("new_classes.txt exists", (base_dir / "automatic_annotation" / "class" / "new_classes.txt").exists())

# Check model files
models = ["yolo12s.pt", "yolo11n.pt", "best.pt", "yolo12m.pt"]
model_found = any((base_dir / "automatic_annotation" / m).exists() for m in models)
check(f"YOLO model found ({', '.join(models)})", model_found)

# Check Python packages
try:
    import streamlit
    check("Streamlit installed", True)
except:
    check("Streamlit installed", False)

try:
    import cv2
    check("OpenCV installed", True)
except:
    check("OpenCV installed", False)

try:
    import ultralytics
    check("Ultralytics (YOLO) installed", True)
except:
    check("Ultralytics (YOLO) installed", False)

# Summary
print("\n" + "=" * 60)
print(f"Passed: {checks_passed} | Failed: {checks_failed}")
if checks_failed == 0:
    print("✓ All checks passed! Ready to launch.")
else:
    print(f"✗ {checks_failed} check(s) failed. Please fix before launching.")
print("=" * 60)
```

## 🎯 Success Indicators

After running through all tabs:
- [ ] Frames extracted to configured directory
- [ ] Augmented images created (if enabled)
- [ ] Annotation files (.txt) generated
- [ ] Preview shows annotated images
- [ ] No errors in logs

## 📝 Next Steps

Once validated:
1. Run your actual annotation pipeline
2. Export dataset if needed
3. Use annotations for training
4. Iterate and improve

Good luck! 🚀
