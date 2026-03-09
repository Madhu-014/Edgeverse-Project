import numpy as np
import cv2
import ultralytics
import time
from ultralytics import YOLO
import os
import shutil
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YOLO-based auto-annotation for frames")
    parser.add_argument("--frames-dir", type=str, default="output_frames", help="Directory containing input frames")
    parser.add_argument("--annot-dir", type=str, default="output_annotation", help="Directory to save annotations")
    parser.add_argument("--old-classes", type=str, default="class/old_classes.txt", help="Path to old classes file")
    parser.add_argument("--new-classes", type=str, default="class/new_classes.txt", help="Path to new classes file")
    
    args = parser.parse_args()
    
    frames_dir = args.frames_dir
    annot_dir = args.annot_dir
    old_class_filename = args.old_classes
    new_class_filename = args.new_classes

    try:
        # Load model - try multiple common model files
        model_files = ["yolo12s.pt", "yolo11n.pt", "best.pt", "yolo12m.pt"]
        model = None
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"✓ Loading model: {model_file}")
                model = YOLO(model_file)
                break
        
        if model is None:
            print(f"❌ ERROR: No YOLO model found. Checked for: {', '.join(model_files)}")
            sys.exit(1)
        
        # Verify class files exist
        if not os.path.exists(old_class_filename):
            print(f"❌ ERROR: {old_class_filename} not found")
            sys.exit(1)
        
        if not os.path.exists(new_class_filename):
            print(f"❌ ERROR: {new_class_filename} not found")
            sys.exit(1)
        
        # Verify frames directory exists and has images
        if not os.path.exists(frames_dir):
            print(f"❌ ERROR: {frames_dir} directory not found")
            sys.exit(1)
        
        # Create annotation directory
        os.makedirs(annot_dir, exist_ok=True)
        print(f"✓ Annotation directory: {annot_dir}")
        
        # Target folders and new filenames (try to copy classes to labelImg if available)
        targets = {
            annot_dir: 'classes.txt',
        }
        
        # Try to copy to labelImg (optional)
        try:
            labelimg_path = '../venv/lib/python3.10/site-packages/labelImg/data'
            if os.path.exists(labelimg_path):
                targets[labelimg_path] = 'predefined_classes.txt'
        except:
            pass
        
        # Create folders and copy classes file
        for folder, new_name in targets.items():
            try:
                os.makedirs(folder, exist_ok=True)
                target_path = os.path.join(folder, new_name)
                shutil.copy(new_class_filename, target_path)
                print(f"✓ Copied classes to {target_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not copy to {folder}: {e}")
        
        # Read class mappings
        old_class_list = []
        new_class_list = []
        
        with open(old_class_filename, 'r') as f:
            for line in f:
                old_class_list.append(line.strip())
        
        with open(new_class_filename, 'r') as f:
            for line in f:
                new_class_list.append(line.strip())
        
        print(f"✓ Old classes ({len(old_class_list)}): {old_class_list}")
        print(f"✓ New classes ({len(new_class_list)}): {new_class_list}")
        
        # Build label mapping dictionary
        label_dict = {}
        for i in range(len(new_class_list)):
            if new_class_list[i] in old_class_list:
                index = old_class_list.index(new_class_list[i])
                label_dict[index] = i
                print(f"✓ Mapped class {index} ({old_class_list[index]}) -> {i} ({new_class_list[i]})")
        
        if not label_dict:
            print("⚠ WARNING: No classes were mapped! Check your class files.")
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        for root, dirs, files in os.walk(frames_dir):
            for f in files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    continue
                
                frame_count += 1
                datapath = os.path.join(root, f)
                frame_name = str(f).split(".")[0]
                
                print(f"\nProcessing frame {frame_count}: {f}")
                
                # Read image
                img = cv2.imread(datapath)
                if img is None:
                    print(f"  ❌ Error reading image {datapath}")
                    continue
                
                # Prepare output paths
                label_filename = os.path.join(annot_dir, frame_name + ".txt")
                img_filename = os.path.join(annot_dir, frame_name + ".jpg")
                
                # Save copy of image
                cv2.imwrite(img_filename, img)
                print(f"  ✓ Saved image: {img_filename}")
                
                # Run YOLO inference
                results = model(img)
                
                annotation_count = 0
                for result in results:
                    boxes = result.boxes.numpy()
                    for box in boxes:
                        b = box.xywhn[0]
                        c = box.cls
                        if int(c[0]) in label_dict:
                            new_id = label_dict[int(c[0])]
                            str_data = f"{new_id} {b[0]} {b[1]} {b[2]} {b[3]}\n"
                            with open(label_filename, "a+") as lf:
                                lf.write(str_data)
                            annotation_count += 1
                
                processed_count += 1
                print(f"  ✓ Generated annotation: {label_filename} ({annotation_count} boxes)")
        
        print(f"\n" + "="*60)
        print(f"✓ Auto-annotation completed successfully!")
        print(f"  Frames processed: {processed_count}/{frame_count}")
        print(f"  Output directory: {annot_dir}")
        print(f"="*60)

    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()