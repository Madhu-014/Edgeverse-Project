import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

def analyze_annotations(annotations_dir: str, output_file: str = None, start_frame: int = 1, end_frame: int = None):
    """Analyze annotation files and count class occurrences
    
    Args:
        annotations_dir: Directory containing annotation .txt files
        output_file: Output file path for results (optional)
        start_frame: Starting frame number
        end_frame: Ending frame number (if None, processes all files)
    """
    
    if not os.path.exists(annotations_dir):
        print(f"❌ ERROR: Annotations directory not found: {annotations_dir}")
        return {}
    
    # Auto-detect end frame if not provided
    if end_frame is None:
        # Count .txt files in directory
        txt_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt') and f.startswith('frame')]
        if txt_files:
            # Extract frame numbers and find max
            frame_nums = []
            for f in txt_files:
                try:
                    num = int(f.replace('frame', '').replace('.txt', ''))
                    frame_nums.append(num)
                except ValueError:
                    pass
            end_frame = max(frame_nums) if frame_nums else 0
        else:
            end_frame = 0
    
    if end_frame == 0:
        print(f"⚠ No frames found in {annotations_dir}")
        return {}
    
    print(f"ℹ Analyzing frames {start_frame} to {end_frame} from: {annotations_dir}")
    
    # Dictionary to count class occurrences
    class_counts = defaultdict(int)
    frames_found = 0
    
    # Loop through desired frame numbers and build file names
    for i in range(start_frame, end_frame + 1):
        fname = f"frame{i}.txt"
        path = os.path.join(annotations_dir, fname)
        
        if not os.path.isfile(path):
            continue
        
        frames_found += 1
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = parts[0]
                    class_counts[class_id] += 1
    
    if frames_found == 0:
        print(f"⚠ No annotation files found in range {start_frame}-{end_frame}")
        return {}
    
    print(f"✓ Found and processed {frames_found} annotation files")
    
    # Write the results to a file if specified
    if output_file:
        try:
            with open(output_file, 'w') as out:
                out.write("Class_ID\tCount\n")
                for class_id in sorted(class_counts, key=lambda x: int(x)):
                    out.write(f"{class_id}\t\t\t{class_counts[class_id]}\n")
            print(f"✓ Class counts written to: {output_file}")
        except Exception as e:
            print(f"❌ Error writing output file: {e}")
    
    return class_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze annotation dataset and count class occurrences")
    parser.add_argument("--annot-dir", type=str, default="output_annotation", help="Annotations directory")
    parser.add_argument("--output", type=str, default="class_counts.txt", help="Output file for results")
    parser.add_argument("--start-frame", type=int, default=1, help="Starting frame number")
    parser.add_argument("--end-frame", type=int, default=None, help="Ending frame number (auto-detect if not provided)")
    
    args = parser.parse_args()
    
    analyze_annotations(
        annotations_dir=args.annot_dir,
        output_file=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
