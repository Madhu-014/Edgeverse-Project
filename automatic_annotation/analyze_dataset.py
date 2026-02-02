import os
from collections import defaultdict

# Directory containing annotation .txt files
annotations_dir = "output_annotation"
output_file = "class_counts.txt"

# ✅ Set frame number range (e.g., frame_001.txt to frame_100.txt)
start_frame = 1
end_frame = 1320

# Dictionary to count class occurrences
class_counts = defaultdict(int)

# Loop through desired frame numbers and build file names
for i in range(start_frame, end_frame + 1):
    fname = f"frame{i}.txt"
    path = os.path.join(annotations_dir, fname)

    if not os.path.isfile(path):
        print(f"⚠️ Skipping missing file: {fname}")
        continue

    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            class_id = parts[0]
            class_counts[class_id] += 1

# Write the results to a file
with open(output_file, 'w') as out:
    out.write("Class_ID\tCount\n")
    for class_id in sorted(class_counts, key=lambda x: int(x)):
        out.write(f"{class_id}\t\t\t{class_counts[class_id]}\n")

print(f"✅ Class counts written to: {output_file}")
