import numpy as np
import cv2
import ultralytics
import time
from ultralytics import YOLO
import os
import shutil

model = YOLO("yolo12s.pt")

old_class_filename = "class/old_classes.txt"
new_class_filename = "class/new_classes.txt"


# Target folders and new filenames
targets = {
    'output_annotation': 'classes.txt',
    '../venv/lib/python3.10/site-packages/labelImg/data':'predefined_classes.txt'
}

# Create folders if they don't exist, and copy file with new name
for folder, new_name in targets.items():
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    target_path = os.path.join(folder, new_name)
    shutil.copy(new_class_filename, target_path)
    print(f"Copied to {target_path}")


old_class_list = []
new_class_list = []

with open(old_class_filename, 'r') as f:
    for line in f:
        old_class_list.append(line.strip())
        
with open(new_class_filename, 'r') as f:
    for line in f:
        new_class_list.append(line.strip())

label_dict = {}

for i in range(len(new_class_list)):
	if(new_class_list[i] in old_class_list):
		index = old_class_list.index(new_class_list[i])
		label_dict[index] = i


path = "output_frames/"
for root, dirs, files in os.walk(path):
	for f in files:
		datapath = os.path.join(root,f)
		print(str(f).split(".")[0])

		img = cv2.imread(datapath)
		if img is None:
			print(f"Error reading image {datapath}")
			continue
		
		label_filename = "output_annotation/"+str(f).split(".")[0] + ".txt"
		img_filename = "output_annotation/"+str(f).split(".")[0] + ".jpg"
		cv2.imwrite(img_filename,img)

		results = model(img)

		for result in results:
			boxes = result.boxes.numpy()
			for box in boxes:
				b = box.xywhn[0]
				c = box.cls
				if(int(c[0]) in label_dict):
					new_id = label_dict[int(c[0])]
					str_data = str(new_id) + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + " " + str(b[3])
					with open(label_filename,"a+") as f:
						f.write(str_data)
						f.write("\n")