import cv2
import numpy as np
import time
import os
from pathlib import Path

annotated_data_folder = Path("annotated_data/")
output_folder_name = "dataset/headlight2/"

annotated_folder_names = [f.name for f in annotated_data_folder.iterdir() if f.is_dir()]
print(annotated_folder_names)

train_val_split = int(100/12)

image_dir_train = "images/train/"
image_dir_val = "images/val/"
label_dir_train = "labels/train/"
label_dir_val = "labels/val/"
image_dir_path_train = os.path.join(output_folder_name, image_dir_train)
image_dir_path_val = os.path.join(output_folder_name, image_dir_val)
label_dir_path_train = os.path.join(output_folder_name, label_dir_train)
label_dir_path_val = os.path.join(output_folder_name, label_dir_val)

# Create the directory
os.makedirs(image_dir_path_train, exist_ok=True)
os.makedirs(image_dir_path_val, exist_ok=True)
os.makedirs(label_dir_path_train, exist_ok=True)
os.makedirs(label_dir_path_val, exist_ok=True)

file_dict = {}

file_count = 0

train_count = sum(1 for f in Path(image_dir_path_train).iterdir() if f.is_file())
val_count = sum(1 for f in Path(image_dir_path_val).iterdir() if f.is_file())

output_file_name = ""

for folder_names in annotated_folder_names:
	folder_path = "annotated_data/" + str(folder_names) + "/"
	for root, dirs, files in os.walk(folder_path):
		for f in files:
			datapath = os.path.join(root,f)
			file_name = str(f).split(".")[0]
			file_format = str(f).split(".")[1]
			file_count += 1
			if(file_count % train_val_split == 0):
				if((file_name not in file_dict) and (file_name != "classes")):
					val_count += 1
					file_dict[file_name] = [val_count,"val"]
			else:
				if((file_name not in file_dict) and (file_name != "classes")):
					train_count += 1
					file_dict[file_name] = [train_count,"train"]
			
			if(file_name in file_dict):
				if(file_format == "jpg"):
					img = cv2.imread(datapath)
					if(file_dict[file_name][1] == "val"):
						output_file_name = image_dir_path_val + str(file_dict[file_name][0]) + "." + file_format
					if(file_dict[file_name][1] == "train"):
						output_file_name = image_dir_path_train + str(file_dict[file_name][0]) + "." + file_format
					cv2.imwrite(output_file_name,img)
				if(file_format == "txt"):
					if(file_dict[file_name][1] == "val"):
						output_file_name = label_dir_path_val + str(file_dict[file_name][0]) + "." + file_format
					if(file_dict[file_name][1] == "train"):
						output_file_name = label_dir_path_train + str(file_dict[file_name][0]) + "." + file_format
					with open(datapath, 'r') as txt_file:
						contents = txt_file.read()
					with open(output_file_name, 'w') as dest_file:
						dest_file.write(contents)
			if(file_name == "classes"):
				with open(datapath, 'r') as txt_file:
					content_class = txt_file.read()
				output_file_name = label_dir_path_val + file_name + "." + file_format
				with open(output_file_name, 'w') as dest_file:
					dest_file.write(content_class)
				output_file_name = label_dir_path_train + file_name + "." + file_format
				with open(output_file_name, 'w') as dest_file:
					dest_file.write(content_class)
									
	print(file_dict)

			
		




