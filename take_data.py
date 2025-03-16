import os
import shutil



files = os.listdir("/Users/hoangnamvu/Downloads/archive/images/train")

for file in files:
    if "cam_11_" in file:
        file_full_path = os.path.join("/Users/hoangnamvu/Downloads/archive/images/train", file)
        shutil.copy(file_full_path,"data")