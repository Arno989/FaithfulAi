import logging
import sys, os
import shutil
from pathlib import Path
from shutil import copyfile

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath("../"))
sys.path.insert(0, PROJECT_ROOT)
images_Root = f"{PROJECT_ROOT}\\Data\\Pre-images"
images_F = f"{PROJECT_ROOT}\\Data\\Pre-images\\textures-Faithful\\block"
images_V = f"{PROJECT_ROOT}\\Data\\Pre-images\\textures-Vanilla\\block"
images_Processed_F = f"{PROJECT_ROOT}\\Data\\Processed-images\\FaithfulBlocks"
images_Processed_V = f"{PROJECT_ROOT}\\Data\\Processed-images\\VanillaBlocks"


for img_part in [images_V, images_F]:
    i = 0
    for file in os.listdir(img_part):
        if file.endswith(".png"): 
            try:
                src_dir = os.path.join(img_part, file)
                if img_part == images_V:
                    dst_dir = os.path.join(images_Processed_V, f"vanilla_{i}.png")
                elif img_part == images_F:
                    dst_dir = os.path.join(images_Processed_F, f"faithful_{i}.png")
                shutil.copy(src_dir ,dst_dir)
                
                i += 1
                print(f"Copied: {src_dir}  to  {dst_dir}")
            except Exception as e:
                print(e)
                
subprocess.run("find . -type f -iname '*.png' -exec pngcrush -ow -rem allb -reduce {} \;", shell=True, check=True, text=True)




# print(PROJECT_ROOT)
# log.info(PROJECT_ROOT)

# path = os.path.expanduser('~/Drive/Company/images/')
# src = os.listdir(os.path.join(path, 'full_res'))


# for dirpath, dirs, files in os.walk(images_Root):	 
# 	path = dirpath.split('/')
# 	log.info(f"| {len(path)*'---'} [{os.path.basename(dirpath)}]")
# 	for f in files:
# 		log.info(f"|{len(path)*'---'} {f}")


# for filename in src:
#     if filename.endswith('.jpg'):
#         basename = os.path.splitext(filename)[0]
#         print(basename) #log to console so I can see it's at least doing something (it's not)
#         dest = os.path.join(path, basename)
#         if not os.path.exists(dest):
#             os.makedirs(dest) #create the folder if it doesn't exist
#         shutil.copyfile(os.path.join(path, 'full_res', filename), os.path.join(dest, 'export.jpg'))
