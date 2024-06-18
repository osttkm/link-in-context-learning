from pathlib import Path
import json
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

root_path = Path('/home/oshita/vlm/Link-Context-Learning/docs/CAP_coco2014_train.jsonl')

def check_image(img_path):
    img_path = Path("/dataset/mscoco2014/train2014/"+img_path)
    if not img_path.exists():
        print(f"{img_path} does not exist.")
    else:
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"{img_path} is broken. {e}")
        else:
            img.close()

data = {}
all_images = []
with open(root_path, 'r') as f:
    for line in f:
        _data = json.loads(line)
        img_path = _data['img_path']
        all_images.append(img_path)

all_images = list(set(all_images))

# Use a ProcessPoolExecutor to parallelize the image checking
with ProcessPoolExecutor() as executor:
    executor.map(check_image, all_images)