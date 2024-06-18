import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image

def validate_image(file_path):
    """画像が存在し、破損していないかを確認する関数"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 画像が破損していないことを確認
        print('File is valid:', file_path)
        return True
    except (IOError, FileNotFoundError):
        print('File not found')
        return False

def process_batch(batch):
    processed_batch = []
    for line in batch:
        try:
            data = json.loads(line)
            img_path = data['img_path']
            filename = Path(img_path).name
            new_img_path = Path(f"/dataset/visual_genome/VG_100K/VG_100K/{filename}")
            # 画像が存在し、破損していないかを確認
            if validate_image(new_img_path):
                data["img_path"] = str(new_img_path)
                processed_batch.append(json.dumps(data))
        except Exception as e:
            print(f"Error processing line: {e}")
    return processed_batch

def batch_generator(lines, batch_size):
    for i in range(0, len(lines), batch_size):
        yield lines[i:i+batch_size]

def read_and_process_jsonl_in_batches(input_path, output_path, batch_size=10000, num_workers=200):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    with ProcessPoolExecutor(max_workers=num_workers) as executor, open(output_path, 'w') as output_file:
        for batch in tqdm(batch_generator(lines, batch_size), desc="Batches"):
            future = executor.submit(process_batch, batch)
            for processed_line in future.result():
                output_file.write(processed_line + '\n')

input_file_path = '/home/oshita/vlm/shikra/data/GC_genome196_train.jsonl'
output_file_path = '/home/oshita/vlm/shikra/data/GC_genome196_train_revised.jsonl'

# Execute the processing
read_and_process_jsonl_in_batches(input_file_path, output_file_path)
