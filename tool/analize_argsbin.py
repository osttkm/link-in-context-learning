import os
from zipfile import ZipFile
import pickle

# ファイルパスの定義
file_path = '/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT/training_args.bin'  # ZIPファイルのパス
extracted_folder_path = './'  # 展開された内容を保存するパス

# ZIPファイルの展開
with ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# 展開されたファイルのリスト取得
extracted_files_path = os.path.join(extracted_folder_path, 'training_args')
extracted_files_in_dir = os.listdir(extracted_files_path)

# data.pklファイルの読み込み
data_pkl_path = os.path.join(extracted_files_path, 'data.pkl')

# 特定のモジュールが必要な場合、ダミーのモジュールを定義することで読み込みが可能になる場合があります。
# 以下の例では、実際にはダミーのモジュール定義やその他の安全な読み込み方法を示していません。
# 実際には、pickleの安全でない読み込みに関連するリスクを理解し、信頼できるファイルのみを処理するようにしてください。
try:
    with open(data_pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    print("data.pklの内容:", data)
except ModuleNotFoundError as e:
    print(f"必要なモジュールが見つかりません: {e}")
