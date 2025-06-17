import os
import random
import shutil
# --------------------------------------------------
# 1. 設定：請依實際情況修改路徑
# --------------------------------------------------

# PASCAL VOC 2012 的根目錄（裡面應包含 JPEGImages 與 SegmentationClass 兩個資料夾）
VOC_ROOT = r'C:\Users\N16131691\Desktop\VOCdevkit\VOC2012'   # ← 請改成你本機 PASCAL VOC 2012 資料夾的路徑

# 輸出 mini-voc 分割資料集的資料夾，程式會自動建立子目錄
OUTPUT_DIR = r'./mini_voc_seg'

# 總共要抽取的張數，以及 train/val 的數目
TOTAL_IMAGES = 300
NUM_TRAIN = 240
NUM_VAL = TOTAL_IMAGES - NUM_TRAIN

# 隨機種子，確保可重現
RANDOM_SEED = 42

# --------------------------------------------------
# 2. 檢查 VOC 原始資料夾結構
# --------------------------------------------------

# 原始影像放在 JPEGImages，檔名為 *.jpg
JPEG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
# 對應的 segmentation masks 放在 SegmentationClass，檔名為 *.png
MASK_DIR = os.path.join(VOC_ROOT, 'SegmentationClass')

if not os.path.isdir(JPEG_DIR):
    raise FileNotFoundError(f"找不到影像資料夾：{JPEG_DIR}")
if not os.path.isdir(MASK_DIR):
    raise FileNotFoundError(f"找不到 segmentation mask 資料夾：{MASK_DIR}")

# --------------------------------------------------
# 3. 掃描所有可用的 sample
# --------------------------------------------------

# 列出 JPEGImages 底下所有 .jpg 檔案，取 basename （去掉副檔名）
all_jpegs = [
    os.path.splitext(f)[0]
    for f in os.listdir(JPEG_DIR)
    if f.lower().endswith('.jpg')
]

# 只保留那些在 SegmentationClass 中也有對應 .png mask 的影像 ID
# 以避免有 image 但沒有 mask 的情況
valid_ids = []
for img_id in all_jpegs:
    mask_path = os.path.join(MASK_DIR, img_id + '.png')
    if os.path.isfile(mask_path):
        valid_ids.append(img_id)

if len(valid_ids) < TOTAL_IMAGES:
    raise RuntimeError(
        f"可用的影像張數不足：找到 {len(valid_ids)} 張，但你希望抽取 {TOTAL_IMAGES} 張。"
    )

# --------------------------------------------------
# 4. 隨機抽樣並切分 train/val
# --------------------------------------------------

random.seed(RANDOM_SEED)
selected_ids = random.sample(valid_ids, TOTAL_IMAGES)

train_ids = selected_ids[:NUM_TRAIN]
val_ids   = selected_ids[NUM_TRAIN:]

# --------------------------------------------------
# 5. 建立輸出資料夾
# --------------------------------------------------

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# train images / masks
TRAIN_IMG_OUT  = os.path.join(OUTPUT_DIR, 'train', 'images')
TRAIN_MASK_OUT = os.path.join(OUTPUT_DIR, 'train', 'masks')
# val images / masks
VAL_IMG_OUT    = os.path.join(OUTPUT_DIR, 'val',   'images')
VAL_MASK_OUT   = os.path.join(OUTPUT_DIR, 'val',   'masks')

make_dir(TRAIN_IMG_OUT)
make_dir(TRAIN_MASK_OUT)
make_dir(VAL_IMG_OUT)
make_dir(VAL_MASK_OUT)

# --------------------------------------------------
# 6. 複製檔案函式
# --------------------------------------------------

def copy_samples(id_list, img_src_dir, mask_src_dir, img_dst_dir, mask_dst_dir):
    """
    將 id_list 中的每個影像 ID，從 img_src_dir 複製 .jpg 到 img_dst_dir，
    從 mask_src_dir 複製 .png 到 mask_dst_dir。
    """
    for img_id in id_list:
        src_img_path  = os.path.join(img_src_dir, img_id + '.jpg')
        src_mask_path = os.path.join(mask_src_dir, img_id + '.png')

        dst_img_path  = os.path.join(img_dst_dir, img_id + '.jpg')
        dst_mask_path = os.path.join(mask_dst_dir, img_id + '.png')

        if not os.path.isfile(src_img_path) or not os.path.isfile(src_mask_path):
            # 如果任一檔案不存在，就跳過並印警告
            print(f"[警告] 找不到對應檔案：{img_id}，跳過。")
            continue

        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_mask_path, dst_mask_path)

# --------------------------------------------------
# 7. 開始複製 train/val 檔
# --------------------------------------------------

print("── 開始複製 TRAIN 範本 ...")
copy_samples(train_ids, JPEG_DIR, MASK_DIR, TRAIN_IMG_OUT, TRAIN_MASK_OUT)
print(f"已將 {len(train_ids)} 張影像及對應的 mask 複製到 {TRAIN_IMG_OUT}＆{TRAIN_MASK_OUT}")

print("── 開始複製 VAL 範本 ...")
copy_samples(val_ids, JPEG_DIR, MASK_DIR, VAL_IMG_OUT, VAL_MASK_OUT)
print(f"已將 {len(val_ids)} 張影像及對應的 mask 複製到 {VAL_IMG_OUT}＆{VAL_MASK_OUT}")

# --------------------------------------------------
# 8. 完成
# --------------------------------------------------

print("==== Mini-VOC-Seg 資料集建立完成 ====")
print(f"總共輸出路徑：{os.path.abspath(OUTPUT_DIR)}")
print(f"  - Train images: {TRAIN_IMG_OUT} ({len(train_ids)} 張)")
print(f"  - Train masks:  {TRAIN_MASK_OUT} ({len(train_ids)} 張)")
print(f"  - Val images:   {VAL_IMG_OUT} ({len(val_ids)} 張)")
print(f"  - Val masks:    {VAL_MASK_OUT} ({len(val_ids)} 張)")
