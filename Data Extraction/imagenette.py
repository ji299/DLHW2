import os
import shutil
import random
import pandas as pd

# ------------------- 參數設定 -------------------
# 1. 請自行確認並修改成你本機的「已解壓縮」Imagenette-160 根目錄
src_root = "C:/Users/N16131691/Desktop/imagenette_160/imagenette2-160"

# 2. 訂義新的「輸出根目錄」。執行程式時，會在底下自動建立 train/、val/ 及各 label 子資料夾
output_root = "./mini_imagenette_160"

# 3. 標籤欄位名稱：使用 noisy_labels_0 作為最終的 clean label
label_column = "noisy_labels_0"

# 4. train: 每個類別 抽 24 張；val: 每個類別 抽  6 張
NUM_PER_LABEL_TRAIN = 24
NUM_PER_LABEL_VAL   = 6

# 5. 隨機種子（讓抽樣可重現）
random.seed(42)

# ------------------- 讀取 CSV -------------------
csv_path = "C:/Users/N16131691/Desktop/imagenette_160/imagenette2-160/noisy_imagenette.csv"  # 請確認檔案路徑正確
df = pd.read_csv(csv_path)

# 把所有不同 label (noisy_labels_0) 列出來
all_labels = sorted(df[label_column].unique())

# ------------------- 建立輸出資料夾 -------------------
# 如果已存在就先刪除整個資料夾，再從頭建立
if os.path.isdir(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root, exist_ok=True)

train_root = os.path.join(output_root, "train")
val_root   = os.path.join(output_root, "val")
os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root,   exist_ok=True)

# 用來存 最終 train/val annotations
train_records = []  # list of tuples: (relative_filepath, label)
val_records   = []

# ------------------- 逐 label 分別抽樣 -------------------
for label in all_labels:
    # 篩出此 label 且 is_valid==False 作為「候選訓練樣本池」
    candidates_train = df[(df[label_column] == label) & (df["is_valid"] == False)]
    # 篩出此 label 且 is_valid==True 作為「候選驗證樣本池」
    candidates_val   = df[(df[label_column] == label) & (df["is_valid"] == True)]

    # 檢查候選池是否至少有足夠數量可抽樣
    if len(candidates_train) < NUM_PER_LABEL_TRAIN:
        raise RuntimeError(f"Label = {label} 的候選訓練池不足 {NUM_PER_LABEL_TRAIN} 張（目前只有 {len(candidates_train)} 張）！")
    if len(candidates_val) < NUM_PER_LABEL_VAL:
        raise RuntimeError(f"Label = {label} 的候選驗證池不足 {NUM_PER_LABEL_VAL} 張（目前只有 {len(candidates_val)} 張）！")

    # 隨機抽樣（不重複）
    sampled_train = random.sample(list(candidates_train.index), NUM_PER_LABEL_TRAIN)
    sampled_val   = random.sample(list(candidates_val.index), NUM_PER_LABEL_VAL)

    # 建立此 label 對應的輸出子資料夾
    train_label_folder = os.path.join(train_root, label)
    val_label_folder   = os.path.join(val_root,   label)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder,   exist_ok=True)

    # ------ 把抽到的「訓練樣本」複製到 ./mini_imagenette_160/train/<label>/  ----------------
    for idx in sampled_train:
        rel_path = df.loc[idx, "path"]        # e.g. "train/n0123456/xxx.JPEG"
        src_file = os.path.join(src_root, rel_path)
        dst_file = os.path.join(train_label_folder, os.path.basename(rel_path))
        # 複製 JPEG 檔
        shutil.copy2(src_file, dst_file)

        # 記錄 annotation：相對於 output_root
        # e.g. "train/n0123456/xxx.JPEG", label="n0123456"
        rel_for_csv = os.path.join("train", label, os.path.basename(rel_path))
        train_records.append((rel_for_csv, label))

    # ------ 把抽到的「驗證樣本」複製到 ./mini_imagenette_160/val/<label>/ ----------------
    for idx in sampled_val:
        rel_path = df.loc[idx, "path"]        # e.g. "val/n0123456/yyy.JPEG"
        src_file = os.path.join(src_root, rel_path)
        dst_file = os.path.join(val_label_folder, os.path.basename(rel_path))
        # 複製 JPEG 檔
        shutil.copy2(src_file, dst_file)

        # 記錄 annotation：相對於 output_root
        # e.g. "val/n0123456/yyy.JPEG", label="n0123456"
        rel_for_csv = os.path.join("val", label, os.path.basename(rel_path))
        val_records.append((rel_for_csv, label))

# ------------------- 寫出 Annotation CSV -------------------
train_df = pd.DataFrame(train_records, columns=["filepath","label"])
val_df   = pd.DataFrame(val_records,   columns=["filepath","label"])

train_df.to_csv(os.path.join(output_root, "train_annotations.csv"), index=False)
val_df.to_csv(  os.path.join(output_root, "val_annotations.csv"),   index=False)

# ------------------- 列印結果 -------------------
print("===== 完成抽樣與複製 =====")
print(f"→ 訓練集總共 {len(train_records)} 張影像（理想值：10 類 × 24 張 = 240 張）")
print(f"→ 驗證集總共 {len(val_records)} 張影像（理想值：10 類 ×  6 張 =  60 張）")
print(f"最終的資料夾結構：\n{output_root}/")
print("  ├─ train/")
for lbl in all_labels:
    cnt = len([f for f in train_records if f[1] == lbl])
    print(f"  │    ├─ {lbl}/ (共 {cnt} 張)")
print("  └─ val/")
for lbl in all_labels:
    cnt = len([f for f in val_records if f[1] == lbl])
    print(f"       ├─ {lbl}/ (共 {cnt} 張)")

print("\nAnnotation CSV 檔：")
print(f"  • 訓練集 annotation → {os.path.join(output_root,'train_annotations.csv')}")
print(f"  • 驗證集 annotation → {os.path.join(output_root,'val_annotations.csv')}")
