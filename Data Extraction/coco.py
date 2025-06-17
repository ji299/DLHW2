import os, json, shutil, random
from collections import defaultdict
# ------------------ Configuration ------------------
# 請依實際路徑調整 ↓↓↓
ANNOTATIONS_DIR  = r'C:\Users\N16131691\Desktop\coco\annotations'
TRAIN_IMAGES_DIR = r'C:\Users\N16131691\Desktop\coco\train2017'
VAL_IMAGES_DIR   = r'C:\Users\N16131691\Desktop\coco\val2017'

NUM_TRAIN = 240          # Train 欲抽取張數
NUM_VAL   = 60           # Val   欲抽取張數
NUM_CATS  = 10           # 隨機挑選類別數 (=10)

OUTPUT_DIR = './mini_coco_det'
TRAIN_OUT  = os.path.join(OUTPUT_DIR, 'train')
VAL_OUT    = os.path.join(OUTPUT_DIR, 'val')

random.seed(42)          # 取消或改值可重新隨機

# ---------------------------------------------------
def prepare_dirs():
    for split in (TRAIN_OUT, VAL_OUT):
        os.makedirs(os.path.join(split, 'images'),      exist_ok=True)
        os.makedirs(os.path.join(split, 'annotations'), exist_ok=True)

# ---------------------------------------------------
def build_cat2img_map(coco_data, selected_cats):
    """category_id → set(image_id)，僅限 selected_cats。"""
    cat2img = defaultdict(set)
    for ann in coco_data['annotations']:
        cid = ann['category_id']
        if cid in selected_cats:
            cat2img[cid].add(ann['image_id'])
    return cat2img

# ---------------------------------------------------
def balanced_sample_image_ids(coco_data, selected_cats, total_num):
    """在 selected_cats 間平均抽取 total_num 張 image_id。"""
    cat2img = build_cat2img_map(coco_data, selected_cats)
    per_cls = total_num // len(selected_cats)
    rem     = total_num - per_cls * len(selected_cats)

    chosen = set()
    # 第一輪：各類先抽 per_cls
    for cid in selected_cats:
        avail = list(cat2img[cid] - chosen)
        need  = min(per_cls, len(avail))
        chosen |= set(random.sample(avail, need))

    # 第二輪：把 rem 依序補齊
    for cid in selected_cats[:rem]:
        avail = list(cat2img[cid] - chosen)
        if avail:
            chosen.add(random.choice(avail))

    # 第三輪：若仍不足，跨類別隨機補
    if len(chosen) < total_num:
        pool = set().union(*[cat2img[c] for c in selected_cats]) - chosen
        need = total_num - len(chosen)
        if len(pool) < need:
            raise RuntimeError(f"影像不足：還差 {need} 張")
        chosen |= set(random.sample(list(pool), need))

    assert len(chosen) == total_num
    return chosen

# ---------------------------------------------------
def filter_and_copy(split_name, coco_json_path, img_dir,
                    out_dir, total_num, selected_cats):
    """產生新 COCO JSON 並複製影像。"""
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img_ids = balanced_sample_image_ids(coco, selected_cats, total_num)

    new_imgs = [img for img in coco['images'] if img['id'] in img_ids]
    new_anns = [ann for ann in coco['annotations']
                if ann['image_id'] in img_ids and ann['category_id'] in selected_cats]
    new_cats = [cat for cat in coco['categories'] if cat['id'] in selected_cats]

    new_coco = {
        'info':      coco.get('info', {}),
        'licenses':  coco.get('licenses', []),
        'images':    new_imgs,
        'annotations': new_anns,
        'categories':  new_cats
    }

    # 1) 另存 JSON
    anno_path = os.path.join(out_dir, 'annotations', f'{split_name}.json')
    with open(anno_path, 'w') as f:
        json.dump(new_coco, f)
    print(f'[{split_name}] JSON 產生 → {anno_path}')

    # 2) 複製影像
    for img in new_imgs:
        src = os.path.join(img_dir, img['file_name'])
        dst = os.path.join(out_dir, 'images', img['file_name'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    print(f'[{split_name}] 已複製 {len(new_imgs)} 張影像到 {os.path.join(out_dir, "images")}')

# ---------------------------------------------------
def choose_random_categories(train_json_path, k=10):
    """讀取 train JSON，隨機選出 k 個“有標註”之類別 id。"""
    with open(train_json_path, 'r') as f:
        coco = json.load(f)

    # 只保留在 annotations 中真正出現過的類別
    anns_cats = {ann['category_id'] for ann in coco['annotations']}
    candidates = [cat['id'] for cat in coco['categories'] if cat['id'] in anns_cats]
    if len(candidates) < k:
        raise RuntimeError('可用類別少於要求數量！')

    selected = sorted(random.sample(candidates, k))
    name_map = {cat['id']: cat['name'] for cat in coco['categories']}
    print(f'◆ 本次隨機抽取類別 ({k}):',
          ', '.join(f'{cid}:{name_map[cid]}' for cid in selected))
    return selected

# ---------------------------------------------------
def main():
    prepare_dirs()

    train_json = os.path.join(ANNOTATIONS_DIR, 'instances_train2017.json')
    val_json   = os.path.join(ANNOTATIONS_DIR, 'instances_val2017.json')

    # 1) 隨機決定 10 個類別，後續 train/val 共用
    selected_cats = choose_random_categories(train_json, NUM_CATS)

    # 2) 產生 Train / Val
    filter_and_copy('train', train_json, TRAIN_IMAGES_DIR,
                    TRAIN_OUT, NUM_TRAIN, selected_cats)

    filter_and_copy('val',   val_json,   VAL_IMAGES_DIR,
                    VAL_OUT,   NUM_VAL,  selected_cats)

if __name__ == '__main__':
    main()
