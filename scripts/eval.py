import os
import glob
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics import JaccardIndex
from tqdm import tqdm
from model import MyMultiTaskModel

class DetectionDataset(Dataset):
    """自訂 COCO 檢測 Dataset，從助教給定路徑讀取影像與標註"""
    def __init__(self, det_root, transform=None):
        # det_root 下應包含 val/images/ 與 val/annotations.json
        self.images_dir = os.path.join(det_root, "val", "images")
        ann_file = os.path.join(det_root, "val", "annotations.json")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        return img, target

class SegmentationDataset(Dataset):
    """自訂分割 Dataset，從助教給定影像與標註資料夾讀取"""
    def __init__(self, seg_root, transform=None, mask_transform=None):
        # seg_root 下應包含 val/images/ 與 val/masks/
        self.img_paths = sorted(glob.glob(os.path.join(seg_root, "val", "images", "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(seg_root, "val", "masks", "*")))
        assert len(self.img_paths) == len(self.mask_paths), "影像與標註數量不一致！"
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask

class ClassificationDataset(Dataset):
    """自訂分類 Dataset，從助教給定根目錄讀取各類別子資料夾"""
    def __init__(self, cls_root, transform=None):
        # cls_root 下每個子資料夾為一個 class，裡面放影像
        self.samples = []
        self.transform = transform
        classes = sorted([d for d in os.listdir(cls_root) if os.path.isdir(os.path.join(cls_root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for fname in os.listdir(os.path.join(cls_root, c)):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_root, c, fname), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_detection(model, det_root, device):
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    ds = DetectionDataset(det_root, transform)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    coco_gt = ds.coco
    results = []
    model.eval()
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Evaluate DET"):
            imgs = [img.to(device) for img in imgs]
            outs = model(imgs)["det"]
            for out, tgt in zip(outs, targets):
                image_id = int(tgt["image_id"].item())
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b
                    results.append({
                        "image_id": image_id,
                        "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)
                    })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return float(coco_eval.stats[0])

def evaluate_segmentation(model, seg_root, device, num_classes):
    img_tf = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    def mask_to_tensor(m):
        import numpy as np
        return torch.as_tensor(np.array(m), dtype=torch.long)

    ds = SegmentationDataset(seg_root, transform=img_tf, mask_transform=mask_to_tensor)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    jacc = JaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    model.eval()
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluate SEG"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)["seg"]
            preds = torch.argmax(logits, dim=1)
            jacc.update(preds, masks)

    return jacc.compute().mean().item()

def evaluate_classification(model, cls_root, device):
    img_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    ds = ClassificationDataset(cls_root, transform=img_tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluate CLS"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)["cls"]
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Unified-OneHead 多任務評估")
    parser.add_argument("--weights",   type=str, required=True, help="模型權重 .pt 路徑")
    parser.add_argument("--det_root",  type=str, required=True, help="助教提供的檢測資料夾路徑")
    parser.add_argument("--seg_root",  type=str, required=True, help="助教提供的分割資料夾路徑")
    parser.add_argument("--cls_root",  type=str, required=True, help="助教提供的分類資料夾路徑 (val 影像根目錄)")
    parser.add_argument("--tasks",     type=str, default="all",
                        choices=["det","seg","cls","all"],
                        help="評估任務：det, seg, cls, all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 載入模型
    model = MyMultiTaskModel()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)


    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,}")

    # 執行評估
    if args.tasks in ("det","all"):
        print("\n>>> Evaluating Detection ...")
        mAP = evaluate_detection(model, args.det_root, device)
        print(f"Detection mAP@[.50:.95]: {mAP:.4f}")

    if args.tasks in ("seg","all"):
        print("\n>>> Evaluating Segmentation ...")
        mIoU = evaluate_segmentation(model, args.seg_root, device, num_classes=21)
        print(f"Segmentation mIoU: {mIoU:.4f}")

    if args.tasks in ("cls","all"):
        print("\n>>> Evaluating Classification ...")
        top1 = evaluate_classification(model, args.cls_root, device)
        print(f"Classification Top-1 Acc: {top1*100:.2f}%")

if __name__ == "__main__":
    main()
