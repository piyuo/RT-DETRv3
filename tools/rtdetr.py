import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO

from olive.data.registry import Registry


# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def cxcywh_to_xyxy(boxes):
    """Convert (cx,cy,w,h) -> (x1,y1,x2,y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# -------------------------------
# Dataset
# -------------------------------
class CocoDetection(Dataset):
    def __init__(self, data_dir, annotation_file):
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.data_dir = data_dir

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.data_dir, path)).convert("RGB")
        img_t = transform(img)

        # load boxes/labels in normalized cxcywh format
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img.width
            cy = (y + h / 2) / img.height
            nw = w / img.width
            nh = h / img.height
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Return numpy arrays (ONNX expects numpy)
        return {
            "images": img_t.unsqueeze(0).numpy(),   # add batch dim
            "im_shape": np.array([[img.height, img.width]], dtype=np.float32),
            "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32),
        }, {
            "boxes": boxes.numpy(),
            "labels": labels.numpy()
        }

    def __len__(self):
        return len(self.ids)


@Registry.register_dataset()
def dataset_load(data_dir, annotation_file, **kwargs):
    return CocoDetection(data_dir, annotation_file)


# -------------------------------
# Postprocess
# -------------------------------
@Registry.register_post_process()
def dataset_post_process(outputs):
    """Turn model raw outputs into usable predictions."""
    pred_logits = torch.from_numpy(outputs["pred_logits"])
    pred_boxes = torch.from_numpy(outputs["pred_boxes"])

    if pred_logits.numel() == 0 or pred_boxes.numel() == 0:
        return {"boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,))}

    probs = pred_logits.softmax(-1)[0, :, :-1]  # drop no-object class
    scores, labels = probs.max(-1)

    return {
        "boxes": pred_boxes[0],
        "labels": labels,
        "scores": scores,
    }


# -------------------------------
# Evaluation (simple IoU/F1)
# -------------------------------
def calculate_iou_f1(preds, targets, iou_threshold=0.5):
    if len(preds["boxes"]) == 0 or len(targets["boxes"]) == 0:
        return 0.0, 0.0

    pred_xyxy = cxcywh_to_xyxy(preds["boxes"])
    tgt_xyxy = cxcywh_to_xyxy(torch.from_numpy(targets["boxes"]))

    ious = []
    tp, fp = 0, 0
    for pb in pred_xyxy:
        iou = box_iou(pb.unsqueeze(0), tgt_xyxy).max().item()
        ious.append(iou)
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    fn = len(tgt_xyxy) - tp

    mean_iou = np.mean(ious) if ious else 0.0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return mean_iou, f1


def box_iou(box1, box2):
    """IoU between two sets of boxes (xyxy)."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def evaluate(outputs, targets):
    mean_iou, f1 = calculate_iou_f1(outputs, targets)
    return {"mean_of_max_iou": mean_iou, "f1_score": f1}
