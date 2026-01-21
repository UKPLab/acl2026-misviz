from typing import Tuple, List
from collections import defaultdict
import json, re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


def _strip_code_fence(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", str(text), flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else str(text).strip()


def _extract_json_array(text):
    if "[" not in text or "]" not in text:
        return text
    start, end = text.find("["), text.rfind("]") + 1
    return text[start:end]


def _to_rect_from_points(coords):
    """
    Accepts:
      - [[x,y], [x,y], ...]
      - [[x1,y1,x2,y2], ...]   (line segments)
      - mixtures of both

    Returns (xmin, ymin, xmax, ymax)
    """
    xs, ys = [], []

    for c in coords:
        if not isinstance(c, (list, tuple)):
            continue

        # Case 1: point
        if len(c) == 2:
            x, y = c
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))

        # Case 2: line segment
        elif len(c) == 4:
            x1, y1, x2, y2 = c
            if all(isinstance(v, (int, float)) for v in c):
                xs.extend([float(x1), float(x2)])
                ys.extend([float(y1), float(y2)])

    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Not enough points to form a rectangle")

    return min(xs), min(ys), max(xs), max(ys)


def extract_rects(predicted_str):
    """
    Parse model output and return a list of rectangles as (xmin, ymin, xmax, ymax).
    Accepts:
      - {"coordinates": [[xmin,ymin],[xmax,ymax]]}
      - {"coordinates": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
    Ignores any extra fields or trailing prose.
    """
    text = _strip_code_fence(predicted_str)
    arr = _extract_json_array(text)

    rects: List[Tuple[float, float, float, float]] = []

    try:
        data = json.loads(arr)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "coordinates" in item:
                    coords = item["coordinates"]
                    if isinstance(coords, list) and len(coords) >= 2:
                        rects.append(_to_rect_from_points(coords))
    except Exception:
        # optional fallback
        pattern = re.compile(
            r'\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
            r'(?:\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]){1,}\s*\]',
            flags=re.DOTALL
        )
        for m in pattern.finditer(text):
            nums = list(map(float, [n for n in m.groups() if n is not None]))
            # rebuild pairs
            pairs = [list(nums[i:i+2]) for i in range(0, len(nums), 2)]
            if len(pairs) >= 2:
                rects.append(_to_rect_from_points(pairs))

    return rects


def iou(box_a,
        box_b):
    """
    Compute IoU between two bbox
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def gt_bbox_to_rect(b):
    xmin = float(b["x"])
    ymin = float(b["y"])
    xmax = xmin + float(b["width"])
    ymax = ymin + float(b["height"])
    return xmin, ymin, xmax, ymax


def index_ground_truth(gt_list):
    gt_index = defaultdict(list)

    for g in gt_list:
        if "bbox" not in g or not g["bbox"]:
            continue
        for b in g["bbox"]:
            gt_index[g["image_path"]].append(gt_bbox_to_rect(b))

    return gt_index


def match_boxes(pred_rects, gt_rects):
    """
    Returns list of IoU scores for each predicted rect
    """
    scores = []

    for p in pred_rects:
        if not gt_rects:
            scores.append(0.0)
            continue

        best = 0.0
        for g in gt_rects:
            best = max(best, iou(p, g))
        scores.append(best)

    return scores


def match_boxes_with_gt(pred_rects, gt_rects):
    """
    Returns list of tuples:
    (pred_rect, best_gt_rect, best_iou)
    """
    matches = []

    for p in pred_rects:
        best_iou = 0.0
        best_gt = None
        for g in gt_rects:
            score = iou(p, g)
            if score > best_iou:
                best_iou = score
                best_gt = g
        matches.append((p, best_gt, best_iou))

    return matches


def evaluate_iou(predictions, ground_truth):
    gt_index = index_ground_truth(ground_truth)

    all_ious = []
    per_image = []

    for p in predictions:
        img_path = p["image_path"]

        if img_path not in gt_index:
            continue

        pred_rects = extract_rects(p["predicted_misleader"])
        if not pred_rects:
            continue

        gt_rects = gt_index[img_path]
        matches = match_boxes_with_gt(pred_rects, gt_rects)

        ious = [m[2] for m in matches]
        all_ious.extend(ious)

        per_image.append({
            "image_path": img_path,
            "num_pred_boxes": len(pred_rects),
            "mean_iou": sum(ious) / len(ious)
        })

    summary = {
        "num_predictions_total": len(predictions),
        "num_images_with_gt_bbox": len(gt_index),
        "num_images_evaluated": len(per_image),
        "num_pred_boxes_evaluated": len(all_ious),
        "mean_iou": sum(all_ious) / len(all_ious) if all_ious else 0.0,
    }

    return summary, per_image


def show_gt_vs_pred_with_iou(img_path, gt_rects, pred_rects):
    """
    Plot the visualization with ground truth and predicted bbox next to each other. Report the IoU score.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    matches = match_boxes_with_gt(pred_rects, gt_rects)
    mean_iou = sum(m[2] for m in matches) / len(matches) if matches else 0.0

    _, axes = plt.subplots(1, 2, figsize=(min(16, w/40), min(8, h/40)))

    # Ground truth
    axes[0].imshow(img)
    axes[0].set_title("Ground truth")
    for i, (x1, y1, x2, y2) in enumerate(gt_rects, 1):
        axes[0].add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor="red", facecolor="none")
        )
        axes[0].text(x1, y1 - 3, f"GT {i}",
                     color="white",
                     bbox=dict(facecolor="red", alpha=0.6, pad=1))
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(img)
    axes[1].set_title(f"Prediction (mean IoU = {mean_iou:.3f})")

    for i, (p, _, score) in enumerate(matches, 1):
        x1, y1, x2, y2 = p
        axes[1].add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor="lime", facecolor="none")
        )
        axes[1].text(
            x1, y1 - 3,
            f"P{i} IoU={score:.2f}",
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=1)
        )

    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_samples(predictions, ground_truth, n=5):
    gt_index = index_ground_truth(ground_truth)

    valid = [
        p for p in predictions
        if p["image_path"] in gt_index
        and extract_rects(p["predicted_misleader"])
    ]

    for p in valid[:n]:
        img_path = f"data/{p['image_path']}"
        gt_rects = gt_index[p["image_path"]]
        pred_rects = extract_rects(p["predicted_misleader"])

        show_gt_vs_pred_with_iou(img_path, gt_rects, pred_rects)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    parser.add_argument('--display_examples', type=int, default=0, help='Show some examples of visualizations with predicted bbox')
    parser.add_argumen('--n_examples', type=int, default=0, help='Number of examples to show')
    args = parser.parse_args()

    print('-----------------------------------------')
    print(f"{args.model} - misviz")
    ground_truth = json.load(open("data/misviz/misviz.json"))
    results = json.load(open(f"results/{args.model}/misviz_{args.split}_bbox.json", encoding="utf-8"))
    scores, _ = evaluate_iou(results, ground_truth)
    print(scores)
    if args.display_examples:
        visualize_samples(results, ground_truth, n=args.n_examples)
    print('-----------------------------------------')