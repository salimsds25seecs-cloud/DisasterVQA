import json
import re
from typing import Dict, List
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

SYNONYMS = {
    "houses": ["house", "home", "homes", "residential building"],
    "roads": ["road", "street", "highway"],
    "vegetation": ["trees", "plants", "crops", "fields"],
}

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # simple plural handling
    if text.endswith("s"):
        text = text[:-1]
    return text

def normalize_object(obj: str) -> str:
    obj = normalize_text(obj)
    if obj.endswith("s"):
        obj = obj[:-1]
    return obj

def object_in_text(obj: str, text: str) -> bool:
    text = normalize_text(text)
    candidates = SYNONYMS.get(obj, [obj])

    for c in candidates:
        c_norm = normalize_object(c)
        if c_norm in text:
            return True
    return False

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def single_label_accuracy(gt, preds, field):
    y_true, y_pred = [], []

    for img, gt_data in gt.items():
        if img not in preds:
            continue

        y_true.append(normalize_text(gt_data.get(field, "")))
        y_pred.append(normalize_text(preds[img].get(field, "")))

    return accuracy_score(y_true, y_pred)

import re
from typing import Dict


def incident_prediction_accuracy(
    gt: Dict,
    preds: Dict,
    gt_field: str = "incident_class",
    pred_field: str = "incident_class",
) -> float:
    """
    Computes accuracy for incident prediction.

    Rules:
    - If prediction has <= 2 words → direct comparison
    - If prediction is a sentence → GT incident keyword must appear
      (supports variants like flood/flooding)
    """

    INCIDENT_VARIANTS = {
        "flood": ["flood", "flooding", "flooded"],
        "fire": ["fire", "burning", "burnt"],
        "earthquake": ["earthquake", "quake"],
        "cyclone": ["cyclone", "hurricane", "typhoon"],
        "landslide": ["landslide", "mudslide"],
        "traffic": ["traffic", "accident", "collision"],
    }

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    correct = 0
    total = 0

    for img, gt_data in gt.items():
        if img not in preds:
            continue

        gt_incident = normalize(gt_data.get(gt_field, ""))
        pred_text = normalize(preds[img].get(pred_field, ""))

        if not gt_incident or not pred_text:
            continue

        total += 1
        pred_words = pred_text.split()

        # Short prediction → strict comparison
        if len(pred_words) <= 2:
            if gt_incident in pred_words:
                correct += 1
            continue

        # Long sentence → keyword / variant search
        variants = INCIDENT_VARIANTS.get(gt_incident, [gt_incident])

        if any(v in pred_text for v in variants):
            correct += 1
        else:
            print(img)
            print(preds[img])

    return correct / total if total > 0 else 0.0

def damaged_objects_sentence_accuracy(
    gt: Dict,
    preds: Dict,
    object_field: str = "damaged_objects",
    pred_text_field: str = "scene_analysis",
) -> float:

    per_image_scores = []

    for img, gt_data in gt.items():
        if img not in preds:
            continue

        gt_objects: List[str] = gt_data.get(object_field, [])
        pred_sentence = preds[img].get(pred_text_field, "")

        if not gt_objects or not pred_sentence:
            continue

        pred_norm = normalize_text(pred_sentence)

        found = 0
        for obj in gt_objects:
            obj_norm = normalize_text(obj)
            if obj_norm in pred_norm:
                found += 1

        per_image_scores.append(found / len(gt_objects))

    return sum(per_image_scores) / len(per_image_scores) if per_image_scores else 0.0

def bleu_score(gt: Dict, preds: Dict):
    smoothie = SmoothingFunction().method4
    scores = []

    for img, gt_data in gt.items():
        if img not in preds:
            continue

        ref = normalize_text(gt_data.get("scene_analysis", "")).split()
        hyp = normalize_text(preds[img].get("scene_analysis", "")).split()

        if ref and hyp:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=smoothie))

    return sum(scores) / len(scores) if scores else 0.0

def evaluate(gt, preds, bf_training):

    results = {}

    # Structured fields
    if bf_training:
        results["incident_class_accuracy"] = incident_prediction_accuracy(gt, preds, "incident_class")
    else:
        results["incident_class_accuracy"] = single_label_accuracy(gt, preds, "incident_class")
    results["damage_level_accuracy"] = single_label_accuracy(gt, preds, "damage_level")
    results["human_presence_accuracy"] = single_label_accuracy(gt, preds, "human_presence")

    # Damaged objects (string-based)
    results["damaged_objects"] = damaged_objects_sentence_accuracy(gt, preds)

    # Scene text quality
    results["scene_analysis_bleu"] = bleu_score(gt, preds)

    return results


if __name__ == "__main__":
    GT_PATH = "Disaster_VQA/val.json"
    PRED_PATH = "after_training.json"

    bf_training = True

    with open(GT_PATH, 'r') as file:
        gt_data = json.load(file)

    with open(PRED_PATH, 'r') as file:
        pred_objects = json.load(file)

    pred_data = {}
    for idx, (key, _) in enumerate(gt_data[0].items()):
        pred_data[key] = pred_objects[idx]

    results = evaluate(gt_data[0], pred_data, bf_training)

    print("\n=== Moondream Evaluation Results ===\n")

    print(f"Incident Class Accuracy : {results['incident_class_accuracy']:.4f}")
    print(f"Damage Level Accuracy   : {results['damage_level_accuracy']:.4f}")
    print(f"Human Presence Accuracy : {results['human_presence_accuracy']:.4f}\n")

    print(f"Damaged Objects Accuracy: {results["damaged_objects"]:.4f}")

    print(f"Scene Analysis BLEU     : {results['scene_analysis_bleu']:.4f}")
