import re

def extract_class(answer):
    if answer == "yes":
        return "fire"
    if answer == "no":
        return "nofire"

def predict_binary_class(text_output):
    text = text_output.strip().lower()
    lead_text = text.lstrip(", ")

    if re.match(r"^(no|nope|nah)\b", lead_text):
        return "nofire"
    if re.match(r"^yes\b", lead_text):
        return "fire"

    nofire_patterns = [
        r"\bno fire\b",
        r"\bnofire\b",
        r"\bno flames?\b",
        r"\bno burning\b",
        r"\bno wildfire\b",
        r"\bno signs? of fire\b",
        r"\bnot a fire\b",
    ]
    fire_patterns = [
        r"\byes\b",
        r"\bfire\b",
        r"\bflames?\b",
        r"\bwildfire\b",
        r"\bblaze\b",
        r"\bburning\b",
        r"\bsmoke\b",
    ]

    if any(re.search(pattern, text) for pattern in nofire_patterns):
        return "nofire"
    if any(re.search(pattern, text) for pattern in fire_patterns):
        return "fire"

    # Fall back to the first token so short answers like "no"/"yes" still work.
    first_token = re.sub(r"^[^a-z]+|[^a-z]+$", "", lead_text.split(maxsplit=1)[0]) if text else ""
    if first_token in {"no", "nope"}:
        return "nofire"
    else:
        return "fire"
    
    return "no answer"


def compute_binary_metrics(targets, predictions):
    assert len(targets) == len(predictions)

    total = len(targets)
    correct = sum(t == p for t, p in zip(targets, predictions))
    accuracy = correct / total if total else 0.0

    def class_f1(positive_class):
        tp = sum((t == positive_class) and (p == positive_class) for t, p in zip(targets, predictions))
        fp = sum((t != positive_class) and (p == positive_class) for t, p in zip(targets, predictions))
        fn = sum((t == positive_class) and (p != positive_class) for t, p in zip(targets, predictions))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    f1_fire = class_f1("fire")
    f1_nofire = class_f1("nofire")

    return {
        "accuracy": accuracy,
        "f1_fire": f1_fire,
        "f1_nofire": f1_nofire,
        "f1_macro": (f1_fire + f1_nofire) / 2 if total else 0.0,
    }

def process_metrics(annotations, predictions):
    targets = []
    predictions_format = []
    for ann, pred in zip(annotations, predictions):
        print(f"Verify: {ann} \n")
        target_class = extract_class(ann["answer"][0])
        predicted_class = predict_binary_class(pred["text_output"])
        targets.append(target_class)
        predictions_format.append(predicted_class)
        pred["target_class"] = target_class
        pred["predicted_class"] = predicted_class
        pred["image_path"] = ann["image"]
    
    print(f"Targets: {targets} \n")
    #print(f"Predictions: {predictions_format} \n")
    metrics = compute_binary_metrics(targets, predictions_format)
    return metrics