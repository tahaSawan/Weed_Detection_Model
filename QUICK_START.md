# Quick Start - Fix Model Issues

## 🚀 Quick Fix (5 Steps)

### 1. Analyze Current Issues
```bash
python analyze_model_performance.py --model best.pt --images Sample-Dataset --find-problems --visualize
```

### 2. Capture Negative Examples (Persons, etc.)
```bash
python add_negative_examples.py --capture-webcam --output negative_examples/ --num 50
```
Show: persons, hands, tools (NOT weeds!)

### 3. Add to Dataset
```bash
python add_negative_examples.py \
    --dataset datasets/yolo_format_single_class \
    --negative-source negative_examples/ \
    --ratio 0.15
```

### 4. Retrain Model
```bash
python improve_model_training.py \
    --data datasets/yolo_format_single_class \
    --weights best.pt \
    --epochs 150 \
    --patience 25
```

### 5. Test New Model
```bash
python webcam_inference.py --weights runs/train/weed_improved/weights/best.pt --conf 0.3
```

---

## 📋 What Each Script Does

- **`analyze_model_performance.py`**: Shows what's wrong (false positives, missing weeds)
- **`add_negative_examples.py`**: Adds "not weed" examples (persons, backgrounds)
- **`improve_model_training.py`**: Trains with better settings (augmentation, hyperparameters)

---

## 💡 Key Points

1. **False Positives** → Add negative examples
2. **Missing Weeds** → Better training (higher resolution, more epochs)
3. **Test confidence** → Try 0.2-0.4 range

See `MODEL_IMPROVEMENT_GUIDE.md` for detailed instructions.

