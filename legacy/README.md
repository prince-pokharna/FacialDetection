# Legacy Implementation Archive

> [!WARNING]
> The scripts in this directory are part of the original prototype and have been superseded by the production open-set Face Recognition Attendance system in `/ml`, `/backend`, and `/frontend`.

## Contents
- `model_train.py`: Closed-set 25-class Softmax classifier using MobileNetV2 (achieved ~60.39% val accuracy). Cannot enroll new identities without full retraining; cannot reliably reject unknown identities.
- `model_evaluation_improved.py`: Evaluation script for closed-set classifier.
- `webcam_integration.py`: OpenCV + Haar Cascade desktop script writing directly to local SQLite. Lacks liveness verification, web interface, and open-set handling.
- `preprocess.py`: Haar Cascade cropping to 128x128 px (conflicting with 160x160 px training input).
- `split.py`, `sub.py`, `subred.py`: Dataset subset and slicing scripts for VGGFace2 subset.
- `training_log.txt`: Raw execution log showing real measured accuracy (~60.39% best validation accuracy).
- `labels.json`: Fixed 25-class label map.
- `test_results/`: Sample static output images from legacy testing.

See [`../docs/AUDIT.md`](../docs/AUDIT.md) for full audit details and [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the new system design.
