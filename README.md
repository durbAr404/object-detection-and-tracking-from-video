# Object-detection-and-tracking-from-video
Multi-class object detection and tracking on a custom recorded video using a pre-trained Mask R-CNN model and IoU based frame to frame tracker, built with PyTorch, torchvision, and OpenCV.



## Pipeline Overview

1. **Video Ingestion** - The input video is loaded using OpenCV's `VideoCapture` and the total frame count is recorded for iteration

2. **Mask R-CNN Inference** - A pre-trained Mask R-CNN model (ResNet-50 FPN backbone, trained on MS COCO) is run on each frame to produce raw bounding boxes, integer class labels, and confidence scores for all 80 COCO object categories

3. **Detection Persistence** - Raw detections are saved to Google Drive as `.pt` files so that the expensive inference step does not need to be repeated across sessions

4. **Class Filtering** - Only detections whose class label belongs to the predefined set of target classes (`CLASSES_TO_TRACK`) are retained

5. **Confidence Filtering** - Each retained detection is compared against a per-class confidence threshold (`CONF_THRESH`); detections that fall below the threshold are discarded as false positives

6. **IoU-Based Tracking** - A greedy frame-to-frame IoU tracker associates filtered detections with existing active tracks; each class is tracked independently to prevent cross-class ID mixing

7. **Track Management** - Active tracks that successfully match a detection are updated with the new bounding box position; unmatched detections spawn new tracks with a unique integer ID; unmatched active tracks are dropped

8. **Output Video Generation** - The tracked objects are rendered onto each frame with colour-coded bounding boxes, class labels, track IDs, and confidence scores and written to `task3.mp4`



## About the Video

The input video was recorded in a public area in front of a shopping mall and contains a variety of MS COCO object categories including pedestrians, vehicles, and street-side objects. The video is **719 frames** in length.



## Tracked Object Classes

| Class | Confidence Threshold | Notes |
|---|---|---|
| `person` | 0.75 | Most frequently tracked object |
| `car` | 0.70 | Clearly distinguishable |
| `traffic light` | 0.45 | Frequently visible |
| `fire hydrant` | 0.55 | Stable detections |
| `dog` | 0.30 | Lowered due to orientation variation |
| `bird` | 0.30 | Lowered due to small size and movement |
| `backpack` | 0.50 | Worn by persons |
| `bench` | 0.20 | Difficult to classify correctly |
| `bicycle` | 0.60 | Occasional appearance |
| `motorcycle` | 0.65 | Occasional appearance |
| `stop sign` | 0.60 | Static object |
| `handbag` | 0.55 | Carried by persons |
| `chair` | 0.55 | Background object |
| `dining table` | 0.60 | Background object |



##  Key Design Decisions

### Detection Filtering
- A universal confidence threshold of `0.5` was used as a starting point
- Thresholds were tuned per class after inspecting detection quality on sample frames
- Classes like `dog`, `bird`, and `bench` required lower thresholds due to pose variation, small size, and background similarity
- High-salience classes (`person`, `car`, `fire hydrant`, `traffic light`) retained higher thresholds to minimise false positives

### IoU-Based Tracking
- Tracking is performed independently per class - no cross-class ID assignment is possible
- Greedy matching: each active track is paired with the highest-IoU detection of the same class
- IoU threshold was reduced from `0.5` to `0.3` after experimentation, improving matching stability for fast-moving objects
- Unmatched detections spawn new track IDs; unmatched tracks are dropped





## Results Summary

| Metric | Detail |
|---|---|
| Model | Mask R-CNN ResNet-50 FPN (COCO pre-trained) |
| Input Video Length | ~719 frames |
| Target Classes | 14 MS COCO categories |
| Tracking Method | Greedy IoU association (threshold = 0.3) |
| Most Stable Tracks | `person`, `car`, `fire hydrant`, `traffic light` |
| Partial Tracks |  `bird` (correct for brief intervals due to motion) |
| Most Challenging Class | `bench` (consistently difficult to classify correctly) |

---

## Findings

- Lowering the IoU threshold from `0.5` to `0.3` significantly improved bounding box matching for moving objects
- Classes with high visual salience and low movement (`fire hydrant`, `traffic light`) produced the most stable and consistent tracks
- Small or fast-moving objects (`person`,`car`, `bird`) were correctly detected for brief intervals but suffered from track interruptions due to rapid orientation changes
- The `bench` class remained problematic throughout, likely due to partial occlusion and background similarity
- Fine-tuning both confidence and IoU thresholds together yielded a noticeable improvement in overall tracking accuracy


## Technologies Used


* Python - Core programming language 
* PyTorch + torchvision - Mask R-CNN model and inference
* OpenCV - Video I/O, frame decoding, annotation 
* NumPy - Array operations and bounding box math 
* Matplotlib - In-notebook frame visualisation 
* Google Colab - Cloud GPU runtime environment 
