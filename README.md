

---

# 🚗 Lane & Object Detection using YOLO + OpenCV

This project detects **lanes** and **objects (vehicles, pedestrians, etc.)** in road videos using **YOLOv3** and **OpenCV**.
It also highlights decision regions (ego lane & right lane) to simulate a basic **collision avoidance system**.

---

## 📌 Features

* Lane detection using **Canny + Hough Transform**
* Object detection using **YOLOv3 pretrained on COCO dataset**
* Decision logic:

  * 🚦 **STOP** if object detected in ego lane
  * ➡️ **MOVE RIGHT** if right lane is free
  * ✅ **CLEAR** if no obstacle
* Configurable regions (polygons) for lane-based decision-making
* Real-time video display with bounding boxes and lane overlays

---

## 📂 Project Structure

```
.
├── lane_object_detection.py   # Main script
├── requirements.txt           # Python dependencies
├── yolov3.cfg                 # YOLOv3 config file (included)
├── coco.names                 # COCO class labels (included)
└── README.md                  # Documentation
```

⚠️ **Note:** The `yolov3.weights` file (\~248 MB) is **not included** due to GitHub’s 100MB file size limit.
You must **download it manually** (instructions below).

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/lane-object-detection.git
   cd lane-object-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv3 model files**

   * Download **YOLOv3 weights** (\~248 MB) from Kaggle:
     👉 [YOLOv3 Weights on Kaggle](https://www.kaggle.com/datasets/shivam316/yolov3-weights)
   * Extract the `.weights` file and place it in the project root folder (next to `yolov3.cfg`).

   ✅ `yolov3.cfg` and `coco.names` are already included in this repo.

---

## ▶️ Usage

Run the script with a video file path:

```bash
python lane_object_detection.py video.mp4
```

* Press **Q** to quit the window.
* Replace `video.mp4` with the path to your own dashcam/road video.

---

## 🎯 Running in Different Modes

By default, the script does **both lane + object detection**.
You can easily switch to **only lane detection** or **only object detection**:

### 🔹 Only Lane Detection

Inside `process_video()` in `lane_object_detection.py`, comment out YOLO detection lines:

```python
# boxes, confidences, class_ids, indexes = detect_objects_yolo(frame, net, output_layers)
# lane_frame = draw_labels(lane_frame, boxes, confidences, class_ids, indexes, class_names)
```

Now the script will **only detect lanes**.

### 🔹 Only Object Detection

Comment out the lane detection function call:

```python
# lane_frame = detect_lanes(frame)
```

Now the script will **only detect objects** with YOLO.

---

## 🟦 Modifying the Polygons (Regions of Interest)

The polygons (ego lane & right lane) are defined **relative to video width and height**, so they can be adjusted per video.

### Example from code:

```python
# Relative polygon coordinates (x_pct, y_pct)
ego_rel   = [(0.30, 1.00), (0.30, 0.50), (0.70, 0.50), (0.70, 1.00)]
right_rel = [(0.70, 1.00), (0.70, 0.50), (0.95, 0.50), (0.95, 1.00)]
```

* `x_pct` and `y_pct` are percentages of frame width/height.
* `(0.30, 1.00)` → 30% of frame width, 100% of frame height (bottom-left).
* `(0.70, 0.50)` → 70% of frame width, 50% of frame height (mid-right).

When the video resolution changes, the polygons **scale automatically** because they’re relative.

👉 To adjust:

* **Ego lane wider** → decrease first `x_pct` and increase last `x_pct`.
* **Shift polygons higher** → reduce `y_pct` values (closer to 0).
* **Cover more of right lane** → increase right polygon’s width (e.g., `0.95 → 1.0`).

This makes the system flexible across different dashcam or CCTV videos.

---

<img width="803" height="448" alt="screenshot-1722315397132" src="https://github.com/user-attachments/assets/2800a874-b6f8-4159-8879-1c927411e2f9" />


---

## 📌 Requirements

* Python 3.7+
* OpenCV
* NumPy

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

* Add **real-time webcam support**
* Improve lane detection with **deep learning (LaneNet / SCNN)**
* Integrate with **Kalman/DeepSORT tracking** for smoother results
* Config file for easy polygon adjustments

---

## 📜 License

This project is open-source under the **MIT License**.

---


