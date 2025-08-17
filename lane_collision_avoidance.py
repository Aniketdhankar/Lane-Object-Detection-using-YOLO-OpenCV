

import cv2
import numpy as np

input_video_path = "add_your_video_path"

# -------------------------------
# Load YOLO model
# -------------------------------
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()

    # Handle different OpenCV versions
    try:
        unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
    except AttributeError:
        unconnected_out_layers = net.getUnconnectedOutLayers()

    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    return net, output_layers

# -------------------------------
# Load class names
# -------------------------------
def load_class_names():
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# -------------------------------
# YOLO Object Detection
# -------------------------------
def detect_objects_yolo(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                 (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w  = int(detection[2] * width)
                h  = int(detection[3] * height)
                x  = cx - w // 2
                y  = cy - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indexes = indexes.flatten() if len(indexes) > 0 else []
    return boxes, confidences, class_ids, indexes

# -------------------------------
# Draw bounding boxes
# -------------------------------
def draw_labels(frame, boxes, confidences, class_ids, indexes, class_names):
    for i in indexes:
        x, y, w, h = boxes[i]
        label = class_names[class_ids[i]]
        conf  = round(confidences[i], 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {conf}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# -------------------------------
# Lane detection (Hough Transform)
# -------------------------------
def detect_lanes(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred= cv2.GaussianBlur(gray, (5,5), 0)
    edges  = cv2.Canny(blurred, 50, 150)
    h, w   = edges.shape
    mask   = np.zeros_like(edges)
    poly   = np.array([[(0, h), (w//2, h//2), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    masked = cv2.bitwise_and(edges, mask)
    lines  = cv2.HoughLinesP(masked, 1, np.pi/180, 50,
                             minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    return frame

# -------------------------------
# Region masks (for decision logic)
# -------------------------------
def region_mask(frame_shape, polygon):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    return mask

def any_box_in_region(boxes, indexes, mask):
    for i in indexes:
        x, y, w, h = boxes[i]
        cx, cy = x + w//2, y + h//2
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            if mask[cy, cx] == 255:
                return True
    return False

# -------------------------------
# Main video processing
# -------------------------------
def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    net, output_layers = load_yolo_model()
    class_names        = load_class_names()

    ret, frame0 = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        return
    H, W = frame0.shape[:2]

    # Define regions of interest
    ego_rel   = [(0.30, 1.00), (0.30, 0.50), (0.70, 0.50), (0.70, 1.00)]
    right_rel = [(0.70, 1.00), (0.70, 0.50), (0.95, 0.50), (0.95, 1.00)]

    ego_poly   = [(int(x*W), int(y*H)) for x,y in ego_rel]
    right_poly = [(int(x*W), int(y*H)) for x,y in right_rel]

    ego_mask   = region_mask((H, W), ego_poly)
    right_mask = region_mask((H, W), right_poly)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lane detection + YOLO
        lane_frame = detect_lanes(frame)
        boxes, confidences, class_ids, indexes = detect_objects_yolo(
            frame, net, output_layers)
        lane_frame = draw_labels(lane_frame, boxes, confidences, class_ids,
                                 indexes, class_names)

        # Decision logic
        if any_box_in_region(boxes, indexes, ego_mask):
            action, color = "STOP", (0, 0, 255)
        elif not any_box_in_region(boxes, indexes, right_mask):
            action, color = "MOVE RIGHT", (0, 255, 0)
        else:
            action, color = "CLEAR", (255, 255, 255)

        # Overlay info
        cv2.polylines(lane_frame, [np.array(ego_poly, np.int32)], True, (255, 0, 0), 2)
        cv2.polylines(lane_frame, [np.array(right_poly, np.int32)], True, (0, 255, 255), 2)
        cv2.putText(lane_frame, action, (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

        cv2.imshow('Lane + Object Detection', lane_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example: python lane_object_detection.py video.mp4
    
     process_video(input_video_path)
    
    

