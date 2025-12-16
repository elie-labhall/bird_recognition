import time
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# -------- CONFIG --------
MODEL_PATH = "best_int8_2.tflite"
VIDEO_SOURCE = "birds2.mp4"   # or 0 for webcam
CONF_THRESH = 0.30
IOU_THRESH = 0.3
# ------------------------


import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)


def xywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


# --- Load TFLite model ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_h = input_details[0]["shape"][1]
in_w = input_details[0]["shape"][2]

# --- Open video source ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("ERROR: Cannot open video source")
    exit()

print("Running bird detection... Press Q to exit")

frame_count = 0
t0 = time.time()

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # -------- PREPROCESS --------
    img = cv2.resize(frame, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # -------- INFERENCE --------
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]
    # out shape is typically (84, 8400) -> transpose to (8400, 84)
    if out.ndim == 3:
        out = out[0]
    out = out.T  # (num_anchors, 4 + num_classes)

    # -------- DECODE YOLOv8 OUTPUT --------
    boxes = out[:, :4]          # cx, cy, w, h (normalized 0–1)
    scores_all = out[:, 4:]     # class scores

    # one-class model → max over classes is just bird score
    scores = scores_all.max(axis=1)
    class_ids = scores_all.argmax(axis=1)

    # keep only confident detections
    mask = scores >= CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes.size > 0:
        # scale boxes to original image size
        boxes[:, 0] *= w  # cx
        boxes[:, 1] *= h  # cy
        boxes[:, 2] *= w  # w
        boxes[:, 3] *= h  # h

        boxes_xyxy = xywh_to_xyxy(boxes)

        # OpenCV NMS needs [x, y, w, h]
        boxes_xywh = np.zeros_like(boxes_xyxy)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        b_list = boxes_xywh.tolist()
        s_list = scores.tolist()

        indices = cv2.dnn.NMSBoxes(b_list, s_list, CONF_THRESH, IOU_THRESH)

        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, bw, bh = b_list[i]
                score = s_list[i]

                x1 = int(max(x, 0))
                y1 = int(max(y, 0))
                x2 = int(min(x + bw, w - 1))
                y2 = int(min(y + bh, h - 1))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"bird {score:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # -------- FPS --------
    fps = 1.0 / (time.time() - t1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Bird Detector (TFLite YOLOv8)", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Average FPS over the whole run
t_total = time.time() - t0
if frame_count > 0 and t_total > 0:
    print(f"Average FPS: {frame_count / t_total:.2f}")

cap.release()
cv2.destroyAllWindows()
