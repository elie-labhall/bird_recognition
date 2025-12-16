import numpy as np
import cv2
import time
from tensorflow.lite.python.interpreter import Interpreter

# --- Model paths ---
TFLITE_MODEL_PATH = "efficientdet_lite0.tflite"
VIDEO_FILE_PATH = "birds.mp4"

# --- Detection parameters ---
INPUT_WIDTH, INPUT_HEIGHT = 320, 320
CONFIDENCE_THRESHOLD = 0.025
BIRD_CLASS_ID = 14  # COCO class ID for bird

# --- Load TFLite model ---
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    if floating_model:
        img = np.float32(img) / 255.0
    else:
        img = np.uint8(img)

    return img


def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    print("Loaded EfficientDet-Lite0")
    print("Press 'q' to exit")

    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        frame_h, frame_w, _ = frame.shape

        # ---- Inference ----
        inp = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()

        # EfficientDet TF-Hub Output Format:
        boxes   = interpreter.get_tensor(output_details[0]['index'])[0]  # (N,4)
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # (N,)
        scores  = interpreter.get_tensor(output_details[2]['index'])[0]  # (N,)
        count   = int(interpreter.get_tensor(output_details[3]['index'])[0])

        # ---- Post-processing ----
        for i in range(count):
            score = scores[i]
            class_id = int(classes[i])

            if score < CONFIDENCE_THRESHOLD:
                continue

            if class_id != BIRD_CLASS_ID:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            x_min = int(xmin * frame_w)
            y_min = int(ymin * frame_h)
            x_max = int(xmax * frame_w)
            y_max = int(ymax * frame_h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"bird {int(score * 100)}%",
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # ---- FPS ----
        t2 = time.time()
        fps = 1 / (t2 - t1)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (frame_w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        cv2.imshow("EfficientDet-Lite0 Bird Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection(VIDEO_FILE_PATH)
