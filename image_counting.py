import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
# from PIL import Image   # <-- optional: currently unused, remove if you don't need PIL

def read_image_to_bgr(path_or_bytes):
    """
    Read an image from disk path or bytes and return a BGR numpy array
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes))

    return img


def process_image(image_source, model: YOLO, classes=None, conf_thres=0.25):
    """
    Process a single image and return (annotated_rgb_numpy, counts_dict).

    - image_source: path to image or bytes
    - model: loaded ultralytics YOLO model
    - classes: List of class indices to keep (None = all)
    - conf_thres: confidence threshhold
    """

    frame = read_image_to_bgr(image_source)
    if frame is None:
        raise ValueError("Could not read image")

    # Run prediction on single image. Passing the numpy frame directly is supported.
    results = model.predict(source=frame, conf=conf_thres)
    counts = defaultdict(int)

    # annotate on a copy
    annotated = frame.copy()

    # Guard so we only access boxes if results exist and contain data
    if results and len(results) > 0 and getattr(results[0].boxes, "data", None) is not None:
        # NOTE: changed .xpu() -> .cpu() to bring tensors to CPU then to numpy
        boxes = results[0].boxes.xyxy.cpu().numpy()               # CHANGED
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        names = model.names

        for (x1, y1, x2, y2), cls_idx, conf in zip(boxes, class_indices, confidences):
            # filter by class if requested (cls_idx is already int)
            if classes and int(cls_idx) not in classes:
                continue

            cls_name = names[int(cls_idx)]
            counts[cls_name] += 1

            # draw box & label (ensure ints for drawing)
            x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{cls_name} {conf:.2f}",
                (x1_i, y1_i - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    # Convert BGR -> RGB for display in Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, dict(counts)
