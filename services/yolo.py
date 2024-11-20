import cv2
import numpy as np

def preprocess_image(image: np.ndarray, input_size: tuple):
    resized = cv2.resize(image, input_size)
    normalized = resized / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0).astype(np.float32)

def draw_boxes(image, boxes, confidences, class_ids, labels):
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        label = f"{labels[cls_id]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
