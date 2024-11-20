import cv2
import numpy as np

def preprocess_image(image: np.ndarray, input_size: tuple):
    resized = cv2.resize(image, input_size)
    normalized = resized / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0).astype(np.float32)

def postprocess_output(outputs, orig_image, conf_threshold=0.5):
    boxes, scores, classes = [], [], []
    for output in outputs[0]:
        confidence = output[4]
        if confidence > conf_threshold:
            x, y, w, h = output[0:4]
            class_id = np.argmax(output[5:])
            score = output[4] * output[5 + class_id]
            boxes.append((x, y, w, h))
            scores.append(score)
            classes.append(class_id)

    h, w, _ = orig_image.shape
    boxes = [(int((x - w / 2) * w), int((y - h / 2) * h), int(w * w), int(h * h)) for x, y, w, h in boxes]
    return boxes, scores, classes

def draw_boxes(image, boxes, scores, classes, labels):
    for box, score, class_id in zip(boxes, scores, classes):
        x, y, w, h = box
        label = labels[class_id]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
