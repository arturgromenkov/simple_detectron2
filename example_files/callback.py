import numpy as np
import cv2


# Example callback for detectrion-classification task your functions also MUST BE NAMED 'callback'
def callback(input_tensor, boxes, class_ids=None, confidences=None, class_names=None):
    img = input_tensor.cpu().numpy()

    # Convert to uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = map(int, boxes[i])

        # Draw rectangle
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)

        label_text = ""
        if class_ids is not None:
            class_id = int(class_ids[i])
            if class_names:
                label_text += class_names[class_id]
            else:
                label_text += str(class_id)
        
        if confidences is not None:
            conf = confidences[i]
            label_text += f" {conf:.2f}"

        # Put label above the box
        if label_text:
            y_text = max(y1 - 10, 0)
            cv2.putText(img_bgr,
                        label_text,
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        1,
                        cv2.LINE_AA)

    return img_bgr
