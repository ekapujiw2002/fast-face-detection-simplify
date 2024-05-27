"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os
import time

import cv2
import numpy as np
import onnx

import box_utils_numpy_helper as box_utils

# onnx runtime
import onnxruntime as ort

# silence all the warning gibberish from onnxruntime
ort.set_default_logger_severity(3)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
    
def preprocess(img_in):
    orig_image = img_in.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image
    
def do_prediction(original_image, threshold=0.7):
    time_time = -time.time()

    orig_image = original_image.copy()
    image_preprocess = preprocess(orig_image)
    
    # feed it to the model
    confidences, boxes = ort_session.run(None, {input_name: image_preprocess})
    # print("Detection cost time:{} ms".format((time.time() - time_time)*1000))
    
    # make prediction
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    
    time_time += time.time()
    
    return boxes, labels, probs, (time_time*1000)

    
onnx_path = "models/version-RFB-320.onnx"
class_names = ["background","face"]

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
result_path = "./detect_imgs_results_onnx"

threshold = 0.7
path = "imgsx"
sum_face = 0

if not os.path.exists(result_path):
    os.makedirs(result_path)
    
    
if __name__ == "__main__":
    vidcam = cv2.VideoCapture(0)

    while True:
        is_success, frame_cam = vidcam.read()
        
        if is_success:
            orig_image = frame_cam.copy()
            image = preprocess(orig_image)
            
            boxes, labels, probs, duration = do_prediction(orig_image)
            print(f"Detection cost time: {duration:.3f} ms")
            
            cv2.putText(orig_image, f"Time = {duration:.3f} ms",
                    (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    2)  # line type
                    
            cv2.putText(orig_image, f"Face = {boxes.shape[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    2)  # line type
            
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.3f}"
                
                print(f"Got {label} at {box}")              

                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                
            cv2.imshow('Result', orig_image) 
            
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    # After the loop release the cap object 
    vidcam.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 