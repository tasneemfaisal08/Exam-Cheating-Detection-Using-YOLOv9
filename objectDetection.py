import cv2
import numpy as np

# Load YOLO model and class names
net = cv2.dnn.readNet('yolov3.weights', 'yolov3-416.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Read input video file
cap = cv2.VideoCapture(0)

# Get video frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video file format and codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detected_video.avi', fourcc, fps, (frame_width, frame_height))

# Process each frame
while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (640,640), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputLayersNames = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(outputLayersNames)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                centerX = int(detection[0] * width)
                centerY = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(centerX - w/2)
                y = int(centerY - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        if label == 'cell phone' or label == 'book':
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            continue
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 0, 0), 2)

    # Write the frame to the output video file
    out.write(img)

    # Display the frame
    cv2.imshow('Detected Video', img)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
