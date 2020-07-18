
import numpy as np
import cv2

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 100, size=(len(classes), 3))

#calculate the distance
def calDist(w,h):
    return (2 * 3.14 * 180) / (w + h * 360)*1000 + 3 #feet


#Loading video
cap = cv2.VideoCapture('Test1.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")


#reading video
while (cap.isOpened()):

    ret, frame = cap.read()
    height, width, channels = frame.shape
    if ret == True:
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        dists = []
        time = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    #computing distance

                    dist = calDist(w,h)
                    dists.append(dist)


                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)




                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                  # print(class_id)



        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + " " + str(round(dists[i],3))
                if i < len(time):
                    label = label+" "+str(round(time[i]))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y-10), (x + w, y), color, -1)
                cv2.putText(frame, label, (x, y), font, 1, (255,255,255))



        #displaying frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

