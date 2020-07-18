
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

xr0=0
xl0=0
z0=1
f=1.9685   # noraml focal length (inches)
#calculate the distance
def calDist(xl,xr,w):
    global xr0,xl0,z0
    if xr0==0 and xl0==0:
        xr0=xr
        xl0=xl
        z0=1
        return 0
    if z0== 1:
        z0= w*f/(xr0-xl0)
        return 0
    return (((xr0-xl0)*z0)/(xr-xl))


#Loading video
cap = cv2.VideoCapture('Test1.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

prevDists=[]
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
        TCC=[]

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


                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # computing distance
                    dist = calDist(x,x+w,w)
                    dists.append(dist)


                    if len(dists) == len(prevDists):
                        for z1,zo in zip(dists,prevDists) :
                            if z1 == 0:
                                TCC.append(1000)
                                continue
                            S= z0/z1
                            if S == 1:
                                TCC.append(1000)
                                continue
                            Tm = 00.4/ (S-1)    # 0.4 comes from considering 24 frames per sec
                            print(Tm)
                            C = Tm + 1
                            T = Tm *((1-(( 1+2*C)**(1/2))) / C)
                            TCC.append(T)

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
                if i < len(TCC):
                    label = label+"  " + str(TCC[i])
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y-10), (x + w, y), color, -1)
                cv2.putText(frame, label, (x, y), font, 1, (255,255,255))

        prevDists=dists
        # print(TCC)

        #displaying frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

