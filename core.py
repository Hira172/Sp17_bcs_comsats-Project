import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolo.utils import Load_Yolo_model, parameters_for_img_processing, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolo.configs import *
import time
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from CollusionDetection.Predictor import time_to_contact




def system(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = [], display_tm = False, realTime = True ):
    # Definition of the  deep sort parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb' # deep sort tensorflow pretrained model
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], [] #parameters for finding fps

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'MPEG') # defining video writer
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .avi

    NUM_CLASS = read_class_names(CLASSES) # reading coco classes in the form of key value
    num_classes = len(NUM_CLASS)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    # calculating parameters for img processing fucntion
    _, original_frame = vid.read()
    nw, nh, ih, iw, dh, dw = parameters_for_img_processing(np.copy(original_frame), [input_size, input_size])

    # colors for detection
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    detection_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    detection_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), detection_colors))
    # random.seed(0)
    random.shuffle(detection_colors) # to shuffle shades of same color
    # random.seed(None)

    newTime = 0
    prevTime = 0
    dummy_time = 1
    t3 = 0

    # loop for video
    while True:
        _, original_frame = vid.read() # _ is bool value for reading correctly or not
        # cv2.imshow("org",original_frame)

        prevTime = newTime
        newTime = time.time()

        # diamentions of original should be equal to the input_size
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size],nw, nh, ih, iw, dh, dw)
        # yolo.predict expect min_ndim=4, original frame ndim=3 means we need 4d array instead of 3d
        # adding 4th d in the start making shape (1, input_size, input_size, 3)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        pred_bbox = Yolo.predict(image_data) # keras predict funtion returns list


        t2 = time.time()
        # print("Time taken by yolo prediction:(in sec)",t2-t1)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)



        # need to optimize below functions
        # removing all the extra bounding boxes to make tracking faster
        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms') #final yolo detected boxes

        # draw yolo detections
        # yoloFrame = draw_bbox(original_frame, bboxes, detection_colors, NUM_CLASS)
        # cv2.imshow("yolo", yoloFrame)

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []


        #tracking
        for bbox in bboxes: #loop to sperate the bounding boxes in the frames
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                x = bbox[0].astype(int)
                y = bbox[1].astype(int)
                w = bbox[2].astype(int)
                h = bbox[3].astype(int)
                scoreVal = bbox[4]
                class_id = bbox[5].astype(int)
                boxes.append([x, y, w-x, h-y])
                # boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(scoreVal)
                label = NUM_CLASS[class_id]
                names.append(label)

        # cv2.imshow("yoloFrame",yoloFrame)

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        # create deep sort object for detection
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature
                      in zip(boxes, scores, names, features)]


        if realTime:
            tracked_bboxes = time_to_contact(original_frame, tracker.matchedBoxes, newTime, prevTime,
                                             key_list, val_list, display_tm=display_tm)
        else:
            tracked_bboxes = time_to_contact(original_frame, tracker.matchedBoxes, dummy_time,
                                             dummy_time - 0.01666666666, key_list, val_list, display_tm=display_tm)

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # print("Count of tracked objects:",len(tracker.tracks))

        # Obtain info from the tracks
        # tracked_bboxes = []
        # for track in tracker.tracks:
        # for track in tracker.tracks:
        #     # if not track.is_confirmed() or track.time_since_update > 5:
        #     #     continue
        #     bbox = track.to_tlbr() # Get the corrected/predicted bounding box
        #     class_name = track.get_class() #Get the class name of particular object
        #     tracking_id = track.track_id # Get the ID for the particular track
        #     index = key_list[val_list.index(class_name)] # Get predicted object index by object name
        #     tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, detection_colors, NUM_CLASS, tracking=True)


        # calculating fps
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        # image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('Tracked', image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            
    cv2.destroyAllWindows()

