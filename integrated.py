import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
import math
from math import *
# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


from sklearn.linear_model import RANSACRegressor

def centre(bbox):
    """bbox is an np array"""
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    center_x = int((p1[0]+p2[0])/2)
    center_y = int((p1[1]+p2[1])/2)
    return center_x, center_y


def ball_vs_line(l1, l2, pt):
    """ calculate on which side of the line the ball is"""
    # line
    x1 = l1[0]
    y1 = l1[1]
    x2 = l2[0]
    y2 = l2[1]

    slope = (y2-y1)/(x2-x1) 
    intercept = -slope*x1 + y1   

    lhs = pt[1]
    rhs = slope*pt[0]+intercept  

    if lhs > rhs:
        return True # above the line
    else:
        return False # below the line


def line_intersection(p1, p2, p3, p4):
    """ calculate intersection point of two lines
        line1: p1, p2
        line2: p3, p4
    """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]

    # calculate the intersecting point of two lines  
    # source: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    # line1: (x1,y1) (x2,y2)
    # line2: (x3,y3) (x4,y4)
    intersect_x = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    intersect_y = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    
    return intersect_x, intersect_y


def traj_line_intersection(intersect_pt, goal_corners):
    """ calculate whether the trajectory line intersection went through the goal post or through its right or left"""
    gp_left = goal_corners[0]
    gp_right = goal_corners[3]

    if intersect_pt[1] > gp_right[1] and intersect_pt[0] > gp_right[0]:
        cross_location = "right"
    elif intersect_pt[1] < gp_left[1] and intersect_pt[0] < gp_left[0]:
        cross_location = "left"
    elif intersect_pt[1] >= gp_left[1] and intersect_pt[1] <= gp_right[1] and intersect_pt[0] <= gp_right[0] and intersect_pt[0] >= gp_left[0]:
        cross_location = "inside"
    else:
        cross_location = "unknown"
    
    return cross_location


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def check(x1, y1, x2, y2, x3,
		y3, x4, y4, x, y):
	
	"""check if the ball is inside the goal post or not"""
	# Calculate area of rectangle ABCD
	A = (area(x1, y1, x2, y2, x3, y3) +
		area(x1, y1, x4, y4, x3, y3))

	# Calculate area of triangle PAB
	A1 = area(x, y, x1, y1, x2, y2)

	# Calculate area of triangle PBC
	A2 = area(x, y, x2, y2, x3, y3)

	# Calculate area of triangle PCD
	A3 = area(x, y, x3, y3, x4, y4)

	# Calculate area of triangle PAD
	A4 = area(x, y, x1, y1, x4, y4)

	# Check if sum of A1, A2, A3
	# and A4 is same as A
	return (A == A1 + A2 + A3 + A4)


def euclidean_distance(x1,y1,x2,y2):

    e = math.sqrt((y2-y1)**2 + (x2-x1)**2)

    return e


def slope(x1, y1, x2, y2):
    s = (y2-y1)/(x2-x1+ 0.00000000000000000000000000000000001)
    return s


def sort_corners(pts):
    if len(pts) >= 4:
        # sort the complete list by x in increasing order
        pts = sorted(pts, key=lambda x: x[0])
        # sort first 2 elements of the list by y in decreasing order
        pts_1 = sorted(pts[:2], key=lambda x: x[1], reverse=True) 
        # sort last 2 elements of the list by y in increasing order
        pts_2 = sorted(pts[2:], key=lambda x: x[1])
        # combine the two lists 
        new_list = pts_1+pts_2
        return new_list
    else:
        return pts


def last_line(goal_points):
    if len(goal_points) >= 4:
        pts_1 = goal_points[:2]
        pts_2 = goal_points[2:]

        p1 = pts_1[0]
        p2 = pts_2[1]

        theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])

        start_x = int(p1[0] + 2000*np.cos(theta))
        start_y = int(p1[1] + 2000*np.sin(theta))

        end_x = int(p1[0] - 2000*np.cos(theta))
        end_y = int(p1[1] - 2000*np.sin(theta))

        return [start_x, start_y, end_x, end_y]
    else:
        return []


def stabilization(gps, gp_queue, gp_queue1, im, im0, template, stab_gap):

    x_diff = 0
    y_diff = 0

    x_diff1 = 0
    y_diff1 = 0



    if gps is not None and len(gps):

        gps[:, :4] = scale_coords(im.shape[2:], gps[:, :4], im0.shape).round() 

        goal_points = []
        for gp in gps:  
            x1 = int ( gp[0].item() )
            y1 = int ( gp[1].item() )
            x2 = int ( gp[2].item() )
            y2 = int ( gp[3].item() )
            
            mid_x = int((x1+x2)/2)
            mid_y = int((y1+y2)/2)
            centre_pt = [mid_x, mid_y]
            goal_points.append(centre_pt)


            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2

            cls = int ( gp[5].item() )

            if cls == 0:
                pts = ( (x1+x2)//2, (y1+y2)//2 )

                if len(gp_queue) < 50: 
                    gp_queue.append(pts)
                else:
                    gp_queue.pop(0)
                    gp_queue.append(pts)

                if len(gp_queue) > stab_gap-1:
                    x_diff = gp_queue[-1][0] - gp_queue[-stab_gap][0]
                    y_diff = gp_queue[-1][1] - gp_queue[-stab_gap][1]

                    # cv2.putText(im0, f'dx: {x_diff}, dy: {y_diff}', (x2, y2), font, 
                    #                 fontScale, (0,0,0), thickness, cv2.LINE_AA)
                    
                    # cv2.putText(template, f'dx: {x_diff}, dy: {y_diff}', (x2, y2), font, 
                    #                 fontScale, (0,0,0), thickness, cv2.LINE_AA)
            
            elif cls == 1:
                pts = ( (x1+x2)//2, (y1+y2)//2 )

                if len(gp_queue1) < 50: 
                    gp_queue1.append(pts)
                else:
                    gp_queue1.pop(0)
                    gp_queue1.append(pts)

                if len(gp_queue1) > stab_gap-1:
                    x_diff1 = gp_queue1[-1][0] - gp_queue1[-stab_gap][0]
                    y_diff1 = gp_queue1[-1][1] - gp_queue1[-stab_gap][1]

                    # cv2.putText(im0, f'dx: {x_diff1}, dy: {y_diff1}', (x2, y2), font, 
                    #                 fontScale, (0,0,0), thickness, cv2.LINE_AA)
                    
                    # cv2.putText(template, f'dx: {x_diff1}, dy: {y_diff1}', (x2, y2), font, 
                    #                 fontScale, (0,0,0), thickness, cv2.LINE_AA)

            # rectangles around corner points of goal post
            im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), (0,0,0), 2)
            # template = cv2.rectangle(template, (x1, y1), (x2, y2), (0,0,0), 2)
            
        # sort goal points here to draw the polygon in the right shape
        goal_points = sort_corners(goal_points)
        pts = np.array(goal_points)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(template, [pts], isClosed=True, color=(0,0,0), thickness=4)

        # draw the last line
        line_pts = last_line(goal_points)
        if len(line_pts) == 4:
            cv2.line(template, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
    
    else:
        line_pts = []  
        goal_points = []

    avg_xdiff = (x_diff1 + x_diff)/2
    avg_ydiff = (y_diff + y_diff1)/2

    # line_pts are two points of the line
    # goal_points are 4 points of the goal post 
    return avg_xdiff, avg_ydiff, line_pts, goal_points


def fresh_start(queue):
 
    queue.clear()

    return 0


def remove_false_positives(queue, temp_cand, x_stab, y_stab):

    popped_element = 0

    if len(temp_cand):
        min = 100000
        final = []

        for cands in temp_cand:
            if cands[2] < min:
                final = cands
                min = cands[2]
        

        if len(queue) < 25:
            queue.append((final[0], final[1]))
        else:
            queue.pop(0)
            queue.append((final[0], final[1]))

        if len(queue)>1:
            x1 = queue[-2][0]
            y1 = queue[-2][1]
            x2 = queue[-1][0]
            y2 = queue[-1][1]

            x1 = x1 - x_stab
            y1 = y1 - y_stab

            x2 = x2 - x_stab
            y2 = y2 - y_stab
        
            return int(x1), int(y1), int(x2), int(y2) #, queue[-2][1], queue[-1][0], queue[-1][1]
        
        else:
            return queue[-1][0], queue[-1][1], queue[-1][0], queue[-1][1]
        
    else:
        return queue[-1][0], queue[-1][1], queue[-1][0], queue[-1][1]


def shot_detection(last_ball_x, last_ball_y, x_center, y_center, min_dist, last_f, frame_idx, im0, p, last_speed, traj_start, traj_end, shot_frame, shot_flag, segment_thres, acc_thres, template):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2


    e = euclidean_distance(last_ball_x, last_ball_y, x_center, y_center)
    print("-----")
    print("euclidean distance:",e)
    print('-----')

    if e > min_dist:

        print(frame_idx, last_f)

        speed = e/max((frame_idx - last_f), 1)
        acc = (speed - last_speed)/(frame_idx - last_f)
        last_speed = speed
 
        print("#"*30)
        print("speed: ", speed)
        print("acceleration:", acc)
        

        last_f = frame_idx


        print('#'*30)

        if shot_flag == False:
            if acc > acc_thres: 
                
                shot_flag = True

                cv2.putText(im0, f'speed: {speed: .2f}; Acceleration: {acc: .2f}; Shot/Pass ', (50,  36), font, 
                    fontScale, (0,0,0), thickness, cv2.LINE_AA)
                cv2.putText(template, f'speed: {speed: .2f}; Acceleration: {acc: .2f}; Shot/Pass ', (50,  36), font, 
                    fontScale, (0,0,0), thickness, cv2.LINE_AA)

                if traj_start == -1:
                    traj_start = (x_center, y_center)
                    shot_frame = frame_idx

                # cv2.imshow("-", template)
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(0)
                template = cv2.resize(template,(810,540),interpolation = cv2.INTER_LINEAR)
                im0 = cv2.resize(im0,(810,540),interpolation = cv2.INTER_LINEAR)
                output_canvas = np.concatenate((template, im0), axis=1)
                cv2.imshow('canvas',output_canvas)
                cv2.waitKey(0)  # 1 millisecond

     
        elif frame_idx - shot_frame > segment_thres:
            traj_end = (x_center, y_center)

            cv2.line(im0, traj_start, traj_end, (0,0,0), 3)
            cv2.line(template, traj_start, traj_end, (0,0,0), 3)

            traj_start = -1
            shot_flag = False

            # cv2.imshow("-", template)
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(0)
            template = cv2.resize(template,(810,540),interpolation = cv2.INTER_LINEAR)
            im0 = cv2.resize(im0,(810,540),interpolation = cv2.INTER_LINEAR)
            output_canvas = np.concatenate((template, im0), axis=1)
            cv2.imshow('canvas',output_canvas)
            cv2.waitKey(1)  # 1 millisecond
            
    return traj_start, traj_end, shot_frame, shot_flag, last_speed, last_f, im0, template


def deflection_detection(last_ball_x, last_ball_y, x_center, y_center, min_dist, angle_thres, last_angle, p, im0, shot_flag, deflection_flag, traj_start, traj_end):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2


    e = euclidean_distance(last_ball_x, last_ball_y, x_center, y_center)
    
    if e > min_dist:
        # if shot_flag:
        #     if deflection_flag != True:
        #         deflection_flag = True

        s = slope(last_ball_x, last_ball_y, x_center, y_center)
        angle = degrees(atan(s))
        angle_diff = abs(angle - last_angle)
        last_angle = angle

        print("angle:", angle)
        print("angle change:", angle_diff)

        if angle_diff > angle_thres: 
            
            cv2.putText(im0, f'angle: {angle: .2f}; angle change: {angle_diff: .2f};', (50,  72), font, 
                fontScale, (0,0,0), thickness, cv2.LINE_AA)

            #import pdb; pdb.set_trace()
            #cv2.imshow(str(p), im0)
            #cv2.waitKey(0)

            # else:

            #     s = slope(last_ball_x, last_ball_y, x_center, y_center)
            #     angle = degrees(atan(s))
            #     angle_diff = abs(angle - last_angle)
            #     last_angle = angle

            #     print("angle:", angle)
            #     print("angle change:", angle_diff)

            #     if angle_diff > angle_thres: 
                    
            #         traj_end = (x_center, y_center)

            #         im0 = cv2.line(im0, traj_start, traj_end, (255,0,255), 3)

            #         traj_start = -1
            #         shot_flag = False
            #         deflection_flag = False

            #         # import pdb; pdb.set_trace()
            #         cv2.imshow(str(p), im0)
            #         cv2.waitKey(0)


        

    return last_angle, deflection_flag, shot_flag, traj_start #, traj_end, traj_start


def RanSac(queue):
    x_train = []
    y_train = []

    for pts in queue:
        x_train.append(pts[0])
        y_train.append(pts[1])

    Ransac = RANSACRegressor()
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    

    Ransac.fit(x_train, y_train)
    inlier_mask = Ransac.inlier_mask_

    
    
    for i, pt in enumerate(queue):
        if not inlier_mask[i]:
            queue.remove(pt)
    
    

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        yolo_weights_2=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
):

    
    # queue_count = 0
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # import pdb; pdb.set_trace()

    # print(yolo_weights_2)

    # Load model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load second model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    model2 = DetectMultiBackend(yolo_weights_2, device=device, dnn=dnn, data=None, fp16=half)
    stride2, names2, pt2 = model2.stride, model2.names, model2.pt
    imgsz2 = check_img_size(imgsz, s=stride2)  # check image size


    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources


    # Second Dataloader 
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset2 = LoadStreams(source, img_size=imgsz2, stride=stride2, auto=pt2)
        nr_sources = len(dataset2)
    else:
        dataset2 = LoadImages(source, img_size=imgsz2, stride=stride2, auto=pt2)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources



    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources


    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    model2.warmup(imgsz=(1 if pt2 else nr_sources, 3, *imgsz2))  # warmup




    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources


    #-----------------------------
    queue = []

    gp_queue = []
    gp_queue1 = []

    last_ball_x = -1
    last_ball_y = -1

    min_dist = 0

    # pivot_points = []

    all_detections = []
    last_f = 0

    last_speed = 0

    traj_end = -1
    traj_start = -1

    threshold_false_positive = 360

    shot_frame = 0
    shot_flag = False

    segment_thres =  30

    acc_thres = 50

    stab_gap = 7

    # last_angle = 0
    # angle_thres = 50
    # deflection_flag = False

    cold_start_thres = 5
    frame_count = 0

    #trajectory_map = cv2.VideoWriter ( 'trajectory_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size) 

    #-----------------------------
    for path, im, im0s, vid_cap, s in dataset:
        # adding the template image here
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        break


    traj_points = []
    line_flags = []
    cross_locations = []
    goal = False
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        
        template = np.full((h, w, 3), (255,255,255), dtype='uint8')
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        #goal-post corners
        pred2 = model2(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        gps = pred2[0]

        # Process detections
        #if gps is not None and len(gps):

        for i, det in enumerate(pred):  # detections per image

        
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count


                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            annotator_t = Annotator(template, line_width=line_thickness, pil=not ascii)
            
            #if cfg.STRONGSORT.ECC:  # camera motion compensation
            #    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            ######################## goal post model ################################
                    
            x_stab, y_stab, line_pts, goal_points  = stabilization(gps, gp_queue, gp_queue1, im, im0, template, stab_gap)

            ######################## goal post model ################################                    

            if det is not None and len(det):

                frame_count = 0 
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy
                
                # pts = det[:, :4].cpu().detach().tolist() # corners of goal post
                # print("pts: ", pts)
                # draw a rectangle from these corners 
                # cv2.polylines(template, pts, isClosed=True, color=[0,0,255], thickness=2)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if gps is not None and len(gps) and len(outputs[i]) > 0:

                    # if len(outputs[i]) > 1 or len(det) >1:
                    #     import pdb; pdb.set_trace()
                    
                    temp_cand = []
                    cand_count = 0

                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
    
                        bboxes = output[0:4]
                        print(bboxes)

                        id = output[4]
                        cls = output[5]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if True: # save_vid or save_crop: #or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            # annotator.box_label(bboxes, label, color=colors(c, True))
                            # annotator_t.box_label(bboxes, label, color=colors(c, True))
                            
                            # draw bounding box around the ball
                            annotator.box_label(bboxes, label, color=(0,0,0))
                            annotator_t.box_label(bboxes, label, color=(0,0,0))

                            # centre point of ball bounding box
                            centre_x, centre_y = centre(bboxes)
                            pt = (centre_x, centre_y) 
                            traj_points.append(pt)
                            

                            print("len(line_pts): ", len(line_pts))
                            print("len(goal_points: ", len(goal_points))

                            # determine on which side of the line ball is 
                            if len(line_pts) == 4:
                                line_flag = ball_vs_line(line_pts[:2], line_pts[2:], pt)
                                line_flags.append(line_flag)
                                print("line_flag: ", line_flag)

                                # debug_img = np.full((h, w, 3), (255,255,255), dtype='uint8')
                                # pts = np.array(goal_points)
                                # pts = pts.reshape((-1, 1, 2))
                                # cv2.polylines(debug_img, [pts], isClosed=True, color=(0,0,0), thickness=4)
                                # cv2.line(debug_img, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
                                # cv2.circle(debug_img, (centre_x, centre_y), radius=2, color=(0,0,0), thickness=2)
                                # cv2.putText(debug_img, txt, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                # debug_img = cv2.resize(debug_img,(810,540),interpolation = cv2.INTER_LINEAR)
                                # cv2.imshow('debug_img', debug_img)
                                # cv2.waitKey(1)
                                 
                                if not line_flag and len(goal_points)==4:
                                    # the ball has gone on the other side of the line
                                    # check the intersection of the ball traj with the last line and if the intersection lies inside the goal post or not
                                    # trajectory line is being considered at an interval of 5 frames
                                    intersect_x, intersect_y = line_intersection(traj_points[-1], traj_points[-5], goal_points[0], goal_points[3]) 
                                    cross_location = traj_line_intersection((intersect_x,intersect_y), goal_points) 
                                    cross_locations.append(cross_location)
                                    
                                    if cross_location == 'right' or cross_location == 'left':
                                        print("it's a shot off target")
                                        text = "it's a shot off target"
                                        
                                    elif cross_location == 'inside':
                                        print("it is a Goal")
                                        text = "it is a Goal"
                                        # make sure the ball stays inside, for all future frames
                                        # check the intersection of the ball with the goal post
                                    else:
                                        print("unknown cross location")
                                        text = "unknown cross location"

                                    # check for goal
                                    # is_in_goalpost = check(goal_points[0][0], goal_points[0][1], goal_points[1][0], goal_points[1][1], goal_points[2][0], goal_points[2][1], goal_points[3][0], goal_points[3][1], centre_x, centre_y)
                                    # if is_in_goalpost:
                                    #     txt = "ball is inside the goalpost"
                                    # else:
                                    #     txt = "ball is not in the goalpost"
                                    
                                    # debugging, draw goal post and ball centre point
                                    # debug_img = np.full((h, w, 3), (255,255,255), dtype='uint8')
                                    # pts = np.array(goal_points)
                                    # pts = pts.reshape((-1, 1, 2))
                                    # cv2.polylines(debug_img, [pts], isClosed=True, color=(0,0,0), thickness=4)
                                    # cv2.line(debug_img, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
                                    # cv2.circle(debug_img, (centre_x, centre_y), radius=2, color=(0,0,0), thickness=2)
                                    # cv2.putText(debug_img, txt, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    # debug_img = cv2.resize(debug_img,(810,540),interpolation = cv2.INTER_LINEAR)
                                    # cv2.imshow('debug_img', debug_img)
                                    # cv2.waitKey(1)
                                    
                                    
                                    # if is_in_goalpost:
                                    #     text = "it is a Goal"
                                    
                                    # if is_in_goalpost or cross_location in ["right", "left", "inside"]:
                                    #     cv2.putText(im0, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    #     cv2.putText(template, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    
                                    cv2.putText(template, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    template = cv2.resize(template,(810,540),interpolation = cv2.INTER_LINEAR)
                                    im0 = cv2.resize(im0,(810,540),interpolation = cv2.INTER_LINEAR)
                                    output_canvas = np.concatenate((template, im0), axis=1)
                                    cv2.imshow('canvas',output_canvas)
                                    cv2.waitKey(0)

                            all_detections.append(bboxes)
                            #the function 

########################################### this right here ##################################################################

                            x_center = int((bboxes[0]+bboxes[2])/2)
                            y_center = int((bboxes[1]+bboxes[3])/2)
                                
                            last_ball_x = x_center
                            last_ball_y = y_center

                            # if len(outputs[i]) == 1:
                            #     if len(queue) < 25:
                            #         queue.append((x_center, y_center, id))
                            #     else:
                            #         queue.pop(0)
                            #         queue.append((x_center, y_center, id)) 
                            # else:
                            #     import pdb; pdb.set_trace()

                            if len(queue):
                                dist = euclidean_distance(queue[-1][0], queue[-1][1], x_center, y_center)

                                if dist < threshold_false_positive:
                                    temp_cand.append((x_center, y_center, dist))
                            
                            else:
                                temp_cand.append((x_center, y_center, 1))
                                
                                # for ball_multiple in outputs[i]:
                                #     # if euclidean_distance(outputs[i][0])

                    print("all_detections: ", all_detections)
                    
                    last_ball_x, last_ball_y, x_center, y_center = remove_false_positives(queue, temp_cand, x_stab, y_stab)

                    traj_start, traj_end, shot_frame, shot_flag, last_speed, last_f, im0, template = shot_detection(last_ball_x, last_ball_y, x_center, y_center, min_dist, last_f, frame_idx, im0, p, last_speed, traj_start, traj_end, shot_frame, shot_flag, segment_thres, acc_thres, template)

                    #last_angle, deflection_flag, shot_flag, traj_start = deflection_detection(last_ball_x, last_ball_y, x_center, y_center, min_dist, angle_thres, last_angle, p, im0, shot_flag, deflection_flag, traj_start, traj_end)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1 
                    thickness = 1


                    for l, tr in enumerate(queue):

                        if shot_flag:
                            cv2.circle(im0, (int (tr[0] -  x_stab), int(tr[1] - y_stab)), 4, (0, 0, 255), -1)
                            cv2.circle(template, (int (tr[0] -  x_stab), int(tr[1] - y_stab)), 4, (0, 0, 255), -1)
                        else:
                            cv2.circle(im0, (int (tr[0] -  x_stab), int(tr[1] - y_stab)), 4, (0, 0, 0), -1)
                            cv2.circle(template, (int (tr[0] -  x_stab), int(tr[1] - y_stab)), 4, (0, 0, 0), -1)
                    
                        if l== len(queue) - 1:
                            # cv2.putText(im0, f'True Ball', (tr[0], tr[1]), font, fontScale, (0,0,0), thickness, cv2.LINE_AA)
                            # cv2.putText(template, f'True Ball', (tr[0], tr[1]), font, fontScale, (0,0,0), thickness, cv2.LINE_AA)
                            pass
                                
                            
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                    
                    #last_f = frame_idx

########################################### this right here ##################################################################


                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                print(frame_count)
                frame_count = frame_count + 1

                if frame_count >= cold_start_thres:
                    frame_count = fresh_start(queue)
                    
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')
                #strongsort_list[i].increment_ages()


            # Stream results
            im0 = annotator.result()
            im_t = annotator_t.result()
            
            if show_vid:
                im_t = cv2.resize(im_t,(810,540),interpolation = cv2.INTER_LINEAR)
                im0 = cv2.resize(im0,(810,540),interpolation = cv2.INTER_LINEAR)
                output_canvas = np.concatenate((im_t, im0), axis=1)
                #cv2.imshow('template', im_t)
                #cv2.imshow(str(p), im0)
                cv2.imshow('canvas',output_canvas)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 15, im0.shape[1], im0.shape[0]
                    
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                    # create a new video writer and write video to that writer
                    save_template_path = "/home/omnoai/Desktop/ammara/Segmentation/shot_detection/template.mp4"
                    template_writer = cv2.VideoWriter(save_template_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                vid_writer[i].write(im0) 
                template_writer.write(im_t)

            prev_frames[i] = curr_frames[i]
    
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--yolo-weights-2', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)