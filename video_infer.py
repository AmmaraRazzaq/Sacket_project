import os

path = '/media/omno/New Volume/OMNO AI Projects/Yolov5_StrongSORT_OSNet_original/'

video_path = 'videos/'

os.chdir(path + video_path)

for file in os.listdir():
    print(file)
    os.chdir(path)
    print(path)
    # print('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights Yolov5m6-Best-High-Speed-1280.pt --imgsz 1280 --save-vid')
    # os.system('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights Yolov5m6-Best-High-Speed-1280.pt --imgsz 1280 --save-vid')

    if file == 'try1.mp4':


        print('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights Yolov5m6-Best-High-Speed-1280.pt --imgsz 1280 --save-vid')
        os.system('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights Yolov5m6-Best-High-Speed-1280.pt --imgsz 1280 --save-vid --show-vid')
        break

    # # if file == 'clip_1.mp4':
    # # os.system('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights ball_player_best.pt --save-vid --classes 1')
    # if file == 'test.mp4':
    # os.system('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights Yolov5m6-Best-High-Speed-1280.pt --imgsz 1280 --save-vid')
        # break