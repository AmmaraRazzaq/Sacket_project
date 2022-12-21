import os

path = '/media/omno/New Volume/OMNO AI Projects/Yolov5_StrongSORT_OSNet/'

video_path = 'videos/'

os.chdir(path + video_path)

for file in os.listdir():
    print(file)
    os.chdir(path)
    print(path)
    print('python track.py --source ' + video_path + file + ' --yolo-weights ball_player_best.pt --save-vid')
    # if file == 'clip_1.mp4':
    # os.system('python tracker_with_tracklets.py --source ' + video_path + file + ' --yolo-weights ball_player_best.pt --save-vid --classes 1')
    if file == 'input.mp4':
        # print('python tracker_with_tracklets_with_goal_post_points.py --source ' + video_path + file + ' --yolo-weights ball_player_best.pt --yolo-weights-2 goalpost_corners_latest.pt --save-vid --classes 1')
        os.system('python tracker_with_tracklets_with_goal_post_points.py --source ' + video_path + file + ' --yolo-weights ball_player_best.pt --yolo-weights-2 goalpost_corners_latest.pt --save-vid --classes 1')
        break