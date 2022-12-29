import gdown 

url_list = [
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_1.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_2.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_3.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_4.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_5.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_6.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_7.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_8.mp4",
    "https://km-video-stream.s3.amazonaws.com/shubham/football_clips/save_9.mp4"]

for url in url_list:
    output = "/home/omno/ammara/shot_detection1/videos/save/" + url.split('/')[-1]
    gdown.download(url, output, quiet=False)
