import cv2
from ultralytics import solutions
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('video_path', type=str)
parser.add_argument('output_path', type=str)
args = parser.parse_args()

video_path = args.video_path
cap = cv2.VideoCapture(video_path)
width, height, fps, fourcc = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(cap.get(cv2.CAP_PROP_FPS)),
    cv2.VideoWriter_fourcc(*'mp4v')
)

output_path = args.output_path
out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

# regions to track (whole frame)
region_points = [
    (0, 0),
    (width, 0),
    (0, height),
    (width, height)
]

# create counter object
counter = solutions.ObjectCounter(show=False, region=region_points, model='yolo11l.pt')

# process video
frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame  = counter.count(frame)
        out.write(frame)
        frame_idx += 1
    else:
        print(f'Error while handling frame {frame_idx}th')
        break

cap.release()
out.release()