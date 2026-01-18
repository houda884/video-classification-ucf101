import cv2, os

VIDEO_DIR = "data/raw_videos"
OUT_DIR = "data/frames"
FRAME_STEP = 10

for cls in os.listdir(VIDEO_DIR):
    for video in os.listdir(f"{VIDEO_DIR}/{cls}"):
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{cls}/{video}")
        count, idx = 0, 0

        save_dir = f"{OUT_DIR}/{cls}/{video[:-4]}"
        os.makedirs(save_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_STEP == 0:
                frame = cv2.resize(frame, (224,224))
                cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
                count += 1
            idx += 1
        cap.release()
