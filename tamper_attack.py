import cv2

cap = cv2.VideoCapture("normal.mp4")
out = cv2.VideoWriter("tampered.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame[50:150,50:150] = 0  # overlay attack
    out.write(frame)

cap.release()
out.release()
