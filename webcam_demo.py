from modeling.MTCNN import MTCNN
import torch
import cv2
import os

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    img_size=160, margin=0, min_face_size=20,
    thresholds=[0.8, 0.9, 0.95], factor=0.709, post_process=True,
    device=device
)

c = 0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        timeF = 1  # 每一帧检测一次
        if c % timeF == 0:
            boxes, probs, points = mtcnn.detect(frame[:, :, ::-1].copy(), landmarks=True)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
            if points is not None:
                for point in points:
                    # print(point)
                    for po in point:
                        cv2.circle(frame, (po[0], po[1]), 1, (0, 0, 255), 4)
        c += 1
        cv2.imshow('webcam', frame)
        #         out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord(
                'e'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
