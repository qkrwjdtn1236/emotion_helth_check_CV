import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance


# # 모델 로드
# https://github.com/JustinShenk/fer  fer2013 pre-trained model
# https://github.com/Microsoft/FERPlus fer2013 dataset 
emotion_model = load_model('emotion_model.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# EAR 계산 함수
def compute_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 카메라 열기
cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for rect in faces:
        x, y, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        face_gray = gray[y:y2, x:x2]

        try:
            # 감정 분석
            face = cv2.resize(face_gray, (64, 64))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            preds = emotion_model.predict(face, verbose=0)[0]
            emotion_probability = np.max(preds)
            label = emotion_labels[np.argmax(preds)]
        except:
            continue

        # 얼굴 컬러 영역
        face_color = frame[y:y2, x:x2]

        try:
            # 피부색 분석 (LAB)
            lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            avg_l = np.mean(l)
            avg_a = np.mean(a)

            if avg_a > 145 and avg_l < 160:
                color_status = "Red Face"
            elif avg_l > 180 and avg_a < 135:
                color_status = "Pale Face"
            else:
                color_status = "Normal"

            # 유분/땀 분석 (HSV)
            forehead = face_color[0:int((y2 - y) * 0.25), :]
            hsv_forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
            _, _, v = cv2.split(hsv_forehead)
            highlight_mask = cv2.inRange(v, 200, 255)
            highlight_ratio = np.sum(highlight_mask > 0) / highlight_mask.size

            if highlight_ratio > 0.08:
                skin_status = "Oily/Sweaty Skin"
            else:
                skin_status = "Normal Skin"

        except:
            color_status = "Unknown"
            skin_status = "Unknown"

        try:
            # 피로도 분석 (눈 감김 + 눈썹 거리)
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            left_brow = landmarks[17:22]
            left_eye_center = np.mean(left_eye, axis=0)
            left_brow_center = np.mean(left_brow, axis=0)
            brow_eye_dist = np.linalg.norm(left_eye_center - left_brow_center)

            print(avg_ear, brow_eye_dist)
            if avg_ear < 0.25 and brow_eye_dist < 90:
                fatigue_status = "Tired"
            else:
                fatigue_status = "Not Tired"
        except:
            fatigue_status = "Unknown"

        # 시각화 출력
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {emotion_probability:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, color_status, (x, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, skin_status, (x, y2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Fatigue: {fatigue_status}", (x, y2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 전체 감정 확률
        for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Facial Expression + Health + Fatigue", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
