# import cv2
# import dlib
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# emotion_model = load_model('emotion_model.hdf5',compile=False)
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # 얼굴 탐지기
# detector = dlib.get_frontal_face_detector()

# # 웹캠 또는 이미지 사용
# cap = cv2.VideoCapture(2)  # 또는 'image.jpg'

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     print(faces)

#     for rect in faces:
#         x, y, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
#         face = gray[y:y2, x:x2]
#         try:
#             face = cv2.resize(face, (64, 64, 1))

#             # 얼굴 이미지 크기를 64x64로 리사이즈 (모델이 요구하는 입력 크기)
#             face = cv2.resize(face, (64, 64))

            
#         except Exception as e:
#             print(f"예측 중 오류 발생: {e}")
#             continue
#         # 전처리
#         face = face.astype("float") / 255.0
#         face = np.expand_dims(face, axis=-1)  # (64, 64, 1)로 채널 차원 추가
#         face = np.expand_dims(face, axis=0)   # (1, 64, 64, 1)로 배치 차원 추가

#         # 감정 예측
#         preds = emotion_model.predict(face, verbose=0)[0]
#         emotion_probability = np.max(preds)
#         label = emotion_labels[np.argmax(preds)]


#         print(preds)
#         # 결과 출력
#         cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(frame, f"{label}: {emotion_probability:.2f}", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

#         # 모든 감정 확률 출력 (원한다면)
#         for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
#             text = f"{emotion}: {prob:.2f}"
#             cv2.putText(frame, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
#     cv2.imwrite('output.jpg',frame)
#     cv2.imshow("Facial Expression", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import AVFoundation


# # 모델 로드
# https://github.com/JustinShenk/fer  fer2013 pre-trained model
# https://github.com/Microsoft/FERPlus fer2013 dataset 
emotion_model = load_model('emotion_model.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 얼굴 탐지기
detector = dlib.get_frontal_face_detector()

# 카메라
cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 실시간 분석 시작
while True:
    ret, frame = cap.read() # 카메라 정보
    if not ret: # 이미지 못불러온다면
        break # 멈춤

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 이미지 표정 감지 전처리
    faces = detector(gray) # 얼굴 인식 처리

    for rect in faces:
        x, y, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom() # 얼굴 위치 추출

        # 얼굴 크롭 (감정 분석용)
        face_gray = gray[y:y2, x:x2] # 얼굴 인식범위 ROI 처리
        try: # 이미지 후처리, 얼굴 인식 못하면 에러 나기에 
            face = cv2.resize(face_gray, (64, 64))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            # 감정 예측
            preds = emotion_model.predict(face, verbose=0)[0]
            emotion_probability = np.max(preds)
            label = emotion_labels[np.argmax(preds)]

        except Exception as e:
            print(f"감정 예측 오류: {e}")
            continue

        # 얼굴 크롭 (색상/피부 분석용)
        face_color = frame[y:y2, x:x2]
        try:
            # LAB 색 채널을 이용한 붉음, 창백함 측정
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
            forehead = face_color[0:int((y2 - y) * 0.25), :]  # 이마 영역
            hsv_forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_forehead)

            highlight_mask = cv2.inRange(v, 200, 255)
            highlight_ratio = np.sum(highlight_mask > 0) / highlight_mask.size

            if highlight_ratio > 0.08:
                skin_status = "Oily/Sweaty Skin"
            else:
                skin_status = "Normal Skin"

        except Exception as e:
            color_status = "Unknown"
            skin_status = "Unknown"
            print(f"피부 분석 오류: {e}")

        # 분석 결과 시각화 출력
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {emotion_probability:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, color_status, (x, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, skin_status, (x, y2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 감정 전체 확률
        for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Facial Expression + Health Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

