import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, c = img.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):

            label = hand_handedness.classification[0].label

            hand_landmarks = results.multi_hand_landmarks[idx]
            thumb_x = hand_landmarks.landmark[4].x
            pinky_x = hand_landmarks.landmark[20].x

            if label == "Right":
                if thumb_x > pinky_x:
                    posisi = "Punggung"
                else:
                    posisi = "Telapak"
                label_display = "KANAN"
            elif label == "Left":
                if thumb_x < pinky_x:
                    posisi = "Punggung"
                else:
                    posisi = "Telapak"
                label_display = "KIRI"

            # Tentukan koordinat teks (ambil landmark 0)
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)

            cv2.putText(img, f"{label_display} - {posisi}",
                        (cx, cy - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 3)
            
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    cv2.imshow("Deteksi tangan depan belakang",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
