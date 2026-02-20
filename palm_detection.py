import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            landmarks = hand_landmarks.landmark
            hand_label = hand_info.classification[0].label  # Left or Right

            finger_count = 0

            # ---- THUMB LOGIC (HAND DEPENDENT) ----
            if hand_label == "Right":
                if landmarks[4].x < landmarks[3].x:
                    finger_count += 1
            else:  # Left hand
                if landmarks[4].x > landmarks[3].x:
                    finger_count += 1

            # ---- OTHER 4 FINGERS ----
            for tip in finger_tips[1:]:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    finger_count += 1

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Display
            cv2.putText(
                frame,
                f"{hand_label} Hand: {finger_count}",
                (30, 80 if hand_label == "Left" else 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

    cv2.imshow("Finger Number Detection (Both Hands)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()