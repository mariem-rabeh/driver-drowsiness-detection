import cv2
import mediapipe as mp
import pygame
from ear_utils import get_average_ear

EAR_THRESHOLD = 0.2231
CONSEC_FRAMES_ALERT = 15
SOUND_PATH = "reports/alert_sound.mp3"

mp_face_mesh = mp.solutions.face_mesh

pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)


def main():
    cap = cv2.VideoCapture(0)
    closed_frame_counter = 0
    alert_active = False

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=True, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status_text = "No face detected"
            status_color = (128, 128, 128)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear = get_average_ear(landmarks, w, h)

                if ear < EAR_THRESHOLD:
                    closed_frame_counter += 1
                else:
                    closed_frame_counter = 0
                    if alert_active:
                        alert_sound.stop()
                    alert_active = False

                if closed_frame_counter >= CONSEC_FRAMES_ALERT and not alert_active:
                    alert_active = True
                    alert_sound.play(loops=-1)  # joue en boucle jusqu'a stop()

                if alert_active:
                    status_text = f"ALERTE SOMNOLENCE ! (EAR: {ear:.3f})"
                    status_color = (0, 0, 255)
                else:
                    status_text = f"Yeux ouverts (EAR: {ear:.3f})"
                    status_color = (0, 255, 0)
            else:
                # Pas de visage detecte : on arrete l'alerte par securite
                if alert_active:
                    alert_sound.stop()
                    alert_active = False

            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, status_color, 2)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    alert_sound.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()