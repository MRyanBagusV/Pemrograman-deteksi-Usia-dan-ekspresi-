import cv2
from deepface import DeepFace
from fer import FER
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def detect_face(frame):
    # Menggunakan Haar Cascade untuk mendeteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Mengembalikan wajah pertama yang terdeteksi
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w], faces[0]
    return None, None

def estimate_age(face):
    if face is not None:
        # Mengkonversi wajah dari BGR ke RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            # Menggunakan DeepFace untuk estimasi usia
            analysis = DeepFace.analyze(face_rgb, actions=['age'], enforce_detection=False)
            age = analysis[0]['age']
            return age
        except Exception as e:
            print(f"Error dalam analisis usia: {e}")
            return None
    return None

def detect_expression(face):
    if face is not None:
        # Menggunakan FER untuk mendeteksi ekspresi wajah
        detector = FER()
        emotion_analysis = detector.detect_emotions(face)
        if emotion_analysis:
            dominant_emotion = max(emotion_analysis[0]['emotions'], key=emotion_analysis[0]['emotions'].get)
            return dominant_emotion
    return None

def count_fingers(frame):
    # Menggunakan MediaPipe untuk mendeteksi tangan
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark tangan pada frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Menghitung jari berdasarkan posisi landmark (misalnya thumb up, index, middle, ring, pinky)
            # Ini adalah pendekatan sederhana. Anda mungkin ingin menggunakan logika lebih kompleks untuk menghitung jari
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            if landmarks:
                # Asumsi sederhana bahwa setiap landmark jari yang lebih tinggi dari landmark sebelumnya menunjukkan jari yang terangkat
                if landmarks[4].y < landmarks[3].y:  # Thumb
                    finger_count += 1
                if landmarks[8].y < landmarks[6].y:  # Index
                    finger_count += 1
                if landmarks[12].y < landmarks[10].y:  # Middle
                    finger_count += 1
                if landmarks[16].y < landmarks[14].y:  # Ring
                    finger_count += 1
                if landmarks[20].y < landmarks[18].y:  # Pinky
                    finger_count += 1

    return finger_count

def capture_and_display_age_expression_fingers():
    # Menginisialisasi kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        return

    print("Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break

        # Mendeteksi wajah
        face, bbox = detect_face(frame)
        if face is not None:
            # Estimasi usia
            age = estimate_age(face)
            if age is not None:
                cv2.putText(frame, f'Usia: {age} tahun', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Mendeteksi ekspresi wajah
            expression = detect_expression(face)
            if expression:
                cv2.putText(frame, f'Ekspresi: {expression}', (bbox[0], bbox[1] + bbox[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Ekspresi tidak terdeteksi', (bbox[0], bbox[1] + bbox[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Tidak ada wajah terdeteksi', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Menghitung jari
        finger_count = count_fingers(frame)
        cv2.putText(frame, f'Jumlah Jari: {finger_count}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Menampilkan video dari kamera dengan teks estimasi usia, ekspresi, dan jumlah jari
        cv2.imshow('Kamera dengan Estimasi Usia, Ekspresi, dan Jumlah Jari', frame)

        # Keluar dari loop saat 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_display_age_expression_fingers()
