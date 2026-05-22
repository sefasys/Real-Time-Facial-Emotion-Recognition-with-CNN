import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- AYARLAR ---
MODEL_PATH = "emotion_model.h5"
IMG_SIZE = (48, 48)
CONFIDENCE_THRESHOLD = 0.40  # Eğer güven %40'ın altındaysa 'Uncertain' de

# Sınıflar (Modelin çıkış sırasına göre)
CLASS_LABELS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

# Renk Kodları (OpenCV BGR formatında: Blue, Green, Red)
COLORS = {
    'Angry': (0, 0, 255),       # Kırmızı
    'Fear': (255, 0, 255),      # Mor (Magenta tonu daha net görünür)
    'Happy': (0, 255, 255),     # Sarı
    'Sad': (255, 0, 0),         # Mavi
    'Surprise': (0, 165, 255),  # Turuncu
    'Uncertain': (128, 128, 128)# Gri
}

def main():
    print("Model yükleniyor...")
    try:
        model = load_model(MODEL_PATH)
        print("Model başarıyla yüklendi!")
    except Exception as e:
        print(f"HATA: Model yüklenemedi. {e}")
        return

    # Yüz tanıma için Haar Cascade
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Kamerayı Başlat
    cap = cv2.VideoCapture(0)

    print("Çıkış için 'q' tuşuna basınız.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aynalama (Kullanıcı kendini ayna gibi görsün diye)
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # ROI (İlgi Alanı) - Yüzü çıkar
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                # Preprocessing
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Tahmin
                prediction = model.predict(roi, verbose=0)[0]
                max_index = np.argmax(prediction)
                confidence = prediction[max_index]

                # --- RENK VE ETİKET MANTIĞI ---
                if confidence < CONFIDENCE_THRESHOLD:
                    label = "Uncertain"
                    color = COLORS['Uncertain']
                    conf_text = f"{int(confidence*100)}%" # Düşük de olsa oranını gör
                else:
                    label = CLASS_LABELS[max_index]
                    color = COLORS[label]
                    conf_text = f"{int(confidence*100)}%"

                # Kutuyu çiz
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Yazıyı yaz (Arkaplanlı yazı daha okunaklı olur)
                text = f"{label} ({conf_text})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Duygu Analizi', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()