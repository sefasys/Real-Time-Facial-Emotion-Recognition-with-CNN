import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm  # İlerleme çubuğu için (pip install tqdm)

# --- AYARLAR ---
SOURCE_DIR = "Data"              # Orijinal verinin olduğu klasör
TARGET_DIR = "Processed_Data"    # İşlenmiş verinin kaydedileceği yer
TARGET_COUNT = 12000             # Her sınıf için hedeflenen sayı
IMG_SIZE = (48, 48)              # CNN için resim boyutu (Genelde 48x48 kullanılır)

# Sınıf isimleri (Klasör isimlerinle birebir aynı olmalı)
# Not: Klasör isimlerinde büyük/küçük harf duyarlılığına dikkat et.
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise"] 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(img_path):
    """Resmi okur, griye çevirir ve yeniden boyutlandırır."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        return resized
    except Exception as e:
        print(f"Hata oluştu: {img_path} - {e}")
        return None

def augment_image(img):
    """Resme rastgele augmentasyon uygular."""
    rows, cols = img.shape
    
    # 1. Rastgele Yatay Çevirme (Flip)
    if random.choice([True, False]):
        img = cv2.flip(img, 1)

    # 2. Rastgele Döndürme (-15 ile +15 derece arası)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    # 3. Rastgele Parlaklık/Gürültü (Opsiyonel basit bir noise)
    noise = np.random.randint(-10, 10, (rows, cols), dtype='int16')
    img = np.clip(img + noise, 0, 255).astype('uint8')

    return img

def main():
    create_dir(TARGET_DIR)
    
    for class_name in CLASSES:
        class_path = os.path.join(SOURCE_DIR, class_name)
        target_class_path = os.path.join(TARGET_DIR, class_name)
        create_dir(target_class_path)
        
        # Klasördeki tüm resimleri listele
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(images)
        
        print(f"--- Sınıf: {class_name} | Mevcut: {current_count} | Hedef: {TARGET_COUNT} ---")
        
        # --- DURUM 1: FAZLA VERİ VARSA (Downsampling) ---
        if current_count > TARGET_COUNT:
            print(f"Fazla veri var. {current_count - TARGET_COUNT} adet silinecek (seçilmeyecek).")
            selected_images = random.sample(images, TARGET_COUNT)
            
            for img_name in tqdm(selected_images, desc=f"{class_name} İşleniyor"):
                img_path = os.path.join(class_path, img_name)
                processed_img = process_image(img_path)
                if processed_img is not None:
                    cv2.imwrite(os.path.join(target_class_path, img_name), processed_img)

        # --- DURUM 2: EKSİK VERİ VARSA (Augmentation) ---
        else:
            needed = TARGET_COUNT - current_count
            print(f"Eksik veri var. {needed} adet sentetik veri üretilecek.")
            
            # Önce mevcutların hepsini işle ve kaydet
            loaded_images = [] # Augmentation için hafızada tut
            for img_name in tqdm(images, desc=f"{class_name} Orijinaller Kaydediliyor"):
                img_path = os.path.join(class_path, img_name)
                processed_img = process_image(img_path)
                if processed_img is not None:
                    save_path = os.path.join(target_class_path, img_name)
                    cv2.imwrite(save_path, processed_img)
                    loaded_images.append(processed_img)
            
            # Şimdi eksik kısım için augmentation yap
            if loaded_images:
                count = 0
                pbar = tqdm(total=needed, desc=f"{class_name} Augmentation Yapılıyor")
                while count < needed:
                    # Rastgele bir resim seç
                    base_img = random.choice(loaded_images)
                    aug_img = augment_image(base_img)
                    
                    # Yeni isimle kaydet
                    new_name = f"aug_{count}_{random.randint(1000,9999)}.jpg"
                    cv2.imwrite(os.path.join(target_class_path, new_name), aug_img)
                    
                    count += 1
                    pbar.update(1)
                pbar.close()

    print("\nPreprocessing tamamlandı! Yeni veri seti: 'Processed_Data' klasöründe.")

if __name__ == "__main__":
    main()