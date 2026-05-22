import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- AYARLAR ---
DATA_DIR = "Processed_Data"      # İşlenmiş verinin yeri
IMG_SIZE = (48, 48)              # Resim boyutu
BATCH_SIZE = 64                  # Her adımda işlenecek resim sayısı
EPOCHS = 50                      # Maksimum eğitim turu
MODEL_NAME = "emotion_model.h5"  # Kaydedilecek model adı
GRAPH_DIR = "Training_Results"   # Grafiklerin kaydedileceği klasör

# Klasör yoksa oluştur
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def create_model(num_classes):
    """
    CNN Mimarisi:
    Conv -> BatchNorm -> ReLU -> Pool -> Dropout bloklarından oluşur.
    Bu yapı overfitting'i azaltırken öğrenmeyi hızlandırır.
    """
    model = Sequential()

    # 1. Blok
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2. Blok
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3. Blok
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 4. Blok (Daha derin özellikler için)
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected (Dense) Katmanlar
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Çıkış Katmanı (Sınıf sayısı kadar nöron)
    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_history(history):
    """Loss ve Accuracy grafiklerini çizer ve kaydeder."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(GRAPH_DIR, "accuracy_loss_graph.png"))
    print(f"Grafikler '{GRAPH_DIR}' klasörüne kaydedildi.")
    # plt.show() # İstersen yorum satırını kaldırıp ekranda görebilirsin

def plot_confusion_matrix(model, val_generator):
    """Modelin tahminlerini analiz eder ve Confusion Matrix çizer."""
    print("Confusion Matrix hesaplanıyor...")
    
    # Tahminleri al
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    # Matrix oluştur
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (True)')
    plt.title('Confusion Matrix')
    
    plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
    
    # Detaylı raporu text dosyasına yaz
    report = classification_report(y_true, y_pred, target_names=class_labels)
    with open(os.path.join(GRAPH_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print("Confusion Matrix ve Rapor kaydedildi.")

def main():
    # 1. Veri Yükleyicileri (Data Generators)
    # Validation split ile veriyi %80 Train, %20 Val olarak ayırıyoruz.
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print("Eğitim verisi yükleniyor...")
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("Doğrulama (Test) verisi yükleniyor...")
    # Shuffle=False yapıyoruz ki Confusion Matrix'te sıralama bozulmasın
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        subset='validation',
        shuffle=False 
    )

    # 2. Modeli Oluştur
    num_classes = train_generator.num_classes
    model = create_model(num_classes)
    model.summary()

    # 3. Callbacks (Yardımcı fonksiyonlar)
    # Eğer validation loss 5 epoch boyunca düşmezse eğitimi durdur (Zaman kaybını önler)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # En iyi modeli kaydet
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, mode='max')

    # 4. Eğitimi Başlat
    print("Eğitim başlıyor...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint]
    )

    # 5. Sonuçları Görselleştir ve Kaydet
    plot_history(history)
    plot_confusion_matrix(model, val_generator)
    
    print(f"\nModel eğitildi ve '{MODEL_NAME}' olarak kaydedildi.")

if __name__ == "__main__":
    main()