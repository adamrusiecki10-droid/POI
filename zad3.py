import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- ZADANIE 1 & 2: Przygotowanie katalogów i wycinanie próbek tekstur ---
def prepare_dataset():
  # Definiujemy 3 przykładowe kategorie tekstur
  categories = ['gres', 'tynk', 'laminat']
  os.makedirs('images', exist_ok=True)

  # Jeśli nie masz własnych zdjęć, generujemy bazowe obrazy tekstur
  for cat in categories:
    cat_dir = os.path.join('images', cat)
    os.makedirs(cat_dir, exist_ok=True)

    for i in range(3):
      if cat == 'gres':
        img = np.random.randint(120, 220, (512, 512), dtype=np.uint8)
        img[::64, :] = 40  # imitacja siatki płytek
        img[:, ::64] = 40
      elif cat == 'tynk':
        img = np.random.randint(60, 200, (512, 512), dtype=np.uint8)  # szum
      else:  # laminat
        base_line = np.linspace(80, 180, 512, dtype=np.uint8)
        img = np.tile(base_line, (512, 1))
        # Bezpieczne dodanie szumu z uwzględnieniem zakresu 0-255
        noise = np.random.randint(-15, 16, (512, 512))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

      cv2.imwrite(os.path.join(cat_dir, f'base_{i}.png'), img)


def crop_patches(patch_size=128):
    categories = ['gres', 'tynk', 'laminat']
    output_base = 'dataset_patches'

    for cat in categories:
        input_dir = os.path.join('images', cat)
        output_dir = os.path.join(output_base, cat)
        os.makedirs(output_dir, exist_ok=True)

        count = 0
        for filename in os.listdir(input_dir):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            # Wycinanie fragmentów o zadanych wymiarach (np. 128x128)
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = img[y:y + patch_size, x:x + patch_size]
                    patch_name = os.path.join(output_dir, f'patch_{count}.png')
                    cv2.imwrite(patch_name, patch)
                    count += 1
    print("[INFO] Pomyślnie wycięto i zapisano próbki tekstur.")


# --- ZADANIE 3 & 4: Obliczanie cech GLCM i zapis do CSV ---
def extract_glcm_features():
    categories = ['gres', 'tynk', 'laminat']
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 stopni
    prop_names = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    data = []

    for cat in categories:
        folder = os.path.join('dataset_patches', cat)
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Zmniejszenie głębi jasności do 5 bitów (64 poziomy): 256 / 4 = 64
            img_reduced = (img // 4).astype(np.uint8)

            # Obliczenie macierzy GLCM
            glcm = graycomatrix(img_reduced, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

            feature_vector = []
            for prop in prop_names:
                values = graycoprops(glcm, prop)  # macierz (len(distances), len(angles))
                feature_vector.extend(values.flatten())

            # Dodanie nazwy kategorii na końcu wektora cech
            feature_vector.append(cat)
            data.append(feature_vector)

    # Generowanie nagłówków kolumn
    columns = []
    for prop in prop_names:
        for d in distances:
            for a in [0, 45, 90, 135]:
                columns.append(f"{prop}_d{d}_a{a}")
    columns.append("category")

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("texture_features.csv", index=False)
    print("[INFO] Zapisano wektory cech do pliku texture_features.csv")
    return df


# --- ZADANIE 5: Klasyfikacja danych (SVM) ---
def classify_textures(df):
    X = df.drop(columns=["category"])
    y = df["category"]

    # Podział na zbiór treningowy (70%) i testowy (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicjalizacja i trenowanie klasyfikatora SVM
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)

    # Testowanie modelu
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[wynik] Dokładność klasyfikatora SVM: {acc * 100:.2f}%")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    prepare_dataset()
    crop_patches(patch_size=128)
    df_features = extract_glcm_features()
    classify_textures(df_features)