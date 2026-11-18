import csv
import numpy as np
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN, KMeans


# --- KROK 1: Funkcja generująca plik, jeśli go nie ma ---
def generate_combined_cloud(filename="combined_clouds.xyz"):
  print("Generowanie pliku z chmurami punktów...")
  # 1. Płaska pozioma powierzchnia (z = 0)
  x_h = np.random.uniform(-5, 5, 1500)
  y_h = np.random.uniform(-5, 5, 1500)
  z_h = np.zeros(1500)
  cloud_h = np.column_stack((x_h, y_h, z_h))

  # 2. Płaska pionowa powierzchnia (y = 15)
  x_v = np.random.uniform(-5, 5, 1500)
  y_v = np.full(1500, 15.0)
  z_v = np.random.uniform(0, 8, 1500)
  cloud_v = np.column_stack((x_v, y_v, z_v))

  # 3. Powierzchnia cylindryczna (x = 15, R = 3)
  theta = np.random.uniform(0, 2 * np.pi, 2000)
  z_c = np.random.uniform(0, 10, 2000)
  x_c = 15.0 + 3.0 * np.cos(theta)
  y_c = 3.0 * np.sin(theta)
  cloud_c = np.column_stack((x_c, y_c, z_c))

  # Łączenie i zapis
  combined = np.vstack((cloud_h, cloud_v, cloud_c))
  np.random.shuffle(combined)
  np.savetxt(filename, combined, fmt="%.6f", delimiter=" ")
  print(f"Pomyślnie utworzono plik: {filename}\n")


# --- KROK 2: Wczytanie pliku XYZ za pomocą pakietu csv ---
def load_xyz_csv(filename):
  points = []
  with open(filename, mode="r", newline="") as file:
    reader = csv.reader(file, delimiter=" ")
    for row in reader:
      clean_row = [float(val) for val in row if val.strip() != ""]
      if len(clean_row) >= 3:
        points.append(clean_row[:3])
  return np.array(points)


# --- KROK 3: Własny RANSAC ---
def fit_plane_ransac(points, iterations=1000, threshold=0.05):
  best_inliers = []
  best_normal = None
  best_d = 0
  n_points = len(points)

  for _ in range(iterations):
    idx = np.random.choice(n_points, 3, replace=False)
    p1, p2, p3 = points[idx]

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_length = np.linalg.norm(normal)

    if norm_length == 0:
      continue

    normal = normal / norm_length
    d = -np.dot(normal, p1)

    distances = np.abs(np.dot(points, normal) + d)
    inliers = np.where(distances < threshold)[0]

    if len(inliers) > len(best_inliers):
      best_inliers = inliers
      best_normal = normal
      best_d = d

  return best_normal, best_d, best_inliers


def classify_surface(normal, inliers_ratio):
  if inliers_ratio < 0.65:
    return "Powierzchnia niepłaska (CYLINDER)"

  nz = abs(normal[2])
  if nz > 0.85:
    return "Płaszczyzna POZIOMA"
  elif nz < 0.15:
    return "Płaszczyzna PIONOWA"
  else:
    return "Płaszczyzna UKOŚNA"


if __name__ == "__main__":
  filename = "combined_clouds.xyz"

  # Automatycznie generujemy plik na start, aby uniknąć błędu braku pliku
  generate_combined_cloud(filename)

  # Wczytanie danych
  points = load_xyz_csv(filename)
  print(f"Wczytano {len(points)} punktów z pliku.\n")

  # K-średnich + Własny RANSAC
  print("=== METODA 1: K-Means + Własny RANSAC ===")
  kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
  labels_km = kmeans.fit_predict(points)

  for cluster_id in range(3):
    cluster_points = points[labels_km == cluster_id]
    normal, d, inliers = fit_plane_ransac(cluster_points)
    ratio = len(inliers) / len(cluster_points)

    print(f"\nKlaster {cluster_id} (liczba punktów: {len(cluster_points)}):")
    print(
        f"  Wektor normalny: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]"
    )
    print(f"  Dopasowanie (inliers): {ratio*100:.1f}%")
    print(f"  Werdykt: {classify_surface(normal, ratio)}")

  # DBSCAN + pyransac3d
  print("\n" + "=" * 45)
  print("=== METODA 2: DBSCAN + pyransac3d ===")
  dbscan = DBSCAN(eps=1.5, min_samples=10)
  labels_db = dbscan.fit_predict(points)

  for cluster_id in set(labels_db):
    if cluster_id == -1:
      continue

    cluster_points = points[labels_db == cluster_id]
    plane_detector = pyrsc.Plane()
    best_eq, best_inliers = plane_detector.fit(cluster_points, thresh=0.05)

    normal = np.array(best_eq[:3])
    ratio = len(best_inliers) / len(cluster_points)

    print(f"\nKlaster {cluster_id} (liczba punktów: {len(cluster_points)}):")
    print(
        f"  Wektor normalny (pyransac3d): [{normal[0]:.4f}, {normal[1]:.4f},"
        f" {normal[2]:.4f}]"
    )
    print(f"  Dopasowanie (inliers): {ratio*100:.1f}%")
    print(f"  Werdykt: {classify_surface(normal, ratio)}")