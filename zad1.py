import numpy as np


def generate_horizontal_plane(
    width=10.0, length=10.0, num_points=1000, z_val=0.0
):
  """a. płaska pozioma powierzchnia o ograniczonej szerokości i długości

  X i Y wyznaczają płaszczyznę poziomą, Z jest stałe (wysokość).
  """
  x = np.random.uniform(-width / 2, width / 2, num_points)
  y = np.random.uniform(-length / 2, length / 2, num_points)
  z = np.full(num_points, z_val)
  return np.column_stack((x, y, z))


def generate_vertical_plane(
    width=10.0, height=5.0, num_points=1000, y_val=0.0
):
  """b. płaska pionowa powierzchnia o ograniczonej szerokości i wysokości

  X wyznacza szerokość, Z wyznacza wysokość, Y jest stałe.
  """
  x = np.random.uniform(-width / 2, width / 2, num_points)
  z = np.random.uniform(0, height, num_points)
  y = np.full(num_points, y_val)
  return np.column_stack((x, y, z))


def generate_cylinder(radius=5.0, height=10.0, num_points=2000):
  """c. powierzchnia cylindryczna o zadanym promieniu i ograniczonej wysokości

  Układ współrzędnych walcowych rzutowany na Cartesian:
  X = R * cos(theta), Y = R * sin(theta), Z = wysokość.
  """
  theta = np.random.uniform(0, 2 * np.pi, num_points)
  z = np.random.uniform(0, height, num_points)
  x = radius * np.cos(theta)
  y = radius * np.sin(theta)
  return np.column_stack((x, y, z))


def save_to_xyz(filename, points):
  """Zapisuje chmurę punktów do pliku tekstowego w formacie XYZ."""
  np.savetxt(filename, points, fmt="%.6f", delimiter=" ")
  print(f"Zapisano plik: {filename}")


if __name__ == "__main__":
  # Generowanie chmury a: poziomej
  cloud_h = generate_horizontal_plane(
      width=10, length=10, num_points=2500, z_val=0.0
  )
  save_to_xyz("horizontal_plane.xyz", cloud_h)

  # Generowanie chmury b: pionowej
  cloud_v = generate_vertical_plane(width=10, height=5, num_points=2500, y_val=5.0)
  save_to_xyz("vertical_plane.xyz", cloud_v)

  # Generowanie chmury c: cylindrycznej
  cloud_c = generate_cylinder(radius=4.0, height=8.0, num_points=4000)
  save_to_xyz("cylinder.xyz", cloud_c)