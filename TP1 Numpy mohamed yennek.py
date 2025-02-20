import numpy as np
import time
import matplotlib.pyplot as plt

# Exercice 1 :
# Création d'un tableau 1D et conversion en float64
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("1D Array:", array_1d)

# Création d'un tableau 2D et affichage de ses dimensions
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Shape:", array_2d.shape, "Size:", array_2d.size)

# Création d'un tableau 3D avec valeurs aléatoires
array_3d = np.random.rand(2, 3, 4)
print("3D Shape:", array_3d.shape, "Dimensions:", array_3d.ndim)

# Exercice 2 :
# Inversion d'un tableau 1D
array_1d_reversed = np.arange(10)[::-1]
print("Reversed 1D:", array_1d_reversed)

# Extraction d'un sous-tableau
array_2d_shape = np.arange(12).reshape(3, 4)
subarray = array_2d_shape[:2, -2:]
print("Subarray:", subarray)

# Remplacement des valeurs supérieures à 5 par 0
array_5x5 = np.random.randint(0, 10, (5, 5))
array_5x5[array_5x5 > 5] = 0
print("Modified 5x5:", array_5x5)

# Exercice 3 :
# Création d'une matrice identité et affichage de ses attributs
identity_matrix = np.eye(3)
print("Identity Matrix:", identity_matrix)
print(identity_matrix.shape, "/", identity_matrix.size, "/", identity_matrix.ndim, "/", identity_matrix.itemsize, "/", identity_matrix.nbytes)

# Création d'un tableau de valeurs équidistantes
evenly_spaced = np.linspace(0, 5, 10)
print("Evenly Spaced:", evenly_spaced, "Dtype:", evenly_spaced.dtype)

# Somme des éléments d'un tableau 3D aléatoire
array_3d_random = np.random.randn(2, 3, 4)
sum_elements = np.sum(array_3d_random)
print("Sum of 3D Array:", sum_elements)

# Exercice 4 :
# Indexation avancée avec fancy indexing
array_random = np.random.randint(0, 50, 20)
selected_elements = array_random[[2, 5, 7, 10, 15]]
print("Selected Elements:", selected_elements)

# Sélection des éléments avec un masque booléen
array_2d_mask = np.random.randint(0, 30, (4, 5))
masked_elements = array_2d_mask[array_2d_mask > 15]
print("Masked Elements:", masked_elements)

# Remplacement des valeurs négatives par 0
array_1d_negative = np.random.randint(-10, 10, 10)
array_1d_negative[array_1d_negative < 0] = 0
print("Modified 1D:", array_1d_negative)

# Exercice 5 :
# Concaténation de tableaux 1D
array1 = np.random.randint(0, 10, 5)
array2 = np.random.randint(0, 10, 5)
concatenated = np.concatenate((array1, array2))
print("Concatenated:", concatenated)

# Division d'un tableau 2D en deux parties égales
array_2d_split = np.random.randint(0, 10, (6, 4))
split_arrays = np.split(array_2d_split, 2, axis=0)
print("Split Arrays:", split_arrays)

# Division d'un tableau en trois parties selon les colonnes
array_2d_column_split = np.random.randint(0, 10, (3, 6))
column_splits = np.split(array_2d_column_split, 3, axis=1)
print("Column Splits:", column_splits)

# Exercice 6 :
# Calculs statistiques sur un tableau 1D
array_stat = np.random.randint(1, 100, 15)
print("Mean:", np.mean(array_stat), "Median:", np.median(array_stat), "Std Dev:", np.std(array_stat), "Variance:", np.var(array_stat))

# Calculs des sommes des lignes et colonnes d'un tableau 2D
array_2d = np.random.randint(1, 50, (4, 4))
print("Row Sums:", np.sum(array_2d, axis=1))
print("Column Sums:", np.sum(array_2d, axis=0))

# Calculs max et min sur un tableau 3D
array_3d = np.random.randint(1, 20, (2, 3, 4))
print("Max:", np.max(array_3d), "Min:", np.min(array_3d))

# Exercice 7 :
# Transformation d'un tableau 1D en 2D
array_reshape = np.arange(1, 13).reshape(3, 4)
print("Reshaped Array:", array_reshape)

# Transposition d'un tableau 2D
array_transpose = np.random.randint(1, 10, (3, 4)).T
print("Transposed Array:", array_transpose)

# Aplatissement d'un tableau 2D en 1D
array_flatten = np.random.randint(1, 10, (2, 3)).flatten()
print("Flattened Array:", array_flatten)

# Exercice 8 :
# Normalisation d'un tableau par la moyenne des colonnes
array_broadcast = np.random.randint(1, 10, (3, 4))
column_mean = array_broadcast.mean(axis=0)
normalized_array = array_broadcast - column_mean
print("Normalized Array:", normalized_array)

# Calcul du produit extérieur entre deux tableaux 1D
array1 = np.random.randint(1, 5, 4)
array2 = np.random.randint(1, 5, 4)
outer_product = np.outer(array1, array2)
print("Outer Product:", outer_product)

# Modification des valeurs > 5 en ajoutant 10
array_large = np.random.randint(1, 10, (4, 5))
array_large[array_large > 5] += 10
print("Large Array:", array_large)

# Exercice 9 :
# Tri d'un tableau 1D
array_sort = np.random.randint(1, 20, 10)
sorted_array = np.sort(array_sort)
print("Sorted Array:", sorted_array)

# Tri d'un tableau 2D en fonction de la deuxième colonne
array_2d_sort = np.random.randint(1, 50, (3, 5))
array_2d_sorted = array_2d_sort[array_2d_sort[:, 1].argsort()]
print("Sorted 2D Array:", array_2d_sorted)

# Recherche des indices des éléments > 50
array_search = np.random.randint(1, 100, 15)
indices = np.where(array_search > 50)
print("Indices:", indices)
print("Corresponding Values:", array_search[indices])

# Exercice 10 :
# Calcul du déterminant d'une matrice
matrix_A = np.random.randint(1, 10, (2, 2))
determinant = np.linalg.det(matrix_A)
print("Déterminant :", determinant)

# Calcul des valeurs propres et vecteurs propres d'une matrice 3x3
matrix_B = np.random.randint(1, 5, (3, 3))
eigenvalues, eigenvectors = np.linalg.eig(matrix_B)
print("Valeurs propres :", eigenvalues)

# Produit matriciel entre deux matrices compatibles
matrix_C = np.random.randint(1, 10, (2, 3))
matrix_D = np.random.randint(1, 10, (3, 2))
matrix_product = np.dot(matrix_C, matrix_D)
print("Produit matriciel :", matrix_product)

# Exercice 11 :
# Génération d'un échantillon uniforme entre 0 et 1
uniform_sample = np.random.uniform(0, 1, 10)
print("Échantillon uniforme :", uniform_sample)

# Génération d'un échantillon normal (loi normale)
normal_sample = np.random.normal(0, 1, (3, 3))
print("Échantillon normal :", normal_sample)

# Histogramme d'une distribution d'entiers aléatoires
random_ints = np.random.randint(1, 100, 20)
plt.hist(random_ints, bins=5, edgecolor="black")
plt.show()

# Exercice 12 :
# Sélection des éléments diagonaux d'une matrice
array_2d = np.random.randint(1, 21, (5, 5))
print("Éléments diagonaux :", np.diagonal(array_2d))

# Détection des nombres premiers dans un tableau
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

array_1d = np.random.randint(1, 51, 10)
primes = array_1d[np.vectorize(is_prime)(array_1d)]
print("Nombres premiers :", primes)

# Sélection des nombres pairs d'un tableau 2D
array_2d_even = np.random.randint(1, 11, (4, 4))
even_numbers = array_2d_even[array_2d_even % 2 == 0]
print("Nombres pairs :", even_numbers)

# Exercice 13 :
# Ajout de NaN dans un tableau aléatoire
array_nan = np.random.randint(1, 11, 10).astype(float)
indices = np.random.choice(len(array_nan), size=3, replace=False)
array_nan[indices] = np.nan
print("Tableau avec NaN :", array_nan)

# Remplacement des valeurs < 5 par NaN dans un tableau 2D
array_2d_nan = np.random.randint(1, 11, (3, 4)).astype(float)
array_2d_nan[array_2d_nan < 5] = np.nan
print("Tableau 2D avec NaN :", array_2d_nan)

# Identification des indices contenant des NaN
array_1d_15 = np.random.randint(1, 21, 15).astype(float)
nan_indices = np.random.choice(len(array_1d_15), size=4, replace=False)
array_1d_15[nan_indices] = np.nan
indices_nans = np.where(np.isnan(array_1d_15))[0]
print("Indices des valeurs NaN :", indices_nans)

# Exercice 14 :
# Calcul de la moyenne et de l'écart-type d'un grand tableau
large_array = np.random.randint(1, 101, 10**6)
start = time.time()
mean_val = np.mean(large_array)
std_dev = np.std(large_array)
end = time.time()
print(f"Moyenne : {mean_val}, Écart-type : {std_dev}, Temps : {end - start:.5f} sec")

# Addition de deux grandes matrices et mesure du temps
matrix_A = np.random.randint(1, 11, (1000, 1000))
matrix_B = np.random.randint(1, 11, (1000, 1000))
start = time.time()
matrix_sum = matrix_A + matrix_B
end = time.time()
print(f"Temps d'exécution de l'addition : {end - start:.5f} sec")

# Somme le long des axes d'un tableau 3D
array_3d = np.random.randint(1, 11, (100, 100, 100))
start = time.time()
sum_axis0 = np.sum(array_3d, axis=0)
sum_axis1 = np.sum(array_3d, axis=1)
sum_axis2 = np.sum(array_3d, axis=2)
end = time.time()
print(f"Temps de calcul des sommes : {end - start:.5f} sec")

# Exercice 15 :
# Calcul des sommes et produits cumulatifs d'un tableau 1D
array_1d = np.arange(1, 11)
print("Somme cumulative :", np.cumsum(array_1d))
print("Produit cumulatif :", np.cumprod(array_1d))

# Somme cumulative le long des axes d'un tableau 2D
array_2d = np.random.randint(1, 21, (4, 4))
print("Somme cumulative par ligne :", np.cumsum(array_2d, axis=1))
print("Somme cumulative par colonne :", np.cumsum(array_2d, axis=0))

# Calcul des valeurs min, max et somme d'un tableau aléatoire
array_random = np.random.randint(1, 51, 10)
print("Minimum :", np.min(array_random))
print("Maximum :", np.max(array_random))
print("Somme :", np.sum(array_random))

# Exercice 16 :
# Génération d'une série de dates journalières à partir d'aujourd'hui
dates_daily = np.arange(np.datetime64('today'), np.datetime64('today') + 10, dtype='datetime64[D]')
print("Dates journalières :", dates_daily)

# Génération d'une série de dates mensuelles
dates_monthly = np.arange('2022-01', '2022-06', dtype='datetime64[M]')
print("Dates mensuelles :", dates_monthly)

# Génération de timestamps aléatoires en 2023
random_days = np.random.randint(0, 365, 10)
timestamps_2023 = np.datetime64('2023-01-01') + random_days
print("Timestamps 2023 :", timestamps_2023)

# Exercice 17 :
# Création d'un tableau avec représentation binaire des entiers
dtype_custom = np.dtype([('nombre', np.int32), ('binaire', 'U10')])
array_binary = np.array([(i, bin(i)) for i in range(5)], dtype=dtype_custom)
print("Tableau avec représentation binaire :", array_binary)

# Création d'un tableau de nombres complexes
dtype_complex = np.dtype([('valeur', np.complex128)])
array_complex = np.array([[complex(1, 2), complex(3, 4), complex(5, 6)],
                          [complex(7, 8), complex(9, 10), complex(11, 12)],
                          [complex(13, 14), complex(15, 16), complex(17, 18)]], dtype=dtype_complex)
print("Tableau de nombres complexes :", array_complex)

# Création d'un tableau structuré contenant des informations sur des livres
dtype_books = np.dtype([('Titre', 'U50'), ('Auteur', 'U50'), ('Pages', np.int32)])
books = np.array([("Le Petit Prince", "Antoine de Saint-Exupéry", 96),
                  ("1984", "George Orwell", 328),
                  ("Les Misérables", "Victor Hugo", 1232)], dtype=dtype_books)
print("Tableau structuré des livres :", books)

