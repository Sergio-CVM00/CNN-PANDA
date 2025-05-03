# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib para graficar

# Cargar el dataset
csv_path = "NewMetadata.csv"
df = pd.read_csv(csv_path)

# Obtener lista única de imágenes en el dataset completo
unique_images = df['image_id'].unique()

# Dividir imágenes en train, val y test
train_images, temp_images = train_test_split(unique_images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Asignar cada imagen a su conjunto correspondiente
def assign_set(image_id):
    if image_id in train_images:
        return 'train'
    elif image_id in val_images:
        return 'val'
    else:
        return 'test'

df['set'] = df['image_id'].apply(assign_set)

# Separar los datasets finales
df_train = df[df['set'] == 'train'].drop(columns=['set'])
df_val = df[df['set'] == 'val'].drop(columns=['set'])
df_test = df[df['set'] == 'test'].drop(columns=['set'])

# Calcular pesos de clases para balancear entrenamiento sin Data Augmentation
labels = df_train['label_type'].unique()
class_weights = compute_class_weight(class_weight="balanced", classes=labels, y=df_train['label_type'])
class_weight_dict = dict(zip(labels, class_weights))

print("Pesos de clases calculados:", class_weight_dict)

# Guardar los nuevos datasets
df_train.to_csv("train_patches_100.csv", index=False)
df_val.to_csv("val_patches_100.csv", index=False)
df_test.to_csv("test_patches_100.csv", index=False)

print("✔ División completada: train, val y test guardados correctamente.")

# -------------------------------
# 🔹 GRAFICAR LA DISTRIBUCIÓN DE ETIQUETAS EN CADA CONJUNTO
def plot_label_distribution(df, title):
    label_counts = df['label_type'].value_counts()
    print(f"\n{title}:")
    print(label_counts)  # Imprimir la cantidad exacta de muestras por clase
    plt.bar(label_counts.index, label_counts.values, color='skyblue')
    plt.title(title)
    plt.xlabel('Etiquetas')
    plt.ylabel('Cantidad de muestras')
    plt.show()
# %%
# Graficar la distribución de etiquetas en cada conjunto
plot_label_distribution(df_train, "Distribución de etiquetas en el conjunto de entrenamiento")
plot_label_distribution(df_val, "Distribución de etiquetas en el conjunto de validación")
plot_label_distribution(df_test, "Distribución de etiquetas en el conjunto de prueba")