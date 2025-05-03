#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from collections import Counter
from sklearn.metrics import cohen_kappa_score, f1_score

# -------------------------------
# 🔹 CONFIGURACIONES GENERALES
TEST_CSV_PATH = "C:/Users/Usuario/Documents/repos/TFG/NewPatches/test_patches_100.csv" 
df = pd.read_csv('test_patches_100.csv')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ocultar warnings innecesarios
BATCH_SIZE = 64                     # Tamaño del batch para la evaluación
NUM_IMAGES = len(df)                  # Número total de imágenes a evaluar
IMAGE_SIZE = (224, 224)             # Resolución de las imágenes
MODEL_NAME = "1746202178_65_trained_model_best.h5"  # ! Modelo a evaluar
IMAGE_DIR = "C:/Users/Usuario/Documents/tfg/output_patches/patches"
LOG_FILENAME = "test_log.csv"

# Mapeo de etiquetas
label_map = {'benign': 0, 'GP3': 1, 'GP4': 2, 'GP5': 3}

# -------------------------------
# 🔹 FUNCIÓN DE PREPROCESAMIENTO CON tf.data
def _process_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)  # ! CAMBIAR EN FUNCION DEL MODELO 
    label = tf.one_hot(label, depth=4)
    return image, label

# -------------------------------
# 🔹 PREPARAR EL DATASET DE EVALUACIÓN
# Cargar CSV y limitar a NUM_IMAGES
df = pd.read_csv(TEST_CSV_PATH)
df = df.head(min(NUM_IMAGES, len(df))).reset_index(drop=True)

# Crear la lista de rutas completas y etiquetas numéricas
image_paths = [os.path.join(IMAGE_DIR, x) for x in df['patch_path']]
labels = [label_map[x] for x in df['label_type']]

# Crear el tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(_process_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 🔹 FUNCIÓN DE PÉRDIDA: FOCAL LOSS (para cargar el modelo correctamente)
def focal_loss(alpha=[0.25, 0.25, 0.25, 0.40], gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.reduce_sum(y_true * alpha, axis=1, keepdims=True)
        focal_weight = tf.pow(1 - y_pred, gamma)
        loss_val = weight * focal_weight * cross_entropy
        return tf.reduce_mean(loss_val)
    return loss

# -------------------------------
# 🔹 CARGAR EL MODELO ENTRENADO
print("📥 Cargando el modelo entrenado...")
model = load_model(MODEL_NAME, custom_objects={"loss": focal_loss()})
print("✅ Modelo cargado correctamente")

# -------------------------------
# 🔹 OBTENER PREDICCIONES
print("\n🔍 Realizando predicciones...")
y_pred = model.predict(dataset, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.array([label_map[x] for x in df['label_type']])

# -------------------------------
# 🔹 CALCULAR MÉTRICAS ADICIONALES
# Kappa Cuadrático de Cohen
kappa = cohen_kappa_score(y_true, y_pred_classes, weights='quadratic')
print(f"\n📈 **Kappa Cuadrático de Cohen**: {kappa:.4f}")

# F1-score (promedio ponderado)
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"📊 **F1-score (weighted)**: {f1:.4f}")

# -------------------------------
# 🔹 EVALUAR EL MODELO
print(f"\n🔍 Iniciando evaluación en {len(df)} imágenes...")
results = model.evaluate(dataset, verbose=1)
loss_val, accuracy_val = results[0], results[1]
print(f"\n📊 Evaluación Final")
print(f"  - Pérdida en prueba: {loss_val:.4f}")
print(f"  - Precisión en prueba: {accuracy_val * 100:.2f}%")
print(f"  - Kappa Cuadrático: {kappa:.4f}")
print(f"  - F1-score (weighted): {f1:.4f}")

# -------------------------------
# 🔹 CONTAR LA DISTRIBUCIÓN DE CLASES (desde el CSV)
class_totals = Counter([label_map[x] for x in df['label_type']])

# -------------------------------
# 🔹 GUARDAR RESULTADOS EN CSV
RUN_ID = int(time.time())  # Generar un RUN_ID único basado en el timestamp
log_data = pd.DataFrame([{
    "Run ID": RUN_ID,  # Añadir Run ID
    "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Modelo": MODEL_NAME,
    "Num_Imagenes": NUM_IMAGES,
    "Precisión (%)": round(accuracy_val * 100, 2),
    "Pérdida": round(loss_val, 4),
    "Kappa Cuadrático": round(kappa, 4),  # Nuevo campo
    "F1-score (weighted)": round(f1, 4),  # Nuevo campo
    "Benign": class_totals.get(0, 0),
    "GP3": class_totals.get(1, 0),
    "GP4": class_totals.get(2, 0),
    "GP5": class_totals.get(3, 0)
}])

if os.path.exists(LOG_FILENAME):
    log_data.to_csv(LOG_FILENAME, mode='a', header=False, index=False)
else:
    log_data.to_csv(LOG_FILENAME, mode='w', header=True, index=False)

print(f"\n✅ Resultados guardados en {LOG_FILENAME}")
# %%
