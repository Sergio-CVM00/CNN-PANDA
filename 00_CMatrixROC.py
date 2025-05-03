#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------------------------------
# !🔹 CONFIGURACIONES GENERALES
TEST_CSV_PATH = "C:/Users/Usuario/Documents/repos/TFG/NewPatches/test_patches_100.csv" 
df = pd.read_csv('test_patches_100.csv')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 64  
NUM_IMAGES = len(df)  
IMAGE_SIZE = (224, 224)  
MODEL_NAME = "1746202178_65_trained_model_best.h5" # !
IMAGE_DIR = "C:/Users/Usuario/Documents/tfg/output_patches/patches"

# Extraer el nombre del modelo sin extensión
model_name_without_extension = os.path.splitext(MODEL_NAME)[0]
output_dir = f"matrix_results/{model_name_without_extension}"
os.makedirs(output_dir, exist_ok=True)

# Mapeo de etiquetas y nombres de clases
label_map = {'benign': 0, 'GP3': 1, 'GP4': 2, 'GP5': 3}
class_names = ["Benign", "GP3", "GP4", "GP5"]

# -------------------------------
# 🔹 FUNCIÓN DE PÉRDIDA: FOCAL LOSS
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
# 🔹 FUNCIÓN DE PREPROCESAMIENTO
def _load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label

# -------------------------------
# 🔹 PREPARAR DATASET
df = pd.read_csv(TEST_CSV_PATH)
df = df.head(min(NUM_IMAGES, len(df))).reset_index(drop=True)
NUM_IMAGES = len(df)
image_paths = [os.path.join(IMAGE_DIR, x) for x in df['patch_path']]
labels = [label_map[x] for x in df['label_type']]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 🔹 CARGAR MODELO
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
# 🔹 GENERAR MÉTRICAS
cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = confusion_matrix(y_true, y_pred_classes, normalize='true')
report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

def quadratic_kappa(cohen_matrix):
    n_classes = cohen_matrix.shape[0]
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2
    
    hist_true = np.sum(cohen_matrix, axis=1)
    hist_pred = np.sum(cohen_matrix, axis=0)
    
    expected = np.outer(hist_true, hist_pred) * weights
    observed = cohen_matrix * weights
    
    total = np.sum(cohen_matrix)
    expected = np.sum(expected) / (total ** 2)
    observed = np.sum(observed) / total
    
    return 1 - (observed / expected) if expected != 0 else 0

kappa = quadratic_kappa(cm)

# -------------------------------
# 🔹 VISUALIZACIÓN Y GUARDADO DE IMÁGENES
# Matriz de Confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Verdadero")
plt.title(f"Matriz de Confusión - {model_name_without_extension}")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# Matriz Normalizada
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Verdadero")
plt.title(f"Matriz de Confusión Normalizada - {model_name_without_extension}")
plt.savefig(f"{output_dir}/confusion_matrix_normalized.png")
plt.close()

# Curvas ROC
y_true_onehot = label_binarize(y_true, classes=[0, 1, 2, 3])
n_classes = len(class_names)
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange']
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title(f'Curvas ROC - {model_name_without_extension}')
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/roc_curves.png")
plt.close()

# Imagen con métricas
metrics_text = [
    f"📊 Reporte de Clasificación - {model_name_without_extension}",
    classification_report(y_true, y_pred_classes, target_names=class_names),
    f"\n📈 Kappa Cuadrático: {kappa:.4f}",
    "\n📉 AUC por Clase:",
    *[f"{class_names[i]}: {roc_auc[i]:.4f}" for i in range(n_classes)]
]

plt.figure(figsize=(12, 8))
plt.text(0.1, 0.1, "\n".join(metrics_text), fontfamily='monospace', fontsize=12)
plt.axis('off')
plt.title(f"Resumen de Métricas - {model_name_without_extension}", pad=20)
plt.savefig(f"{output_dir}/metrics_summary.png")
plt.close()

# -------------------------------
# 🔹 IMPRESIÓN EN CONSOLA
print("\n" + "\n".join(metrics_text))

# -------------------------------
# 🔹 GUARDADO DE LOGS
CSV_FILENAME = "00_confusion_matrix_log.csv"
current_time = int(time.time())  # Usamos esto como Run ID

hyperparams = {
    "Run ID": current_time,
    "Modelo": MODEL_NAME,
    "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Learning Rate": 0.0005,          
    "Batch Size": BATCH_SIZE,
    "Dropout Rate": 0.5,             
    "Épocas": 15,                    
    "Image Size": str(IMAGE_SIZE),
    "Accuracy Global": round(report['accuracy'], 4),
    "Matriz Confusión": str(cm.tolist()),
    "Kappa Cuadrático": round(kappa, 4),
    "AUC Benign": round(roc_auc[0], 4),
    "AUC GP3": round(roc_auc[1], 4),
    "AUC GP4": round(roc_auc[2], 4),
    "AUC GP5": round(roc_auc[3], 4)
}

for cls in class_names:
    hyperparams[f"{cls} Precision"] = round(report[cls]['precision'], 2)
    hyperparams[f"{cls} Recall"] = round(report[cls]['recall'], 2)
    hyperparams[f"{cls} F1"] = round(report[cls]['f1-score'], 2)

log_df = pd.DataFrame([hyperparams])

if os.path.exists(CSV_FILENAME):
    log_df.to_csv(CSV_FILENAME, mode='a', header=False, index=False)
else:
    log_df.to_csv(CSV_FILENAME, index=False)

print(f"\n✅ Resultados guardados en {CSV_FILENAME}")
print(f"📁 Imágenes guardadas en: {output_dir}")
# %%