#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 
from keras import regularizers

train_data = pd.read_csv('train_patches_100.csv')
LOG_FILE = "train_log.csv"      
EPOCH_LOG_FILE = "epoch_log.csv"
RUN_ID = int(time.time()) 
# -------------------------------
# CONFIGURACIONES GENERALES
PERCENTAGE = 100      
TOTAL_IMAGENES = len(train_data)
TOTAL_TRAIN_IMAGES = int((TOTAL_IMAGENES * PERCENTAGE) / 100)
SHUFFLE_BUFFER = TOTAL_TRAIN_IMAGES

IMAGE_SIZE = (224, 224) 
LAYERS = 12      
EPOCHS = 50
PATIENCE = 5 # Early stopping
PATIENCE_LR = 2    # lr_scheduler  
BATCH_SIZE = 16                
LEARNING_RATE = 0.00003            
DROPOUT_RATE = 0.5
ALPHA = [0.25, 0.75, 0.25, 0.75]     # FOCAL LOSS values         
      
# -------------------------------
# RUTAS DEL DATASET
IMAGE_DIR = "C:/Users/Usuario/Documents/tfg/output_patches/patches"
TRAIN_CSV_PATH = "C:/Users/Usuario/Documents/repos/TFG/NewPatches/train_patches_100.csv"  
VAL_CSV_PATH = "C:/Users/Usuario/Documents/repos/TFG/NewPatches/val_patches_100.csv"              

# -------------------------------
# FUNCION DE PERDIDA: FOCAL LOSS (optimizada con tf.function)
def focal_loss(alpha=ALPHA, gamma=2.0):  # Aumentar alpha para GP5
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
# FUNCION DE CARGA
def load_dataset(csv_path, image_dir, max_images=None):
    df = pd.read_csv(csv_path)
    print(f"Cargando dataset desde: {csv_path}")
    
    possible_path_columns = ['image_path', 'filename', 'file_path', 'path', 'image', 'patch_path']
    path_column = None
    
    for col in possible_path_columns:
        if col in df.columns:
            path_column = col
            print(f"Usando columna '{col}' para rutas de imgaenes")
            break
    
    if path_column is None:
        raise KeyError(f"No se encontro una columna de rutas de imgaenes. Columnas disponibles: {df.columns.tolist()}")
    
    # Verificar la columna de etiquetas
    label_column = 'label_type'
    if label_column not in df.columns:
        possible_label_columns = ['label', 'class', 'category', 'tipo']
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                print(f"Usando columna '{col}' para etiquetas")
                break
        
        if label_column not in df.columns:
            raise KeyError(f"No se encontro una columna de etiquetas. Columnas disponibles: {df.columns.tolist()}")
    
    label_map = {'benign': 0, 'GP3': 1, 'GP4': 2, 'GP5': 3}
    image_paths = [os.path.join(image_dir, p) for p in df[path_column]]
    labels = [label_map[label] for label in df[label_column]]
    
    # Limitar la cantidad de imgaenes si se especifica max_images
    if max_images is not None:
        image_paths = image_paths[:max_images]
        labels = labels[:max_images]
    
    print(f"DistribuciÃ³n de etiquetas: {Counter(labels)}")
    
    return image_paths, labels

# -------------------------------
# FUNCION PARA AJUSTAR EL THRESHOLD
def adjust_threshold(model, val_dataset, target_class=3):
    # Obtener las predicciones del modelo
    y_pred = model.predict(val_dataset)
    
    # Obtener las probabilidades de la clase objetivo (GP5)
    y_pred_target = y_pred[:, target_class]
    
    # Obtener las etiquetas verdaderas
    y_true = np.concatenate([y for x, y in val_dataset], axis=0)
    y_true_target = y_true[:, target_class]
    
    # Calcular la curva de Precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true_target, y_pred_target)
    
    # Encontrar el threshold que maximiza F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_threshold

# -------------------------------
# PREPRACION DEL DATASET CON DATA AUGMENTATION (optimizada con tf.function)
@tf.function
def parse_function(filename, label):
    """
    Parsea un archivo de imagen, lo decodifica, redimensiona, aplica data augmentation
    condicionalmente para las clases minoritarias GP3 y GP5, y lo preprocesa para DenseNet121.

    Args:
        filename (tf.Tensor): Tensor de string con la ruta al archivo de imagen.
        label (tf.Tensor): Tensor de entero con la etiqueta de la clase.

    Returns:
        tuple: Una tupla conteniendo la imagen preprocesada (tf.Tensor) y
               la etiqueta en formato one-hot (tf.Tensor).
    """
    # --- Lectura y Decodificación ---
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3) # Decodifica como imagen a color (3 canales)
    image = tf.image.resize(image, IMAGE_SIZE) # Redimensiona a tamaño fijo

    # --- Data Augmentation Condicional ---
    # Determina si se debe aplicar augmentation (si la etiqueta es GP3(1) o GP5(3))
    should_augment = tf.logical_or(tf.equal(label, 1),
                                   tf.equal(label, 3))

    # Aplica cada augmentation solo si should_augment es True
    image = tf.cond(
        should_augment,
        lambda: tf.image.random_flip_left_right(image), # Volteo horizontal aleatorio
        lambda: image # No hacer nada si no se debe aumentar
    )
    image = tf.cond(
        should_augment,
        lambda: tf.image.random_flip_up_down(image), # Volteo vertical aleatorio
        lambda: image
    )
    image = tf.cond(
        should_augment,
        lambda: tf.image.random_brightness(image, max_delta=0.3), # Brillo aleatorio
        lambda: image
    )
    image = tf.cond(
        should_augment,
        lambda: tf.image.random_contrast(image, lower=0.7, upper=1.3), # Contraste aleatorio
        lambda: image
    )
    image = tf.cond(
        should_augment,
        lambda: tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), # Rotación aleatoria 90 grados
        lambda: image
    )

    # --- Normalización Específica del Modelo ---
    # Preprocesa la imagen según los requerimientos de DenseNet121
    # (escalado de píxeles, etc.)
    image = tf.keras.applications.densenet.preprocess_input(image)

    # --- Codificación de Etiqueta ---
    # Convierte la etiqueta numérica a formato one-hot
    # Asume 4 clases en total (0, 1, 2, 3)
    label = tf.one_hot(label, depth=4)

    return image, label

# Cargar paths y etiquetas
train_paths, train_labels = load_dataset(TRAIN_CSV_PATH, IMAGE_DIR, TOTAL_TRAIN_IMAGES)
val_paths, val_labels = load_dataset(VAL_CSV_PATH, IMAGE_DIR)

# Crear un indice aleatorio para el shuffle que se aplicara una vez
random_indices = tf.random.shuffle(tf.range(len(train_paths)))
train_paths = tf.gather(tf.constant(train_paths), random_indices)
train_labels = tf.gather(tf.constant(train_labels), random_indices)

# Convertir a tensores para la validacion
val_paths = tf.constant(val_paths)
val_labels = tf.constant(val_labels)

# Crear datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# -------------------------------
# CREACION DEL MODELO DenseNet121
def create_model(input_shape):
    base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Descongelar mas capas (12 en lugar de 5)
    for layer in base_model.layers[:-LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-LAYERS:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Añadir capas densas adicionales
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # mas neuronas + regularizacion
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Reduccion de dropout para retener patrones
    
    x = Dense(512, activation='relu')(x)  # Capa adicional
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    predictions = Dense(4, activation='softmax')(x)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    
    return model

# Obtener la forma de entrada de una muestra del dataset
for batch in train_dataset.take(1):
    input_shape = batch[0].shape[1:]
    break

# Crear el modelo DenseNet121
model = create_model(input_shape)

# -------------------------------
# CALLBACK PARA GUARDAR LOG POR Epoca
class EpochLogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        log_data = pd.DataFrame([{
            "Run ID": RUN_ID,
            "Epoca": epoch + 1,
            "Learning Rate": K.get_value(self.model.optimizer.lr),
            "Dropout Rate": DROPOUT_RATE,
            "Batch Size": BATCH_SIZE,
            "Train Loss": logs['loss'],
            "Train Acc": logs['accuracy'],
            "Val Loss": logs['val_loss'],
            "Val Acc": logs['val_accuracy'],
            "Total Images": TOTAL_TRAIN_IMAGES,
            "Shuffle Buffer": SHUFFLE_BUFFER,
            "Image Size": IMAGE_SIZE
        }])
        log_data.to_csv(EPOCH_LOG_FILE, mode='a', header=not os.path.exists(EPOCH_LOG_FILE), index=False)

# -------------------------------
# ENTRENAMIENTO CON VALIDACIÓN
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitorea la perdida en validacion (no accuracy)
    factor=0.2,          # Reduce el LR multiplicando por 0.2 
    patience=PATIENCE_LR,         
    min_lr=1e-6,         # LR minimo permitido (antes 1e-7)
    verbose=1            # Muestra mensajes cuando reduzca el LR
)
early_stopping = EarlyStopping(
    monitor='val_loss',  # Usa val_loss, no val_accuracy
    patience=PATIENCE,          #
    restore_best_weights=True  # Recupera los mejores pesos al final
)

from datetime import datetime
# Formato de fecha: YYYYMMDD_HHMMSS
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join("V7models", f"{current_time}_{RUN_ID}")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, f"model_RunID_{RUN_ID}_epoch_{{epoch:02d}}_val_acc_{{val_accuracy:.4f}}.h5"),
    monitor='val_loss',
    mode='min',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

# Convertir train_labels a un array de NumPy
train_labels_np = np.array(train_labels)
# Si es un tensor de TensorFlow, a un array de NumPy
if isinstance(train_labels_np, tf.Tensor):
    train_labels_np = train_labels_np.numpy()
# Calcular los pesos de clase
classes = np.unique(train_labels_np)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels_np)
# Convertir a diccionario para model.fit
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Pesos de clase calculados:", class_weight_dict)

# Entrenar el modelo
start_time = time.time()
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_scheduler, early_stopping, EpochLogCallback(), checkpoint_callback],
                    class_weight=class_weight_dict) 
training_time = time.time() - start_time

# -------------------------------
# AJUSTAR THRESHOLD PARA GP5
best_threshold = adjust_threshold(model, val_dataset, target_class=3)

# Guardar el threshold en un archivo para uso futuro
with open(os.path.join(MODELS_DIR, f"best_threshold_RunID_{RUN_ID}.txt"), "w") as f:
    f.write(f"Best Threshold for GP5: {best_threshold}")

print(f" Mejor threshold para GP5: {best_threshold}")

# -------------------------------
# SELECCIONAR EL MEJOR MODELO
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
best_model_file = None
best_val_acc = 0.0

for model_file in model_files:
    val_acc = float(model_file.split('_val_acc_')[1].replace('.h5', ''))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_file = model_file

if best_model_file:
    best_model_path = os.path.join(MODELS_DIR, best_model_file)
    custom_objects = {'loss': focal_loss()}
    best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
    best_model.save(f"{RUN_ID}_{PERCENTAGE}_trained_model_best.h5")
    print(f" Mejor modelo guardado como '{RUN_ID}_{PERCENTAGE}_trained_model_best.h5' con val_accuracy: {best_val_acc:.4f}")

    for model_file in model_files:
        if model_file != best_model_file:
            os.remove(os.path.join(MODELS_DIR, model_file))
    print(" Modelos no optimos eliminados.")
else:
    print(" No se encontro ningun modelo guardado.")

# -------------------------------
# GUARDAR METRICAS EN EL LOG GENERAL
log_data = pd.DataFrame([{
    "Run ID": RUN_ID,
    "Fecha": pd.Timestamp.now(),
    "Modelo": f"{RUN_ID}_{PERCENTAGE}_best_model.h5",
    "Tiempo (s)": round(training_time, 2),
    "Epocas": len(history.history['loss']),
    "Learning Rate": LEARNING_RATE,
    "Dropout Rate": DROPOUT_RATE,
    "Batch Size": BATCH_SIZE,
    "Train Acc": max(history.history['accuracy']),
    "Val Acc": max(history.history['val_accuracy']),
    "Total Images": TOTAL_TRAIN_IMAGES,
    "Percentage" : PERCENTAGE,
    "Shuffle Buffer": SHUFFLE_BUFFER,
    "Image Size": IMAGE_SIZE,
    "Best Threshold GP5": best_threshold  # Guardar el threshold en el log
}])
log_data.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
print(" Entrenamiento completado y metricas guardadas.")

# -------------------------------
# GRAFICAR LA PRECISION POR EPOCA
plt.figure(figsize=(10, 5))
# Ajustar el eje x para que comience desde 1
epochs_range = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Precision por Epoca')
plt.xlabel('Epoca')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# %%
