import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import multiprocessing
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**45) # Desactivar límite de tamaño para openCV
Image.MAX_IMAGE_PIXELS = None  # Desactivar límite de tamaño para PIL

# Configuración manual
limit = 0  # Límite de imágenes a procesar (0 = sin límite)
batch_size = 50  # Número de imágenes a procesar por lote
use_opencv = True  # Usar OpenCV en lugar de PIL

# Rutas de archivos
csv_path = r"C:\Users\Usuario\Documents\tfg\kaggle\half_input\prostate-cancer-grade-assessment\train.csv"
image_folder = r"C:\Users\Usuario\Documents\tfg\kaggle\half_input\prostate-cancer-grade-assessment\train_images"
mask_folder = r"C:\Users\Usuario\Documents\tfg\kaggle\half_input\prostate-cancer-grade-assessment\train_label_masks"
output_folder = r"C:\Users\Usuario\Documents\tfg\output_patches"
patches_folder = os.path.join(output_folder, "patches")
progress_file = os.path.join(output_folder, "processed_images.txt")
failed_images_file = os.path.join(output_folder, "failed_images.txt")
metadata_csv = os.path.join(output_folder, "NewMetadata.csv")

# Definir parámetros
target_size = 375
threshold = 0.7  # 70% de los píxeles deben cumplir la condición
image_limit = limit  # Límite de imágenes (0 = sin límite)

# Mapeo de etiquetas
radboud_labels = {
    "benign": {1, 2},
    "GP3": {3},
    "GP4": {4},
    "GP5": {5}
}
karolinska_labels = {
    "benign": {1},
    "GP3": set(),
    "GP4": set(),
    "GP5": set()
}

# Funciones para cargar imágenes y máscaras con PIL
def load_large_image_pil(image_path, mode='RGB'):
    try:
        with Image.open(image_path) as img:
            return np.array(img.convert(mode))
    except Exception as e:
        print(f"Error cargando imagen PIL {image_path}: {e}")
        return None

def load_mask_pil(mask_path):
    try:
        with Image.open(mask_path) as img:
            mask = np.array(img)
            if len(mask.shape) > 2:
                mask = mask[..., 0]
            return mask
    except Exception as e:
        print(f"Error cargando máscara PIL {mask_path}: {e}")
        return None

# Funciones para cargar imágenes y máscaras con OpenCV
def load_large_image_cv2(image_path):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"OpenCV no pudo cargar la imagen: {image_path}")
            return None
    except Exception as e:
        print(f"Error cargando imagen OpenCV {image_path}: {e}")
        return None

# Función para extraer parches
def extract_patches(image_path, mask_path, image_id, output_dir, label_mapping, isup_grade, image_to_provider):
    provider = image_to_provider.get(image_id)
    if provider is None:
        print(f"Proveedor no encontrado para la imagen {image_id}")
        return []

    # Cargar imagen y máscara según el método seleccionado
    image = load_large_image_cv2(image_path)
    mask = load_mask_pil(mask_path)  # carga la mascara con PIL para obtener las etiquetas correctas.

    if image is None or mask is None:
        print(f"Error cargando {image_path} o {mask_path}")
        return []

    h, w = image.shape[:2]
    patch_idx = 0
    patch_counts = {label: 0 for label in label_mapping.keys()}
    patch_metadata_list = []
    benign_patch_count = 0  # Contador de parches benignos

    for y in range(0, h, target_size):
        for x in range(0, w, target_size):
            if y + target_size > h or x + target_size > w:
                continue

            patch = image[y:y + target_size, x:x + target_size]
            patch_mask = mask[y:y + target_size, x:x + target_size]

            total_pixels = patch_mask.size

            for label_name, valid_labels in label_mapping.items():
                # Si isup_grade == 0, solo extraer parches benignos
                if isup_grade == 0 and label_name != "benign":
                    continue
                # Si isup_grade > 0, solo extraer parches malignos
                if isup_grade > 0 and label_name == "benign":
                    continue

                valid_pixels = np.isin(patch_mask, list(valid_labels)).sum()
                valid_ratio = valid_pixels / total_pixels

                if valid_ratio >= threshold:
                    # Limitar la cantidad de parches benignos a 40 por imagen
                    if isup_grade == 0 and label_name == "benign" and benign_patch_count >= 40:
                        continue

                    patch_counts[label_name] += 1
                    patch_filename = f"{image_id}_{patch_idx}_{label_name}.png"
                    patch_path = os.path.join(patches_folder, patch_filename)

                    cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))  # guardamos el patch usando opencv.

                    patch_metadata_list.append([image_id, provider, patch_path, label_name, x, y])
                    patch_idx += 1

                    if isup_grade == 0 and label_name == "benign":
                        benign_patch_count += 1

    print(f"Imagen {image_id}: {patch_counts}")
    return patch_metadata_list

def load_progress():
    # Cargar imágenes ya procesadas si existe el archivo
    processed_images = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
    
    # Cargar imágenes fallidas si existe el archivo
    failed_images = set()
    if os.path.exists(failed_images_file):
        with open(failed_images_file, 'r') as f:
            failed_images = set(line.strip() for line in f)
    
    return processed_images, failed_images

def load_existing_metadata():
    # Cargar metadata existente si existe el archivo
    if os.path.exists(metadata_csv):
        return pd.read_csv(metadata_csv)
    return pd.DataFrame(columns=["image_id", "provider", "patch_path", "label_type", "x", "y"])

def save_batch_metadata(new_metadata, batch_num):
    # Guardar metadata del batch actual
    df_new = pd.DataFrame(new_metadata, columns=["image_id", "provider", "patch_path", "label_type", "x", "y"])
    batch_csv = f"{metadata_csv.replace('.csv', '')}_batch_{batch_num}.csv"
    df_new.to_csv(batch_csv, index=False)
    
    # Actualizar el archivo principal de metadata
    df_existing = load_existing_metadata()
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(metadata_csv, index=False)
    
    return df_combined

def mark_as_processed(image_id):
    with open(progress_file, 'a') as f:
        f.write(f"{image_id}\n")

def mark_as_failed(image_id):
    with open(failed_images_file, 'a') as f:
        f.write(f"{image_id}\n")

def process_image(image_id, df, image_folder, mask_folder, patches_folder, radboud_labels, karolinska_labels, image_to_provider):
    try:
        image_path = os.path.join(image_folder, f"{image_id}.tiff")
        mask_path = os.path.join(mask_folder, f"{image_id}_mask.tiff")
        
        if not os.path.exists(mask_path):
            print(f"La máscara para la imagen {image_id} no existe. Omitiendo esta imagen.")
            mark_as_failed(image_id)
            return []
        
        provider = image_to_provider.get(image_id)
        isup_grade = df[df['image_id'] == image_id]['isup_grade'].values[0]
        
        if provider == "radboud":
            metadata = extract_patches(image_path, mask_path, image_id, patches_folder, radboud_labels, isup_grade, image_to_provider)
        elif provider == "karolinska":
            metadata = extract_patches(image_path, mask_path, image_id, patches_folder, karolinska_labels, isup_grade, image_to_provider)
        
        mark_as_processed(image_id)
        return metadata
    
    except Exception as e:
        print(f"Error procesando imagen {image_id}: {e}")
        mark_as_failed(image_id)
        return []

def process_images_in_batches():
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(patches_folder, exist_ok=True)
    os.makedirs(grids_folder, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    image_to_provider = dict(zip(df['image_id'], df['data_provider']))
    
    df_filtered = df[(df['isup_grade'] >= 0)]
    image_list = df_filtered['image_id'].tolist()
    
    processed_images, failed_images = load_progress()
    print(f"Imágenes previamente procesadas: {len(processed_images)}")
    print(f"Imágenes previamente fallidas: {len(failed_images)}")
    
    pending_images = [img for img in image_list if img not in processed_images and img not in failed_images]
    print(f"Imágenes pendientes por procesar: {len(pending_images)}")
    
    if image_limit > 0 and len(pending_images) > image_limit:
        print(f"Limitando el procesamiento a {image_limit} imágenes de {len(pending_images)} pendientes")
        pending_images = pending_images[:image_limit]
    
    # Usar multiprocesamiento para procesar imágenes en paralelo
    num_processes = multiprocessing.cpu_count()  # Número de núcleos de CPU disponibles
    pool = multiprocessing.Pool(processes=num_processes)
    
    batch_num = 0
    for i in range(0, len(pending_images), batch_size):
        batch_num += 1
        batch_end = min(i + batch_size, len(pending_images))
        print(f"Procesando lote {batch_num} ({i + 1}-{batch_end} de {len(pending_images)})")
        
        batch_images = pending_images[i:batch_end]
        results = pool.starmap(process_image, [(img, df, image_folder, mask_folder, patches_folder, radboud_labels, karolinska_labels, image_to_provider) for img in batch_images])
        
        batch_metadata = [item for sublist in results for item in sublist]
        
        if batch_metadata:
            df_combined = save_batch_metadata(batch_metadata, batch_num)
            print(f"Metadata guardada. Total acumulado: {len(df_combined)}")
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    print(f"Configuración:")
    print(f"- Límite de imágenes: {'Sin límite' if image_limit == 0 else image_limit}")
    print(f"- Tamaño de lote: {batch_size}")
    print(f"- Método de carga: {'OpenCV' if use_opencv else 'PIL'}")
    
    # Procesar imágenes en lotes
    process_images_in_batches()