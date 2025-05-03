#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Cargar el archivo CSV
df = pd.read_csv('00_confusion_matrix_log.csv')

# Extraer información del nombre del modelo
def extract_model_info(model_name):
    pattern = r'(\d+)_(\d+)_trained_model'
    match = re.search(pattern, model_name)
    
    if match:
        model_id = match.group(1)
        percentage = int(match.group(2))
        return model_id, percentage
    else:
        return None, None

# Aplicar la función para extraer información
df[['Model_ID', 'Percentage']] = pd.DataFrame(
    df['Modelo'].apply(extract_model_info).tolist(), 
    index=df.index
)

# Ordenar por porcentaje para visualizaciones consistentes
df = df.sort_values(by='Percentage')

# Definir colores específicos para cada clase
class_colors = {
    'Benign': '#2E8B57',  # Verde mar
    'GP3': '#4169E1',     # Azul real
    'GP4': '#FF8C00',     # Naranja oscuro
    'GP5': '#DC143C'      # Carmesí
}

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 10))

# 1. Gráfico de accuracy vs porcentaje
plt.subplot(2, 2, 1)
sns.lineplot(x='Percentage', y='Accuracy Global', data=df, marker='o', linewidth=2, color='#9370DB')
plt.title('Accuracy Global vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('Accuracy Global')
plt.grid(True)

# 2. Gráfico de Kappa Cuadrático vs porcentaje
plt.subplot(2, 2, 2)
sns.lineplot(x='Percentage', y='Kappa Cuadrático', data=df, marker='o', linewidth=2, color='#20B2AA')
plt.title('Kappa Cuadrático vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('Kappa Cuadrático')
plt.grid(True)

# 3. Gráfico de AUC por clase vs porcentaje
plt.subplot(2, 2, 3)
sns.lineplot(x='Percentage', y='AUC Benign', data=df, marker='o', linewidth=2, 
             label='Benign', color=class_colors['Benign'])
sns.lineplot(x='Percentage', y='AUC GP3', data=df, marker='s', linewidth=2, 
             label='GP3', color=class_colors['GP3'])
sns.lineplot(x='Percentage', y='AUC GP4', data=df, marker='^', linewidth=2, 
             label='GP4', color=class_colors['GP4'])
sns.lineplot(x='Percentage', y='AUC GP5', data=df, marker='D', linewidth=2, 
             label='GP5', color=class_colors['GP5'])
plt.title('AUC por Clase vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

# 4. Gráfico de F1-score por clase vs porcentaje
plt.subplot(2, 2, 4)
sns.lineplot(x='Percentage', y='Benign F1', data=df, marker='o', linewidth=2, 
             label='Benign', color=class_colors['Benign'])
sns.lineplot(x='Percentage', y='GP3 F1', data=df, marker='s', linewidth=2, 
             label='GP3', color=class_colors['GP3'])
sns.lineplot(x='Percentage', y='GP4 F1', data=df, marker='^', linewidth=2, 
             label='GP4', color=class_colors['GP4'])
sns.lineplot(x='Percentage', y='GP5 F1', data=df, marker='D', linewidth=2, 
             label='GP5', color=class_colors['GP5'])
plt.title('F1-Score por Clase vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metricas_vs_porcentaje.png', dpi=300)
plt.show()

# Crear una visualización adicional para precision y recall
plt.figure(figsize=(15, 12))

# Precision por clase
plt.subplot(2, 1, 1)
sns.lineplot(x='Percentage', y='Benign Precision', data=df, marker='o', linewidth=2, 
             label='Benign', color=class_colors['Benign'])
sns.lineplot(x='Percentage', y='GP3 Precision', data=df, marker='s', linewidth=2, 
             label='GP3', color=class_colors['GP3'])
sns.lineplot(x='Percentage', y='GP4 Precision', data=df, marker='^', linewidth=2, 
             label='GP4', color=class_colors['GP4'])
sns.lineplot(x='Percentage', y='GP5 Precision', data=df, marker='D', linewidth=2, 
             label='GP5', color=class_colors['GP5'])
plt.title('Precision por Clase vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall por clase
plt.subplot(2, 1, 2)
sns.lineplot(x='Percentage', y='Benign Recall', data=df, marker='o', linewidth=2, 
             label='Benign', color=class_colors['Benign'])
sns.lineplot(x='Percentage', y='GP3 Recall', data=df, marker='s', linewidth=2, 
             label='GP3', color=class_colors['GP3'])
sns.lineplot(x='Percentage', y='GP4 Recall', data=df, marker='^', linewidth=2, 
             label='GP4', color=class_colors['GP4'])
sns.lineplot(x='Percentage', y='GP5 Recall', data=df, marker='D', linewidth=2, 
             label='GP5', color=class_colors['GP5'])
plt.title('Recall por Clase vs. Porcentaje de Datos')
plt.xlabel('Porcentaje de Datos Usados (%)')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('precision_recall_vs_porcentaje.png', dpi=300)
plt.show()
# %%
