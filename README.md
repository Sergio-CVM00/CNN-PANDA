# Clasificación de Cáncer de Próstata mediante Aprendizaje Profundo

Repositorio para el entrenamiento y evaluación de modelos de clasificación de grados de cáncer de próstata (ISUP) utilizando parches de imágenes histopatológicas.

---

## 📁 Estructura del Repositorio
Principales scripts y su función:
- **`procesamiento_imagenes.py`**: Extrae parches de imágenes TIFF.
- **`balanceo.py`**: Divide los datos en conjuntos (train/val/test) y balancea las clases.
- **`00_train.py`**: Entrena un modelo basado en DenseNet121 con aumento de datos y *focal loss*.
- **`00_test.py`**: Evalúa el modelo en el conjunto de test y genera métricas (Kappa, F1-score).
- **`00_CMatrixROC.py`**: Genera matrices de confusión, curvas ROC y reportes de clasificación.
- **`00_visualize_metrics.py`**: Visualiza métricas guardadas en CSV (accuracy, AUC, etc.).

---
