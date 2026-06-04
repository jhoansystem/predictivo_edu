"""
=============================================================================
SCRIPT DE PREPARACIÓN DE DATOS Y ENTRENAMIENTO DE MODELOS
Sistema de Alerta Temprana Académica basado en Machine Learning
=============================================================================
Este script:
1. Lee el CSV limpio (Resultados_2015-2024_limpio.csv)
2. Agrega los datos a nivel de estudiante-año
3. Crea 3 datasets separados para cada modelo
4. Guarda todo en un archivo Excel con 3 hojas
5. Entrena los 3 modelos (Reg. Lineal, Reg. Logística, Árbol de Decisión)
6. Serializa los modelos y el scaler con joblib
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# ============================================================================
# PASO 1: CARGA DE DATOS LIMPIOS
# ============================================================================
print("=" * 60)
print("PASO 1: Cargando datos limpios...")
print("=" * 60)

df = pd.read_csv('Resultados_2015-2024_limpio.csv', low_memory=False)
print(f"  Dimensiones del dataset limpio: {df.shape}")
print(f"  Columnas: {list(df.columns)}")

# ============================================================================
# PASO 2: AGREGACIÓN A NIVEL ESTUDIANTE-AÑO
# ============================================================================
print("\n" + "=" * 60)
print("PASO 2: Agregando datos a nivel estudiante-año...")
print("=" * 60)

# --- Datos base del estudiante ---
datos_estudiante = df.groupby(['matricula', 'ano']).agg(
    grado=('grado', 'first'),
    grupo=('grupo', 'first')
).reset_index()
datos_estudiante['cohorte'] = datos_estudiante['ano'].astype(str) + '-' + datos_estudiante['grado'].astype(str)
print(f"  Estudiantes únicos por año: {len(datos_estudiante)}")

# --- Promedio anual de notas ---
promedio_anual = df.groupby(['matricula', 'ano']).agg(
    promedio_nota=('nota', 'mean')
).reset_index()

# --- Intensidad total semanal ---
# Asegurar que intensidad_semannal sea numérica
df['intensidad_semannal'] = pd.to_numeric(df['intensidad_semannal'], errors='coerce').fillna(1).astype(int)
intensidad_total = df.groupby(['matricula', 'ano']).agg(
    intensidad_total_semanal=('intensidad_semannal', 'sum')
).reset_index()

# --- Cantidad de asignaturas ---
asignaturas_count = df.groupby(['matricula', 'ano']).agg(
    cantidad_asignaturas=('nommat', 'nunique')
).reset_index()

# --- Total de faltas ---
faltas_total = df.groupby(['matricula', 'ano']).agg(
    total_faltas=('faltas', 'sum')
).reset_index()

# --- Porcentaje de inasistencia ---
merged_f = faltas_total.merge(intensidad_total, on=['matricula', 'ano'])
merged_f['porc_inasistencia'] = (merged_f['total_faltas'] / merged_f['intensidad_total_semanal'].replace(0, np.nan)) * 100
merged_f['porc_inasistencia'] = merged_f['porc_inasistencia'].fillna(0)

# --- Materias perdidas (nota < 3.0) ---
df['materia_perdida'] = (df['nota'] < 3.0).astype(int)
materias_perdidas = df.groupby(['matricula', 'ano']).agg(
    materias_perdidas=('materia_perdida', 'sum')
).reset_index()

# --- Total de refuerzos ---
df['refuerzo_num'] = pd.to_numeric(df['refuerzo'], errors='coerce').fillna(0).astype(int)
refuerzos_count = df.groupby(['matricula', 'ano']).agg(
    total_refuerzos=('refuerzo_num', 'sum')
).reset_index()

# --- Nota mínima y máxima ---
nota_min = df.groupby(['matricula', 'ano']).agg(nota_minima=('nota', 'min')).reset_index()
nota_max = df.groupby(['matricula', 'ano']).agg(nota_maxima=('nota', 'max')).reset_index()
nota_std = df.groupby(['matricula', 'ano']).agg(nota_std=('nota', 'std')).reset_index()
nota_std['nota_std'] = nota_std['nota_std'].fillna(0)

# --- Promedio por área del saber (pivot) ---
# Mapear codmat a áreas del saber simplificadas
area_mapping = {}
for codmat in df['codmat'].unique():
    area_mapping[codmat] = codmat  # Usar codmat directamente como ID de área

df['area_id'] = df['codmat'].astype(str)
promedio_area = df.groupby(['matricula', 'ano', 'area_id']).agg(
    prom_area=('nota', 'mean')
).reset_index()

# Pivot: una columna por área
promedio_area_pivot = promedio_area.pivot_table(
    index=['matricula', 'ano'],
    columns='area_id',
    values='prom_area',
    aggfunc='mean'
).reset_index()

# Renombrar columnas de áreas
promedio_area_pivot.columns = [
    f'prom_{col}.' if col not in ['matricula', 'ano'] else col
    for col in promedio_area_pivot.columns
]

# Seleccionar solo las áreas más comunes (al menos 50% de cobertura)
min_coverage = len(datos_estudiante) * 0.3
area_cols = [c for c in promedio_area_pivot.columns if c.startswith('prom_')]
valid_area_cols = [c for c in area_cols if promedio_area_pivot[c].notna().sum() >= min_coverage]
promedio_area_pivot = promedio_area_pivot[['matricula', 'ano'] + valid_area_cols]

# --- Target: promovido binario ---
df['promovido_clean'] = df['promovido'].fillna('N')
promovido_target = df.groupby(['matricula', 'ano']).agg(
    promovido=('promovido_clean', 'first')
).reset_index()
promovido_target['promovido_bin'] = (promovido_target['promovido'] == 'S').astype(int)

# ============================================================================
# PASO 3: CONSTRUCCIÓN DE LOS 3 DATASETS
# ============================================================================
print("\n" + "=" * 60)
print("PASO 3: Construyendo los 3 datasets para modelado...")
print("=" * 60)

def merge_features(base, other, keys=['matricula', 'ano']):
    """Helper para merge sin duplicar columnas."""
    cols = [c for c in other.columns if c in keys or c not in base.columns]
    return base.merge(other[cols], on=keys, how='left')

# --- 3.1 Dataset para Regresión Lineal ---
# Target: promedio_nota (continuo 0.0-5.0)
df_lr = datos_estudiante.copy()
for mdf in [promedio_anual, intensidad_total, asignaturas_count, faltas_total,
            merged_f[['matricula', 'ano', 'porc_inasistencia']],
            materias_perdidas, refuerzos_count, nota_min, nota_max, nota_std]:
    df_lr = merge_features(df_lr, mdf)
df_lr = merge_features(df_lr, promedio_area_pivot)

print(f"  Dataset Regresión Lineal: {df_lr.shape}")
print(f"    Target: promedio_nota (rango: {df_lr['promedio_nota'].min():.2f} - {df_lr['promedio_nota'].max():.2f})")

# --- 3.2 Dataset para Regresión Logística ---
# Target: promovido_bin (0=No, 1=Sí)
df_log = datos_estudiante.copy()
for mdf in [promedio_anual, intensidad_total, asignaturas_count, faltas_total,
            merged_f[['matricula', 'ano', 'porc_inasistencia']],
            materias_perdidas, refuerzos_count, nota_min, nota_max, nota_std]:
    df_log = merge_features(df_log, mdf)
df_log = merge_features(df_log, promedio_area_pivot)
df_log = merge_features(df_log, promovido_target[['matricula', 'ano', 'promovido', 'promovido_bin']])

print(f"  Dataset Regresión Logística: {df_log.shape}")
print(f"    Distribución target: {df_log['promovido_bin'].value_counts().to_dict()}")

# --- 3.3 Dataset para Árboles de Decisión ---
# Mismas variables + interacciones
df_dt = df_log.copy()
df_dt['promedio_x_asignaturas'] = df_dt['promedio_nota'] * df_dt['cantidad_asignaturas']
df_dt['faltas_x_materias_perdidas'] = df_dt['total_faltas'] * df_dt['materias_perdidas']

print(f"  Dataset Árboles de Decisión: {df_dt.shape}")

# ============================================================================
# PASO 4: IMPUTACIÓN DE NULOS Y LIMPIEZA FINAL
# ============================================================================
print("\n" + "=" * 60)
print("PASO 4: Imputación de nulos y limpieza final...")
print("=" * 60)

# Imputar nulos en columnas de promedios por área con el promedio general del estudiante
for dataset_name, dataset in [('RL', df_lr), ('RLog', df_log), ('AD', df_dt)]:
    area_cols_in_ds = [c for c in dataset.columns if c.startswith('prom_')]
    for col in area_cols_in_ds:
        n_nulls = dataset[col].isna().sum()
        if n_nulls > 0:
            dataset[col] = dataset[col].fillna(dataset['promedio_nota'])
    # Reemplazar infinitos
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.fillna(0, inplace=True)
    print(f"  {dataset_name}: Nulos restantes = {dataset.isna().sum().sum()}, Infinitos = {np.isinf(dataset.select_dtypes(include=[np.number])).sum().sum()}")

# ============================================================================
# PASO 5: GUARDAR EN EXCEL CON 3 HOJAS
# ============================================================================
print("\n" + "=" * 60)
print("PASO 5: Guardando datasets en Excel...")
print("=" * 60)

excel_path = 'dataset_modelos_aprendizaje.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_lr.to_excel(writer, sheet_name='Regresion_Lineal', index=False)
    df_log.to_excel(writer, sheet_name='Regresion_Logistica', index=False)
    df_dt.to_excel(writer, sheet_name='Arboles_Decision', index=False)

print(f"  Archivo guardado: {excel_path}")

# También guardar CSVs individuales (para compatibilidad con notebooks)
df_lr.to_csv('dataset_regresion_lineal.csv', index=False)
df_log.to_csv('dataset_regresion_logistica.csv', index=False)
df_dt.to_csv('dataset_arboles_decision.csv', index=False)
print("  CSVs individuales guardados.")

# ============================================================================
# PASO 6: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ============================================================================
print("\n" + "=" * 60)
print("PASO 6: Entrenamiento de modelos...")
print("=" * 60)

os.makedirs('modelos', exist_ok=True)

# --- 6.1 Regresión Lineal ---
print("\n  --- 6.1 Regresión Lineal ---")
# Seleccionar features numéricas (excluir identificadores y target)
lr_exclude = ['matricula', 'ano', 'grupo', 'cohorte', 'promedio_nota']
lr_features = [c for c in df_lr.columns if c not in lr_exclude and df_lr[c].dtype in ['int64', 'float64']]

X_lr = df_lr[lr_features].values
y_lr = df_lr['promedio_nota'].values

X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

scaler_lr = StandardScaler()
X_lr_train_scaled = scaler_lr.fit_transform(X_lr_train)
X_lr_test_scaled = scaler_lr.transform(X_lr_test)

modelo_lr = LinearRegression()
modelo_lr.fit(X_lr_train_scaled, y_lr_train)
y_lr_pred = modelo_lr.predict(X_lr_test_scaled)

print(f"    R²: {r2_score(y_lr_test, y_lr_pred):.4f}")
print(f"    MAE: {mean_absolute_error(y_lr_test, y_lr_pred):.4f}")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_lr_test, y_lr_pred)):.4f}")

joblib.dump(modelo_lr, 'modelos/modelo_regresion_lineal.joblib')
joblib.dump(scaler_lr, 'modelos/scaler_regresion_lineal.joblib')
joblib.dump(lr_features, 'modelos/features_regresion_lineal.joblib')
print("    Modelo guardado.")

# --- 6.2 Regresión Logística ---
print("\n  --- 6.2 Regresión Logística ---")
log_exclude = ['matricula', 'ano', 'grupo', 'cohorte', 'promovido', 'promovido_bin']
log_features = [c for c in df_log.columns if c not in log_exclude and df_log[c].dtype in ['int64', 'float64']]

X_log = df_log[log_features].values
y_log = df_log['promovido_bin'].values

X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42, stratify=y_log)

scaler_log = StandardScaler()
X_log_train_scaled = scaler_log.fit_transform(X_log_train)
X_log_test_scaled = scaler_log.transform(X_log_test)

modelo_log = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
modelo_log.fit(X_log_train_scaled, y_log_train)
y_log_pred = modelo_log.predict(X_log_test_scaled)
y_log_proba = modelo_log.predict_proba(X_log_test_scaled)[:, 1]

print(f"    Accuracy: {accuracy_score(y_log_test, y_log_pred):.4f}")
print(f"    Precision: {precision_score(y_log_test, y_log_pred):.4f}")
print(f"    Recall: {recall_score(y_log_test, y_log_pred):.4f}")
print(f"    F1-Score: {f1_score(y_log_test, y_log_pred):.4f}")
print(f"    ROC-AUC: {roc_auc_score(y_log_test, y_log_proba):.4f}")

joblib.dump(modelo_log, 'modelos/modelo_regresion_logistica.joblib')
joblib.dump(scaler_log, 'modelos/scaler_regresion_logistica.joblib')
joblib.dump(log_features, 'modelos/features_regresion_logistica.joblib')
print("    Modelo guardado.")

# --- 6.3 Árbol de Decisión ---
print("\n  --- 6.3 Árbol de Decisión ---")
dt_exclude = ['matricula', 'ano', 'grupo', 'cohorte', 'promovido', 'promovido_bin']
dt_features = [c for c in df_dt.columns if c not in dt_exclude and df_dt[c].dtype in ['int64', 'float64']]

X_dt = df_dt[dt_features].values
y_dt = df_dt['promovido_bin'].values

X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42, stratify=y_dt)

modelo_dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
modelo_dt.fit(X_dt_train, y_dt_train)
y_dt_pred = modelo_dt.predict(X_dt_test)
y_dt_proba = modelo_dt.predict_proba(X_dt_test)[:, 1]

print(f"    Accuracy: {accuracy_score(y_dt_test, y_dt_pred):.4f}")
print(f"    Precision: {precision_score(y_dt_test, y_dt_pred):.4f}")
print(f"    Recall: {recall_score(y_dt_test, y_dt_pred):.4f}")
print(f"    F1-Score: {f1_score(y_dt_test, y_dt_pred):.4f}")
print(f"    ROC-AUC: {roc_auc_score(y_dt_test, y_dt_proba):.4f}")

joblib.dump(modelo_dt, 'modelos/modelo_arbol_decision.joblib')
joblib.dump(dt_features, 'modelos/features_arbol_decision.joblib')
print("    Modelo guardado.")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  Archivos generados:")
print(f"    - dataset_modelos_aprendizaje.xlsx (3 hojas)")
print(f"    - dataset_regresion_lineal.csv")
print(f"    - dataset_regresion_logistica.csv")
print(f"    - dataset_arboles_decision.csv")
print(f"    - modelos/modelo_regresion_lineal.joblib")
print(f"    - modelos/scaler_regresion_lineal.joblib")
print(f"    - modelos/features_regresion_lineal.joblib")
print(f"    - modelos/modelo_regresion_logistica.joblib")
print(f"    - modelos/scaler_regresion_logistica.joblib")
print(f"    - modelos/features_regresion_logistica.joblib")
print(f"    - modelos/modelo_arbol_decision.joblib")
print(f"    - modelos/features_arbol_decision.joblib")
print(f"\n  ¡Proceso completado exitosamente!")
