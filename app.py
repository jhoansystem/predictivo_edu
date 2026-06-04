import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página
st.set_page_config(
    page_title="SATA | Alerta Temprana Académica",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para apariencia premium (Rediseñado)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── BASE ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(160deg, #060f1e 0%, #0a192f 50%, #0d2137 100%) !important;
        min-height: 100vh;
    }
    .main .block-container {
        padding-top: 2rem;
    }

    /* ── TEXTOS GENERALES ── */
    p, li, span, .stMarkdown p {
        color: #a8b2d8 !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e6f1ff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    strong, b {
        color: #ccd6f6 !important;
    }
    a { color: #64ffda !important; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1d35 0%, #112240 100%) !important;
        border-right: 1px solid rgba(100,255,218,0.12) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892b0 !important;
    }
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #ccd6f6 !important;
    }

    /* ── RADIO NAV (sidebar) ── */
    /* Etiqueta del grupo "Navegación" */
    section[data-testid="stSidebar"] .stRadio > label > div > p {
        color: #64ffda !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin-bottom: 0.4rem !important;
    }
    /* Cada ítem de radio */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #a8b2d8 !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        padding: 6px 10px !important;
        border-radius: 7px !important;
        margin: 2px 0 !important;
        transition: all 0.2s !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(100,255,218,0.07) !important;
        color: #64ffda !important;
    }
    /* Círculo del radio seleccionado */
    section[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] div {
        border-color: #64ffda !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] div {
        background-color: #64ffda !important;
        border-color: #64ffda !important;
    }

    /* ── MÉTRICAS ── */
    div[data-testid="metric-container"] {
        background: rgba(17,34,64,0.85) !important;
        border: 1px solid rgba(100,255,218,0.14) !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
        backdrop-filter: blur(8px) !important;
        transition: border-color 0.2s !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(100,255,218,0.35) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8892b0 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 1.2px !important;
    }

    /* ── BOTONES ── */
    .stButton > button {
        background: transparent !important;
        color: #64ffda !important;
        border: 1.5px solid #64ffda !important;
        padding: 0.55rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.4px !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: rgba(100,255,218,0.1) !important;
        box-shadow: 0 4px 18px rgba(100,255,218,0.22) !important;
        transform: translateY(-1px) !important;
    }

    /* ── SELECTBOX / DROPDOWN ── */
    .stSelectbox label,
    .stSelectbox > label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.9px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #112240 !important;
        border: 1px solid rgba(100,255,218,0.22) !important;
        border-radius: 9px !important;
        color: #e6f1ff !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #64ffda !important;
    }
    /* Texto seleccionado dentro del dropdown */
    .stSelectbox div[data-baseweb="select"] span {
        color: #e6f1ff !important;
    }
    /* Lista desplegable */
    ul[data-baseweb="menu"] {
        background-color: #112240 !important;
        border: 1px solid rgba(100,255,218,0.2) !important;
        border-radius: 9px !important;
    }
    ul[data-baseweb="menu"] li {
        color: #ccd6f6 !important;
        background-color: transparent !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: rgba(100,255,218,0.08) !important;
        color: #64ffda !important;
    }

    /* ── NUMBER INPUT ── */
    .stNumberInput label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
    }
    .stNumberInput input {
        background-color: #112240 !important;
        border: 1px solid rgba(100,255,218,0.2) !important;
        border-radius: 8px !important;
        color: #e6f1ff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    .stNumberInput input:focus {
        border-color: #64ffda !important;
        box-shadow: 0 0 0 2px rgba(100,255,218,0.1) !important;
    }
    .stNumberInput button {
        background: rgba(100,255,218,0.06) !important;
        border-color: rgba(100,255,218,0.15) !important;
        color: #64ffda !important;
    }

    /* ── SLIDER ── */
    .stSlider label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #64ffda !important;
        border-color: #64ffda !important;
        box-shadow: 0 0 0 4px rgba(100,255,218,0.18) !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
        color: #8892b0 !important;
    }

    /* ── FORMULARIO ── */
    div[data-testid="stForm"] {
        background: rgba(11,29,53,0.7) !important;
        border: 1px solid rgba(100,255,218,0.12) !important;
        border-radius: 14px !important;
        padding: 1.8rem !important;
        backdrop-filter: blur(10px) !important;
    }

    /* ── DATAFRAME / TABLA ── */
    .stDataFrame {
        border: 1px solid rgba(100,255,218,0.12) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    .stDataFrame th {
        background-color: #112240 !important;
        color: #64ffda !important;
        font-weight: 600 !important;
    }
    .stDataFrame td {
        color: #ccd6f6 !important;
        background-color: rgba(10,25,47,0.6) !important;
    }

    /* ── ALERTAS ── */
    div[data-testid="stAlert"][data-type="success"],
    .stSuccess {
        background: rgba(16,185,129,0.1) !important;
        border: 1px solid rgba(16,185,129,0.35) !important;
        border-radius: 10px !important;
        color: #6ee7b7 !important;
    }
    div[data-testid="stAlert"][data-type="warning"],
    .stWarning {
        background: rgba(245,158,11,0.1) !important;
        border: 1px solid rgba(245,158,11,0.35) !important;
        border-radius: 10px !important;
        color: #fcd34d !important;
    }
    div[data-testid="stAlert"][data-type="error"],
    .stError {
        background: rgba(239,68,68,0.1) !important;
        border: 1px solid rgba(239,68,68,0.35) !important;
        border-radius: 10px !important;
    }

    /* ── BARRA DE PROGRESO ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #64ffda, #00bfa5) !important;
        border-radius: 10px !important;
    }
    .stProgress > div > div > div {
        background: rgba(100,255,218,0.08) !important;
        border-radius: 10px !important;
    }

    /* ── EXPANDER ── */
    details {
        background: rgba(17,34,64,0.5) !important;
        border: 1px solid rgba(100,255,218,0.1) !important;
        border-radius: 9px !important;
    }
    details summary {
        color: #a8b2d8 !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
    }
    details summary:hover {
        color: #64ffda !important;
    }

    /* ── DIVISORES ── */
    hr {
        border-color: rgba(100,255,218,0.1) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #060f1e; }
    ::-webkit-scrollbar-thumb { background: #233554; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d4373; }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos (con caché)
@st.cache_data
def cargar_datos():
    try:
        # Cargamos el dataset de Regresión Logística que contiene la mayoría de las columnas de agregación
        df = pd.read_csv('Datasets/dataset_regresion_logistica.csv')
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None

# Función para cargar modelos
def _patch_sklearn_compat(model):
    """Parche de compatibilidad para modelos guardados con scikit-learn < 1.5.
    La versión 1.5+ eliminó el atributo 'multi_class' de LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        if not hasattr(model, 'multi_class'):
            model.multi_class = 'auto'
        if not hasattr(model, 'l1_ratio'):
            model.l1_ratio = None
    return model

def cargar_modelo(tipo):
    try:
        if tipo == 'logistica':
            model = joblib.load('Modelos/modelo_regresion_logistica.joblib')
            model = _patch_sklearn_compat(model)
            scaler = joblib.load('Modelos/scaler_regresion_logistica.joblib')
            features = joblib.load('Modelos/features_regresion_logistica.joblib')
            return model, scaler, features
        elif tipo == 'arbol':
            model = joblib.load('Modelos/modelo_arbol_decision.joblib')
            features = joblib.load('Modelos/features_arbol_decision.joblib')
            return model, None, features
        elif tipo == 'lineal':
            model = joblib.load('Modelos/modelo_regresion_lineal.joblib')
            scaler = joblib.load('Modelos/scaler_regresion_lineal.joblib')
            features = joblib.load('Modelos/features_regresion_lineal.joblib')
            return model, scaler, features
    except Exception as e:
        st.error(f"Error al cargar el modelo o componentes: {e}")
        return None, None, None


# Cargar dataset base
df = cargar_datos()

# ============================================================================
# PANEL LATERAL (SIDEBAR)
# ============================================================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem;'>🏫 SATA · ML</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #64ffda; margin-top: -1rem;'>Sistema de Alerta Temprana Académica</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navegación principal
    seccion = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Análisis Exploratorio", "🔮 Predicción del Modelo", "🧠 Quiz de Conocimiento"]
    )
    
    st.markdown("---")
    st.markdown("""
    **Acerca del Proyecto:**
    Este sistema predictivo de Machine Learning ayuda a prevenir la reprobación académica mediante la identificación de patrones de riesgo en estudiantes de bachillerato (6° a 11°).
    
    **Tecnologías:**
    - Python 3.11
    - Streamlit
    - Scikit-Learn
    - Plotly
    
    *Bootcamp Talento Tech Colombia - 2025*
    """)

# ============================================================================
# SECCIÓN 1: INICIO (HOME)
# ============================================================================
if seccion == "🏠 Inicio":
    st.markdown("<h1 style='color: #e6f1ff;'>🏫 Sistema de Alerta Temprana Académica</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #64ffda;'>Machine Learning aplicado a la prevención de la deserción escolar</p>", unsafe_allow_html=True)
    st.write("---")
    
    # Métricas clave
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Estudiantes Únicos", f"{df['matricula'].nunique():,}")
        with col2:
            st.metric("Registros Agregados", f"{len(df):,}")
        with col3:
            tasa_aprobacion = (df['promovido_bin'] == 1).mean() * 100
            st.metric("Tasa de Aprobación General", f"{tasa_aprobacion:.1f}%")
        with col4:
            anos_cobertura = df['ano'].nunique()
            st.metric("Años de Datos", f"{anos_cobertura} años")
            
    # Contenido principal
    st.markdown("### 🎯 Objetivo del Sistema")
    st.markdown("""
    El sistema tiene como propósito principal **predecir si un estudiante de bachillerato aprobará o no aprobará el año lectivo escolar**, a partir de su rendimiento actual (calificaciones, desviación de notas), historial de inasistencias, cantidad de materias reprobadas en periodos previos y el apoyo en actividades de refuerzo escolar.
    
    Esta predicción se genera usando modelos de inteligencia artificial y se orienta bajo las directrices del **SIEE (Sistema Institucional de Evaluación de Estudiantes)** en Colombia.
    """)
    
    # Metodología
    st.markdown("### 🛠️ Metodología CRISP-ML")
    st.markdown("""
    El proyecto fue desarrollado siguiendo el estándar de calidad **CRISP-ML**, el cual consta de 7 fases:
    1. **Comprensión del Negocio:** Definición de objetivos pedagógicos y normativos.
    2. **Comprensión de los Datos:** Análisis exploratorio de 331,787 registros institucionales históricos (2015-2024).
    3. **Preparación de los Datos:** Agregación a nivel de estudiante-año, imputación inteligente de vacíos en áreas específicas y feature engineering.
    4. **Modelado:** Entrenamiento y optimización de Regresión Lineal, Regresión Logística y Árboles de Decisión.
    5. **Evaluación:** Validación rigurosa de métricas de negocio y de modelado (Accuracy > 96.5%).
    6. **Despliegue:** Implementación de esta plataforma web interactiva en Streamlit Cloud.
    7. **Monitoreo y Mantenimiento:** Planificación para re-entrenamiento del modelo con nuevos datos anuales.
    """)

# ============================================================================
# SECCIÓN 2: EDA (ANÁLISIS EXPLORATORIO DE DATOS)
# ============================================================================
elif seccion == "📊 Análisis Exploratorio":
    st.markdown("<h1>📊 Análisis Exploratorio de Datos (EDA)</h1>", unsafe_allow_html=True)
    st.write("Visualización interactiva del dataset institucional consolidado.")
    st.write("---")
    
    if df is not None:
        tipo_grafico = st.selectbox(
            "Selecciona la vista de análisis:",
            ["Estadísticas Descriptivas del Dataset", 
             "Distribución de Estudiantes y Aprobación", 
             "Análisis de Notas y Rendimiento",
             "Relación de Inasistencias y Pérdidas",
             "Mapa de Correlaciones Numéricas"]
        )
        
        # 2.1 Estadísticas Descriptivas
        if tipo_grafico == "Estadísticas Descriptivas del Dataset":
            st.markdown("### 📋 Estadísticas Descriptivas Generales")
            st.markdown("A continuación se presenta un resumen estadístico detallado de las variables del dataset después del proceso de limpieza e imputación:")
            st.dataframe(df.describe().T, use_container_width=True)
            
        # 2.2 Distribución de Estudiantes
        elif tipo_grafico == "Distribución de Estudiantes y Aprobación":
            st.markdown("### 👥 Distribución y Aprobación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Estudiantes por Grado Escolar")
                df_grado = df['grado'].value_counts().reset_index()
                df_grado.columns = ['Grado', 'Cantidad']
                df_grado = df_grado.sort_values('Grado')
                fig_grado = px.bar(df_grado, x='Grado', y='Cantidad', color='Cantidad',
                                   color_continuous_scale='teal', template='plotly_dark')
                st.plotly_chart(fig_grado, use_container_width=True)
                
            with col2:
                st.markdown("#### Tasa de Promoción por Año")
                tasa_ano = df.groupby('ano')['promovido_bin'].mean().reset_index()
                tasa_ano['Tasa (%)'] = tasa_ano['promovido_bin'] * 100
                fig_ano = px.line(tasa_ano, x='ano', y='Tasa (%)', markers=True,
                                  color_discrete_sequence=['#64ffda'], template='plotly_dark')
                fig_ano.update_yaxes(range=[50, 100])
                st.plotly_chart(fig_ano, use_container_width=True)
                
        # 2.3 Análisis de Notas
        elif tipo_grafico == "Análisis de Notas y Rendimiento":
            st.markdown("### 📈 Notas y Rendimiento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribución de la Nota Promedio General")
                fig_hist_nota = px.histogram(df, x='promedio_nota', nbins=50,
                                             color_discrete_sequence=['#64ffda'], template='plotly_dark')
                st.plotly_chart(fig_hist_nota, use_container_width=True)
                
            with col2:
                st.markdown("#### Nota Promedio por Grado Escolar")
                fig_box_nota = px.box(df, x='grado', y='promedio_nota', color='grado',
                                      template='plotly_dark')
                st.plotly_chart(fig_box_nota, use_container_width=True)
                
        # 2.4 Inasistencias y Pérdidas
        elif tipo_grafico == "Relación de Inasistencias y Pérdidas":
            st.markdown("### ⚠️ Inasistencias, Pérdidas y Promoción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribución de Materias Perdidas según Promoción")
                # Cambiar 0/1 por texto
                df_temp = df.copy()
                df_temp['Estado'] = df_temp['promovido_bin'].map({1: 'Promovido (Aprueba)', 0: 'No Promovido (Reprueba)'})
                fig_box_perdidas = px.box(df_temp, x='Estado', y='materias_perdidas', color='Estado',
                                          color_discrete_map={'Promovido (Aprueba)': '#10b981', 'No Promovido (Reprueba)': '#f59e0b'},
                                          template='plotly_dark')
                st.plotly_chart(fig_box_perdidas, use_container_width=True)
                
            with col2:
                st.markdown("#### Distribución de Inasistencias Totales según Promoción")
                fig_box_faltas = px.box(df_temp, x='Estado', y='total_faltas', color='Estado',
                                        color_discrete_map={'Promovido (Aprueba)': '#10b981', 'No Promovido (Reprueba)': '#f59e0b'},
                                        template='plotly_dark')
                st.plotly_chart(fig_box_faltas, use_container_width=True)
                
        # 2.5 Mapa de Correlaciones
        elif tipo_grafico == "Mapa de Correlaciones Numéricas":
            st.markdown("### 🔗 Matriz de Correlación de Variables")
            st.markdown("Análisis estadístico de la correlación lineal (Pearson) entre variables clave:")
            
            # Filtramos columnas de promedio por área para no sobrecargar el mapa
            cols_corr = ['grado', 'promedio_nota', 'intensidad_total_semanal', 'cantidad_asignaturas',
                         'total_faltas', 'porc_inasistencia', 'materias_perdidas', 'total_refuerzos',
                         'nota_minima', 'nota_maxima', 'nota_std', 'promovido_bin']
            
            corr_matrix = df[cols_corr].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('#0a192f')
            ax.set_facecolor('#0a192f')
            
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap=sns.diverging_palette(220, 150, as_cmap=True),
                fmt=".2f", 
                linewidths=0.5,
                ax=ax,
                annot_kws={"size": 9, "color": "white"}
            )
            
            # Cambiar colores de etiquetas de ejes a blanco
            ax.tick_params(colors='white')
            plt.xticks(rotation=45, ha='right')
            
            st.pyplot(fig)
            plt.close()

# ============================================================================
# SECCIÓN 3: PREDICCIÓN (INTERFACE PREDICTIVA)
# ============================================================================
elif seccion == "🔮 Predicción del Modelo":
    st.markdown("<h1>🔮 Predicción del Desempeño Escolar</h1>", unsafe_allow_html=True)
    st.write("Ingrese los indicadores del estudiante para calcular el nivel de riesgo de reprobación académica.")
    st.write("---")
    
    # Selector de Modelo
    modelo_elegido = st.selectbox(
        "Seleccione el modelo de clasificación a utilizar:",
        ["Regresión Logística (Recomendado)", "Árbol de Decisión"]
    )
    
    # Cargar el modelo correspondiente
    tipo_mod = 'logistica' if modelo_elegido == "Regresión Logística (Recomendado)" else 'arbol'
    model, scaler, feature_names = cargar_modelo(tipo_mod)
    
    if model is not None and feature_names is not None:
        st.markdown("### 📝 Formulario de Datos del Estudiante")
        
        # Formulario de entrada
        with st.form("form_prediccion"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grado = st.slider("Grado Escolar:", min_value=6, max_value=11, value=9)
                promedio_nota = st.number_input("Promedio General de Notas (0.0 a 5.0):", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
                total_faltas = st.number_input("Total de Faltas en el Año:", min_value=0, max_value=300, value=10)
                cantidad_asignaturas = st.number_input("Cantidad de Asignaturas Cursadas:", min_value=1, max_value=25, value=11)
                
            with col2:
                materias_perdidas = st.number_input("Materias Perdidas (Nota < 3.0):", min_value=0, max_value=15, value=1)
                total_refuerzos = st.number_input("Total de Refuerzos Tomados:", min_value=0, max_value=50, value=2)
                porc_inasistencia = st.slider("Porcentaje de Inasistencia (%):", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
                intensidad_total_semanal = st.number_input("Intensidad Horaria Semanal (Total horas):", min_value=1, max_value=200, value=100)
                
            with col3:
                nota_minima = st.number_input("Nota Mínima Obtenida (0.0 a 5.0):", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
                nota_maxima = st.number_input("Nota Máxima Obtenida (0.0 a 5.0):", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
                nota_std = st.number_input("Desviación Estándar de Notas (nota_std):", min_value=0.0, max_value=2.5, value=0.6, step=0.05)
                
            submit_pred = st.form_submit_button("🚀 Realizar Predicción")
            
        if submit_pred:
            # Construir el registro de predicción según las columnas de entrenamiento
            # Crear un diccionario inicial con los datos básicos
            data_dict = {
                'grado': grado,
                'promedio_nota': promedio_nota,
                'intensidad_total_semanal': intensidad_total_semanal,
                'cantidad_asignaturas': cantidad_asignaturas,
                'total_faltas': total_faltas,
                'porc_inasistencia': porc_inasistencia,
                'materias_perdidas': materias_perdidas,
                'total_refuerzos': total_refuerzos,
                'nota_minima': nota_minima,
                'nota_maxima': nota_maxima,
                'nota_std': nota_std
            }
            
            # Si el modelo es el Árbol de Decisión, calcular interacciones
            if tipo_mod == 'arbol':
                data_dict['promedio_x_asignaturas'] = promedio_nota * cantidad_asignaturas
                data_dict['faltas_x_materias_perdidas'] = total_faltas * materias_perdidas
                
            # Rellenar las columnas de promedios por áreas (ej. prom_1., prom_10.) con el promedio general ingresado
            for col in feature_names:
                if col not in data_dict:
                    if col.startswith('prom_'):
                        data_dict[col] = promedio_nota
                    else:
                        data_dict[col] = 0.0  # Fallback de seguridad
                        
            # Crear DataFrame con el orden exacto de las columnas de entrenamiento
            X_df = pd.DataFrame([data_dict])[feature_names]
            
            # Escalar si es Regresión Logística
            if scaler is not None:
                X_input = scaler.transform(X_df.values)
            else:
                X_input = X_df.values
                
            # Predicción
            pred_class = model.predict(X_input)[0]
            pred_proba = model.predict_proba(X_input)[0][1] # Probabilidad de aprobación (clase 1)
            
            st.write("---")
            st.markdown("### 🏆 Resultado del Análisis de Alerta Temprana")
            
            # Mostrar resultado con "bombas y platillos"
            col_res1, col_res2 = st.columns([1.5, 2])
            
            with col_res1:
                if pred_class == 1:
                    st.success("### ✅ ESTUDIANTE PROMOVIDO")
                    st.markdown("""
                    El modelo clasifica al estudiante con **Bajo Riesgo Académico**.
                    
                    **Métricas del modelo:**
                    - Probabilidad de Aprobación: **{:.1f}%**
                    """.format(pred_proba * 100))
                    
                    # Efectos visuales de celebración
                    st.balloons()
                    st.snow()
                else:
                    st.warning("### ⚠️ ALERTA: NO PROMOVIDO (EN RIESGO)")
                    st.markdown("""
                    El modelo clasifica al estudiante con **Alto Riesgo Académico** de reprobación.
                    
                    **Métricas del modelo:**
                    - Probabilidad de Aprobación: **{:.1f}%**
                    - Nivel de Riesgo: **Alto**
                    """.format(pred_proba * 100))
                    
            with col_res2:
                # Mostrar barra de progreso/métrica
                st.markdown("#### Nivel de Aprobación Estimado")
                st.progress(float(pred_proba))
                
                # Recomendaciones Pedagógicas (SIEE Colombia)
                st.markdown("#### 💡 Recomendaciones Pedagógicas:")
                if pred_class == 1:
                    st.markdown("""
                    1. **Mantener el ritmo actual:** Incentivar al estudiante a sostener su nivel académico en el último periodo.
                    2. **Participación de liderazgo:** Apoyar a compañeros con mayor dificultad académica actuando como monitor en asignaturas fuertes.
                    """)
                else:
                    st.markdown("""
                    1. **Plan de Apoyo Pedagógico Inmediato:** Matricular al estudiante en actividades de refuerzo prioritarias para las materias perdidas.
                    2. **Seguimiento a Inasistencias:** Programar cita con acudiente para investigar el motivo de las inasistencias y crear compromisos.
                    3. **Flexibilización Curricular:** Ajustar entregas académicas y evaluaciones de acuerdo a las dificultades detectadas en la nota mínima.
                    """)

# ============================================================================
# SECCIÓN 4: QUIZ DE MACHINE LEARNING (PAGADÓGICO)
# ============================================================================
elif seccion == "🧠 Quiz de Conocimiento":
    st.markdown("<h1>🧠 Quiz Pedagógico: Python & Machine Learning</h1>", unsafe_allow_html=True)
    st.write("Prueba tus conocimientos sobre Inteligencia Artificial y Ciencia de Datos con este cuestionario educativo.")
    st.write("---")
    
    preguntas = [
        {
            "id": 1,
            "pregunta": "¿Qué es el Machine Learning o Aprendizaje Automático?",
            "opciones": [
                "Un programa informático que sigue una serie de reglas fijas codificadas por un programador.",
                "Una rama de la Inteligencia Artificial que permite a las computadoras aprender y mejorar a partir de los datos sin ser programadas explícitamente.",
                "Un sistema operativo diseñado para gestionar grandes cantidades de archivos."
            ],
            "correcta": 1,
            "explicacion": "El Aprendizaje Automático se centra en la creación de algoritmos que analizan datos y extraen patrones para hacer predicciones u tomar decisiones sin intervención humana directa."
        },
        {
            "id": 2,
            "pregunta": "¿Cuál es la principal diferencia entre Regresión y Clasificación?",
            "opciones": [
                "La Regresión predice valores continuos (números), mientras que la Clasificación predice etiquetas discretas (categorías).",
                "La Regresión solo sirve para datos de texto y la Clasificación para imágenes.",
                "No hay diferencia, son dos nombres para el mismo tipo de algoritmo."
            ],
            "correcta": 0,
            "explicacion": "Correcto. Por ejemplo, predecir el promedio numérico de un estudiante es Regresión, mientras que clasificar si Aprueba o Reprueba es Clasificación."
        },
        {
            "id": 3,
            "pregunta": "En un modelo de Regresión Lineal, ¿qué mide la métrica R² (R-cuadrado)?",
            "opciones": [
                "El porcentaje de registros que fueron clasificados correctamente.",
                "La proporción de la variabilidad de la variable objetivo que es explicada por las características del modelo.",
                "El promedio del valor absoluto de los errores de predicción."
            ],
            "correcta": 1,
            "explicacion": "El R² indica qué tan bien se ajustan las características a la tendencia del promedio real. Varía entre 0 y 1 (o 100% de varianza explicada)."
        },
        {
            "id": 4,
            "pregunta": "¿Qué representa una Matriz de Confusión en Clasificación?",
            "opciones": [
                "Una tabla que resume el número de predicciones correctas e incorrectas del modelo frente a los valores reales.",
                "Un conjunto de fórmulas matemáticas complejas para calcular derivadas.",
                "Una lista con todos los nombres de las columnas que contienen errores."
            ],
            "correcta": 0,
            "explicacion": "La Matriz de Confusión contiene Verdaderos Positivos, Verdaderos Negativos, Falsos Positivos y Falsos Negativos, lo que ayuda a evaluar el desempeño detallado del clasificador."
        },
        {
            "id": 5,
            "pregunta": "¿Qué es el sobreajuste (overfitting) de un modelo?",
            "opciones": [
                "Cuando un modelo aprende de forma excelente los datos de entrenamiento pero no logra generalizar con datos nuevos (prueba).",
                "Cuando un modelo no logra aprender nada durante el proceso de entrenamiento y sus resultados son aleatorios.",
                "El proceso de comprimir el tamaño del archivo .joblib del modelo para que ocupe menos espacio."
            ],
            "correcta": 0,
            "explicacion": "El overfitting ocurre cuando el modelo memoriza el ruido y detalles específicos del conjunto de entrenamiento, fallando al enfrentarse a nuevos estudiantes en producción."
        },
        {
            "id": 6,
            "pregunta": "¿Por qué es crucial dividir los datos en conjuntos de Entrenamiento (Train) y Prueba (Test)?",
            "opciones": [
                "Porque Scikit-Learn genera un error si no se realiza la división.",
                "Para poder evaluar la capacidad de generalización del modelo frente a datos nuevos que no ha visto durante el entrenamiento.",
                "Para entrenar dos modelos diferentes al mismo tiempo."
            ],
            "correcta": 1,
            "explicacion": "Dividir los datos nos permite simular cómo se comportará el modelo en producción al probarlo en un set de datos de test que nunca fue analizado por el algoritmo durante el fit."
        },
        {
            "id": 7,
            "pregunta": "¿Qué es un Árbol de Decisión?",
            "opciones": [
                "Un modelo lineal que dibuja una línea recta para separar los datos.",
                "Un algoritmo basado en la toma de decisiones sucesivas a modo de flujo que divide los datos según reglas lógicas.",
                "Una estructura de base de datos relacional para guardar notas."
            ],
            "correcta": 1,
            "explicacion": "Los Árboles de Decisión dividen los datos secuencialmente por condiciones (ej. ¿Materias perdidas > 2?) facilitando una representación gráfica muy fácil de interpretar por los humanos."
        },
        {
            "id": 8,
            "pregunta": "¿Qué es la métrica de Sensibilidad o Exhaustividad (Recall)?",
            "opciones": [
                "La proporción de predicciones positivas que fueron correctas.",
                "La proporción de casos positivos reales que fueron identificados correctamente por el modelo.",
                "La velocidad con la que el modelo realiza una predicción."
            ],
            "correcta": 1,
            "explicacion": "El Recall o Sensibilidad mide la capacidad del modelo para capturar a todos los estudiantes en riesgo (positivos reales) sin dejar pasar a ninguno desapercibido."
        }
    ]
    
    respuestas_usuario = {}
    
    # Renderizar preguntas
    for p in preguntas:
        st.markdown(f"#### Pregunta {p['id']}: {p['pregunta']}")
        respuestas_usuario[p['id']] = st.radio(
            "Selecciona una opción:",
            p['opciones'],
            key=f"q_{p['id']}",
            index=None
        )
        st.write("")
        
    if st.button("📝 Calificar Quiz"):
        correctas = 0
        respondidas = 0
        
        # Verificar respuestas
        for p in preguntas:
            resp = respuestas_usuario[p['id']]
            if resp is not None:
                respondidas += 1
                idx_resp = p['opciones'].index(resp)
                if idx_resp == p['correcta']:
                    correctas += 1
                    
        if respondidas < len(preguntas):
            st.warning("⚠️ Por favor responde todas las preguntas del quiz antes de calificar.")
        else:
            puntaje = (correctas / len(preguntas)) * 100
            
            st.markdown("### 📊 Tu Resultado Final")
            if puntaje == 100:
                st.success(f"🏆 ¡Puntaje Perfecto! **{correctas} / {len(preguntas)}** ({puntaje:.0f}%)")
                st.balloons()
            elif puntaje >= 70:
                st.info(f"🌟 ¡Muy buen trabajo! **{correctas} / {len(preguntas)}** ({puntaje:.0f}%)")
            else:
                st.warning(f"💪 Sigue intentándolo, el aprendizaje es un camino constante. **{correctas} / {len(preguntas)}** ({puntaje:.0f}%)")
                
            # Mostrar retroalimentación expandible
            st.markdown("#### 📖 Explicaciones detalladas:")
            for p in preguntas:
                with st.expander(f"Pregunta {p['id']} - Retroalimentación"):
                    resp_u = respuestas_usuario[p['id']]
                    idx_u = p['opciones'].index(resp_u)
                    es_correcta = idx_u == p['correcta']
                    
                    st.write(f"**Tu respuesta:** {resp_u} " + ("✅" if es_correcta else "❌"))
                    st.write(f"**Respuesta correcta:** {p['opciones'][p['correcta']]}")
                    st.write(f"**Explicación:** {p['explicacion']}")
