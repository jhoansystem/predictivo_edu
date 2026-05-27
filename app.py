import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Get directory of the current script to resolve relative paths robustly
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set page configuration with a premium look
st.set_page_config(
    page_title="PredictivoEdu - Analítica Predictiva Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean glassmorphism and modern feel
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card-green {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 20px;
        border-radius: 12px;
        color: #34d399;
        margin-top: 15px;
    }
    .prediction-card-red {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 20px;
        border-radius: 12px;
        color: #f87171;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_data():
    path = os.path.join(base_dir, "resultados_estudiantes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
        return df
    else:
        # Fallback dummy data if not found
        st.warning("No se encontró el dataset 'resultados_estudiantes.csv'. Cargando datos de muestra simulados.")
        np.random.seed(42)
        n = 1000
        grades = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        df_dummy = pd.DataFrame({
            'ano': np.random.choice(range(2015, 2025), n),
            'matricula': range(1000, 1000+n),
            'grado': np.random.choice(grades, n),
            'promedio_nota': np.random.uniform(1.5, 4.8, n),
            'nota_min': np.random.uniform(0.5, 3.5, n),
            'nota_max': np.random.uniform(3.0, 5.0, n),
            'total_faltas': np.random.poisson(5, n),
            'total_refuerzos': np.random.poisson(1, n),
            'materias_reprobadas': np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            'total_materias': [11] * n
        })
        df_dummy['promovido'] = np.where(
            (df_dummy['promedio_nota'] >= 3.0) & (df_dummy['materias_reprobadas'] <= 2) & (df_dummy['total_faltas'] < 15), 
            'S', 'N'
        )
        df_dummy['promovido_num'] = df_dummy['promovido'].map({'S': 1, 'N': 0})
        return df_dummy

df = load_data()

# Grade mapping
grado_map = {
    'P1': 0, 'P2': 0, 'P3': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
    'AC': 6
}
df['grado_num'] = df['grado'].astype(str).map(grado_map).fillna(6).astype(int)

# Sidebar Navigation Panel
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=80)
st.sidebar.title("🎓 PredictivoEdu")
st.sidebar.subheader("Menú de Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio / Ciclo de Vida", "📊 Dashboard EDA", "🔮 Modelado Predictivo", "📝 Trivia de Machine Learning"]
)

# Add links at the bottom of the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Enlaces Rápidos:**")
st.sidebar.markdown("[🌐 Landing Page (HTML)](file:///c:/Users/ADMIN/Desktop/ProyectoTalentoTech/index.html)")
st.sidebar.markdown("[💻 Código en GitHub](https://github.com/jhoansystem/predictivo_edu.git)")
st.sidebar.markdown("[📊 Despliegue Streamlit](https://predictivoedu.streamlit.app/)")

# ==========================================
# PAGE 1: INICIO & CICLO DE VIDA (CRISP-ML)
# ==========================================
if page == "🏠 Inicio / Ciclo de Vida":
    st.title("🏫 Sistema Predictivo de Retención Académica")
    st.write("Bienvenido a **PredictivoEdu**, una plataforma analítica diseñada para predecir y prevenir el bajo rendimiento académico escolar aplicando el ciclo de vida completo de Machine Learning (Metodología CRISP-ML).")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Propósito y Objetivos")
        st.markdown("""
        *   **Propósito Principal**: Apoyar a directivos y docentes en la detección temprana de estudiantes en riesgo de reprobación escolar.
        *   **Objetivos Clave**:
            1.  Limpiar y normalizar 10 años de registros transaccionales (2015-2024).
            2.  Visualizar comportamientos de calificaciones e inasistencias en tableros dinámicos.
            3.  Entrenar modelos estadísticos capaces de predecir promedios académicos y probabilidades de promoción en tiempo real.
        """)
        
    with col2:
        st.subheader("💡 Problema a Resolver")
        st.info("""
        La falta de herramientas analíticas oportunas impide que los centros educativos tomen medidas preventivas antes de que finalice el año escolar. 
        Este sistema procesa datos de calificaciones, fallas a clase e historial de refuerzos pedagógicos para generar predicciones preventivas individuales y grupales.
        """)

    st.markdown("---")
    st.subheader("🔄 Fases del Ciclo de Vida del ML (CRISP-ML)")
    
    with st.expander("1. Identificación del Problema & Negocio"):
        st.write("Definición de las metas de analítica educativa y definición de variables clave para la retención (notas promedio y promoción).")
        
    with st.expander("2. Recolección de Datos"):
        st.write("Carga y exploración del dataset original `Resultados2015-2024.csv` con más de 840,000 registros escolares transaccionales.")
        
    with st.expander("3. Preparación de Datos (ETL)"):
        st.write("Remoción de anomalías, imputación de valores faltantes por medianas de asignatura, y agregación del dataset a nivel estudiante-año para mejorar la eficiencia de procesamiento.")
        
    with st.expander("4. Ingeniería de Modelos (Entrenamiento)"):
        st.write("Partición de datos (70/30), estandarización de características, y entrenamiento en paralelo de una Regresión Lineal y una Regresión Logística.")
        
    with st.expander("5. Evaluación del Modelo"):
        st.write("Monitoreo de métricas como RMSE, R2, Accuracy, F1-Score y área ROC-AUC en conjuntos de prueba.")
        
    with st.expander("6. Despliegue, Mantenimiento & Actualización"):
        st.write("Modelos persistidos con `joblib` y expuestos de manera web con Streamlit y HTML. Monitoreo constante del drift de los datos.")

# ==========================================
# PAGE 2: DASHBOARD EDA
# ==========================================
elif page == "📊 Dashboard EDA":
    st.title("📊 Análisis Exploratorio de Datos Interactivo")
    st.write("Filtra el conjunto de datos limpio para explorar relaciones, estadísticas descriptivas y distribuciones de rendimiento escolar.")
    
    # Filters Layout
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        years = sorted(df['ano'].unique())
        selected_year = st.selectbox("Selecciona Año Académico:", ["Todos"] + list(years))
    with col_f2:
        grades = sorted(df['grado'].unique())
        selected_grade = st.selectbox("Selecciona Grado Escolar:", ["Todos"] + list(grades))
        
    # Apply Filters
    df_filtered = df.copy()
    if selected_year != "Todos":
        df_filtered = df_filtered[df_filtered['ano'] == selected_year]
    if selected_grade != "Todos":
        df_filtered = df_filtered[df_filtered['grado'] == selected_grade]
        
    # KPI Cards
    st.markdown("### 📈 Indicadores Clave de Rendimiento (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_students = len(df_filtered)
    kpi1.metric("Total Estudiantes Filtrados", f"{total_students:,}")
    
    avg_grade = df_filtered['promedio_nota'].mean() if total_students > 0 else 0.0
    kpi2.metric("Promedio General de Notas", f"{avg_grade:.3f}")
    
    promoted_rate = (df_filtered['promovido'] == 'S').mean() * 100 if total_students > 0 else 0.0
    kpi3.metric("Tasa de Promoción Escolar", f"{promoted_rate:.1f}%")
    
    avg_absences = df_filtered['total_faltas'].mean() if total_students > 0 else 0.0
    kpi4.metric("Promedio de Faltas Anuales", f"{avg_absences:.1f}")
    
    st.markdown("---")
    
    # Graphs Section
    st.markdown("### 📊 Gráficos de Análisis Estadístico")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.subheader("Caja y Bigotes (Box Plot) de Calificaciones")
        if total_students > 0:
            fig_box = px.box(
                df_filtered, 
                x='promovido', 
                y='promedio_nota', 
                color='promovido',
                title="Calificación Promedio vs Estado de Promoción",
                labels={'promovido': '¿Fue Promovido?', 'promedio_nota': 'Nota Promedio Ponderada'},
                color_discrete_map={'S': '#10b981', 'N': '#ef4444'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No hay datos para graficar con el filtro seleccionado.")
            
    with g_col2:
        st.subheader("Distribución de Promoción Escolar (Gráfico de Barras)")
        if total_students > 0:
            promo_counts = df_filtered['promovido'].value_counts().reset_index()
            promo_counts.columns = ['Estado', 'Estudiantes']
            promo_counts['Estado'] = promo_counts['Estado'].map({'S': 'Promovido (S)', 'N': 'No Promovido (N)'})
            fig_bar = px.bar(
                promo_counts,
                x='Estado',
                y='Estudiantes',
                color='Estado',
                title="Conteo de Alumnos por Estado de Promoción",
                color_discrete_map={'Promovido (S)': '#10b981', 'No Promovido (N)': '#ef4444'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No hay datos para graficar con el filtro seleccionado.")

    st.markdown("---")
    st.subheader("🔍 Resumen de Análisis Estadístico del Subconjunto de Datos")
    if total_students > 0:
        st.dataframe(df_filtered[['promedio_nota', 'total_faltas', 'total_refuerzos', 'materias_reprobadas']].describe().T)
    else:
        st.write("Sin datos para describir estadísticamente.")

# ==========================================
# PAGE 3: MODELADO PREDICTIVO
# ==========================================
elif page == "🔮 Modelado Predictivo":
    st.title("🔮 Simulador de Modelado Predictivo Integrado")
    st.write("Ingresa los datos escolares del estudiante para ejecutar simultáneamente los modelos de **Regresión Lineal** (predicción de promedio final) y **Regresión Logística** (probabilidad de promoción).")
    
    # Load Models and Scalers
    models_loaded = False
    try:
        lr_model = joblib.load(os.path.join(base_dir, 'modelo_lineal.joblib'))
        scaler_reg = joblib.load(os.path.join(base_dir, 'scaler_lineal.joblib'))
        log_model = joblib.load(os.path.join(base_dir, 'modelo_logistico.joblib'))
        scaler_clf = joblib.load(os.path.join(base_dir, 'scaler_logistico.joblib'))
        models_loaded = True
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib del modelo: {e}")
        st.info("Asegúrate de ejecutar primero el cuaderno de modelado o el script de entrenamiento para generar los modelos.")
        
    st.markdown("---")
    
    col_in, col_out = st.columns([1, 1.2])
    
    with col_in:
        st.subheader("📥 Datos del Estudiante")
        
        # User input features
        in_grado = st.selectbox("Grado Escolar del Alumno:", list(grado_map.keys()), index=10) # Default to 11
        in_grado_num = grado_map[in_grado]
        
        in_faltas = st.slider("Total Inasistencias / Faltas en el Año:", min_value=0, max_value=80, value=5, step=1)
        in_refuerzos = st.slider("Total Refuerzos Académicos Presentados:", min_value=0, max_value=20, value=1, step=1)
        in_materias = st.number_input("Número de Asignaturas Cursadas:", min_value=1, max_value=15, value=11, step=1)
        
        # Failed subjects is a critical variable for promotion
        in_failed = st.slider("Asignaturas Reprobadas (Nota < 3.0):", min_value=0, max_value=int(in_materias), value=0, step=1)
        
        btn_pred = st.button("🚀 Simular Desempeño Académico", type="primary")

    with col_out:
        st.subheader("📤 Resultados de la Simulación")
        
        if btn_pred:
            if not models_loaded:
                st.warning("Los modelos no están entrenados. Usando lógica de simulación interna como fallback de demostración.")
                # Fallback Simulation Logic
                pred_nota = max(1.0, min(5.0, 4.2 - 0.03 * in_faltas + 0.1 * in_refuerzos - 0.25 * in_failed))
                prob_promovido = 1.0 / (1.0 + np.exp(-( (pred_nota - 3.0)*5.0 - 0.8 * in_failed - 0.15 * in_faltas + 0.5 )))
                pred_promovido_class = 1 if prob_promovido >= 0.5 else 0
            else:
                # 1. RUN REGRESSION: Predict average grade (promedio_nota)
                # Features: ['grado_num', 'total_faltas', 'total_refuerzos', 'total_materias']
                feat_reg = np.array([[in_grado_num, in_faltas, in_refuerzos, in_materias]])
                feat_reg_scaled = scaler_reg.transform(feat_reg)
                pred_nota = lr_model.predict(feat_reg_scaled)[0]
                pred_nota = max(0.0, min(5.0, pred_nota)) # Clip between 0 and 5
                
                # 2. RUN CLASSIFICATION: Predict promotion (promovido_num)
                # Features: ['promedio_nota', 'total_faltas', 'total_refuerzos', 'materias_reprobadas', 'grado_num']
                feat_clf = np.array([[pred_nota, in_faltas, in_refuerzos, in_failed, in_grado_num]])
                feat_clf_scaled = scaler_clf.transform(feat_clf)
                pred_promovido_class = log_model.predict(feat_clf_scaled)[0]
                prob_promovido = log_model.predict_proba(feat_clf_scaled)[0][1]
            
            # Display results beautifully
            st.markdown("#### 1. Estimación de Calificación Final (Regresión Lineal)")
            st.metric("Nota Promedio Final Estimada", f"{pred_nota:.2f} / 5.0", delta=f"{pred_nota-3.0:.2f} respecto al mínimo de aprobación")
            
            st.markdown("---")
            
            st.markdown("#### 2. Clasificación de Promoción Escolar (Regresión Logística)")
            st.write(f"**Probabilidad de Promoción Calculada**: `{prob_promovido*100:.2f}%`")
            
            # Display nice status and celebrate if promoted
            if pred_promovido_class == 1:
                st.markdown(f"""
                <div class="prediction-card-green">
                    <h3>🎉 ¡ESTUDIANTE PROMOVIDO!</h3>
                    <p>El estudiante presenta un riesgo de reprobación sumamente bajo. Cumple con los criterios de calificaciones, inasistencias y promoción institucional.</p>
                </div>
                """, unsafe_allow_html=True)
                # Celebrate (bombas y platillos!)
                st.balloons()
                st.snow()
            else:
                st.markdown(f"""
                <div class="prediction-card-red">
                    <h3>⚠️ ALERTA: NO PROMOVIDO</h3>
                    <p>El estudiante se encuentra en una situación crítica y presenta un alto riesgo de reprobación anual. Se recomienda tutoría pedagógica y citación a acudientes.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Haz clic en el botón de la izquierda para realizar la predicción simulada.")

# ==========================================
# PAGE 4: TRIVIA / QUIZ ML
# ==========================================
elif page == "📝 Trivia de Machine Learning":
    st.title("📝 Trivia de Conocimiento en Machine Learning")
    st.write("Demuestra tus conocimientos sobre el ciclo de vida de los proyectos de ML y la metodología CRISP-ML respondiendo este quiz interactivo.")
    
    # Initialize session state for score and answers
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        
    st.markdown("---")
    
    with st.form("quiz_form"):
        # Question 1
        q1 = st.radio(
            "1. ¿Qué significa la sigla CRISP-ML?",
            [
                "Creative Process for Machine Learning",
                "Cross-Industry Standard Process for Machine Learning",
                "Critical Review of Systems in Python for Machine Learning",
                "None of the above"
            ]
        )
        
        # Question 2
        q2 = st.radio(
            "2. En el ciclo de vida del ML, la fase de ETL (Limpieza de datos) pertenece a:",
            [
                "Identificación del Problema",
                "Evaluación del Modelo",
                "Preparación de Datos (Data Preparation)",
                "Despliegue"
            ]
        )
        
        # Question 3
        q3 = st.radio(
            "3. ¿Cuál es la principal diferencia entre Regresión Lineal y Regresión Logística?",
            [
                "La regresión lineal predice variables continuas; la regresión logística predice probabilidades y clases categóricas.",
                "La regresión lineal se escribe en Python y la logística en R.",
                "La regresión lineal no requiere normalizar los datos.",
                "La regresión logística se usa exclusivamente para bases de datos relacionales."
            ]
        )
        
        # Question 4
        q4 = st.radio(
            "4. ¿Qué técnica aplicamos en el ETL para optimizar la velocidad y rendimiento de la memoria de Streamlit?",
            [
                "Eliminar el 90% de los datos aleatoriamente.",
                "Realizar una agregación de transacciones a nivel de estudiante-año.",
                "Entrenar un modelo de Deep Learning.",
                "Desactivar las visualizaciones gráficas de la aplicación."
            ]
        )
        
        # Question 5
        q5 = st.radio(
            "5. ¿Cuál de las siguientes métricas se usa para evaluar modelos de CLASIFICACIÓN (como la Regresión Logística)?",
            [
                "Error Cuadrático Medio (MSE)",
                "R-Squared (R2)",
                "Matriz de Confusión y ROC-AUC Score",
                "Coeficiente de correlación de Pearson"
            ]
        )
        
        submit_quiz = st.form_submit_button("Enviar Respuestas")
        
        if submit_quiz:
            score = 0
            # Q1 check (Option 2)
            if q1 == "Cross-Industry Standard Process for Machine Learning":
                score += 1
            # Q2 check (Option 3)
            if q2 == "Preparación de Datos (Data Preparation)":
                score += 1
            # Q3 check (Option 1)
            if q3 == "La regresión lineal predice variables continuas; la regresión logística predice probabilidades y clases categóricas.":
                score += 1
            # Q4 check (Option 2)
            if q4 == "Realizar una agregación de transacciones a nivel de estudiante-año.":
                score += 1
            # Q5 check (Option 3)
            if q5 == "Matriz de Confusión y ROC-AUC Score":
                score += 1
                
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            
    if st.session_state.quiz_submitted:
        st.markdown("### 🏆 Resultados del Quiz")
        st.write(f"Tu puntaje final es: **{st.session_state.quiz_score} / 5**")
        
        if st.session_state.quiz_score == 5:
            st.success("🥇 ¡Excelente! Tienes un dominio completo del ciclo de vida de proyectos de Machine Learning.")
            st.balloons()
        elif st.session_state.quiz_score >= 3:
            st.info("👍 ¡Buen trabajo! Conoces los fundamentos del ciclo de vida y los modelos de regresión.")
        else:
            st.warning("📚 Te sugerimos revisar los cuadernos Jupyter del proyecto para afianzar los conceptos del modelado predictivo.")
            
        # Explanations
        with st.expander("Ver Respuestas Correctas y Explicación"):
            st.markdown("""
            *   **Pregunta 1**: La respuesta correcta es *Cross-Industry Standard Process for Machine Learning*, la cual es una adaptación moderna de la metodología clásica CRISP-DM para incorporar las fases de aseguramiento de calidad del ML.
            *   **Pregunta 2**: La respuesta correcta es *Preparación de Datos (Data Preparation)*. El proceso de Extracción, Transformación y Carga (ETL) es central en esta fase.
            *   **Pregunta 3**: La respuesta correcta es *La regresión lineal predice variables continuas; la regresión logística predice probabilidades y clases categóricas.* Es la diferencia matemática e instrumental fundamental entre ambos métodos de regresión.
            *   **Pregunta 4**: La respuesta correcta es *Realizar una agregación de transacciones a nivel de estudiante-año.* Esto reduce las filas del dataset original de ~840,000 a ~10,400, facilitando una carga rápida y de bajo consumo en Streamlit.
            *   **Pregunta 5**: La respuesta correcta es *Matriz de Confusión y ROC-AUC Score*. El MSE y R2 se aplican para problemas de regresión continua.
            """)
