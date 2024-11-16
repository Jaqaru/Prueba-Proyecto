from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predicción del Compromiso Estudiantil",
    page_icon="https://images.emojiterra.com/twitter/v13.1/512px/1f393.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barra lateral con opciones de menú
with st.sidebar:
    st.title("Opciones")
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Predicción Individual", "Evaluación por Dataset"],
        icons=["graph-up", "file-earmark-arrow-up"],
        menu_icon="pencil-square"
    )
    st.info("Seleccione una opción para comenzar.")

# Cargar el modelo entrenado
model = pickle.load(open('models/Algoritmo_RF.pkl', 'rb'))

# Función para asignar color según el nivel de compromiso
def get_engagement_color(prediction):
    if prediction == "L":
        return "red"  # Bajo compromiso
    elif prediction == "M":
        return "orange"  # Compromiso medio
    else:
        return "green"  # Alto compromiso

# Función para determinar el GradeClass basado en GPA
def get_grade_class(gpa):
    if gpa >= 3.5:
        return 'A', 'green'
    elif 3.0 <= gpa < 3.5:
        return 'B', 'blue'
    elif 2.5 <= gpa < 3.0:
        return 'C', 'yellow'
    elif 2.0 <= gpa < 2.5:
        return 'D', 'orange'
    else:
        return 'F', 'red'

# Opción de entrada individual de datos
if selected == "Predicción Individual":
    st.header('Predicción de Compromiso Estudiantil')
    st.subheader('Entrada de Usuario')

    # Función para obtener la entrada del usuario
    def get_user_input():
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Variables de entrada con sus etiquetas
                age = st.selectbox('Edad', [15, 16, 17, 18])
                gender = st.radio('Género', options=[0, 1], format_func=lambda x: "Masculino" if x == 0 else "Femenino")
                ethnicity = st.selectbox('Etnicidad', [0, 1, 2, 3], format_func=lambda x: {0: "Caucásico", 1: "Afroamericano", 2: "Asiático", 3: "Otro"}[x])
                parental_education = st.selectbox('Nivel de Educación de los Padres', [0, 1, 2, 3, 4], format_func=lambda x: {0: "Ninguno", 1: "Secundaria", 2: "Algún Colegio", 3: "Licenciatura", 4: "Posgrado"}[x])
            with col2:
                study_time = st.slider('Horas de Estudio Semanales', 0, 20, 10)
                absences = st.slider('Ausencias (días)', 0, 30, 0)
                tutoring = st.radio('¿Recibe Tutoría?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                parental_support = st.selectbox('Apoyo de los Padres', [0, 1, 2, 3, 4], format_func=lambda x: {0: "Ninguno", 1: "Bajo", 2: "Moderado", 3: "Alto", 4: "Muy Alto"}[x])
                extracurricular = st.radio('¿Participa en Actividades Extracurriculares?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                sports = st.radio('¿Practica Deportes?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                music = st.radio('¿Toca Música?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                volunteering = st.radio('¿Hace Voluntariado?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

        # Crear un DataFrame con la entrada del usuario
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Ethnicity': [ethnicity],
            'ParentalEducation': [parental_education],
            'StudyTimeWeekly': [study_time],
            'Absences': [absences],
            'Tutoring': [tutoring],
            'ParentalSupport': [parental_support],
            'Extracurricular': [extracurricular],
            'Sports': [sports],
            'Music': [music],
            'Volunteering': [volunteering]
        })
        return user_data

    # Obtener datos de entrada
    user_input = get_user_input()

    # Botón para realizar la predicción
    if st.button("Evaluar Compromiso"):
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input) if hasattr(model, 'predict_proba') else None

        # Suponiendo que el modelo predice el GPA, lo puedes obtener aquí (esto depende de cómo está entrenado el modelo)
        gpa = np.random.uniform(2.0, 4.0)  # Esto es solo un ejemplo, deberías obtener el GPA real desde el modelo o las características

        # Obtener el GradeClass y su color
        grade_class, grade_color = get_grade_class(gpa)

        st.subheader('Resultado de Predicción')
        classification_result = str(prediction[0])
        color = get_engagement_color(classification_result)

        # Mostrar el resultado de la clasificación
        st.markdown(f"<h3 style='color: {color};'>Nivel de Compromiso Predicho: {classification_result}</h3>", unsafe_allow_html=True)

        # Mostrar el GradeClass con el color correspondiente
        st.markdown(f"<h3 style='color: {grade_color};'>Grade Class Predicho: {grade_class}</h3>", unsafe_allow_html=True)

        # Nivel de confianza como barra de progreso con porcentaje
        if probability is not None:
            prob = np.max(probability) * 100  # Probabilidad máxima
            st.subheader('Nivel de Confianza')
            st.progress(int(prob))
            st.write(f"{prob:.2f}%")

# Opción para evaluación en lote de un archivo de datos
if selected == "Evaluación por Dataset":
    st.header('Evaluación de Compromiso para Datos Cargados')
    uploaded_file = st.file_uploader("Suba su dataset (CSV)", type=["csv"])

    if uploaded_file:
        st.subheader('Datos de Entrada')
        df = pd.read_csv(uploaded_file)

        # Seleccionar las nuevas columnas relevantes
        X = df[['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
                'Extracurricular', 'Sports', 'Music', 'Volunteering']]

        # Realizar predicciones y calcular probabilidades
        prediction = model.predict(X)
        probability = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        # Suponiendo que el dataset tenga una columna de GPA, calcula el GradeClass para cada fila
        df['GradeClass'], df['GradeColor'] = zip(*df['GPA'].apply(get_grade_class))  # Asumiendo que tienes la columna GPA en el CSV

        # Asignar resultados y colores al DataFrame
        df['Nivel de Compromiso'] = prediction
        df['Color'] = df['Nivel de Compromiso'].apply(get_engagement_color)

        # Agregar probabilidad si está disponible
        if probability is not None:
            df['Nivel de Confianza'] = [f"{(np.max(p) * 100):.2f}%" for p in probability]

        # Mostrar tabla con colores de predicción y GradeClass
        st.write(df.style.applymap(lambda color: f'background-color: {color}', subset=['Color']))

# Ocultar algunos elementos de la interfaz de Streamlit
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)