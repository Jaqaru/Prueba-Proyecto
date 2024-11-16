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

# Opción de entrada individual de datos
if selected == "Predicción Individual":
    st.header('Predicción de Compromiso Estudiantil')
    st.subheader('Entrada de Usuario')

    # Función para obtener la entrada del usuario
    def get_user_input():
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                raisedhands = st.slider('Manos Levantadas (Count)', 0, 100, 10)
                VisITedResources = st.slider('Recursos Visitados (Count)', 0, 100, 10)
            with col2:
                AnnouncementsView = st.slider('Anuncios Vistos (Count)', 0, 100, 10)
                Discussion = st.slider('Participación en Discusiones (Count)', 0, 100, 10)

        # Crear un DataFrame
        user_data = pd.DataFrame({
            'raisedhands': [raisedhands],
            'VisITedResources': [VisITedResources],
            'AnnouncementsView': [AnnouncementsView],
            'Discussion': [Discussion]
        })
        return user_data

    # Obtener datos de entrada
    user_input = get_user_input()

    # Botón para realizar la predicción
    if st.button("Evaluar Compromiso"):
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input) if hasattr(model, 'predict_proba') else None

        st.subheader('Resultado de Predicción')
        classification_result = str(prediction[0])
        color = get_engagement_color(classification_result)

        # Mostrar el resultado de la clasificación
        st.markdown(f"<h3 style='color: {color};'>Nivel de Compromiso Predicho: {classification_result}</h3>", unsafe_allow_html=True)

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

        # Seleccionar columnas relevantes
        X = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']]

        # Realizar predicciones y calcular probabilidades
        prediction = model.predict(X)
        probability = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        # Asignar resultados y colores al DataFrame
        df['Nivel de Compromiso'] = prediction
        df['Color'] = df['Nivel de Compromiso'].apply(get_engagement_color)

        # Agregar probabilidad si está disponible
        if probability is not None:
            df['Nivel de Confianza'] = [f"{(np.max(p) * 100):.2f}%" for p in probability]

        # Mostrar tabla con colores de predicción
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