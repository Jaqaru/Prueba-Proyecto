from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Student Engagement Prediction",  # Título de la página
    page_icon="https://images.emojiterra.com/twitter/v13.1/512px/1f393.png",  # Ícono de la página
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barra lateral con opciones de menú
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Predict Engagement", "Evaluate Dataset"],  # Opciones del menú
        icons=["graph-up", "file-earmark-arrow-up"],  # Iconos de las opciones
        menu_icon="pencil-square"
    )

# Cargar el modelo entrenado
model = pickle.load(open('C:/Users/LG/PROYECTO DE PRACTICA/Nueva carpeta/models/grad_hp.pkl', 'rb'))

# Función para asignar color según el nivel de compromiso
def get_engagement_color(prediction):
    if prediction == "L":
        return "red"  # Bajo compromiso: rojo
    elif prediction == "M":
        return "orange"  # Compromiso medio: naranja
    else:
        return "green"  # Alto compromiso: verde

# Opción de entrada individual de datos
if selected == "Predict Engagement":
    st.header('Predict Student Engagement')
    st.subheader('User Input')

    # Función para obtener la entrada del usuario
    def get_user_input():
        raisedhands = st.slider('Raised Hands (Count)', 0, 100, 10)  # Valor mínimo, máximo y por defecto
        VisITedResources = st.slider('Visited Resources (Count)', 0, 100, 10)
        AnnouncementsView = st.slider('Announcements Viewed (Count)', 0, 100, 10)
        Discussion = st.slider('Discussion Participation (Count)', 0, 100, 10)

        # Crear un diccionario con los datos
        user_data = {
            'raisedhands': raisedhands,
            'VisITedResources': VisITedResources,
            'AnnouncementsView': AnnouncementsView,
            'Discussion': Discussion
        }

        # Convertir a DataFrame
        features = pd.DataFrame(user_data, index=[0])
        return features

    # Obtener los datos del usuario
    user_input = get_user_input()

    # Botón para realizar la predicción
    if st.button("Evaluate Engagement"):
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input) if hasattr(model, 'predict_proba') else None

        st.subheader('Prediction Result')
        classification_result = str(prediction[0])

        # Obtener el color según el nivel de compromiso
        color = get_engagement_color(classification_result)

        # Mostrar el resultado de la clasificación con el color correspondiente
        st.markdown(f"<h3 style='color: {color};'>Predicted Engagement Level: {classification_result}</h3>", unsafe_allow_html=True)

        # Mostrar probabilidad de la predicción si está disponible
        if probability is not None:
            prob = np.max(probability) * 100  # Probabilidad más alta
            st.subheader('Confidence Level')
            st.success(f"{prob:.2f}%")

# Opción para evaluación en lote de un archivo de datos
if selected == "Evaluate Dataset":
    st.header('Evaluate Engagement for Uploaded Data')
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file:
        st.subheader('Input Data')
        df = pd.read_csv(uploaded_file)

        # Extraer las columnas requeridas
        X = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']]

        # Realizar las predicciones y calcular probabilidades
        prediction = model.predict(X)
        probability = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        # Crear un DataFrame para mostrar los resultados
        df['Predicted Engagement'] = prediction

        # Agregar columna de confianza si existe la probabilidad
        if probability is not None:
            confidence = [f"{(np.max(p) * 100):.2f}%" for p in probability]
            df['Confidence Level'] = confidence

        # Asignar colores a las predicciones de compromiso
        df['Color'] = df['Predicted Engagement'].apply(get_engagement_color)

        # Mostrar los resultados con los colores
        st.write(df.style.applymap(lambda v: f'background-color: {v}', subset=['Predicted Engagement']).apply(
            lambda v: f'background-color: {v}', subset=['Predicted Engagement']))
            
# Ocultar elementos de la interfaz de Streamlit
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)