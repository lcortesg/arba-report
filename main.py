import streamlit as st
import cv2
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import json
#import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from scipy.signal import butter, filtfilt
from numpy import diff

left_video_name = 'p5l_arba.mp4'
right_video_name = 'p5r_arba.mp4'

left_video = cv2.VideoCapture(left_video_name)
left_frame_count = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
right_video = cv2.VideoCapture(right_video_name)
right_frame_count = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))

@st.cache
def butter_lowpass_filter(data, cutoff=6, fs=120, order=8):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalise frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)  # Filter data
    return y

@st.cache
def load_video_frame(video_name, id):
    video = cv2.VideoCapture(video_name)
    video.set(cv2.CAP_PROP_POS_FRAMES, id)
    _, frame = video.read()
    return frame

@st.cache
def load_data(df_left, df_right, report_left, report_right):
    data_left = json.load(report_left)
    data_right = json.load(report_right)
    return df_left, df_right, data_left, data_right

@st.cache
def get_data(df, list):
    data = {}
    for descriptor in list:
        data[descriptor] = df[descriptor]
    return pd.DataFrame(data)

@st.cache
def get_pos(df, list, min_range, max_range):
    data = {}
    for descriptor in list:
        data[descriptor] = butter_lowpass_filter(df[descriptor][min_range:max_range])
    return pd.DataFrame(data)

@st.cache
def get_gradient(df, fs=120):
    data = {}
    for descriptor in df:
        data_df = df[descriptor].tolist()
        data_delta = np.gradient(data_df, 1/fs)
        data[descriptor] = butter_lowpass_filter(data_delta)
    return pd.DataFrame(data)

@st.cache
def get_gradient_simp(df, fs=120):
    data_delta = np.gradient(df, 1/fs)
    data = butter_lowpass_filter(data_delta)
    return data

@st.cache
def get_candidates(data_left, data_right):
    lsc = data_left["Metatarso I"]["x_max"]
    rsc = data_right["Metatarso D"]["x_min"]

    left_ankle_xmin = data_left["Tobillo I"]["x_min"]
    left_ankle_ymax = data_left["Tobillo I"]["y_max"]

    right_ankle_xmax = data_right["Tobillo D"]["x_max"]
    right_ankle_ymax = data_right["Tobillo D"]["y_max"]

    lcc = []
    for value in left_ankle_xmin:
        lcc.append([x for x in left_ankle_ymax if x > value][0])

    rcc = []
    for value in right_ankle_xmax:
        rcc.append([x for x in right_ankle_ymax if x > value][0])

    return lcc, lsc, rcc, rsc


im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="ABMA Report",
    page_icon=im,
    layout="wide",
)


if 'count1' not in st.session_state:
    st.session_state.count1 = 0
if 'count2' not in st.session_state:
    st.session_state.count2 = 0
if 'count3' not in st.session_state:
    st.session_state.count3 = 0
if 'count4' not in st.session_state:
    st.session_state.count4 = 0
if "cycle_left_1" not in st.session_state:
    st.session_state.cycle_left_1 = 1
if "cycle_left_2" not in st.session_state:
    st.session_state.cycle_left_2 = 1
if "cycle_right_1" not in st.session_state:
    st.session_state.cycle_right_1 = 1
if "cycle_right_2" not in st.session_state:
    st.session_state.cycle_right_2 = 1

df_left = pd.read_csv('p5l_arba.csv')
df_right= pd.read_csv('p5r_arba.csv')
report_left = open('p5l_arba.json')
report_right = open('p5r_arba.json')
df_left, df_right, data_left, data_right = load_data(df_left, df_right, report_left, report_right)
lcc, lsc, rcc, rsc = get_candidates(data_left,data_right)

#select_plane = st.sidebar.selectbox(
#    "Seleccionar el Plano",
#    ("","Sagital", "Frontal","lala")
#)
#if select_plane=="Sagital":
#    select_contact = st.sidebar.selectbox(
#        "Seleccionar contacto",
#        ("Contacto inicial", "Apoyo medio")
#    )
#elif select_plane=="Frontal":
#    select_contact = st.sidebar.selectbox(
#        "Seleccionar contacto",
#        ("Contacto inicial2", "Apoyo medio2")
#    )

window = 5
default_cycle = 1


#st.sidebar.button("Recomendaciones")
#st.sidebar.button("Exportar")


st.title('Reporte ABMA')

#colti1, colti2 = st.columns(2)

#with colti1:
#    st.title('Reporte ABMA')

#with colti2:
#    logo = Image.open('assets/logos/lanek.png')
#    #logo = cv2.imread('assets/logos/lanek0.png',cv2.IMREAD_UNCHANGED)
#    st.image(logo, width = 200, channels='BGR')

st.header('Datos paciente')

coldp1, coldp2 = st.columns(2)

with coldp1:
    st.write('Nombre: ', 'Nombre1 Nombre2 Apellido1 Apellido2')
    st.write('Edad: ', 30)
    st.write('Mail: ', 'nombre.apellido@lanek.cl')
    #st.write({
    #    'Nombre': 'Nombre1 Nombre2 Apellido1 Apellido2', 
    #    'Edad': '30',
    #    'Mail': 'nombre.apellido@lanek.cl',
    #    })

with coldp2:
    st.write('Fecha de evaluación: ', '05/04/2022')
    st.write('Fecha de reportería: ', '06/04/2022')
    #st.write({
    #    'Fecha de evaluación': '05/04/2022', 
    #    'Fecha de reportería': '06/04/2022',
    #    })



st.header('Datos clínicos')

antecedentes = st.sidebar.text_input('Antecedentes', key='antecedentes')
st.write('Antecedentes:', antecedentes)

km_semanales = st.sidebar.text_input('Kilómetros semanales', key='km_semanales')
st.write('Kilómetros semanales:', km_semanales)

graphics = False
show_angles = st.sidebar.checkbox('Mostrar ángulos', value=False)
if show_angles:
    graphics = True
    ang_list = st.sidebar.multiselect(
        'Seleccionar ángulos a graficar',
        ['Tronco', 'Cadera', 'Rodilla', 'Tobillo'],
        ['Tronco', 'Cadera', 'Rodilla', 'Tobillo'],
        key="ang_list")


show_x_pos = st.sidebar.checkbox('Mostrar posiciones X', value=False)
if show_x_pos:
    graphics = True
    x_pos_list = st.sidebar.multiselect(
        'Seleccionar posiciones x a graficar',
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'], 
        key="x_pos_list")

show_y_pos = st.sidebar.checkbox('Mostrar posiciones Y', value=False)
if show_y_pos:
    graphics = True
    y_pos_list = st.sidebar.multiselect(
        'Seleccionar posiciones y a graficar',
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        key="y_pos_list")

show_xy_pos = st.sidebar.checkbox('Mostrar plano XY', value=False)
if show_xy_pos:
    graphics = True
    xy_pos_list = st.sidebar.multiselect(
        'Seleccionar posiciones x/y a graficar',
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        key="xy_pos_list")

show_heatmaps = st.sidebar.checkbox('Mostrar mapas de calor', value=False)
if show_heatmaps:
    graphics = True
    heat_list = st.sidebar.multiselect(
        'Seleccionar mapas de calor a graficar',
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        ['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'],
        key="heat_list")


st.header('Visual')
colvis1, colvis2 = st.columns(2)


with colvis1:

    st.subheader("Extremidad izquierda, contacto inicial")

    #if st.button('Next', key="next_left_1"):
    #    st.session_state.count1 += 1
    #if st.button('Previous', key="prev_left_1"):
    #    st.session_state.count1 -= 1

    st.session_state.cycle_left_1 = st.slider('Desfase de contacto izquierdo', -window, window, 0, 1, format=None, key='left_contact', help='Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado')
    ciclo_l1 = st.slider('Ciclo de carrera izquierda contacto', 0, len(lcc)-1, default_cycle, 1, format=None, key='left_contact_slider', help='Cambia entre los diferentes ciclos de carrera')
    id1 = lcc[ciclo_l1]+st.session_state.cycle_left_1

    frame = load_video_frame(left_video_name, id1)
    #left_video.set(cv2.CAP_PROP_POS_FRAMES, id1)
    #_, frame = left_video.read()


    st.image(frame, caption=f"Tronco LI: {round(df_left.iloc[id1]['Tronco LI'],1)}, Cadera LI: {round(df_left.iloc[id1]['Cadera LI'],1)}, Rodilla LI: {round(df_left.iloc[id1]['Rodilla LI'],1)}, Tobillo LI: {round(df_left.iloc[id1]['Tobillo LI'],1)}", channels='BGR')

    #st.write({
    #    'Tronco LI': round(df_left.iloc[id1]['Tronco LI'],1), 
    #    'Cadera LI': round(df_left.iloc[id1]['Cadera LI'],1),
    #    'Rodilla LI': round(df_left.iloc[id1]['Rodilla LI'],1),
    #    'Tobillo LI': round(df_left.iloc[id1]['Tobillo LI'],1)
    #    })

    st.write('Tronco LI = ',round(df_left.iloc[id1]['Tronco LI'],1))
    st.write('Cadera LI = ',round(df_left.iloc[id1]['Cadera LI'],1))
    st.write('Rodilla LI = ',round(df_left.iloc[id1]['Rodilla LI'],1))
    st.write('Tobillo LI = ',round(df_left.iloc[id1]['Tobillo LI'],1))

    comentario_left_1 = st.text_input('Comentarios', key='comentario_left_1')
    st.write('comentario:', comentario_left_1)


    st.subheader("Extremidad izquierda, apoyo medio")

    #if st.button('Next', key="next_left_2"):
    #    st.session_state.count3 += 1
    #if st.button('Previous', key="prev_left_2"):
    #    st.session_state.count3 -= 1
    st.session_state.cycle_left_2 = st.slider('Desfase de separación izquierdo', -window, window, 0, 1, format=None, key='left_separation', help='Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado')
    ciclo_l2 = st.slider('Ciclo de carrera izquierda separación', 0, len(lsc)-1, default_cycle, 1, format=None, key='left_separation_slider', help='Cambia entre los diferentes ciclos de carrera')
    id3 = lsc[ciclo_l2]+st.session_state.cycle_left_2

    frame = load_video_frame(left_video_name, id3)
    #left_video.set(cv2.CAP_PROP_POS_FRAMES, id3)
    #_, frame = left_video.read()


    st.image(frame,caption=f"Tronco LI: {round(df_left.iloc[id3]['Tronco LI'],1)}, Cadera LI: {round(df_left.iloc[id3]['Cadera LI'],1)}, Rodilla LI: {round(df_left.iloc[id3]['Rodilla LI'],1)}, Tobillo LI: {round(df_left.iloc[id3]['Tobillo LI'],1)}", channels='BGR')

    #st.write({
    #    'Tronco LI': round(df_left.iloc[id3]['Tronco LI'],1), 
    #    'Cadera LI': round(df_left.iloc[id3]['Cadera LI'],1),
    #    'Rodilla LI': round(df_left.iloc[id3]['Rodilla LI'],1),
    #    'Tobillo LI': round(df_left.iloc[id3]['Tobillo LI'],1)
    #    })

    st.write('Tronco LI = ',round(df_left.iloc[id3]['Tronco LI'],1))
    st.write('Cadera LI = ',round(df_left.iloc[id3]['Cadera LI'],1))
    st.write('Rodilla LI = ',round(df_left.iloc[id3]['Rodilla LI'],1))
    st.write('Tobillo LI = ',round(df_left.iloc[id3]['Tobillo LI'],1))

    comentario_left_2 = st.text_input('Comentarios', key='comentario_left_2')
    st.write('comentario:', comentario_left_2)

    #st.write(f'Average run cycle: {round(120/np.average(np.diff(lcc)),3)} [Hz]')

with colvis2:

    st.subheader("Extremidad derecha, contacto inicial")

    #if st.button('Next', key="next_right_1"):
    #    st.session_state.count2 += 1
    #if st.button('Previous', key="prev_right_1"):
    #    st.session_state.count2 -= 1

    st.session_state.cycle_right_1 = st.slider('Desfase de contacto derecho', -window, window, 0, 1, format=None, key='right_contact', help='Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado')
    ciclo_r1 = st.slider('Ciclo de carrera derecha contacto', 0, len(rcc)-1, default_cycle, 1, format=None, key='right_contact_slider', help='Cambia entre los diferentes ciclos de carrera')
    id2 = rcc[ciclo_r1]+st.session_state.cycle_right_1

    frame = load_video_frame(right_video_name, id2)
    #right_video.set(cv2.CAP_PROP_POS_FRAMES, id2)
    #_, frame = right_video.read()

    st.image(frame,caption=f"Tronco LD: {round(df_right.iloc[id2]['Tronco LD'],1)}, Cadera LD: {round(df_right.iloc[id2]['Cadera LD'],1)}, Rodilla LD: {round(df_right.iloc[id2]['Rodilla LD'],1)}, Tobillo LD: {round(df_right.iloc[id2]['Tobillo LD'],1)}",channels='BGR')

    #st.write({
    #    'Tronco LD': round(df_right.iloc[id2]['Tronco LD'],1), 
    #    'Cadera LD': round(df_right.iloc[id2]['Cadera LD'],1),
    #    'Rodilla LD': round(df_right.iloc[id2]['Rodilla LD'],1),
    #    'Tobillo LD': round(df_right.iloc[id2]['Tobillo LD'],1)
    #    })

    st.write('Tronco LD = ', round(df_right.iloc[id2]['Tronco LD'],1))
    st.write('Cadera LD = ', round(df_right.iloc[id2]['Cadera LD'],1))
    st.write('Rodilla LD = ', round(df_right.iloc[id2]['Rodilla LD'],1))
    st.write('Tobillo LD = ', round(df_right.iloc[id2]['Tobillo LD'],1))

    comentario2 = st.text_input('Comentario', key='comentario_right_1')
    st.write('comentario:', comentario2)

    st.subheader("Extremidad derecha, apoyo medio")

    #if st.button('Next', key="next_right_2"):
    #    st.session_state.count4 += 1
    #if st.button('Previous', key="prev_right_2"):
    #    st.session_state.count4 -= 1
    st.session_state.cycle_right_2 = st.slider('Desfase de separación derecho', -window, window, 0, 1, format=None, key='right_separation', help='Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado')
    ciclo_r2 = st.slider('Ciclo de carrera derecha separación', 0, len(rsc)-1, default_cycle, 1, format=None, key='right_separation_slider', help='Cambia entre los diferentes ciclos de carrera')
    id4 = rsc[ciclo_r2]+st.session_state.cycle_right_2

    frame = load_video_frame(right_video_name, id4)
    #right_video.set(cv2.CAP_PROP_POS_FRAMES, id4)
    #_, frame = right_video.read()

    st.image(frame,caption=f"Tronco LD: {round(df_right.iloc[id4]['Tronco LD'],1)}, Cadera LD: {round(df_right.iloc[id4]['Cadera LD'],1)}, Rodilla LD: {round(df_right.iloc[id4]['Rodilla LD'],1)}, Tobillo LD: {round(df_right.iloc[id4]['Tobillo LD'],1)}",channels='BGR')

    st.write('Tronco LD = ', round(df_right.iloc[id4]['Tronco LD'],1))
    st.write('Cadera LD = ', round(df_right.iloc[id4]['Cadera LD'],1))
    st.write('Rodilla LD = ', round(df_right.iloc[id4]['Rodilla LD'],1))
    st.write('Tobillo LD = ', round(df_right.iloc[id4]['Tobillo LD'],1))

    comentario4 = st.text_input('Comentario', key='comentario_right_2')
    st.write('comentario:', comentario4)

    #st.write(f'Average run cycle: {round(120/np.average(np.diff(rcc)),3)} [Hz]')

st.subheader('Adicionales')

colad1, colad2 = st.columns(2)
with colad1:
    st.write('Cadencia izquierda: ', f'{round(120/np.average(np.diff(lcc)),3)} [Hz]')
    st.write('Ciclo de carrera izquierda: ', f'{round(60*120/np.average(np.diff(lcc)),3)} ciclos por minuto')
    #st.write({
    #    'Cadencia izquierda': f'{round(120/np.average(np.diff(lcc)),3)} [Hz]', 
    #    'Ciclo de carrera izquierda': f'{round(60*120/np.average(np.diff(lcc)),3)} ciclos por minuto',
    #    })

with colad2:
    st.write('Cadencia derecha: ', f'{round(120/np.average(np.diff(rcc)),3)} [Hz]')
    st.write('Ciclo de carrera derecha: ', f'{round(60*120/np.average(np.diff(rcc)),3)} ciclos por minuto')
    #st.write({
    #    'Cadencia derecha': f'{round(120/np.average(np.diff(rcc)),3)} [Hz]', 
    #    'Ciclo de carrera derecha': f'{round(60*120/np.average(np.diff(rcc)),3)} ciclos por minuto',
    #    })


if graphics: 
    st.header('Gráficos')

if show_angles:
    st.subheader('Ángulos articulares')
    colang1, colang2 = st.columns(2)

    with colang1:
        ang_left_list = [s + " LI" for s in ang_list]

        #ang_left_list = st.multiselect(
        #    'Seleccionar ángulos',
        #    ['Tronco LI', 'Cadera LI', 'Rodilla LI', 'Tobillo LI'],
        #    ['Tronco LI', 'Cadera LI', 'Rodilla LI', 'Tobillo LI'])

        #ang_left_list=[]
        #if st.checkbox('Tronco L', value=True, help="Ángulo del tronco respecto a la vertical"): ang_left_list.append('Tronco LI')
        #if st.checkbox('Cadera L', value=True, help="Ángulo entre el tronco y la rodilla"): ang_left_list.append('Cadera LI')
        #if st.checkbox('Rodilla L', value=True, help="Ángulo entre la cadera y el tobillo"): ang_left_list.append('Rodilla LI')
        #if st.checkbox('Tobillo L', value=True, help="Ángulo del tobillo respecto a la horizontal"): ang_left_list.append('Tobillo LI')

        #min_range, max_range = st.slider("Rango de ciclos de carrera izquierda", 0, 25, [1, 3], 1, format=None, key='left_angle_slider')
        left_range = st.slider('Cantidad de ciclos de carrera izquierda', 1, len(lcc)-1, default_cycle, 1, format=None, key='left_angle_slider')
        ang_left = get_pos(df_left, ang_left_list, lcc[0], lcc[left_range])
        st.line_chart(ang_left)

    with colang2:
        ang_right_list = [s + " LD" for s in ang_list]
        #ang_right_list = st.multiselect(
        #    'Seleccionar ángulos a graficar',
        #    ['Tronco LD', 'Cadera LD', 'Rodilla LD', 'Tobillo LD'],
        #    ['Tronco LD', 'Cadera LD', 'Rodilla LD', 'Tobillo LD'])
        #ang_right_list=[]
        #if st.checkbox('Tronco R', value=True, help="Ángulo del tronco respecto a la vertical"): ang_right_list.append('Tronco LD')
        #if st.checkbox('Cadera R', value=True, help="Ángulo entre el tronco y la rodilla"): ang_right_list.append('Cadera LD')
        #if st.checkbox('Rodilla R', value=True, help="Ángulo entre la cadera y el tobillo"): ang_right_list.append('Rodilla LD')
        #if st.checkbox('Tobillo R', value=True, help="Ángulo del tobillo respecto a la horizontal"): ang_right_list.append('Tobillo LD')
        #right_range = st.number_input('Cantidad de ciclos de carrera derecha', 1, len(rcc)-1, default_cycle, 1)
        right_range = st.slider('Cantidad de ciclos de carrera derecha', 1, len(rcc)-1, default_cycle, 1, format=None, key='right_angle_slider')
        ang_right = get_pos(df_right, ang_right_list, rcc[0], rcc[right_range])
        st.line_chart(ang_right)

if show_x_pos:
    st.subheader('Posiciones en el eje X')
    colposx1, colposx2 = st.columns(2)

    with colposx1:
        x_left_list = [s + " I_x" for s in x_pos_list]
        #x_left_list=[]
        #if st.checkbox('Acromion Lx', value=True): x_left_list.append('Acromion I_x')
        #if st.checkbox('Cadera Lx', value=True): x_left_list.append('Cadera I_x')
        #if st.checkbox('Rodilla Lx', value=True): x_left_list.append('Rodilla I_x')
        #if st.checkbox('Tobillo Lx', value=True): x_left_list.append('Tobillo I_x')
        #if st.checkbox('Metatarso Lx', value=True): x_left_list.append('Metatarso I_x')

        left_range = st.slider('Cantidad de ciclos de carrera izquierda', 1, len(lcc)-1, default_cycle, 1, format=None, key='left_x_slider')
        x_left = get_pos(df_left, x_left_list, lcc[0], lcc[left_range])
        st.line_chart(x_left)

        """#### Velocidades eje x"""
        x_left_vel = get_gradient(x_left)
        st.line_chart(x_left_vel)

        """#### Aceleraciones eje x"""
        x_left_acc = get_gradient(x_left_vel)
        st.line_chart(x_left_acc)
   
   


    with colposx2:
        x_right_list = [s + " D_x" for s in x_pos_list]
        #x_right_list=[]
        #if st.checkbox('Acromion Rx', value=True): x_right_list.append('Acromion D_x')
        #if st.checkbox('Cadera Rx', value=True): x_right_list.append('Cadera D_x')
        #if st.checkbox('Rodilla Rx', value=True): x_right_list.append('Rodilla D_x')
        #if st.checkbox('Tobillo Rx', value=True): x_right_list.append('Tobillo D_x')
        #if st.checkbox('Metatarso Rx', value=True): x_right_list.append('Metatarso D_x')

        right_range = st.slider('Cantidad de ciclos de carrera derecha', 1, len(rcc)-1, default_cycle, 1, format=None, key='right_x_slider')
        x_right = get_pos(df_right, x_right_list, rcc[0], rcc[right_range])
        st.line_chart(x_right)

        """#### Velocidades eje x"""
        x_right_vel = get_gradient(x_right)
        st.line_chart(x_right_vel)


        """#### Aceleraciones eje x"""
        x_right_acc = get_gradient(x_right_vel)
        st.line_chart(x_right_acc)

if show_y_pos:
    st.subheader('Posiciones en el eje Y')
    colposy1, colposy2 = st.columns(2)

    with colposy1:
        y_left_list = [s + " I_y" for s in y_pos_list]
        #y_left_list=[]
        #if st.checkbox('Acromion Ly', value=True): y_left_list.append('Acromion I_y')
        #if st.checkbox('Cadera Ly', value=True): y_left_list.append('Cadera I_y')
        #if st.checkbox('Rodilla Ly', value=True): y_left_list.append('Rodilla I_y')
        #if st.checkbox('Tobillo Ly', value=True): y_left_list.append('Tobillo I_y')
        #if st.checkbox('Metatarso Ly', value=True): y_left_list.append('Metatarso I_y')

        left_range = st.slider('Cantidad de ciclos de carrera izquierda', 1, len(lcc)-1, default_cycle, 1, format=None, key='left_y_slider')

        y_left = get_pos(df_left, y_left_list, lcc[0], lcc[left_range])
        st.line_chart(-y_left)

        """#### Velocidades eje y"""
        y_left_vel = get_gradient(y_left)
        st.line_chart(-y_left_vel)

        """#### Aceleraciones eje y"""
        y_left_acc = get_gradient(y_left_vel)
        st.line_chart(-y_left_acc)

    with colposy2:
        y_right_list = [s + " D_y" for s in y_pos_list]
        #y_right_list=[]
        #if st.checkbox('Acromion Ry', value=True): y_right_list.append('Acromion D_y')
        #if st.checkbox('Cadera Ry', value=True): y_right_list.append('Cadera D_y')
        #if st.checkbox('Rodilla Ry', value=True): y_right_list.append('Rodilla D_y')
        #if st.checkbox('Tobillo Ry', value=True): y_right_list.append('Tobillo D_y')
        #if st.checkbox('Metatarso Ry', value=True): y_right_list.append('Metatarso D_y')

        right_range = st.slider('Cantidad de ciclos de carrera derecha', 1, len(rcc)-1, default_cycle, 1, format=None, key='right_y_slider')
        y_right = get_pos(df_right, y_right_list, rcc[0], rcc[right_range])
        st.line_chart(-y_right)

        """#### Velocidades eje y"""
        y_right_vel = get_gradient(y_right)
        st.line_chart(-y_right_vel)

        """#### Aceleraciones eje y"""
        y_right_acc = get_gradient(y_right_vel)
        st.line_chart(-y_right_acc)



if show_xy_pos:
    st.subheader('Posiciones en el plano XY')
    colposxy1, colposxy2 = st.columns(2)

    with colposxy1:
        ciclo_xy= st.slider('Ciclo de carrera izquierda', 0, len(lcc)-1, 1, 1, format=None, key='left_xy_slider', help='Cambia entre los diferentes ciclos de carrera')
        min_range, max_range = st.slider("Rango de ciclos de carrera izquierda", lcc[ciclo_xy], lcc[ciclo_xy+1], [lcc[ciclo_xy], lcc[ciclo_xy+1]], 1, format=None, key='left_position_slider')

        acro_x = df_left['Acromion I_x'][min_range:max_range]
        acro_y = -df_left['Acromion I_y'][min_range:max_range]
        hip_x = df_left['Cadera I_x'][min_range:max_range]
        hip_y = -df_left['Cadera I_y'][min_range:max_range]
        knee_x = df_left['Rodilla I_x'][min_range:max_range]
        knee_y = -df_left['Rodilla I_y'][min_range:max_range]
        ankle_x = df_left['Tobillo I_x'][min_range:max_range]
        ankle_y = -df_left['Tobillo I_y'][min_range:max_range]
        meta_x = df_left['Metatarso I_x'][min_range:max_range]
        meta_y = -df_left['Metatarso I_y'][min_range:max_range]

        trace_acro = go.Scatter(x=acro_x, y=acro_y, name='Acromion L')
        trace_hip = go.Scatter(x=hip_x, y=hip_y, name='Cadera L')
        trace_knee = go.Scatter(x=knee_x, y=knee_y, name='Rodilla L')
        trace_ankle = go.Scatter(x=ankle_x, y=ankle_y, name='Tobillo L')
        trace_meta = go.Scatter(x=meta_x, y=meta_y, name='Metatarso L')

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        #if st.checkbox('Acromion L', value=True): fig.add_trace(trace_acro)
        #if st.checkbox('Cadera L', value=True): fig.add_trace(trace_hip)
        #if st.checkbox('Rodilla L', value=True): fig.add_trace(trace_knee)
        #if st.checkbox('Tobillo L', value=True): fig.add_trace(trace_ankle)
        #if st.checkbox('Metatarso L', value=True): fig.add_trace(trace_meta)
        if "Acromion" in xy_pos_list: fig.add_trace(trace_acro)
        if "cadera" in xy_pos_list: fig.add_trace(trace_hip)
        if "Rodilla" in xy_pos_list: fig.add_trace(trace_knee)
        if "Tobillo" in xy_pos_list: fig.add_trace(trace_ankle)
        if "Metatarso" in xy_pos_list: fig.add_trace(trace_meta)
        st.plotly_chart(fig, use_container_width=True)

        acro_x_vel = get_gradient_simp(acro_x.tolist())
        acro_y_vel = get_gradient_simp(acro_y.tolist())
        hip_x_vel = get_gradient_simp(hip_x.tolist())
        hip_y_vel = get_gradient_simp(hip_y.tolist())
        knee_x_vel = get_gradient_simp(knee_x.tolist())
        knee_y_vel = get_gradient_simp(knee_y.tolist())
        ankle_x_vel = get_gradient_simp(ankle_x.tolist())
        ankle_y_vel = get_gradient_simp(ankle_y.tolist())
        meta_x_vel = get_gradient_simp(meta_x.tolist())
        meta_y_vel = get_gradient_simp(meta_y.tolist())

        trace_acro_vel = go.Scatter(x=acro_x_vel, y=acro_y_vel, name='Acromion L')
        trace_hip_vel = go.Scatter(x=hip_x_vel, y=hip_y_vel, name='Cadera L')
        trace_knee_vel = go.Scatter(x=knee_x_vel, y=knee_y_vel, name='Rodilla L')
        trace_ankle_vel = go.Scatter(x=ankle_x_vel, y=ankle_y_vel, name='Tobillo L')
        trace_meta_vel = go.Scatter(x=meta_x_vel, y=meta_y_vel, name='Metatarso L')

        """#### Velocidades plano xy"""
        fig2 = make_subplots(specs=[[{"secondary_y": False}]])
        if "Acromion" in xy_pos_list: fig2.add_trace(trace_acro_vel)
        if "cadera" in xy_pos_list: fig2.add_trace(trace_hip_vel)
        if "Rodilla" in xy_pos_list: fig2.add_trace(trace_knee_vel)
        if "Tobillo" in xy_pos_list: fig2.add_trace(trace_ankle_vel)
        if "Metatarso" in xy_pos_list: fig2.add_trace(trace_meta_vel)
        st.plotly_chart(fig2, use_container_width=True)

        acro_x_acc = get_gradient_simp(acro_x_vel)
        acro_y_acc = get_gradient_simp(acro_y_vel)
        hip_x_acc = get_gradient_simp(hip_x_vel)
        hip_y_acc = get_gradient_simp(hip_y_vel)
        knee_x_acc = get_gradient_simp(knee_x_vel)
        knee_y_acc = get_gradient_simp(knee_y_vel)
        ankle_x_acc = get_gradient_simp(ankle_x_vel)
        ankle_y_acc = get_gradient_simp(ankle_y_vel)
        meta_x_acc = get_gradient_simp(meta_x_vel)
        meta_y_acc = get_gradient_simp(meta_y_vel)

        trace_acro_acc = go.Scatter(x=acro_x_acc, y=acro_y_acc, name='Acromion L')
        trace_hip_acc = go.Scatter(x=hip_x_acc, y=hip_y_acc, name='Cadera L')
        trace_knee_acc = go.Scatter(x=knee_x_acc, y=knee_y_acc, name='Rodilla L')
        trace_ankle_acc = go.Scatter(x=ankle_x_acc, y=ankle_y_acc, name='Tobillo L')
        trace_meta_acc = go.Scatter(x=meta_x_acc, y=meta_y_acc, name='Metatarso L')

        """#### Aceleraciones plano xy"""
        fig3 = make_subplots(specs=[[{"secondary_y": False}]])
        if "Acromion" in xy_pos_list: fig3.add_trace(trace_acro_acc)
        if "cadera" in xy_pos_list: fig3.add_trace(trace_hip_acc)
        if "Rodilla" in xy_pos_list: fig3.add_trace(trace_knee_acc)
        if "Tobillo" in xy_pos_list: fig3.add_trace(trace_ankle_acc)
        if "Metatarso" in xy_pos_list: fig3.add_trace(trace_meta_acc)
        st.plotly_chart(fig3, use_container_width=True)

        """#### Magnitud de aceleraciones plano xy"""
        acro_mag = np.sqrt(acro_x_acc**2 + acro_y_acc**2)
        hip_mag = np.sqrt(hip_x_acc**2 +  hip_y_acc**2)
        knee_mag = np.sqrt(knee_x_acc**2 + knee_y_acc**2)
        ankle_mag = np.sqrt(ankle_x_acc**2 + ankle_y_acc**2)
        meta_mag = np.sqrt(meta_x_acc**2 + meta_y_acc**2)

        xy_df = pd.DataFrame(list(zip(acro_mag, hip_mag, knee_mag, ankle_mag, meta_mag)), columns =['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'])
        st.line_chart(get_data(xy_df,xy_pos_list))


 
        

    with colposxy2:

        ciclo_xy= st.slider('Ciclo de carrera dercha', 0, len(rcc)-1, 1, 1, format=None, key='right_xy_slider', help='Cambia entre los diferentes ciclos de carrera')
        min_range, max_range = st.slider("Rango de ciclos de carrera derecha", rcc[ciclo_xy], rcc[ciclo_xy+1], [rcc[ciclo_xy], rcc[ciclo_xy+1]], 1, format=None, key='right_position_slider')


        acro_x = df_right['Acromion D_x'][min_range:max_range]
        acro_y = -df_right['Acromion D_y'][min_range:max_range]
        hip_x = df_right['Cadera D_x'][min_range:max_range]
        hip_y = -df_right['Cadera D_y'][min_range:max_range]
        knee_x = df_right['Rodilla D_x'][min_range:max_range]
        knee_y = -df_right['Rodilla D_y'][min_range:max_range]
        ankle_x = df_right['Tobillo D_x'][min_range:max_range]
        ankle_y = -df_right['Tobillo D_y'][min_range:max_range]
        meta_x = df_right['Metatarso D_x'][min_range:max_range]
        meta_y = -df_right['Metatarso D_y'][min_range:max_range]


        trace_acro = go.Scatter(x=acro_x, y=acro_y, name='Acromion R')
        trace_hip = go.Scatter(x=hip_x, y=hip_y, name='Cadera R')
        trace_knee = go.Scatter(x=knee_x, y=knee_y, name='Rodilla R')
        trace_ankle = go.Scatter(x=ankle_x, y=ankle_y, name='Tobillo R')
        trace_meta = go.Scatter(x=meta_x, y=meta_y, name='Metatarso R')


        fig = make_subplots(specs=[[{"secondary_y": False}]])
        #if st.checkbox('Acromion R', value=True): fig.add_trace(trace_acro)
        #if st.checkbox('Cadera R', value=True): fig.add_trace(trace_hip)
        #if st.checkbox('Rodilla R', value=True): fig.add_trace(trace_knee)
        #if st.checkbox('Tobillo R', value=True): fig.add_trace(trace_ankle)
        #if st.checkbox('Metatarso R', value=True): fig.add_trace(trace_meta)
        if "Acromion" in xy_pos_list: fig.add_trace(trace_acro)
        if "cadera" in xy_pos_list: fig.add_trace(trace_hip)
        if "Rodilla" in xy_pos_list: fig.add_trace(trace_knee)
        if "Tobillo" in xy_pos_list: fig.add_trace(trace_ankle)
        if "Metatarso" in xy_pos_list: fig.add_trace(trace_meta)
        st.plotly_chart(fig, use_container_width=True)

        acro_x_vel = get_gradient_simp(acro_x.tolist())
        acro_y_vel = get_gradient_simp(acro_y.tolist())
        hip_x_vel = get_gradient_simp(hip_x.tolist())
        hip_y_vel = get_gradient_simp(hip_y.tolist())
        knee_x_vel = get_gradient_simp(knee_x.tolist())
        knee_y_vel = get_gradient_simp(knee_y.tolist())
        ankle_x_vel = get_gradient_simp(ankle_x.tolist())
        ankle_y_vel = get_gradient_simp(ankle_y.tolist())
        meta_x_vel = get_gradient_simp(meta_x.tolist())
        meta_y_vel = get_gradient_simp(meta_y.tolist())

        trace_acro_vel = go.Scatter(x=acro_x_vel, y=acro_y_vel, name='Acromion R')
        trace_hip_vel = go.Scatter(x=hip_x_vel, y=hip_y_vel, name='Cadera R')
        trace_knee_vel = go.Scatter(x=knee_x_vel, y=knee_y_vel, name='Rodilla R')
        trace_ankle_vel = go.Scatter(x=ankle_x_vel, y=ankle_y_vel, name='Tobillo R')
        trace_meta_vel = go.Scatter(x=meta_x_vel, y=meta_y_vel, name='Metatarso R')

        """#### Velocidades plano xy"""
        fig2 = make_subplots(specs=[[{"secondary_y": False}]])
        if "Acromion" in xy_pos_list: fig2.add_trace(trace_acro_vel)
        if "cadera" in xy_pos_list: fig2.add_trace(trace_hip_vel)
        if "Rodilla" in xy_pos_list: fig2.add_trace(trace_knee_vel)
        if "Tobillo" in xy_pos_list: fig2.add_trace(trace_ankle_vel)
        if "Metatarso" in xy_pos_list: fig2.add_trace(trace_meta_vel)
        st.plotly_chart(fig2, use_container_width=True)

        acro_x_acc = get_gradient_simp(acro_x_vel)
        acro_y_acc = get_gradient_simp(acro_y_vel)
        hip_x_acc = get_gradient_simp(hip_x_vel)
        hip_y_acc = get_gradient_simp(hip_y_vel)
        knee_x_acc = get_gradient_simp(knee_x_vel)
        knee_y_acc = get_gradient_simp(knee_y_vel)
        ankle_x_acc = get_gradient_simp(ankle_x_vel)
        ankle_y_acc = get_gradient_simp(ankle_y_vel)
        meta_x_acc = get_gradient_simp(meta_x_vel)
        meta_y_acc = get_gradient_simp(meta_y_vel)

        trace_acro_acc = go.Scatter(x=acro_x_acc, y=acro_y_acc, name='Acromion R')
        trace_hip_acc = go.Scatter(x=hip_x_acc, y=hip_y_acc, name='Cadera R')
        trace_knee_acc = go.Scatter(x=knee_x_acc, y=knee_y_acc, name='Rodilla R')
        trace_ankle_acc = go.Scatter(x=ankle_x_acc, y=ankle_y_acc, name='Tobillo R')
        trace_meta_acc = go.Scatter(x=meta_x_acc, y=meta_y_acc, name='Metatarso R')

        """#### Aceleraciones plano xy"""
        fig3 = make_subplots(specs=[[{"secondary_y": False}]])
        if "Acromion" in xy_pos_list: fig3.add_trace(trace_acro_acc)
        if "cadera" in xy_pos_list: fig3.add_trace(trace_hip_acc)
        if "Rodilla" in xy_pos_list: fig3.add_trace(trace_knee_acc)
        if "Tobillo" in xy_pos_list: fig3.add_trace(trace_ankle_acc)
        if "Metatarso" in xy_pos_list: fig3.add_trace(trace_meta_acc)
        st.plotly_chart(fig3, use_container_width=True)

        """#### Magnitud de aceleraciones plano xy"""
        acro_mag = np.sqrt(acro_x_acc**2 + acro_y_acc**2)
        hip_mag = np.sqrt(hip_x_acc**2 +  hip_y_acc**2)
        knee_mag = np.sqrt(knee_x_acc**2 + knee_y_acc**2)
        ankle_mag = np.sqrt(ankle_x_acc**2 + ankle_y_acc**2)
        meta_mag = np.sqrt(meta_x_acc**2 + meta_y_acc**2)


        xy_df = pd.DataFrame(list(zip(acro_mag, hip_mag, knee_mag, ankle_mag, meta_mag)), columns =['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'])
        st.line_chart(get_data(xy_df,xy_pos_list))



if show_heatmaps:
    st.subheader('Mapas de calor')
    colheat1, colheat2 = st.columns(2)

    with colheat1:
        if "Acromion" in heat_list:
            acro_left = px.density_heatmap(x=df_left["Acromion I_x"], y=-df_left["Acromion I_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")#,text_auto=True)
            st.plotly_chart(acro_left, use_container_width=True)

        if "Cadera" in heat_list:
            hip_left = px.density_heatmap(x=df_left["Cadera I_x"], y=-df_left["Cadera I_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(hip_left, use_container_width=True)

        if "Rodilla" in heat_list:
            knee_left = px.density_heatmap(x=df_left["Rodilla I_x"], y=-df_left["Rodilla I_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(knee_left, use_container_width=True)

        if "Tobillo" in heat_list:
            ankle_left = px.density_heatmap(x=df_left["Tobillo I_x"], y=-df_left["Tobillo I_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(ankle_left, use_container_width=True)

        if "Metatarso" in heat_list:
            meta_left = px.density_heatmap(x=df_left["Metatarso I_x"], y=-df_left["Metatarso I_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(meta_left, use_container_width=True)

    with colheat2:
        if "Acromion" in heat_list:
            acro_right = px.density_heatmap(x=df_right["Acromion D_x"], y=-df_right["Acromion D_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(acro_right, use_container_width=True)

        if "Cadera" in heat_list:
            hip_right = px.density_heatmap(x=df_right["Cadera D_x"], y=-df_right["Cadera D_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(hip_right, use_container_width=True)

        if "Rodilla" in heat_list:
            knee_right = px.density_heatmap(x=df_right["Rodilla D_x"], y=-df_right["Rodilla D_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(knee_right, use_container_width=True)

        if "Tobillo" in heat_list:
            ankle_right = px.density_heatmap(x=df_right["Tobillo D_x"], y=-df_right["Tobillo D_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(ankle_right, use_container_width=True)

        if "Metatarso" in heat_list:
            meta_right = px.density_heatmap(x=df_right["Metatarso D_x"], y=-df_right["Metatarso D_y"], nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(meta_right, use_container_width=True)


        

