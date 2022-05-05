"""
@file     : graphics_service.py
@brief   : Adds visual elements.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""


import streamlit as st
import cv2
import numpy as np
from source.services.Data.data_service import Data
from source.services.Math.math_service import Math

class Graphics:
    @st.cache
    def load_video_frame(video_name, id):
        video = cv2.VideoCapture(video_name)
        video.set(cv2.CAP_PROP_POS_FRAMES, id)
        _, frame = video.read()
        return frame


    def get_video_info(video):
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height

    # @st.cache(suppress_st_warning=True)
    def show_angle(dataframe, angle, candidates, fps, plane, default_cycle, slider, key):
        if plane == "Left":
            angle = angle + " LI"
        elif plane == "Right":
            angle = angle + " LD"

        cycles = st.slider(
            slider,
            1,
            len(candidates) - 1,
            default_cycle,
            1,
            format=None,
            key=key,
        )

        st.line_chart(Math.get_pos_cycle(dataframe, angle, cycles, candidates, fps))


    def show_positions(
        dataframe, position, candidates, fps, plane, default_cycle, slider, key, coordinate
    ):
        if plane == "Left":
            position = position + " I_"
        elif plane == "Right":
            position = position + " D_"
        position = position + coordinate

        cycles = st.slider(
            slider,
            1,
            len(candidates) - 1,
            default_cycle,
            1,
            format=None,
            key=key,
        )

        pos = Math.get_pos_cycle(dataframe, position, cycles, candidates, fps)
        st.line_chart(pos)

        """#### Velocidades eje x"""
        vel = Math.get_gradient(pos, fps)
        st.line_chart(vel)

        """#### Aceleraciones eje x"""
        acc = Math.get_gradient(vel, fps)
        st.line_chart(acc)


    # @st.cache(suppress_st_warning=True)
    def show_image(
        candidates,
        video,
        plane,
        dataframe,
        header,
        slider1,
        key1,
        help1,
        slider2,
        key2,
        help2,
        key3,
        window,
        default_cycle,
    ):
        window = 5
        st.subheader(header)
        st.session_state.cycle = st.slider(
            slider1,
            -window,
            window,
            0,
            1,
            format=None,
            key=key1,
            help=help1,
        )
        cycle = st.slider(
            slider2,
            0,
            len(candidates) - 1,
            default_cycle,
            1,
            format=None,
            key=key2,
            help=help2,
        )
        id_number = candidates[cycle] + st.session_state.cycle

        frame = Graphics.load_video_frame(video, id_number)
        if plane == "Left":
            text = " LI"
        elif plane == "Right":
            text = " LD"

        caption = f"Tronco{text}: {Math.trunc(dataframe.iloc[id_number]['Tronco'+text],1)}, Cadera{text}: {round(dataframe.iloc[id_number]['Cadera'+text],1)}, Rodilla{text}: {round(dataframe.iloc[id_number]['Rodilla'+text],1)}, Tobillo{text}: {round(dataframe.iloc[id_number]['Tobillo'+text],1)}"
        st.image(frame, caption=caption, channels="BGR")

        st.write(f"Tronco{text}: {Math.trunc(dataframe.iloc[id_number]['Tronco'+text],1)}")
        st.write(f"Cadera{text}: {Math.trunc(dataframe.iloc[id_number]['Cadera'+text],1)}")
        st.write(f"Rodilla{text}: {Math.trunc(dataframe.iloc[id_number]['Rodilla'+text],1)}")
        st.write(f"Tobillo{text}: {Math.trunc(dataframe.iloc[id_number]['Tobillo'+text],1)}")

        comentario = st.text_input("Comentarios", key=key3)
        st.write("comentario:", comentario)



    def show_temp_params(cc, sc, fps, plane, factor=2):
        decimals=1
        # Promedio de cadencia
        CAD = Math.trunc(fps * factor/2 * 60 / np.average(np.diff(cc)), decimals)
    
        # Promedio de tiempo de apoyo
        TA_list = []
        for i in range(len(sc)-3):
            TA_list.append(sc[i]-cc[i])
        TA = Math.trunc(np.average(TA_list) / fps * factor, decimals)

        # Promedio de tiempo de balanceo
        TB_list = []
        for i in range(len(sc)-3):
            TB_list.append(cc[i+1]-sc[i])
        TB = Math.trunc(np.average(TB_list) / fps * factor, decimals)

        # Promedio de tiempo de paso
        TP = Math.trunc(fps / factor / np.average(np.diff(cc)), decimals)

        st.write(f"Cadencia {plane}: {CAD} pasos por minuto")
        st.write(f"Tiempo de apoyo {plane[0:-1]}o: {TA} segundos")
        st.write(f"Tiempo de balanceo {plane[0:-1]}o: {TB} segundos")
        st.write(f"Tiempo de paso {plane[0:-1]}o: {TP} segundos")
