"""
@file     : report_service.py
@brief   : Handles report creation.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""


import cv2
import numpy as np
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from source.services.Math.math_service import Math
from source.services.Data.data_service import Data
from source.services.Database.database_service import Database
from source.services.Graphics.graphics_service import Graphics
from source.services.AWS.S3 import S3


class Report:

    window = 5
    default_cycle = 1
    factor = 2

    def __init__(
        self,
        db_info=[],
    ):

        exam_info = db_info[0]
        patient_info = db_info[1]
        professional_info = db_info[2]

        self.name = f"{patient_info[1]} {patient_info[2]}"
        self.email = f"{patient_info[4]}"
        birthdate = patient_info[7].split("-")
        self.age = Math.calculateAge(date(int(birthdate[0]), int(birthdate[1]), int(birthdate[2])))
        self.exam_date = f"{exam_info[9]}"
        self.report_date = f"{exam_info[10]}"

        self.left_video_name = "p5l_arba.mp4"
        self.right_video_name = "p5r_arba.mp4"

        left_video = cv2.VideoCapture(self.left_video_name)
        left_frame_count = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fpsl, self.widthl, self.heightl = Graphics.get_video_info(left_video)
        right_video = cv2.VideoCapture(self.right_video_name)
        right_frame_count = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fpsr, self.widthr, self.heightr = Graphics.get_video_info(right_video)

        self.fpsl = self.fpsl * self.factor
        self.fpsr = self.fpsr * self.factor

        if "count1" not in st.session_state:
            st.session_state.count1 = 0
        if "count2" not in st.session_state:
            st.session_state.count2 = 0
        if "count3" not in st.session_state:
            st.session_state.count3 = 0
        if "count4" not in st.session_state:
            st.session_state.count4 = 0
        if "cycle" not in st.session_state:
            st.session_state.cycle = 1
        if "cycle_left_1" not in st.session_state:
            st.session_state.cycle_left_1 = 1
        if "cycle_left_2" not in st.session_state:
            st.session_state.cycle_left_2 = 1
        if "cycle_right_1" not in st.session_state:
            st.session_state.cycle_right_1 = 1
        if "cycle_right_2" not in st.session_state:
            st.session_state.cycle_right_2 = 1

        df_left = pd.read_csv("assets/csv/p5l_arba.csv")
        df_right = pd.read_csv("assets/csv/p5r_arba.csv")
        report_left = open("assets/json/p5l_arba.json")
        report_right = open("assets/json/p5r_arba.json")
        self.df_left, self.df_right, self.data_left, self.data_right = Data.load_data(
            df_left, df_right, report_left, report_right
        )

        candidates = Math.get_candidates(self.data_left, self.data_right)

        self.lcc = candidates["lcc"]
        self.lmc = candidates["lmc"]
        self.lsc = candidates["lsc"]
        self.ltp = candidates["ltp"]
        self.rcc = candidates["rcc"]
        self.rmc = candidates["rmc"]
        self.rsc = candidates["rsc"]
        self.rtp = candidates["rtp"]

        # select_plane = st.sidebar.selectbox(
        #    "Seleccionar el Plano",
        #    ("","Sagital", "Frontal","lala")
        # )
        # if select_plane=="Sagital":
        #    select_contact = st.sidebar.selectbox(
        #        "Seleccionar contacto",
        #        ("Contacto inicial", "Apoyo medio")
        #    )
        # elif select_plane=="Frontal":
        #    select_contact = st.sidebar.selectbox(
        #        "Seleccionar contacto",
        #        ("Contacto inicial2", "Apoyo medio2")
        #    )

        self.create_title("Reporte ABMA")
        self.create_patient_data()
        self.create_clinical_data()

        graphics = False
        show_angles = st.sidebar.checkbox("Mostrar ángulos", value=False)
        if show_angles:
            graphics = True
            self.ang_list = st.sidebar.selectbox(
                "Seleccionar ángulos a graficar",
                ("Tronco", "Cadera", "Rodilla", "Tobillo"),
                key="ang_list",
            )

        show_x_pos = st.sidebar.checkbox("Mostrar posiciones X", value=False)
        if show_x_pos:
            graphics = True
            self.x_pos_list = st.sidebar.selectbox(
                "Seleccionar posiciones x a graficar",
                ("Acromion", "Cadera", "Rodilla", "Tobillo", "Metatarso"),
                key="x_pos_list",
            )

        show_y_pos = st.sidebar.checkbox("Mostrar posiciones Y", value=False)
        if show_y_pos:
            graphics = True
            self.y_pos_list = st.sidebar.selectbox(
                "Seleccionar posiciones y a graficar",
                ("Acromion", "Cadera", "Rodilla", "Tobillo", "Metatarso"),
                key="y_pos_list",
            )

        show_xy_pos = st.sidebar.checkbox("Mostrar plano XY", value=False)
        if show_xy_pos:
            graphics = True
            self.xy_pos_list = st.sidebar.selectbox(
                "Seleccionar posiciones x/y a graficar",
                ("Acromion", "Cadera", "Rodilla", "Tobillo", "Metatarso"),
                key="xy_pos_list",
            )

        show_heatmaps = st.sidebar.checkbox("Mostrar mapas de calor", value=False)
        if show_heatmaps:
            graphics = True
            self.heat_list = st.sidebar.multiselect(
                "Seleccionar mapas de calor a graficar",
                ["Acromion", "Cadera", "Rodilla", "Tobillo", "Metatarso"],
                ["Acromion", "Cadera", "Rodilla", "Tobillo", "Metatarso"],
                key="heat_list",
            )

        self.create_images()

        self.create_parameters()

        if graphics:
            st.header("Gráficos")
            if show_angles:
                self.create_angles()
            if show_x_pos:
                self.create_x_positions()
            if show_y_pos:
                self.create_y_positions()
            if show_xy_pos:
                self.create_xy_positions()
            if show_heatmaps:
                self.create_heatmaps()

    def create_title(self, text):
        st.title(text)

    def create_header(self, text):
        st.header(text)
    
    def create_patient_data(self):
        self.create_header("Datos paciente")
        coldp1, coldp2 = st.columns(2)

        with coldp1:
            st.write("Nombre: ", f"{self.name}")
            st.write("Edad: ", f"{self.age}")
            st.write("Mail: ", f"{self.email}")

        with coldp2:
            st.write("Fecha de evaluación: ", f"{self.exam_date}")
            st.write("Fecha de reportería: ", f"{self.report_date}")
    
    def create_clinical_data(self):
        self.create_header("Datos clínicos")
        antecedentes = st.sidebar.text_input("Antecedentes", key="antecedentes")
        st.write("Antecedentes:", antecedentes)

        km_semanales = st.sidebar.text_input("Kilómetros semanales", key="km_semanales")
        st.write("Kilómetros semanales:", km_semanales)

    def create_images(self):
        st.header("Visual")
        colvis1, colvis2 = st.columns(2)

        with colvis1:
            Graphics.show_image(
                candidates=self.lcc,
                video=self.left_video_name,
                plane="Left",
                dataframe=self.df_left,
                header="Extremidad izquierda, contacto inicial",
                slider1="Desfase de contacto izquierdo",
                key1="left_contact",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera izquierda contacto",
                key2="left_contact_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_left_1",
                window=self.window,
                default_cycle=self.default_cycle,
            )

            Graphics.show_image(
                candidates=self.lmc,
                video=self.left_video_name,
                plane="Left",
                dataframe=self.df_left,
                header="Extremidad izquierda, apoyo medio",
                slider1="Desfase de apoyo medio izquierdo",
                key1="left_medium",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera izquierda apoyo medio",
                key2="left_medium_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_left_2",
                window=self.window,
                default_cycle=self.default_cycle,
            )

            Graphics.show_image(
                candidates=self.lsc,
                video=self.left_video_name,
                plane="Left",
                dataframe=self.df_left,
                header="Extremidad izquierda, despegue metatarso",
                slider1="Desfase de separación izquierdo",
                key1="left_separation",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera izquierda despegue metatarso",
                key2="left_separation_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_left_3",
                window=self.window,
                default_cycle=self.default_cycle,
            )

        with colvis2:

            Graphics.show_image(
                candidates=self.rcc,
                video=self.right_video_name,
                plane="Right",
                dataframe=self.df_right,
                header="Extremidad derecha, contacto inicial",
                slider1="Desfase de contacto derecho",
                key1="right_contact",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera derecha contacto",
                key2="right_contact_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_right_1",
                window=self.window,
                default_cycle=self.default_cycle,
            )

            Graphics.show_image(
                candidates=self.rmc,
                video=self.right_video_name,
                plane="Right",
                dataframe=self.df_right,
                header="Extremidad derecha, apoyo medio",
                slider1="Desfase de apoyo medio derecho",
                key1="right_medium",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera derecha apoyo medio",
                key2="right_medium_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_right_2",
                window=self.window,
                default_cycle=self.default_cycle,
            )

            Graphics.show_image(
                candidates=self.rsc,
                video=self.right_video_name,
                plane="Right",
                dataframe=self.df_right,
                header="Extremidad derecha, despegue metatarso",
                slider1="Desfase de separación derecho",
                key1="right_separation",
                help1="Cambia entre los cuadros adyacentes al ciclo de carrera seleccionado",
                slider2="Ciclo de carrera derecha despegue metatarso",
                key2="right_separation_slider",
                help2="Cambia entre los diferentes ciclos de carrera",
                key3="comentario_right_3",
                window=self.window,
                default_cycle=self.default_cycle,
            )

    def create_parameters(self):
        st.subheader("Parámetros temporales adicionales")

        colad1, colad2 = st.columns(2)

        with colad1:
            Graphics.show_temp_params(cc=self.lcc, sc=self.lsc, fps=self.fpsl, plane="izquierda", factor=self.factor)

        with colad2:
            Graphics.show_temp_params(cc=self.rcc, sc=self.rsc, fps=self.fpsr, plane="derecha", factor=self.factor)

    def create_angles(self):

        st.subheader("Ángulos articulares")
        colang1, colang2 = st.columns(2)

        with colang1:
            Graphics.show_angle(
                dataframe=self.df_left,
                angle=self.ang_list,
                candidates=self.lcc,
                fps=self.fpsl,
                plane="Left",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera izquierda",
                key="left_angle_slider",
            )

        with colang2:
            Graphics.show_angle(
                dataframe=self.df_right,
                angle=self.ang_list,
                candidates=self.rcc,
                fps=self.fpsr,
                plane="Right",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera derecha",
                key="right_angle_slider",
            )

    def create_x_positions(self):
        st.subheader("Posiciones en el eje X")
        colposx1, colposx2 = st.columns(2)

        with colposx1:
            Graphics.show_positions(
                dataframe=self.df_left,
                position=self.x_pos_list,
                candidates=self.lcc,
                fps=self.fpsl,
                plane="Left",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera izquierda",
                key="left_x_slider",
                coordinate="x",
            )

        with colposx2:
            Graphics.show_positions(
                dataframe=self.df_right,
                position=self.x_pos_list,
                candidates=self.rcc,
                fps=self.fpsr,
                plane="Right",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera derecha",
                key="right_x_slider",
                coordinate="x",
            )

    def create_y_positions(self):
        st.subheader("Posiciones en el eje Y")
        colposy1, colposy2 = st.columns(2)

        with colposy1:
            Graphics.show_positions(
                dataframe=self.df_left,
                position=self.y_pos_list,
                candidates=self.lcc,
                fps=self.fpsl,
                plane="Left",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera izquierda",
                key="left_y_slider",
                coordinate="y",
            )

        with colposy2:
            Graphics.show_positions(
                dataframe=self.df_right,
                position=self.y_pos_list,
                candidates=self.rcc,
                fps=self.fpsr,
                plane="Right",
                default_cycle=self.default_cycle,
                slider="Cantidad de ciclos de carrera derecha",
                key="right_y_slider",
                coordinate="y",
            )

    def create_xy_positions(self):
        st.subheader("Posiciones en el plano XY")
        colposxy1, colposxy2 = st.columns(2)

        with colposxy1:
            ciclo_xy = st.slider(
                "Ciclo de carrera izquierda",
                0,
                len(self.lcc) - 1,
                1,
                1,
                format=None,
                key="left_xy_slider",
                help="Cambia entre los diferentes ciclos de carrera",
            )
            min_range, max_range = st.slider(
                "Rango de ciclos de carrera izquierda",
                self.lcc[ciclo_xy],
                self.lcc[ciclo_xy + 1],
                [self.lcc[ciclo_xy], self.lcc[ciclo_xy + 1]],
                1,
                format=None,
                key="left_position_slider",
            )

            acro_x = Math.butter_lowpass_filter(
                self.df_left["Acromion I_x"][min_range:max_range], fs=self.fpsl
            )
            acro_y = Math.butter_lowpass_filter(
                -self.df_left["Acromion I_y"][min_range:max_range], fs=self.fpsl
            )
            hip_x = Math.butter_lowpass_filter(
                self.df_left["Cadera I_x"][min_range:max_range], fs=self.fpsl
            )
            hip_y = Math.butter_lowpass_filter(
                -self.df_left["Cadera I_y"][min_range:max_range], fs=self.fpsl
            )
            knee_x = Math.butter_lowpass_filter(
                self.df_left["Rodilla I_x"][min_range:max_range], fs=self.fpsl
            )
            knee_y = Math.butter_lowpass_filter(
                -self.df_left["Rodilla I_y"][min_range:max_range], fs=self.fpsl
            )
            ankle_x = Math.butter_lowpass_filter(
                self.df_left["Tobillo I_x"][min_range:max_range], fs=self.fpsl
            )
            ankle_y = Math.butter_lowpass_filter(
                -self.df_left["Tobillo I_y"][min_range:max_range], fs=self.fpsl
            )
            meta_x = Math.butter_lowpass_filter(
                self.df_left["Metatarso I_x"][min_range:max_range], fs=self.fpsl
            )
            meta_y = Math.butter_lowpass_filter(
                -self.df_left["Metatarso I_y"][min_range:max_range], fs=self.fpsl
            )

            trace_acro = go.Scatter(x=acro_x, y=acro_y, name="Acromion L")
            trace_hip = go.Scatter(x=hip_x, y=hip_y, name="Cadera L")
            trace_knee = go.Scatter(x=knee_x, y=knee_y, name="Rodilla L")
            trace_ankle = go.Scatter(x=ankle_x, y=ankle_y, name="Tobillo L")
            trace_meta = go.Scatter(x=meta_x, y=meta_y, name="Metatarso L")

            fig = make_subplots(specs=[[{"secondary_y": False}]])
            # if st.checkbox('Acromion L', value=True): fig.add_trace(trace_acro)
            # if st.checkbox('Cadera L', value=True): fig.add_trace(trace_hip)
            # if st.checkbox('Rodilla L', value=True): fig.add_trace(trace_knee)
            # if st.checkbox('Tobillo L', value=True): fig.add_trace(trace_ankle)
            # if st.checkbox('Metatarso L', value=True): fig.add_trace(trace_meta)
            if "Acromion" in self.xy_pos_list:
                fig.add_trace(trace_acro)
            if "cadera" in self.xy_pos_list:
                fig.add_trace(trace_hip)
            if "Rodilla" in self.xy_pos_list:
                fig.add_trace(trace_knee)
            if "Tobillo" in self.xy_pos_list:
                fig.add_trace(trace_ankle)
            if "Metatarso" in self.xy_pos_list:
                fig.add_trace(trace_meta)
            st.plotly_chart(fig, use_container_width=True)

            acro_x_vel = Math.get_gradient_simp(acro_x.tolist(), self.fpsl)
            acro_y_vel = Math.get_gradient_simp(acro_y.tolist(), self.fpsl)
            hip_x_vel = Math.get_gradient_simp(hip_x.tolist(), self.fpsl)
            hip_y_vel = Math.get_gradient_simp(hip_y.tolist(), self.fpsl)
            knee_x_vel = Math.get_gradient_simp(knee_x.tolist(), self.fpsl)
            knee_y_vel = Math.get_gradient_simp(knee_y.tolist(), self.fpsl)
            ankle_x_vel = Math.get_gradient_simp(ankle_x.tolist(), self.fpsl)
            ankle_y_vel = Math.get_gradient_simp(ankle_y.tolist(), self.fpsl)
            meta_x_vel = Math.get_gradient_simp(meta_x.tolist(), self.fpsl)
            meta_y_vel = Math.get_gradient_simp(meta_y.tolist(), self.fpsl)

            trace_acro_vel = go.Scatter(x=acro_x_vel, y=acro_y_vel, name="Acromion L")
            trace_hip_vel = go.Scatter(x=hip_x_vel, y=hip_y_vel, name="Cadera L")
            trace_knee_vel = go.Scatter(x=knee_x_vel, y=knee_y_vel, name="Rodilla L")
            trace_ankle_vel = go.Scatter(x=ankle_x_vel, y=ankle_y_vel, name="Tobillo L")
            trace_meta_vel = go.Scatter(x=meta_x_vel, y=meta_y_vel, name="Metatarso L")

            """#### Velocidades plano xy"""
            fig2 = make_subplots(specs=[[{"secondary_y": False}]])
            if "Acromion" in self.xy_pos_list:
                fig2.add_trace(trace_acro_vel)
            if "cadera" in self.xy_pos_list:
                fig2.add_trace(trace_hip_vel)
            if "Rodilla" in self.xy_pos_list:
                fig2.add_trace(trace_knee_vel)
            if "Tobillo" in self.xy_pos_list:
                fig2.add_trace(trace_ankle_vel)
            if "Metatarso" in self.xy_pos_list:
                fig2.add_trace(trace_meta_vel)
            st.plotly_chart(fig2, use_container_width=True)

            acro_x_acc = Math.get_gradient_simp(acro_x_vel, self.fpsl)
            acro_y_acc = Math.get_gradient_simp(acro_y_vel, self.fpsl)
            hip_x_acc = Math.get_gradient_simp(hip_x_vel, self.fpsl)
            hip_y_acc = Math.get_gradient_simp(hip_y_vel, self.fpsl)
            knee_x_acc = Math.get_gradient_simp(knee_x_vel, self.fpsl)
            knee_y_acc = Math.get_gradient_simp(knee_y_vel, self.fpsl)
            ankle_x_acc = Math.get_gradient_simp(ankle_x_vel, self.fpsl)
            ankle_y_acc = Math.get_gradient_simp(ankle_y_vel, self.fpsl)
            meta_x_acc = Math.get_gradient_simp(meta_x_vel, self.fpsl)
            meta_y_acc = Math.get_gradient_simp(meta_y_vel, self.fpsl)

            trace_acro_acc = go.Scatter(x=acro_x_acc, y=acro_y_acc, name="Acromion L")
            trace_hip_acc = go.Scatter(x=hip_x_acc, y=hip_y_acc, name="Cadera L")
            trace_knee_acc = go.Scatter(x=knee_x_acc, y=knee_y_acc, name="Rodilla L")
            trace_ankle_acc = go.Scatter(x=ankle_x_acc, y=ankle_y_acc, name="Tobillo L")
            trace_meta_acc = go.Scatter(x=meta_x_acc, y=meta_y_acc, name="Metatarso L")

            """#### Aceleraciones plano xy"""
            fig3 = make_subplots(specs=[[{"secondary_y": False}]])
            if "Acromion" in self.xy_pos_list:
                fig3.add_trace(trace_acro_acc)
            if "cadera" in self.xy_pos_list:
                fig3.add_trace(trace_hip_acc)
            if "Rodilla" in self.xy_pos_list:
                fig3.add_trace(trace_knee_acc)
            if "Tobillo" in self.xy_pos_list:
                fig3.add_trace(trace_ankle_acc)
            if "Metatarso" in self.xy_pos_list:
                fig3.add_trace(trace_meta_acc)
            st.plotly_chart(fig3, use_container_width=True)

            # """#### Magnitud de aceleraciones plano xy"""
            acro_mag = np.sqrt(acro_x_acc**2 + acro_y_acc**2)
            hip_mag = np.sqrt(hip_x_acc**2 + hip_y_acc**2)
            knee_mag = np.sqrt(knee_x_acc**2 + knee_y_acc**2)
            ankle_mag = np.sqrt(ankle_x_acc**2 + ankle_y_acc**2)
            meta_mag = np.sqrt(meta_x_acc**2 + meta_y_acc**2)

            # xy_df = pd.DataFrame(list(zip(acro_mag, hip_mag, knee_mag, ankle_mag, meta_mag)), columns =['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'])
            # st.line_chart(Data.get_data(xy_df,xy_pos_list))

        with colposxy2:

            ciclo_xy = st.slider(
                "Ciclo de carrera dercha",
                0,
                len(self.rcc) - 1,
                1,
                1,
                format=None,
                key="right_xy_slider",
                help="Cambia entre los diferentes ciclos de carrera",
            )
            min_range, max_range = st.slider(
                "Rango de ciclos de carrera derecha",
                self.rcc[ciclo_xy],
                self.rcc[ciclo_xy + 1],
                [self.rcc[ciclo_xy], self.rcc[ciclo_xy + 1]],
                1,
                format=None,
                key="right_position_slider",
            )

            acro_x = Math.butter_lowpass_filter(
                self.df_right["Acromion D_x"][min_range:max_range], self.fpsr
            )
            acro_y = Math.butter_lowpass_filter(
                -self.df_right["Acromion D_y"][min_range:max_range], self.fpsr
            )
            hip_x = Math.butter_lowpass_filter(
                self.df_right["Cadera D_x"][min_range:max_range], self.fpsr
            )
            hip_y = Math.butter_lowpass_filter(
                -self.df_right["Cadera D_y"][min_range:max_range], self.fpsr
            )
            knee_x = Math.butter_lowpass_filter(
                self.df_right["Rodilla D_x"][min_range:max_range], self.fpsr
            )
            knee_y = Math.butter_lowpass_filter(
                -self.df_right["Rodilla D_y"][min_range:max_range], self.fpsr
            )
            ankle_x = Math.butter_lowpass_filter(
                self.df_right["Tobillo D_x"][min_range:max_range], self.fpsr
            )
            ankle_y = Math.butter_lowpass_filter(
                -self.df_right["Tobillo D_y"][min_range:max_range], self.fpsr
            )
            meta_x = Math.butter_lowpass_filter(
                self.df_right["Metatarso D_x"][min_range:max_range], self.fpsr
            )
            meta_y = Math.butter_lowpass_filter(
                -self.df_right["Metatarso D_y"][min_range:max_range], self.fpsr
            )

            trace_acro = go.Scatter(x=acro_x, y=acro_y, name="Acromion R")
            trace_hip = go.Scatter(x=hip_x, y=hip_y, name="Cadera R")
            trace_knee = go.Scatter(x=knee_x, y=knee_y, name="Rodilla R")
            trace_ankle = go.Scatter(x=ankle_x, y=ankle_y, name="Tobillo R")
            trace_meta = go.Scatter(x=meta_x, y=meta_y, name="Metatarso R")

            fig = make_subplots(specs=[[{"secondary_y": False}]])
            # if st.checkbox('Acromion R', value=True): fig.add_trace(trace_acro)
            # if st.checkbox('Cadera R', value=True): fig.add_trace(trace_hip)
            # if st.checkbox('Rodilla R', value=True): fig.add_trace(trace_knee)
            # if st.checkbox('Tobillo R', value=True): fig.add_trace(trace_ankle)
            # if st.checkbox('Metatarso R', value=True): fig.add_trace(trace_meta)
            if "Acromion" in self.xy_pos_list:
                fig.add_trace(trace_acro)
            if "cadera" in self.xy_pos_list:
                fig.add_trace(trace_hip)
            if "Rodilla" in self.xy_pos_list:
                fig.add_trace(trace_knee)
            if "Tobillo" in self.xy_pos_list:
                fig.add_trace(trace_ankle)
            if "Metatarso" in self.xy_pos_list:
                fig.add_trace(trace_meta)
            st.plotly_chart(fig, use_container_width=True)

            acro_x_vel = Math.get_gradient_simp(acro_x.tolist(), self.fpsr)
            acro_y_vel = Math.get_gradient_simp(acro_y.tolist(), self.fpsr)
            hip_x_vel = Math.get_gradient_simp(hip_x.tolist(), self.fpsr)
            hip_y_vel = Math.get_gradient_simp(hip_y.tolist(), self.fpsr)
            knee_x_vel = Math.get_gradient_simp(knee_x.tolist(), self.fpsr)
            knee_y_vel = Math.get_gradient_simp(knee_y.tolist(), self.fpsr)
            ankle_x_vel = Math.get_gradient_simp(ankle_x.tolist(), self.fpsr)
            ankle_y_vel = Math.get_gradient_simp(ankle_y.tolist(), self.fpsr)
            meta_x_vel = Math.get_gradient_simp(meta_x.tolist(), self.fpsr)
            meta_y_vel = Math.get_gradient_simp(meta_y.tolist(), self.fpsr)

            trace_acro_vel = go.Scatter(x=acro_x_vel, y=acro_y_vel, name="Acromion R")
            trace_hip_vel = go.Scatter(x=hip_x_vel, y=hip_y_vel, name="Cadera R")
            trace_knee_vel = go.Scatter(x=knee_x_vel, y=knee_y_vel, name="Rodilla R")
            trace_ankle_vel = go.Scatter(x=ankle_x_vel, y=ankle_y_vel, name="Tobillo R")
            trace_meta_vel = go.Scatter(x=meta_x_vel, y=meta_y_vel, name="Metatarso R")

            """#### Velocidades plano xy"""
            fig2 = make_subplots(specs=[[{"secondary_y": False}]])
            if "Acromion" in self.xy_pos_list:
                fig2.add_trace(trace_acro_vel)
            if "cadera" in self.xy_pos_list:
                fig2.add_trace(trace_hip_vel)
            if "Rodilla" in self.xy_pos_list:
                fig2.add_trace(trace_knee_vel)
            if "Tobillo" in self.xy_pos_list:
                fig2.add_trace(trace_ankle_vel)
            if "Metatarso" in self.xy_pos_list:
                fig2.add_trace(trace_meta_vel)
            st.plotly_chart(fig2, use_container_width=True)

            acro_x_acc = Math.get_gradient_simp(acro_x_vel, self.fpsr)
            acro_y_acc = Math.get_gradient_simp(acro_y_vel, self.fpsr)
            hip_x_acc = Math.get_gradient_simp(hip_x_vel, self.fpsr)
            hip_y_acc = Math.get_gradient_simp(hip_y_vel, self.fpsr)
            knee_x_acc = Math.get_gradient_simp(knee_x_vel, self.fpsr)
            knee_y_acc = Math.get_gradient_simp(knee_y_vel, self.fpsr)
            ankle_x_acc = Math.get_gradient_simp(ankle_x_vel, self.fpsr)
            ankle_y_acc = Math.get_gradient_simp(ankle_y_vel, self.fpsr)
            meta_x_acc = Math.get_gradient_simp(meta_x_vel, self.fpsr)
            meta_y_acc = Math.get_gradient_simp(meta_y_vel, self.fpsr)

            trace_acro_acc = go.Scatter(x=acro_x_acc, y=acro_y_acc, name="Acromion R")
            trace_hip_acc = go.Scatter(x=hip_x_acc, y=hip_y_acc, name="Cadera R")
            trace_knee_acc = go.Scatter(x=knee_x_acc, y=knee_y_acc, name="Rodilla R")
            trace_ankle_acc = go.Scatter(x=ankle_x_acc, y=ankle_y_acc, name="Tobillo R")
            trace_meta_acc = go.Scatter(x=meta_x_acc, y=meta_y_acc, name="Metatarso R")

            """#### Aceleraciones plano xy"""
            fig3 = make_subplots(specs=[[{"secondary_y": False}]])
            if "Acromion" in self.xy_pos_list:
                fig3.add_trace(trace_acro_acc)
            if "cadera" in self.xy_pos_list:
                fig3.add_trace(trace_hip_acc)
            if "Rodilla" in self.xy_pos_list:
                fig3.add_trace(trace_knee_acc)
            if "Tobillo" in self.xy_pos_list:
                fig3.add_trace(trace_ankle_acc)
            if "Metatarso" in self.xy_pos_list:
                fig3.add_trace(trace_meta_acc)
            st.plotly_chart(fig3, use_container_width=True)

            # """#### Magnitud de aceleraciones plano xy"""
            acro_mag = np.sqrt(acro_x_acc**2 + acro_y_acc**2)
            hip_mag = np.sqrt(hip_x_acc**2 + hip_y_acc**2)
            knee_mag = np.sqrt(knee_x_acc**2 + knee_y_acc**2)
            ankle_mag = np.sqrt(ankle_x_acc**2 + ankle_y_acc**2)
            meta_mag = np.sqrt(meta_x_acc**2 + meta_y_acc**2)

            # xy_df = pd.DataFrame(list(zip(acro_mag, hip_mag, knee_mag, ankle_mag, meta_mag)), columns =['Acromion', 'Cadera', 'Rodilla', 'Tobillo', 'Metatarso'])
            # st.line_chart(Data.get_data(xy_df,xy_pos_list))

    def create_heatmaps(self):
        st.subheader("Mapas de calor")
        colheat1, colheat2 = st.columns(2)

        with colheat1:
            if "Acromion" in self.heat_list:
                acro_left = px.density_heatmap(
                    x=self.df_left["Acromion I_x"],
                    y=-self.df_left["Acromion I_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )  # ,text_auto=True)
                st.plotly_chart(acro_left, use_container_width=True)

            if "Cadera" in self.heat_list:
                hip_left = px.density_heatmap(
                    x=self.df_left["Cadera I_x"],
                    y=-self.df_left["Cadera I_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(hip_left, use_container_width=True)

            if "Rodilla" in self.heat_list:
                knee_left = px.density_heatmap(
                    x=self.df_left["Rodilla I_x"],
                    y=-self.df_left["Rodilla I_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(knee_left, use_container_width=True)

            if "Tobillo" in self.heat_list:
                ankle_left = px.density_heatmap(
                    x=self.df_left["Tobillo I_x"],
                    y=-self.df_left["Tobillo I_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(ankle_left, use_container_width=True)

            if "Metatarso" in self.heat_list:
                meta_left = px.density_heatmap(
                    x=self.df_left["Metatarso I_x"],
                    y=-self.df_left["Metatarso I_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(meta_left, use_container_width=True)

        with colheat2:
            if "Acromion" in self.heat_list:
                acro_right = px.density_heatmap(
                    x=self.df_right["Acromion D_x"],
                    y=-self.df_right["Acromion D_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(acro_right, use_container_width=True)

            if "Cadera" in self.heat_list:
                hip_right = px.density_heatmap(
                    x=self.df_right["Cadera D_x"],
                    y=-self.df_right["Cadera D_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(hip_right, use_container_width=True)

            if "Rodilla" in self.heat_list:
                knee_right = px.density_heatmap(
                    x=self.df_right["Rodilla D_x"],
                    y=-self.df_right["Rodilla D_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(knee_right, use_container_width=True)

            if "Tobillo" in self.heat_list:
                ankle_right = px.density_heatmap(
                    x=self.df_right["Tobillo D_x"],
                    y=-self.df_right["Tobillo D_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(ankle_right, use_container_width=True)

            if "Metatarso" in self.heat_list:
                meta_right = px.density_heatmap(
                    x=self.df_right["Metatarso D_x"],
                    y=-self.df_right["Metatarso D_y"],
                    nbinsx=20,
                    nbinsy=20,
                    marginal_x="histogram",
                    marginal_y="histogram",
                )
                st.plotly_chart(meta_right, use_container_width=True)

            
