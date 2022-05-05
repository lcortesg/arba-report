"""
@file     : math_service.py
@brief   : Process data and mathematical operations.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""


import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy import diff
from datetime import date

class Math:

    @st.cache
    def trunc(values, decs=0):
        return np.trunc(values*10**decs)/(10**decs)

    @st.cache
    def calculateAge_old(birthDate):
        today = date.today()
        age = (
            today.year
            - birthDate.year
            - ((today.month, today.day) < (birthDate.month, birthDate.day))
            )
        return age


    @st.cache
    def calculateAge(born):
        today = date.today()
        try:
            birthday = born.replace(year=today.year)

        # raised when birth date is February 29
        # and the current year is not a leap year
        except ValueError:
            birthday = born.replace(year=today.year, month=born.month + 1, day=1)

        if birthday > today:
            return today.year - born.year - 1
        else:
            return today.year - born.year

    
    @st.cache
    def butter_lowpass_filter(data, fs=120, cutoff=6, order=8):
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq  # Normalise frequency
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)  # Filter data
        return y

    @st.cache
    def get_pos(df, list, min_range, max_range, fps):
        data = {}
        for descriptor in list:
            if descriptor == "Rodilla LI" or descriptor == "Rodilla LD":
                data[descriptor] = Math.butter_lowpass_filter(
                    -df[descriptor][min_range:max_range],
                    fs=fps,
                )
            else:
                data[descriptor] = Math.butter_lowpass_filter(
                    df[descriptor][min_range:max_range],
                    fs=fps,
                )
        return pd.DataFrame(data)


    @st.cache
    def get_pos_simp(df, descriptor, min_range, max_range, fps):
        data = {}
        if descriptor == "Rodilla LI" or descriptor == "Rodilla LD":
            data[descriptor] = Math.butter_lowpass_filter(
                -df[descriptor][min_range:max_range], fs=fps
            )
        else:
            data[descriptor] = Math.butter_lowpass_filter(
                df[descriptor][min_range:max_range], fs=fps
            )
        return pd.DataFrame(data)


    @st.cache
    def get_pos_cycle(df, descriptor, cycle, cc, fps):
        data = {}
        for i in range(cycle):
            if descriptor == "Rodilla LI" or descriptor == "Rodilla LD":
                data[f"{descriptor} ciclo {i+1}"] = Math.butter_lowpass_filter(
                    -df[descriptor][cc[i] : cc[i + 1]], fs=fps
                )
            else:
                data[f"{descriptor} ciclo {i+1}"] = Math.butter_lowpass_filter(
                    df[descriptor][cc[i] : cc[i + 1]], fs=fps
                )
        return pd.DataFrame.from_dict(data, orient="index").transpose()


    @st.cache
    def get_gradient(df, fs=120):
        data = {}
        for descriptor in df:
            data_df = df[descriptor].tolist()
            data_df = [x for x in data_df if np.isnan(x) == False]
            data_delta = np.gradient(data_df, 1 / fs)
            data[descriptor] = Math.butter_lowpass_filter(data_delta)
        return pd.DataFrame.from_dict(data, orient="index").transpose()


    @st.cache
    def get_gradient_simp(df, fs=120):
        data_delta = np.gradient(df, 1 / fs)
        data = Math.butter_lowpass_filter(data_delta)
        return data


    @st.cache
    def get_candidates(data_left, data_right):
        candidates = {}
        # lsc = data_left["Metatarso I"]["x_max"]
        # rsc = data_right["Metatarso D"]["x_min"]

        left_ankle_xmin = data_left["Tobillo I"]["x_min"]
        left_ankle_xmax = data_left["Tobillo I"]["x_max"]
        left_ankle_ymax = data_left["Tobillo I"]["y_max"]

        left_meta_xmax = data_left["Metatarso I"]["x_max"]
        left_meta_ymax = data_left["Metatarso I"]["y_max"]

        right_ankle_xmax = data_right["Tobillo D"]["x_max"]
        right_ankle_ymax = data_right["Tobillo D"]["y_max"]

        right_meta_xmin = data_right["Metatarso D"]["x_min"]
        right_meta_ymax = data_right["Metatarso D"]["y_max"]

        # Contacto

        lcc = []
        for value in left_ankle_xmin:
            lcc.append([x for x in left_ankle_ymax if x > value][0])

        rcc = []
        for value in right_ankle_xmax:
            rcc.append([x for x in right_ankle_ymax if x > value][0])

        # Apoyo medio y tiempo de paso

        ltp = []
        lmc = []
        for i in range(len(lcc) - 1):
            ltp.append(int((lcc[i + 1] - lcc[i]) / 2))
            lmc.append(lcc[i] + int((lcc[i + 1] - lcc[i]) / 2))

        rtp = []
        rmc = []
        for i in range(len(rcc) - 1):
            rtp.append(int((rcc[i + 1] - rcc[i]) / 2))
            rmc.append(rcc[i] + int((rcc[i + 1] - rcc[i]) / 2))

        # Separación
        lsc0 = []
        # for value in left_meta_xmax:
        #    lsc.append([x for x in left_meta_ymax if x > value][0])
        for value in left_meta_xmax:
            for i in range(len(left_meta_ymax)):
                if left_meta_ymax[i] > value:
                    if value != left_meta_xmax[0]:
                        lsc0.append(left_meta_ymax[i - 1])
                    break

        lsc = []
        for i in range(len(lsc0)):
            lsc.append(lsc0[i] + int((lmc[i] - lsc0[i]) / 2))

        rsc0 = []
        # for value in right_meta_xmin:
        #    rsc.append([x for x in right_meta_ymax if x > value][0])
        for value in right_meta_xmin:
            for i in range(len(right_meta_ymax)):
                if right_meta_ymax[i] > value:
                    if value != right_meta_xmin[0]:
                        rsc0.append(right_meta_ymax[i - 1])
                    break

        rsc = []
        for i in range(len(rsc0)):
            rsc.append(rsc0[i] + int((rmc[i] - rsc0[i]) / 2))

        # Tiempo de apoyo
        # Tiempo de balanceo
        # Velocidad de marcha
        # Velocidad de zancada

        # Contrucción de diccionario
        # Izquierda
        candidates["lcc"] = lcc
        candidates["lmc"] = lmc
        candidates["lsc"] = lsc
        candidates["ltp"] = ltp
        # Derecha
        candidates["rcc"] = rcc
        candidates["rmc"] = rmc
        candidates["rsc"] = rsc
        candidates["rtp"] = rtp

        return candidates