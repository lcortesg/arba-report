"""
@file     : data_service.py
@brief   : Handles data fetching.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""


import streamlit as st
import pandas as pd
import json

class Data:
    
    @st.cache(allow_output_mutation=True)
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