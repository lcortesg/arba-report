"""
@file     : main.py
@brief   : Main function of streamlit report.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""

import streamlit as st
from PIL import Image

from source.services.Report.report_service import Report
from source.services.Math.math_service import Math
from source.services.Data.data_service import Data
from source.services.Database.database_service import Database
from source.services.Graphics.graphics_service import Graphics
from source.services.AWS.S3 import S3


def main():
    favicon = Image.open("assets/logos/favicon.png")
    st.set_page_config(
        page_title="ABMA Report",
        page_icon=favicon,
        layout="wide",
    )

    exam_id = st.sidebar.text_input("ID Examen", key="exam_id")
    if exam_id == "":
        st.sidebar.warning("Ingresa un ID de examen")

    else:
        data_base = Database(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            port=st.secrets["DB_PORT"],
        )
        
        #db_success, db_info = data_base.fetch_db_info(exam_id=exam_id)
        db_success = True
        db_info = [["","","","","","","","","","2022-03-21","2022-04-01"],["", "Nombre", "Apellido", "", "nombre@apellido.com", "", "", "1889-04-20"],[]]
        if db_success:
            Report(db_info=db_info)


if __name__ == "__main__":
    main()
