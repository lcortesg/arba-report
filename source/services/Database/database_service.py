"""
@file     : database_service.py
@brief   : Handles database connection.
@date    : 2022/04/21
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bugs    : None.
"""


import streamlit as st
import mysql.connector
from mysql.connector import connect, Error
import pymysql

class Database:

    def __init__(
        self,
        host="",
        database="",
        user="",
        password="",
        port=0
    ):
    
        self.host=host,
        self.database=database,
        self.user=user,
        self.password=password
        self.port=port

    def connect_db(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            if connection.is_connected():
                db_Info = connection.get_server_info()
                st.sidebar.success(f"Connected to MySQL Server version {db_Info}")
                cursor = connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                st.sidebar.success(f"You're connected to database: {record}")
                cursor.execute("select * from Exams")
                return cursor.fetchall()

        except Error as e:
            st.sidebar.error(f"Error while connecting to MySQL {e}")

        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                st.sidebar.error("MySQL connection is closed")


    def connect_to_db(self, db):
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.password,
            db=self.database,
        )

        cur = conn.cursor()
        cur.execute(f"select * from {db}")
        return cur.fetchall()


    def fetch_db_info(self, exam_id):
        exam_info = []
        patient_info = []
        professional_info = []
        patient_id = ""
        professional_id = ""

        exam_success = False
        patient_success = False
        professional_success = False

        db_exams = self.connect_to_db(db="Exams")
        for exam in db_exams:
            if exam_id == exam[0]:
                exam_info = exam
                exam_success = True
                st.sidebar.success("ID de examen válido")
                status = exam[2]
                patient_id = exam[7]
                professional_id = exam[8]
                if status != "Procesado":
                    st.sidebar.warning(f"Status: {status}")
                break

        if not exam_success:
            st.sidebar.error("ID de examen inválido")
            return False

        if exam_success:
            db_patients = self.connect_to_db(db="Patients")
            for patient in db_patients:
                if patient_id == patient[0]:
                    patient_info = patient
                    patient_success = True
                    st.sidebar.success("ID de paciente válido")
                    break
        if not patient_success:
            st.sidebar.error("ID de paciente inválido")
            return False

        if exam_success and patient_success:
            db_professionals = self.connect_to_db(db="Professionals")
            for professional in db_professionals:
                if professional_id == professional[0]:
                    professional_info = professional
                    professional_success = True
                    st.sidebar.success("ID de profesional válido")
                    break

        if not professional_success:
            st.sidebar.error("ID de professional inválido")
            # return False

        db_success = exam_success and patient_success
        db_info = [exam_info, patient_info, professional_info]
        return db_success, db_info