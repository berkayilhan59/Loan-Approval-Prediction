import pandas as pd
import streamlit as st
from PIL import Image
import pickle

@st.cache
def get_data():
    dataframe = pd.read_csv("loan_approval_dataset-2.csv")
    return dataframe

st.set_page_config(layout= "wide", page_title="Data Hunters", page_icon="")

st.title(" :rainbow[Data Hunters]")

main_tab, loan_tab = st.tabs(["Ana Sayfa", "Kredi Onay Sistemi"])

left_col, right_col = main_tab.columns(2)

# Ana Sayfa

left_col.write("VERİ SETİ HİKAYESİ")
left_col.write("Banka geçmişte kredi verilen müşterilerle ilgli bir veri seti toplamıştı.\n"
               "Verileri kullanarak gelecekte kredi verebilecek potansiyele müşterilerin belirlenmesi,\n"
               "izlenmesi ve yönetim süreci ele alınır. Bu süreci otomatikleştirmek için kredi onayını\n"
               "alacak olan müşteriler belirlenir.")

left_col.write("PROBLEM TANIMI")
left_col.write("Bankaya kredi başvurusunda bulunan müşterilerin, kredi onayı"
               "alabilmeleri için belirli kriterleri karşılamaları gerekmektedir."
               "Bu kriterler arasında yıllık gelir, bağımlı kişi sayısı, eğitim durumu,"
               "kendi işinde çalışma durumu,CIBIL skoru (kredi puanı), kredi miktarı, kredi süresi ve"
               "kredi süresi ve sahip olunan varlık değerleri gibi faktörler bulunmaktadır."
               "Müşterilerin bu kriterlere uygun olmaması durumunda, kredi başvuruları"
               "reddedilmektedir. Ancak, müşteriler genellikle hangi kriterlerin yetersiz olduğunu"
               "anlayamamakta ve sürecin şeffaflığı konusunda endişeler yaşamaktadırlar."
               "Bu durum, müşteri memnuniyetini düşürmekte ve bankanın itibarını olumsuz"
               "yönde etkileyebilmektedir.Bu nedenle, kredi onay sürecinin daha anlaşılır"
               "ve erişebilir hale getirilmesi gerekmektedir.")

right_col.image("Toprak-2.png")

# Modeli yükle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    print(type(model))

def run():
    # Input fields
    loan_id = st.text_input("Loan ID")
    no_of_dependents = st.number_input("Number of Dependents", value=0)
    education = st.selectbox("Education", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income ($)", value=0)
    loan_amount = st.number_input("Loan Amount ($)", value=0)
    loan_term = st.number_input("Loan Term (months)", value=0)
    cibil_score = st.number_input("CIBIL Score", value=0)
    residential_assets_value = st.number_input("Residential Assets Value ($)", value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value ($)", value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value ($)", value=0)
    bank_asset_value = st.number_input("Bank Asset Value ($)", value=0)

    # Prediction button
    if st.button("Predict Loan Approval"):
        # Format the input features
        features = [
            no_of_dependents,
            1 if education == "Graduate" else 0,
            1 if self_employed == "Yes" else 0,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]
        # Make prediction
        prediction = model.predict([features])[0]

        # Display prediction result
        if prediction == 0:
            st.error(f"Loan ID: {loan_id} - Loan not approved.")
        else:
            st.success(f"Loan ID: {loan_id} - Loan approved.")
run()
