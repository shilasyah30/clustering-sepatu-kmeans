import streamlit as st

st.set_page_config(
    page_title="Clustering Produk Sepatu",
    page_icon="👟",
    layout="wide"
)

st.title("👟 Sistem Pengelompokan Produk Sepatu")

st.subheader("Penerapan Algoritma K-Means")

st.write(
    "Aplikasi ini digunakan untuk mengelompokkan "
    "produk sepatu berdasarkan rata-rata rating "
    "dan jumlah penilaian konsumen."
)

st.success("Aplikasi Streamlit berhasil dijalankan!")
