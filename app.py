"""Aplikasi tugas akhir: K-Means pada rating dan jumlah penilaian produk sepatu.

Alur utama mengikuti notebook Colab yang diberikan pengguna:
- agregasi berdasarkan product_name
- rata-rata review_rating dan jumlah rating valid
- produk dengan >= 3 rating
- StandardScaler, KMeans(random_state=42, n_init=10)
- uji K=2..8 dengan Silhouette Score
"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Clustering Produk Sepatu | K-Means",
    page_icon="👟",
    layout="wide",
)

st.title("👟 Sistem Pengelompokan Produk Sepatu")
st.caption("Penerapan Algoritma K-Means berdasarkan Rata-rata Rating dan Jumlah Penilaian Konsumen")
st.info(
    "Unggah dataset Amazon Shoes dalam format **.xlsx** atau **.csv**. "
    "Aplikasi akan menjalankan tahapan pengolahan data dan pengujian K-Means "
    "sesuai alur penelitian di Google Colab."
)

KOLUMN_WAJIB = {"product_name", "review_rating", "verified_purchase"}
FITUR_UTAMA = ["rata_rata_rating", "jumlah_penilaian"]


@st.cache_data(show_spinner=False)
def baca_dataset(file_bytes: bytes, nama_file: str) -> pd.DataFrame:
    konten = BytesIO(file_bytes)
    if nama_file.lower().endswith(".xlsx"):
        return pd.read_excel(konten, engine="openpyxl")
    if nama_file.lower().endswith(".csv"):
        try:
            return pd.read_csv(konten, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
    raise ValueError("Format file tidak didukung. Gunakan .xlsx atau .csv.")


@st.cache_data(show_spinner=False)
def siapkan_data(df: pd.DataFrame, minimum_penilaian: int):
    data_produk = df[["product_name", "review_rating", "verified_purchase"]].copy()

    # Ikuti pembersihan dalam notebook Google Colab.
    data_bersih = data_produk.dropna(subset=["product_name"]).copy()
    data_bersih["review_rating"] = pd.to_numeric(
        data_bersih["review_rating"], errors="coerce"
    )
    data_bersih = data_bersih.dropna(subset=["review_rating"]).copy()
    data_bersih = data_bersih[
        data_bersih["review_rating"].between(1, 5)
    ].copy()
    data_bersih["verified_purchase"] = (
        data_bersih["verified_purchase"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0})
    )

    data_cluster = data_bersih.groupby("product_name").agg(
        rata_rata_rating=("review_rating", "mean"),
        jumlah_penilaian=("review_rating", "count"),
        persentase_terverifikasi=("verified_purchase", "mean"),
    ).reset_index()
    data_cluster["persentase_terverifikasi"] *= 100

    data_kmeans = data_cluster[
        data_cluster["jumlah_penilaian"] >= minimum_penilaian
    ].copy()
    return data_produk, data_bersih, data_cluster, data_kmeans


@st.cache_data(show_spinner=False)
def uji_cluster(data_kmeans: pd.DataFrame):
    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(data_kmeans[FITUR_UTAMA])

    percobaan = []
    maks_k = min(8, len(data_kmeans) - 1, len(np.unique(fitur_scaled, axis=0)))
    for k in range(2, maks_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(fitur_scaled)
        # Silhouette Score membutuhkan sekurangnya 2 label dan < n_sampel.
        if not 2 <= len(np.unique(labels)) < len(data_kmeans):
            continue
        percobaan.append(
            {"Jumlah Cluster": k, "Silhouette Score": silhouette_score(fitur_scaled, labels)}
        )
    return pd.DataFrame(percobaan), fitur_scaled


@st.cache_data(show_spinner=False)
def jalankan_model(data_kmeans: pd.DataFrame, k: int):
    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(data_kmeans[FITUR_UTAMA])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(fitur_scaled)
    hasil = data_kmeans.copy()
    hasil["cluster"] = labels

    evaluasi = {
        "Silhouette Score": silhouette_score(fitur_scaled, labels),
        "Davies-Bouldin Index": davies_bouldin_score(fitur_scaled, labels),
        "Calinski-Harabasz Index": calinski_harabasz_score(fitur_scaled, labels),
    }
    ringkasan = hasil.groupby("cluster").agg(
        jumlah_produk=("cluster", "size"),
        rata_rata_rating=("rata_rata_rating", "mean"),
        rata_rata_penilaian=("jumlah_penilaian", "mean"),
    ).round(2)
    ringkasan["persentase_produk"] = (
        ringkasan["jumlah_produk"] / len(hasil) * 100
    ).round(2)
    return hasil, evaluasi, ringkasan


@st.cache_data(show_spinner=False)
def bandingkan_rating_saja(data_kmeans: pd.DataFrame):
    scaled_rating = StandardScaler().fit_transform(data_kmeans[["rata_rata_rating"]])
    if len(np.unique(scaled_rating)) < 2:
        return None
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_rating)
    if len(np.unique(labels)) < 2:
        return None
    perbandingan = data_kmeans.copy()
    perbandingan["cluster_rating"] = labels
    return perbandingan, silhouette_score(scaled_rating, labels)


def excel_hasil(hasil, ringkasan, percobaan, banding):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        hasil.to_excel(writer, sheet_name="Clustering Model Awal", index=False)
        ringkasan.to_excel(writer, sheet_name="Ringkasan Cluster")
        percobaan.to_excel(writer, sheet_name="Pengujian K", index=False)
        if banding is not None:
            banding.to_excel(writer, sheet_name="Clustering Rating", index=False)
    buffer.seek(0)
    return buffer.getvalue()


with st.sidebar:
    st.header("⚙️ Pengaturan")
    minimum_penilaian = st.number_input(
        "Minimal penilaian per produk", min_value=1, max_value=100, value=3, step=1,
        help="Pada penelitian di Colab digunakan minimal 3 penilaian per produk."
    )
    st.caption("Algoritma: K-Means, random_state=42, n_init=10")
    st.caption("Fitur: rata_rata_rating dan jumlah_penilaian")

unggahan = st.file_uploader(
    "Pilih dataset Amazon Shoes", type=["xlsx", "csv"],
    help="Gunakan dataset sumber dari penelitianmu, bukan file Excel hasil clustering."
)

if unggahan is None:
    st.markdown("### 📂 Mulai dari mengunggah dataset")
    st.write(
        "File yang dipakai di Google Colab bernama "
        "`amazon_uk_shoes_products_dataset_2021_12.xlsx`. "
        "Jika kamu masih punya file tersebut, unggah melalui tombol di atas."
    )
    st.stop()

try:
    df = baca_dataset(unggahan.getvalue(), unggahan.name)
except Exception as exc:
    st.error(f"File tidak berhasil dibaca: {exc}")
    st.stop()

kolom_kurang = sorted(KOLUMN_WAJIB - set(df.columns))
if kolom_kurang:
    st.error(
        "Dataset belum memiliki kolom yang diperlukan: "
        + ", ".join(f"`{kol}`" for kol in kolom_kurang)
    )
    st.write("Kolom yang terbaca:", ", ".join(map(str, df.columns)))
    st.stop()

try:
    data_produk, data_bersih, data_cluster, data_kmeans = siapkan_data(
        df, minimum_penilaian
    )
except Exception as exc:
    st.error(f"Terjadi kesalahan saat preprocessing: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Baris data awal", f"{len(df):,}")
m2.metric("Baris data bersih", f"{len(data_bersih):,}")
m3.metric("Produk unik", f"{len(data_cluster):,}")
m4.metric("Produk dianalisis", f"{len(data_kmeans):,}")

if len(data_kmeans) < 3 or len(data_kmeans[FITUR_UTAMA].drop_duplicates()) < 2:
    st.warning(
        "Data yang memenuhi kriteria belum cukup atau tidak cukup bervariasi "
        "untuk pengujian K-Means. Coba file lain atau turunkan batas minimal penilaian."
    )
    st.stop()

with st.spinner("Menguji jumlah cluster K=2 sampai K=8..."):
    percobaan, fitur_scaled = uji_cluster(data_kmeans)
if percobaan.empty:
    st.warning("Tidak ada hasil pengujian cluster yang valid pada dataset ini.")
    st.stop()

k_optimal = int(percobaan.loc[percobaan["Silhouette Score"].idxmax(), "Jumlah Cluster"])
opsi = ["Otomatis (Silhouette terbaik)"] + [
    f"K = {int(k)}" for k in percobaan["Jumlah Cluster"]
]
mode = st.selectbox("Pilih jumlah cluster", opsi)
k = k_optimal if mode.startswith("Otomatis") else int(mode.split("=")[1].strip())
hasil, evaluasi, ringkasan = jalankan_model(data_kmeans, k)

st.success(f"Clustering berhasil! Jumlah cluster yang digunakan: **K = {k}**.")
tab_awal, tab_proses, tab_k, tab_hasil, tab_evaluasi, tab_banding = st.tabs([
    "📄 Data Awal", "🧹 Preprocessing", "📈 Pengujian K", "👟 Hasil Cluster",
    "📊 Evaluasi & Unduh", "🔍 Model Pembanding"
])

with tab_awal:
    st.subheader("Dataset yang diunggah")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Menampilkan 100 baris pertama dari {len(df):,} baris.")
    st.write("**Nama kolom:**", ", ".join(df.columns.astype(str)))

with tab_proses:
    st.subheader("Data setelah pembersihan")
    st.dataframe(data_bersih.head(100), use_container_width=True)
    st.subheader("Agregasi berdasarkan nama produk")
    st.dataframe(data_cluster.head(100), use_container_width=True)
    st.caption(
        "Jumlah penilaian = banyaknya rating valid per nama produk; "
        "rata-rata rating = rata-rata rating valid. "
        "Persentase terverifikasi hanya informasi tambahan, bukan fitur K-Means."
    )
    st.subheader(f"Seleksi produk (minimal {minimum_penilaian} penilaian)")
    st.dataframe(data_kmeans.head(100), use_container_width=True)

with tab_k:
    st.subheader("Pengujian K menggunakan Silhouette Score")
    st.dataframe(percobaan.round(5), hide_index=True, use_container_width=True)
    st.line_chart(percobaan.set_index("Jumlah Cluster")["Silhouette Score"])
    st.write(f"**K terbaik berdasarkan Silhouette Score: {k_optimal}**")
    st.caption(
        "Pengujian mengikuti notebook: nilai K=2 sampai K=8, "
        "KMeans(random_state=42, n_init=10)."
    )

with tab_hasil:
    st.subheader("Visualisasi pengelompokan produk")
    fig, ax = plt.subplots(figsize=(9, 5))
    titik = ax.scatter(
        hasil["jumlah_penilaian"], hasil["rata_rata_rating"],
        c=hasil["cluster"], cmap="viridis", alpha=0.7, s=45
    )
    ax.set_xlabel("Jumlah Penilaian Produk")
    ax.set_ylabel("Rata-rata Rating Produk")
    ax.set_title("K-Means Clustering Produk Sepatu")
    ax.grid(alpha=0.3)
    fig.colorbar(titik, ax=ax, ticks=sorted(hasil["cluster"].unique()), label="Cluster")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Jumlah dan karakteristik setiap cluster")
    st.dataframe(ringkasan, use_container_width=True)
    st.subheader("Tabel hasil clustering")
    st.dataframe(
        hasil[["product_name", "rata_rata_rating", "jumlah_penilaian", "cluster"]],
        use_container_width=True
    )
    st.caption("Nomor cluster hanya penanda kelompok, bukan urutan peringkat kualitas produk.")

with tab_evaluasi:
    st.subheader("Evaluasi model K-Means")
    ea, eb, ec = st.columns(3)
    ea.metric("Silhouette Score", f"{evaluasi['Silhouette Score']:.4f}")
    eb.metric("Davies-Bouldin Index", f"{evaluasi['Davies-Bouldin Index']:.4f}")
    ec.metric("Calinski-Harabasz Index", f"{evaluasi['Calinski-Harabasz Index']:.4f}")
    st.caption("Nilai evaluasi dihitung ulang dari data yang diunggah dan K yang dipilih.")
    st.download_button(
        "⬇️ Unduh hasil clustering (CSV)",
        data=hasil.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_clustering_produk_sepatu.csv",
        mime="text/csv"
    )
    # Model pembanding juga disertakan di ekspor jika data mendukung.
    pembanding = bandingkan_rating_saja(data_kmeans)
    st.download_button(
        "⬇️ Unduh laporan hasil (Excel)",
        data=excel_hasil(hasil, ringkasan, percobaan,
                         pembanding[0] if pembanding else None),
        file_name="hasil_clustering_produk_sepatu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with tab_banding:
    st.subheader("Model pembanding: rata-rata rating saja")
    st.write(
        "Notebook Colab juga menguji model pembanding dengan satu variabel "
        "(rata-rata rating). **Model utama penelitian tetap menggunakan dua variabel**, "
        "yaitu rating dan jumlah penilaian."
    )
    pembanding = bandingkan_rating_saja(data_kmeans)
    if pembanding is None:
        st.warning("Model pembanding tidak dapat dihitung pada dataset ini.")
    else:
        hasil_rating, sil_rating = pembanding
        st.dataframe(pd.DataFrame([
            {"Model": "Rating dan jumlah penilaian", "Silhouette Score": evaluasi["Silhouette Score"]},
            {"Model": "Rating saja (K=2)", "Silhouette Score": sil_rating},
        ]).round(4), hide_index=True, use_container_width=True)
        st.dataframe(
            hasil_rating.groupby("cluster_rating").agg(
                jumlah_produk=("cluster_rating", "size"),
                rata_rata_rating=("rata_rata_rating", "mean"),
                rata_rata_penilaian=("jumlah_penilaian", "mean")
            ).round(2), use_container_width=True
        )
        st.caption(
            "Model pembanding mengikuti langkah pengujian rating saja di notebook. "
            "Nilai yang lebih tinggi pada model rating saja tidak otomatis mengubah "
            "tujuan penelitian yang menggunakan dua variabel."
        )

st.divider()
st.caption("Sumber metode: notebook Google Colab penelitian K-Means produk sepatu.")
