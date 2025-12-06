import streamlit as st
import pandas as pd
import altair as alt
import joblib 
import os 
import numpy as np
import plotly.express as px

# Konfigurasi Halaman (Harus di awal)
st.set_page_config(
    page_title="InSight Padi Dashboard", # Nama page title di browser
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==============================
# KONSTANTA DAN PREPARASI
# ==============================

WEATHER_COLS = [
    'suhu_rata_rata_c', 'curah_hujan_mm', 
    'kelembapan_rata_rata_persen', 'kecepatan_angin_maks_kmh', 
    'radiasi_matahari_mj'
]

# Tambahkan Kordinat Statis untuk mapping lokasi titik pada peta
PROVINCE_COORDS = {
    'Aceh': [4.6951, 96.7495], 'Sumatera Utara': [2.1931, 99.1707], 'Sumatera Barat': [-0.8354, 100.4071],
    'Riau': [0.5105, 101.6256], 'Jambi': [-1.6669, 102.666], 'Sumatera Selatan': [-3.3194, 104.1039],
    'Bengkulu': [-3.8003, 102.2472], 'Lampung': [-4.869, 105.087], 'Bangka Belitung': [-2.3242, 106.4168],
    'Kepulauan Riau': [0.3809, 104.4552], 'DKI Jakarta': [-6.2088, 106.8456], 'Jawa Barat': [-6.9213, 107.6045],
    'Jawa Tengah': [-7.1594, 110.1403], 'DI Yogyakarta': [-7.8016, 110.3642], 'Jawa Timur': [-7.2608, 112.7262],
    'Banten': [-6.4058, 106.0640], 'Bali': [-8.3762, 115.1524], 'Nusa Tenggara Barat': [-8.6528, 117.6177],
    'Nusa Tenggara Timur': [-8.5835, 120.5707], 'Kalimantan Barat': [0.2789, 111.475], 'Kalimantan Tengah': [-1.6738, 113.8443],
    'Kalimantan Selatan': [-3.0016, 115.2875], 'Kalimantan Timur': [0.5056, 116.4194], 'Kalimantan Utara': [2.8105, 116.2084],
    'Sulawesi Utara': [1.2592, 124.7891], 'Sulawesi Tengah': [-1.4307, 121.4456], 'Sulawesi Selatan': [-4.0536, 119.9702],
    'Sulawesi Tenggara': [-4.1466, 122.1022], 'Gorontalo': [0.7186, 122.4678], 'Sulawesi Barat': [-2.5695, 119.3444],
    'Maluku': [-3.2384, 129.567], 'Maluku Utara': [0.9392, 127.6749], 'Papua': [-4.2699, 138.6834],
    'Papua Barat': [-1.3283, 133.578], 'Papua Selatan': [-7.8016, 138.5], 'Papua Tengah': [-3.8016, 135.5],
    'Papua Pegunungan': [-4.2016, 139.0]
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "panen_predictor_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "etl_panen_cuaca_2024.csv")


# ==============================
# LOAD DATA & MODEL (CACHING)
# ==============================

@st.cache_data
def load_data():
    """Memuat DataFrame yang sudah diproses dari ETL."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan: {DATA_PATH}. Pastikan Anda sudah menjalankan etl.py")
        return pd.DataFrame() 

@st.cache_resource
def load_model():
    """Memuat model prediksi Random Forest yang sudah dilatih."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model prediksi tidak ditemukan: {MODEL_PATH}. Pastikan Anda sudah menjalankan etl.py dan menyimpan model.")
        return None
        
df = load_data()
model = load_model()

if df.empty:
    st.stop()
    
# ==============================
# UTILITY PREDIKSI
# ==============================

def create_prediction_input(df_all, selected_province, selected_month):
    """
    Membuat input fitur yang benar untuk prediksi, termasuk lagging dan one-hot encoding.
    """
    
    df_prov = df_all[df_all["province"] == selected_province].copy()
    target_month_index = df_prov[df_prov["month"] == selected_month].index
    
    if target_month_index.empty:
        return None, None
    
    current_idx = target_month_index[0]

    all_weather_lags = []
    for col in WEATHER_COLS:
        for lag in range(1, 4): 
            all_weather_lags.append(f'{col}_lag_{lag}')
            
    pred_input = df_prov.loc[[current_idx], all_weather_lags]
    
    if pred_input.isna().any(axis=1).values[0]:
        return None, None
        
    actual_production = df_prov.loc[current_idx, 'produksi']
        
    all_provinces = sorted(df_all["province"].unique())
    ohe_cols = [f'province_{p}' for p in all_provinces]
    
    ohe_input = pd.DataFrame(0.0, index=[0], columns=ohe_cols)
    ohe_input.loc[0, f'province_{selected_province}'] = 1.0
    
    final_input_data = pd.concat([
        pred_input.reset_index(drop=True), 
        ohe_input.reset_index(drop=True)
    ], axis=1)

    X_train_cols = all_weather_lags + ohe_cols
    
    final_input = final_input_data.reindex(columns=X_train_cols, fill_value=0.0)
    
    return final_input, actual_production


# FUNGSI BARU: PREDIKSI SEMUA PROVINSI
def predict_all_provinces(df_all, model, selected_month):
    """Melakukan prediksi untuk semua provinsi pada bulan yang dipilih."""
    results = []
    all_provinces = sorted(df_all["province"].unique())

    for p in all_provinces:
        # Gunakan fungsi yang sudah ada untuk mendapatkan input per provinsi
        final_input, actual = create_prediction_input(df_all, p, selected_month)
        
        if final_input is not None and model is not None:
            try:
                pred = model.predict(final_input)[0]
                results.append({
                    'province': p,
                    'month': selected_month,
                    'produksi_aktual': actual,
                    'produksi_prediksi': pred,
                    'error': pred - actual
                })
            except Exception:
                # Lewati jika ada error prediksi (misal: data lag tidak lengkap)
                continue
    
    return pd.DataFrame(results)


# ==============================
# STREAMLIT UI START
# ==============================

# JUDUL UTAMA
st.title("Integrasi Data Cuaca dan Pertanian untuk Prediksi Panen Nasional") 
st.subheader("Visualisasi Data Historis dan Prediksi Produksi Padi")

# --- Seleksi Provinsi dan Bulan (Pindah ke Sidebar) ---
provinsi_list = ["Semua Provinsi"] + sorted(df["province"].unique())
bulan_list_with_all = ["Semua Bulan (Tren)"] + sorted(df["month"].unique())

st.sidebar.title("⚙️ Pengaturan Filter")
provinsi = st.sidebar.selectbox("Pilih Provinsi", provinsi_list)
bulan = st.sidebar.selectbox("Pilih Bulan", bulan_list_with_all)

# Info mode di sidebar
if provinsi == "Semua Provinsi" and bulan != "Semua Bulan (Tren)":
    st.sidebar.info(f"Mode: Nasional. Prediksi untuk semua provinsi di **{bulan}**.")
elif bulan == "Semua Bulan (Tren)":
    st.sidebar.warning("Mode: Tren. Prediksi dinonaktifkan.")
else:
    st.sidebar.success(f"Mode: Detil. Analisis **{provinsi}** di **{bulan}**.")

st.markdown("---")


# Logika Filter Data Utama
is_single_province = provinsi != "Semua Provinsi"
is_single_month = bulan != "Semua Bulan (Tren)"

if is_single_province and is_single_month:
    # Case 1: Single Province, Single Month (Standard for Prediction/Detail)
    df_filtered = df[(df["province"] == provinsi) & (df["month"] == bulan)]
elif is_single_province and not is_single_month:
    # Case 2: Single Province, All Months (Used for Trend Chart)
    df_filtered = df[df["province"] == provinsi]
elif not is_single_province and is_single_month:
    # Case 3: All Provinces, Single Month (Used for Map/National Ranking/All Prediction)
    df_filtered = df[df["month"] == bulan]
else:
    # Case 4: All Provinces, All Months (Too big)
    df_filtered = pd.DataFrame() 


# ==============================
# BAGIAN PETA INDONESIA (SCATTER GEO)
# ==============================

st.header("🌍 Peta Sebaran Produksi Padi (Titik)")

# 1. Tentukan data peta berdasarkan seleksi
if not is_single_month:
    st.warning("Peta titik hanya dapat divisualisasikan untuk satu bulan tertentu. Menampilkan bulan terakhir.")
    # Gunakan data bulan terakhir yang tersedia untuk konteks visual jika mode tren dipilih
    latest_month = sorted(df['month'].unique())[-1]
    df_map_data = df[df['month'] == latest_month].copy()
    map_title = f"Produksi Padi (ton) per Provinsi Bulan Terakhir ({latest_month})"
elif is_single_province:
    # Sesuai permintaan: Hanya tampilkan provinsi yang dipilih di peta
    df_map_data = df[(df["month"] == bulan) & (df["province"] == provinsi)].copy()
    map_title = f"Produksi Padi (ton) di {provinsi} Bulan {bulan}"
else:
    # Tampilkan semua provinsi untuk bulan yang dipilih (Mode Nasional)
    df_map_data = df[df["month"] == bulan].copy()
    map_title = f"Produksi Padi (ton) per Provinsi Bulan {bulan} (Visualisasi Titik)"


# 2. Tambahkan koordinat ke data peta
df_map_data['latitude'] = df_map_data['province'].apply(lambda x: PROVINCE_COORDS.get(x, [None, None])[0])
df_map_data['longitude'] = df_map_data['province'].apply(lambda x: PROVINCE_COORDS.get(x, [None, None])[1])
df_map_data.dropna(subset=['latitude', 'longitude'], inplace=True)


if df_map_data.empty:
    st.info("Tidak ada data yang cukup untuk ditampilkan pada peta berdasarkan pilihan Anda.")
else:
    try:
        fig_map = px.scatter_geo(
            df_map_data, 
            lat='latitude', 
            lon='longitude', 
            hover_name="province", 
            size="produksi", 
            color="produksi",
            projection="mercator",
            scope='asia',
            color_continuous_scale="Darkmint", 
            title=map_title,
            height=500,
            template="plotly_dark" 
        )
        
        # Atur batas geografis ke Indonesia/Asia Tenggara
        fig_map.update_geos(
            center={"lat": -2.0, "lon": 118.0}, 
            lataxis_range=[-10, 6], 
            lonaxis_range=[95, 141],
            showcountries=True,
            countrycolor="Gray"
        )

        # Kustomisasi colorbar
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                bgcolor='black', 
                bordercolor='gray',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membuat peta titik: {e}")
        st.warning("Peta tidak dapat ditampilkan karena masalah konfigurasi geografis.")


st.markdown("---")

# ==============================
# (1) PREDIKSI PRODUKSI PANEN
# Logika: Hanya tampil jika Single Month
# ==============================

if model is None:
    st.error("Model prediksi tidak dimuat. Prediksi tidak dapat dilakukan.")
elif not is_single_month:
    st.header("🤖 Prediksi Produksi Padi")
    st.info("Prediksi hanya dapat dilakukan untuk satu bulan tertentu.")
elif is_single_province:
    # --- Case A: Single Province, Single Month (UI Improvement: Use Container) ---
    st.header(f"🤖 Prediksi Produksi Padi ({provinsi} - {bulan})")

    if not df_filtered.empty:
        
        pred_input_df, actual_production = create_prediction_input(df, provinsi, bulan)
        
        if pred_input_df is not None:
            
            try:
                pred = model.predict(pred_input_df)[0]
            except ValueError as e:
                st.error(f"Error saat prediksi: {e}")
                st.stop()
                
            delta_val = pred - actual_production
            
            # Group metrics inside a bordered container
            with st.container(border=True):
                st.markdown(f"**Ringkasan Prediksi {provinsi} pada {bulan}**")
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    st.metric(
                        label=f"Produksi Aktual ({bulan})",
                        value=f"{actual_production:,.0f} ton", 
                        help="Data produksi yang tercatat dari BPS (data target)."
                    )
                
                with pred_col2:
                    st.metric(
                        label=f"Prediksi Model ({bulan})",
                        value=f"{pred:,.0f} ton", 
                        help="Prediksi model Random Forest menggunakan cuaca 3 bulan sebelumnya."
                    )
                    
                with pred_col3:
                    st.metric(
                        label="Perbedaan (Delta)",
                        value=f"{delta_val:,.0f} ton",
                        delta=f"{delta_val:,.0f} ton",
                        delta_color="off", 
                        help="Selisih antara Prediksi dan Aktual."
                    )
        else:
            st.warning(f"Data *lagging* cuaca untuk {bulan} tidak tersedia.")
            
else: 
    # --- Case B: All Provinces, Single Month (Fixed Visualization Logic: Grouped Bars) ---
    st.header(f"🤖 Hasil Prediksi Nasional Bulan {bulan}")
    
    df_prediction_results = predict_all_provinces(df, model, bulan)
    
    if not df_prediction_results.empty:
        st.subheader(f"Perbandingan Aktual vs Prediksi Produksi Padi Bulan {bulan}")
        
        # Prepare data for Altair chart (melt for easy comparison)
        df_chart = df_prediction_results.melt(
            id_vars='province', 
            value_vars=['produksi_aktual', 'produksi_prediksi'],
            var_name='Tipe Data',
            value_name='Produksi (ton)'
        ).sort_values(by='Produksi (ton)', ascending=False)

        
        # Visualisasi Grouped Bar Chart (Consistent Display)
        chart = alt.Chart(df_chart).mark_bar().encode(
            # X-axis (Value)
            x=alt.X('Produksi (ton)', axis=alt.Axis(format=',.0f'), title="Produksi (ton)"), 
            # Y-axis (Category)
            y=alt.Y('province', title='Provinsi', sort='-x'), 
            
            # Grouping Field: PENTING UNTUK DIPISAH KOLOM
            column=alt.Column('Tipe Data', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
            
            color=alt.Color('Tipe Data', scale=alt.Scale(range=['#5a8c54', '#4c78a8'])),
            tooltip=['province', alt.Tooltip('Produksi (ton)', format=',.0f'), 'Tipe Data']
        ).properties(
            title=f"Perbandingan Produksi Padi Aktual vs Prediksi ({bulan})"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        # Display aggregated results
        total_aktual = df_prediction_results['produksi_aktual'].sum()
        total_prediksi = df_prediction_results['produksi_prediksi'].sum()
        
        col_total1, col_total2, _ = st.columns(3)
        with col_total1:
            st.metric("Total Produksi Aktual Nasional", f"{total_aktual:,.0f} ton")
        with col_total2:
            st.metric("Total Prediksi Nasional", f"{total_prediksi:,.0f} ton", 
                      delta=f"{total_prediksi - total_aktual:,.0f} ton")
            
        # Optionally display a detailed table
        with st.expander("Lihat Data Prediksi Detail Per Provinsi"):
            st.dataframe(
                df_prediction_results[['province', 'produksi_aktual', 'produksi_prediksi', 'error']]
                .rename(columns={'produksi_aktual': 'Aktual', 'produksi_prediksi': 'Prediksi', 'error': 'Selisih (P-A)'})
                .set_index('province')
                .sort_values(by='Aktual', ascending=False)
                .style.format('{:,.0f}')
            )
            
    else:
        st.error(f"Gagal melakukan prediksi untuk bulan {bulan} di semua provinsi. Cek ketersediaan data lagging (cuaca 3 bulan sebelumnya).")
        
st.markdown("---")

# ==============================
# (2) VISUALISASI DATA HISTORIS
# ==============================

st.header("📈 Visualisasi Data Historis")

# --- KASUS A: PROVINSI TUNGGAL + SEMUA BULAN (TREN) ---
if is_single_province and not is_single_month:
    st.subheader(f"⏳ Tren Produksi Padi Tahunan di {provinsi}")
    
    df_trend = df_filtered.sort_values(by='month')
    
    if not df_trend.empty:
        trend_chart = alt.Chart(df_trend).mark_line(point=True, color="#5a8c54").encode(
            x=alt.X('month', title="Bulan (2024)"), 
            y=alt.Y('produksi', title="Produksi (ton)"),
            tooltip=[alt.Tooltip('month', title='Bulan'), alt.Tooltip('produksi', title='Produksi', format=',.0f')]
        ).properties(
            title=f"Tren Produksi Padi di {provinsi} Sepanjang 2024"
        ).interactive() 
        
        st.altair_chart(trend_chart, use_container_width=True)
    else:
        st.info(f"Tidak ada data tren yang tersedia untuk {provinsi}.")

# --- KASUS B: SEMUA PROVINSI + BULAN TUNGGAL (RANKING AKTUAL) ---
elif not is_single_province and is_single_month:
    st.subheader(f"🥇 Peringkat Produksi Padi Nasional Bulan {bulan} (Data Aktual)")
    
    df_ranking = df_filtered.sort_values(by='produksi', ascending=False).head(10)

    if not df_ranking.empty:
        rank_chart = alt.Chart(df_ranking).mark_bar().encode(
            x=alt.X('produksi', title="Produksi (ton)", sort="-y"), 
            y=alt.Y('province', title="Provinsi", sort="-x"),
            tooltip=[alt.Tooltip('province'), alt.Tooltip('produksi', format=',.0f')],
            color=alt.value("#5a8c54")
        ).properties(
            title=f"10 Provinsi Produsen Padi Terbesar Bulan {bulan}"
        ).interactive() 
        
        st.altair_chart(rank_chart, use_container_width=True)
    else:
        st.info(f"Tidak ada data untuk peringkat bulan {bulan}.")

# --- KASUS C: PROVINSI TUNGGAL + BULAN TUNGGAL (DETAIL METRIK - UI Improvement: Tabs) ---
elif is_single_province and is_single_month:
    
    tab1, tab2 = st.tabs(["🌤 Data Cuaca Bulan Ini", "🌾 Data Panen Bulan Ini"])
    
    if not df_filtered.empty:
        data_row = df_filtered.iloc[0] 
        
        # --- GRAFIK CUACA (di Tab 1) ---
        with tab1:
            st.subheader(f"Parameter Cuaca untuk {provinsi} ({bulan})")
            cuaca_chart = pd.DataFrame({
                "Parameter": [
                    "Suhu Rata-rata (°C)", "Suhu Maks (°C)", "Suhu Min (°C)", 
                    "Kelembapan (%)", "Curah Hujan (mm)", 
                    "Kecepatan Angin (km/h)", "Radiasi (MJ)"
                ],
                "Nilai": [
                    data_row["suhu_rata_rata_c"], data_row["suhu_maksimum_c"], data_row["suhu_minimum_c"],
                    data_row["kelembapan_rata_rata_persen"], data_row["curah_hujan_mm"],
                    data_row["kecepatan_angin_maks_kmh"], data_row["radiasi_matahari_mj"]
                ]
            })

            chart = alt.Chart(cuaca_chart).mark_bar().encode(
                x=alt.X("Parameter", sort=None, axis=None), 
                y=alt.Y("Nilai", title="Nilai"),
                tooltip=["Parameter", alt.Tooltip("Nilai", format=",.1f")],
                color=alt.condition(
                    alt.datum.Parameter == "Curah Hujan (mm)", 
                    alt.value("orange"),  
                    alt.value("#4c78a8") 
                )
            ).properties(
                title=f"Variabel Cuaca Kunci"
            )
            st.altair_chart(chart, use_container_width=True)

        # --- GRAFIK PANEN (di Tab 2) ---
        with tab2:
            st.subheader("Luas Panen, Produksi, dan Produktivitas")
            panen_chart = pd.DataFrame({
                "Kategori": ["Luas Panen (ha)", "Produksi (ton)", "Produktivitas (ku/ha)"],
                "Nilai": [data_row["luas_panen"], data_row["produksi"], data_row["produktivitas"]]
            })

            chart2 = alt.Chart(panen_chart).mark_bar(color="#5a8c54").encode(
                x=alt.X("Kategori", sort=None), 
                y=alt.Y("Nilai", title="Nilai"), 
                tooltip=["Kategori", alt.Tooltip("Nilai", format=",.0f")] 
            ).properties(title="Metrik Hasil Panen")

            st.altair_chart(chart2, use_container_width=True)

# --- KASUS D: SEMUA PROVINSI + SEMUA BULAN ---
else:
    st.info("Pilih kombinasi Provinsi/Bulan yang lebih spesifik di **Sidebar** untuk melihat visualisasi.")


# --- BAGIAN DATA MENTAH (DISEMBUNYIKAN) ---
st.markdown("---")
with st.expander("Lihat Data Mentah Terpilih (CSV)"):
    if not df_filtered.empty:
        st.dataframe(df_filtered)
    else:
        st.info("Tidak ada data yang tersedia untuk seleksi ini.")

st.caption("Data sumber: Open-Meteo & BPS")