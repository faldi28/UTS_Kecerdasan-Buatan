import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage

# Load models and scaler for supervised learning
with open('decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load K-Means and Hierarchical Clustering models for unsupervised learning
kmeans_model = pickle.load(open('k-means_model.pkl', 'rb'))
hc_model = pickle.load(open('hierarchical_clustering_model.pkl', 'rb'))

# Set page configuration
st.set_page_config(
    page_title="AI Learning Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar menu
with st.sidebar:
    st.title("📊 AI Learning Dashboard")
    st.markdown("Pilih jenis pembelajaran:")
    menu_option = st.radio(
        "Menu Utama",
        ["Supervised Learning", "Unsupervised Learning"]
    )
    st.markdown("---")
    st.markdown("Dibuat oleh Yohanes dan Jhonatan")

# Supervised Learning Section
if menu_option == "Supervised Learning":
    st.title("🧠 Supervised Learning: Heart Disease Prediction")
    st.write("Gunakan model prediksi untuk mendeteksi kemungkinan penyakit jantung berdasarkan fitur masukan.")

    # Input fields for supervised learning
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])

    if st.button("Predict Heart Disease"):
        # Prepare input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_data_scaled = scaler.transform(input_data)

        # Predictions
        dt_prediction = decision_tree_model.predict(input_data_scaled)
        rf_prediction = random_forest_model.predict(input_data_scaled)
        lr_prediction = logistic_regression_model.predict(input_data_scaled)

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Decision Tree Prediction:** {'Heart disease detected' if dt_prediction == 1 else 'No heart disease detected'}")
        st.write(f"**Random Forest Prediction:** {'Heart disease detected' if rf_prediction == 1 else 'No heart disease detected'}")
        st.write(f"**Logistic Regression Prediction:** {'Heart disease detected' if lr_prediction == 1 else 'No heart disease detected'}")

# Unsupervised Learning Section
elif menu_option == "Unsupervised Learning":
    st.title("Unsupervised Learning: K-Means & Hierarchical Clustering")
    st.write("Isi form berikut untuk mendapatkan prediksi klaster dari kedua model.")

    # Form untuk input pengguna
    gender = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
    race_ethnicity = st.selectbox("Ras/Etnis", ["kelompok A", "kelompok B", "kelompok C", "kelompok D", "kelompok E"])
    parental_education = st.selectbox(
        "Tingkat Pendidikan Orang Tua",
        ["beberapa sekolah menengah", "sekolah menengah", "beberapa kuliah", "gelar asosiasi", "gelar sarjana", "gelar master"]
    )
    lunch = st.selectbox("Makanan", ["standar", "gratis/berkurang"])
    test_prep = st.selectbox("Kursus Persiapan Ujian", ["tidak ada", "selesai"])
    math_score = st.slider("Nilai Matematika", 0, 100, 50)
    reading_score = st.slider("Nilai Membaca", 0, 100, 50)
    writing_score = st.slider("Nilai Menulis", 0, 100, 50)

    # Encode categorical features
    encoder = LabelEncoder()
    data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race_ethnicity],
        'parental level of education': [parental_education],
        'lunch': [lunch],
        'test preparation course': [test_prep],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })

        # Tambahkan final score (contoh: rata-rata dari tiga nilai)
    data['final score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3

    # Tampilkan nilai final score
    st.write(f"Nilai Rata-Rata (Final Score): **{data['final score'][0]:.2f}**")
    # Encode categorical variables
    data_encoded = data.copy()
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        data_encoded[col] = encoder.fit_transform(data[col])

    # Urutkan kolom sesuai dengan fitur model
    required_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                         'test preparation course', 'math score', 'reading score', 
                         'writing score', 'final score']
    data_encoded = data_encoded[required_features]


    # Visualisasi Klaster
    if st.button("Prediksi Klaster"):
        # Tambahkan data dummy agar AgglomerativeClustering dapat bekerja
        dummy_data = data_encoded.copy()
        dummy_data.loc[1] = [0] * len(required_features)  # Data dummy dengan nilai nol
        combined_data = pd.concat([data_encoded, dummy_data])  # Gabungkan data pengguna dan data dummy

        # K-Means prediction
        kmeans_cluster = kmeans_model.predict(data_encoded)

        # Hierarchical Clustering prediction
        hc_cluster = hc_model.fit_predict(combined_data)  # Gunakan data gabungan untuk HC

        # Ambil klaster pengguna (indeks pertama)
        user_hc_cluster = hc_cluster[0]  # Ambil klaster untuk pengguna saja

        # Tampilkan hasil
        st.write(f"**K-Means Klaster yang Diprediksi:** {kmeans_cluster[0]}")
        st.write(f"**Hierarchical Clustering Klaster yang Diprediksi:** {user_hc_cluster}")

        # Visualisasi: Scatter Plot K-Means dan Hierarchical
        st.write("### Visualisasi Scatter Plot:")
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Visualisasi: K-Means
        cluster_data_kmeans = data_encoded.copy()
        cluster_data_kmeans['Cluster'] = kmeans_cluster[0]
        sns.scatterplot(data=cluster_data_kmeans, x='math score', y='reading score', hue='Cluster', ax=ax[0], palette='viridis', s=100, edgecolor='black')
        ax[0].set_title('K-Means Clustering')

        # Visualisasi: Hierarchical Clustering
        cluster_data_hc = data_encoded.copy()
        cluster_data_hc['Cluster'] = hc_cluster[:len(data_encoded)]  # Only take the first part of hc_cluster for the user
        sns.scatterplot(data=cluster_data_hc, x='math score', y='reading score', hue='Cluster', ax=ax[1], palette='viridis', s=100, edgecolor='black')
        ax[1].set_title('Hierarchical Clustering')

        st.pyplot(fig)
