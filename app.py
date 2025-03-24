import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

st.set_page_config(page_title="Phân tích Xu hướng Mạng Xã Hội", layout="wide")
st.title("Ứng dụng Phân tích Dữ liệu Mạng Xã Hội")

# Hàm hiển thị biểu đồ Hashtag phổ biến
def show_popular_hashtags():
    st.header("Biểu đồ các Hashtag phổ biến")
    try:
        df = pd.read_csv("data/cleaned_data.csv")
        hashtag_counts = df['hashtags'].explode().value_counts().head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=hashtag_counts.values, y=hashtag_counts.index, ax=ax, palette="viridis")
        ax.set_xlabel("Tần suất sử dụng")
        ax.set_ylabel("Hashtag")
        ax.set_title("Top 20 Hashtag phổ biến nhất")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")

# Hàm phân cụm chủ đề
def cluster_topics():
    st.header("Phân cụm chủ đề từ file CSV")
    uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu văn bản", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Chọn cột chứa văn bản để phân tích", data.columns)
        num_clusters = st.slider("Số cụm (K)", 2, 10, 3)

        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(data[text_col].fillna(""))

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_tfidf)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_tfidf.toarray())

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis')
        ax.set_title("Kết quả phân cụm chủ đề")
        ax.set_xlabel("Thành phần chính 1")
        ax.set_ylabel("Thành phần chính 2")
        st.pyplot(fig)

        data['cluster'] = clusters
        st.write("Dữ liệu sau khi phân cụm:")
        st.dataframe(data[[text_col, 'cluster']])

# Hàm dự đoán xu hướng
def predict_trend():
    st.header("Dự đoán xu hướng từ hashtag hoặc nội dung")
    model = joblib.load("models/random_forest_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    user_input = st.text_area("Nhập hashtag hoặc nội dung", "")

    if st.button("Dự đoán"):
        if user_input.strip():
            input_vec = vectorizer.transform([user_input])
            pred = model.predict(input_vec)[0]
            proba = model.predict_proba(input_vec)[0][1]

            if pred == 1:
                st.success(f"Có dấu hiệu là xu hướng (xác suất: {proba:.2%})")
            else:
                st.info(f"Không có dấu hiệu là xu hướng (xác suất: {proba:.2%})")
        else:
            st.warning("Vui lòng nhập nội dung.")

# Hàm đánh giá mô hình
def evaluate_model():
    st.header("Đánh Giá Hiệu Suất Mô Hình")
    df = pd.read_csv('./data/cleaned_data.csv')
    model = joblib.load("models/random_forest_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    X = vectorizer.transform(df['text'].fillna(""))
    y = df['is_trending']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.subheader("Các Chỉ Số Đánh Giá")
    col1, col2, col3 = st.columns(3)
    col1.metric("F1 Score", f"{f1:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Không Trending', 'Trending'], yticklabels=['Không Trending', 'Trending'])
    plt.title('Ma Trận Nhầm Lẫn')
    plt.xlabel('Nhãn Dự Đoán')
    plt.ylabel('Nhãn Thực Tế')
    st.pyplot(plt.gcf())
    st.subheader("Báo Cáo Chi Tiết")
    report = classification_report(y_test, y_pred, target_names=['Không Trending', 'Trending'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

st.sidebar.header("Chức Năng")
page = st.sidebar.radio("Chọn chức năng", ["Biểu đồ Hashtag phổ biến", "Phân cụm chủ đề từ CSV", "Dự đoán xu hướng hashtag", "Đánh Giá Mô Hình"])

if page == "Biểu đồ Hashtag phổ biến":
    show_popular_hashtags()
elif page == "Phân cụm chủ đề từ CSV":
    cluster_topics()
elif page == "Dự đoán xu hướng hashtag":
    predict_trend()
elif page == "Đánh Giá Mô Hình":
    evaluate_model()