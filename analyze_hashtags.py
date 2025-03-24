import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Đọc dữ liệu
data_path = "../data/cleaned_data.csv"
df = pd.read_csv(data_path)

if 'hashtags' not in df.columns:
    print("Cột 'hashtags' không tồn tại trong dữ liệu.")
else:
    print("Dữ liệu đã tải thành công!")
    
    # Hiển thị thông tin cơ bản về dữ liệu
    print(df.info())
    print(df.head())
    
    # Trực quan hóa tần suất xuất hiện của hashtags
    plt.figure(figsize=(12, 6))
    df['hashtags'].value_counts().head(20).plot(kind='bar', color='skyblue')
    plt.title("Top 20 Hashtags Phổ Biến", fontsize=14)
    plt.xlabel("Hashtag", fontsize=12)
    plt.ylabel("Số lần xuất hiện", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
    
    # Biểu đồ tương tác với Plotly
    fig = px.bar(df['hashtags'].value_counts().head(20), 
                 title="Top 20 Hashtags Phổ Biến", 
                 labels={'index': 'Hashtag', 'value': 'Số lần xuất hiện'},
                 color_discrete_sequence=['skyblue'])
    fig.show()
    
    # WordCloud của hashtags
    hashtags_text = ' '.join(df['hashtags'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hashtags_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Hashtags")
    plt.show()

# Phân cụm dữ liệu
if 'like_count' in df.columns and 'retweet_cc' in df.columns:
    X = df[['like_count', 'retweet_cc']].dropna()
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster_kmeans'] = kmeans.fit_predict(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=300, min_samples=5)
    df['cluster_dbscan'] = dbscan.fit_predict(X)
    
    # Giảm chiều dữ liệu với PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)
    df['pca_1'], df['pca_2'] = reduced_data[:, 0], reduced_data[:, 1]
    
    fig = px.scatter(df, x='pca_1', y='pca_2', color=df['cluster_kmeans'].astype(str), title="Phân Cụm K-Means")
    fig.show()
    
    fig = px.scatter(df, x='pca_1', y='pca_2', color=df['cluster_dbscan'].astype(str), title="Phân Cụm DBSCAN")
    fig.show()
    
# Cải tiến mô hình dự đoán xu hướng
if 'is_trending' in df.columns:
    features = df[['like_count', 'retweet_cc']].dropna()
    labels = df.loc[features.index, 'is_trending']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    print("Báo cáo phân loại:")
    print(classification_report(y_test))
