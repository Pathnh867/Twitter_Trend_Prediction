import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Đường dẫn tới file dữ liệu CSV
DATA_PATH = "data/cleaned_data.csv"

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

# Kiểm tra cột 'text'
if "text" not in df.columns:
    raise ValueError("Cột 'content' không tồn tại trong file dữ liệu")

# Lấy nội dung văn bản, xử lý thiếu dữ liệu nếu có
texts = df["text"].fillna("")

# Khởi tạo vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Huấn luyện vectorizer
vectorizer.fit(texts)

# ✅ Lưu vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Đã lưu vectorizer vào models/tfidf_vectorizer.pkl")
