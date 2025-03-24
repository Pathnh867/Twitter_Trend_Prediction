import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Load data đã làm sạch
df = pd.read_csv("data/cleaned_data.csv")

# Loại bỏ giá trị null
df = df.dropna(subset=["text", "is_trending"])

# Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df["text"])
y = df["is_trending"]

# In tỷ lệ 0/1
print("Tỉ lệ lớp 0 và 1:", y.value_counts(normalize=True))

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình với class_weight
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Lưu model
joblib.dump(model, "models/random_forest_model.pkl")
print("✅ Đã lưu model mới.")
print("Tỉ lệ 0 và 1:", y.value_counts(normalize=True))
