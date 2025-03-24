# 📊 Dự đoán Xu hướng từ Dữ liệu Mạng Xã hội (Hashtags, Chủ đề)

## 📌 Giới thiệu
Đề tài nhằm khai thác dữ liệu từ mạng xã hội (Twitter) để phân tích các hashtag phổ biến, phân cụm các chủ đề liên quan và dự đoán xu hướng đang nổi bật.

## 🎯 Mục tiêu chính
- Phân tích các hashtag phổ biến trên mạng xã hội.
- Phân cụm chủ đề từ nội dung bài viết bằng KMeans.
- Áp dụng mô hình học máy (Random Forest) để dự đoán xu hướng.

## 🛠️ Công nghệ sử dụng
- Ngôn ngữ: Python
- Thư viện: `pandas`, `sklearn`, `matplotlib`, `seaborn`, `plotly`

## 📂 Cấu trúc thư mục
```
.
├── data/                   # Chứa dữ liệu thô và đã xử lý
│   ├── raw_twitter_data.csv
│   └── output.csv
├── models/                # Mô hình được huấn luyện (nếu có)
├── notebooks/             # Notebook phân tích chính
│   └── report.ipynb
├── scripts/               # Các file xử lý độc lập
│   ├── clean_data.py
│   ├── analyze_hashtags.py
│   ├── sentiment_analysis.py
│   ├── predict_trends.py
│   └── network_analysis.py
└── README.md              # Mô tả dự án
```

## 🚀 Hướng dẫn chạy dự án
1. Cài đặt các thư viện cần thiết (nếu chưa có):
```bash
pip install -r requirements.txt
```

2. Mở file `report.ipynb` bằng Jupyter Notebook hoặc VSCode để xem báo cáo.

## ✅ Kết quả đạt được
- Trực quan hóa 20 hashtag phổ biến nhất
- Phân cụm chủ đề bằng KMeans + PCA
- Mô hình Random Forest đạt độ chính xác tương đối trong việc dự đoán xu hướng

## 📌 Hướng phát triển
- Kết hợp phân tích thời gian để theo dõi sự thay đổi xu hướng
- Dùng mô hình embedding nâng cao (BERT, Word2Vec)
- Kết nối mạng lưới người dùng để phân tích ảnh hưởng

---
*Thực hiện bởi: NHOM 12*  
*Môn học: Khai thác Dữ liệu - 03/2025*
