import pandas as pd

# Đọc dữ liệu thô
raw_data = pd.read_csv('../data/raw_twitter_data.csv')

# Làm sạch dữ liệu (ví dụ: loại bỏ giá trị NaN, chuẩn hóa cột)
cleaned_data = raw_data.dropna()

# Chuyển đổi cột 'created_at' thành định dạng datetime
cleaned_data['created_at'] = pd.to_datetime(cleaned_data['created_at'])

# Tách hashtags thành danh sách
cleaned_data['hashtags_list'] = cleaned_data['hashtags'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Lưu dữ liệu đã làm sạch
cleaned_data.to_csv('../data/social_media_data.csv', index=False)
print("Đã làm sạch và lưu dữ liệu vào '../data/social_media_data.csv'")