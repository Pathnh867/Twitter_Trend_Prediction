# scripts/sentiment_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Đọc dữ liệu
data = pd.read_csv('../data/social_media_data.csv')

# Hàm phân tích cảm xúc
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return 'Tích cực'
    elif analysis.sentiment.polarity < 0:
        return 'Tiêu cực'
    else:
        return 'Trung lập'

# Áp dụng phân tích cảm xúc
data['sentiment'] = data['text'].apply(get_sentiment)

# Thống kê cảm xúc
sentiment_counts = data['sentiment'].value_counts()

# Trực quan hóa
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm')
plt.title('Phân bố cảm xúc của bài đăng')
plt.xlabel('Cảm xúc')
plt.ylabel('Số lượng bài đăng')
plt.show()

# In kết quả
print("\nPhân bố cảm xúc:")
print(sentiment_counts)

# Phân tích cảm xúc theo hashtag '#AI'
ai_posts = data[data['hashtags'].str.contains('#AI', na=False)]
ai_sentiment = ai_posts['sentiment'].value_counts()
print("\nCảm xúc của bài đăng chứa #AI:")
print(ai_sentiment)

# Lưu dữ liệu với cột sentiment
data.to_csv('../data/output.csv', index=False)
print("\nĐã lưu kết quả phân tích cảm xúc vào '../data/output.csv'")