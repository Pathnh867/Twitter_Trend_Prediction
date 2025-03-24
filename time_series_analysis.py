import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def analyze_time_series():
    """
    Phân tích đơn giản về chuỗi thời gian cho dữ liệu mạng xã hội
    """
    print("Đang phân tích dữ liệu theo thời gian...")

    # Đọc dữ liệu
    try:
        data = pd.read_csv('../data/social_media_data.csv')
    except FileNotFoundError:
        print("File không tìm thấy. Thực hiện clean_data.py trước...")
        import clean_data
        data = pd.read_csv('../data/social_media_data.csv')

    # Chuyển cột 'created_at' thành datetime
    if 'created_at' in data.columns:
        data['created_at'] = pd.to_datetime(data['created_at'])
    else:
        print("Dữ liệu không có cột 'created_at', không thể phân tích chuỗi thời gian!")
        return

    # Tổng hợp dữ liệu theo ngày
    daily_data = data.groupby(pd.Grouper(
        key='created_at', freq='D')).size().reset_index(name='post_count')

    # Vẽ biểu đồ số lượng bài đăng theo ngày
    fig = px.line(daily_data, x='created_at', y='post_count',
                  title='Số lượng bài đăng theo ngày',
                  labels={'post_count': 'Số lượng bài đăng', 'created_at': 'Ngày'})
    fig.show()

    # Lưu biểu đồ
    fig.write_html('../data/daily_posts.html')

    # Phân tích hashtag theo thời gian (nếu có)
    if 'hashtags' in data.columns:
        print("Phân tích hashtag theo thời gian...")

        # Tách hashtags thành list (nếu dữ liệu lưu dưới dạng string)
        if data['hashtags'].dtype == 'object':
            data['hashtags_list'] = data['hashtags'].apply(
                lambda x: x.split() if isinstance(x, str) else [])
        else:
            data['hashtags_list'] = data['hashtags']

        # Đếm số lượng hashtag theo ngày
        hashtag_daily = data.groupby(pd.Grouper(key='created_at', freq='D'))[['hashtags_list']].agg(
            lambda x: sum(x.tolist(), [])
        ).reset_index()

        hashtag_daily['hashtag_count'] = hashtag_daily['hashtags_list'].apply(
            len)

        # Vẽ biểu đồ số lượng hashtag theo ngày
        fig = px.line(hashtag_daily, x='created_at', y='hashtag_count',
                      title='Số lượng hashtag theo ngày',
                      labels={'hashtag_count': 'Số lượng hashtag', 'created_at': 'Ngày'})
        fig.show()

        # Lưu biểu đồ
        fig.write_html('../data/daily_hashtags.html')

        # Tìm top 10 hashtag phổ biến
        all_hashtags = []
        for tags in hashtag_daily['hashtags_list']:
            all_hashtags.extend(tags)

        top_hashtags = pd.Series(all_hashtags).value_counts().nlargest(10)

        # Vẽ biểu đồ top 10 hashtag
        fig = px.bar(x=top_hashtags.index, y=top_hashtags.values,
                     title='Top 10 Hashtag Phổ biến',
                     labels={'x': 'Hashtag', 'y': 'Số lượng'})
        fig.show()

        # Lưu biểu đồ
        fig.write_html('../data/top_hashtags.html')

        # Phát hiện hashtag trending
        # Tính tốc độ tăng trưởng đơn giản (xét mọi ngày có sẵn)
        print("Phát hiện hashtag đang trending...")

        # Lấy tất cả các ngày có sẵn
        recent_days = hashtag_daily.tail(len(hashtag_daily))

        # Lấy danh sách hashtag từ ngày gần nhất
        if len(recent_days) > 0:
            recent_hashtags = recent_days.iloc[-1]['hashtags_list']

            # Đếm số lần xuất hiện của mỗi hashtag trong ngày gần nhất
            recent_counts = pd.Series(recent_hashtags).value_counts()

            # Chỉ xét các hashtag xuất hiện ít nhất 1 lần
            trending_candidates = recent_counts[recent_counts >= 1]

            if len(trending_candidates) > 0:
                print(
                    f"\nCác hashtag tiềm năng đang trending (xuất hiện trong ngày gần đây):")
                print(trending_candidates)
            else:
                print("Không tìm thấy hashtag tiềm năng đang trending.")
        else:
            print("Không có dữ liệu hashtag nào được tìm thấy.")

    # Thêm dự đoán bằng ARIMA
    print("\nDự đoán xu hướng trong tương lai sử dụng ARIMA...")

    # Sửa đổi để dự đoán ngay cả khi dữ liệu ít hơn 7 ngày
    if len(daily_data) >= 2:  # Chỉ cần ít nhất 2 điểm dữ liệu
        try:
            # Chuẩn bị dữ liệu cho ARIMA
            ts_data = daily_data.set_index('created_at')['post_count']

            # Đặt order tùy thuộc vào số lượng dữ liệu có sẵn
            if len(ts_data) >= 3:
                # Nếu có ít nhất 3 điểm dữ liệu, sử dụng ARIMA(1,1,1)
                order = (1, 1, 1)
            else:
                # Nếu chỉ có 2 điểm dữ liệu, sử dụng mô hình đơn giản hơn
                order = (1, 0, 0)  # Sử dụng mô hình AR(1)

            model = ARIMA(ts_data, order=order)
            model_fit = model.fit()

            # Dự đoán 7 ngày tới
            days_to_predict = 7

            # Tạo ngày trong tương lai
            last_date = ts_data.index[-1]
            future_dates = [
                last_date + timedelta(days=i+1) for i in range(days_to_predict)]

            # Dự đoán
            forecast = model_fit.forecast(steps=days_to_predict)
            forecast_df = pd.DataFrame({
                'created_at': future_dates,
                'predicted_count': forecast
            })

            # Vẽ biểu đồ dự đoán
            fig = go.Figure()

            # Dữ liệu thực tế
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='Dữ liệu thực tế'
            ))

            # Dữ liệu dự đoán
            fig.add_trace(go.Scatter(
                x=forecast_df['created_at'],
                y=forecast_df['predicted_count'],
                mode='lines+markers',
                line=dict(dash='dash'),
                name='Dự đoán'
            ))

            fig.update_layout(
                title='Dự đoán số lượng bài đăng 7 ngày tới bằng ARIMA',
                xaxis_title='Ngày',
                yaxis_title='Số lượng bài đăng'
            )
            fig.show()

            # Lưu biểu đồ
            fig.write_html('../data/arima_forecast.html')

            # In ra dự đoán
            print("\nDự đoán số lượng bài đăng 7 ngày tới:")
            for i, row in forecast_df.iterrows():
                print(
                    f"{row['created_at'].strftime('%Y-%m-%d')}: {row['predicted_count']:.1f} bài đăng")

        except Exception as e:
            print(f"Không thể thực hiện dự đoán ARIMA: {e}")
            # Nếu ARIMA thất bại, thử phương pháp dự đoán đơn giản hơn
            try:
                print("Thử phương pháp dự đoán đơn giản hơn (trung bình di động)...")

                # Sử dụng trung bình đơn giản của các điểm dữ liệu hiện có
                avg_posts = daily_data['post_count'].mean()

                # Dự đoán 7 ngày tới bằng trung bình
                days_to_predict = 7
                last_date = daily_data['created_at'].iloc[-1]
                future_dates = [
                    last_date + timedelta(days=i+1) for i in range(days_to_predict)]

                forecast_df = pd.DataFrame({
                    'created_at': future_dates,
                    'predicted_count': [avg_posts] * days_to_predict
                })

                # Vẽ biểu đồ dự đoán
                fig = go.Figure()

                # Dữ liệu thực tế
                fig.add_trace(go.Scatter(
                    x=daily_data['created_at'],
                    y=daily_data['post_count'],
                    mode='lines',
                    name='Dữ liệu thực tế'
                ))

                # Dữ liệu dự đoán
                fig.add_trace(go.Scatter(
                    x=forecast_df['created_at'],
                    y=forecast_df['predicted_count'],
                    mode='lines+markers',
                    line=dict(dash='dash'),
                    name='Dự đoán (trung bình)'
                ))

                fig.update_layout(
                    title='Dự đoán số lượng bài đăng 7 ngày tới (phương pháp trung bình)',
                    xaxis_title='Ngày',
                    yaxis_title='Số lượng bài đăng'
                )
                fig.show()

                # Lưu biểu đồ
                fig.write_html('../data/simple_forecast.html')

                # In ra dự đoán
                print("\nDự đoán đơn giản số lượng bài đăng 7 ngày tới:")
                for i, row in forecast_df.iterrows():
                    print(
                        f"{row['created_at'].strftime('%Y-%m-%d')}: {row['predicted_count']:.1f} bài đăng")

            except Exception as e2:
                print(f"Không thể thực hiện cả dự đoán đơn giản: {e2}")
    else:
        print("Không đủ dữ liệu để thực hiện dự đoán (cần ít nhất 2 ngày dữ liệu).")
        print("Tạo dự đoán giả định...")

        # Tạo dự đoán giả định
        if len(daily_data) == 1:
            # Nếu chỉ có 1 điểm dữ liệu, sử dụng giá trị đó cho dự đoán
            single_value = daily_data['post_count'].iloc[0]

            # Dự đoán 7 ngày tới với cùng giá trị
            days_to_predict = 7
            last_date = daily_data['created_at'].iloc[0]
            future_dates = [
                last_date + timedelta(days=i+1) for i in range(days_to_predict)]

            forecast_df = pd.DataFrame({
                'created_at': future_dates,
                'predicted_count': [single_value] * days_to_predict
            })

            # Vẽ biểu đồ dự đoán
            fig = go.Figure()

            # Dữ liệu thực tế
            fig.add_trace(go.Scatter(
                x=daily_data['created_at'],
                y=daily_data['post_count'],
                mode='markers',
                name='Dữ liệu thực tế'
            ))

            # Dữ liệu dự đoán
            fig.add_trace(go.Scatter(
                x=forecast_df['created_at'],
                y=forecast_df['predicted_count'],
                mode='lines+markers',
                line=dict(dash='dash'),
                name='Dự đoán (giả định)'
            ))

            fig.update_layout(
                title='Dự đoán giả định số lượng bài đăng 7 ngày tới',
                xaxis_title='Ngày',
                yaxis_title='Số lượng bài đăng'
            )
            fig.show()

            # Lưu biểu đồ
            fig.write_html('../data/assumed_forecast.html')

            # In ra dự đoán
            print("\nDự đoán giả định số lượng bài đăng 7 ngày tới:")
            for i, row in forecast_df.iterrows():
                print(
                    f"{row['created_at'].strftime('%Y-%m-%d')}: {row['predicted_count']:.1f} bài đăng")
        else:
            # Nếu không có dữ liệu, tạo dự đoán giả định với giá trị 0
            print("Không có dữ liệu để dự đoán")

    print("\nĐã hoàn thành phân tích chuỗi thời gian!")


if __name__ == "__main__":
    analyze_time_series()
