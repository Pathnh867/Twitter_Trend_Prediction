
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv('../data/output.csv')

# Tách hashtags thành danh sách (nếu chưa có cột hashtags_list)
if 'hashtags_list' not in data.columns:
    data['hashtags_list'] = data['hashtags'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Tạo biểu đồ mạng
G = nx.Graph()
for hashtags in data['hashtags_list']:
    for i in range(len(hashtags)):
        for j in range(i + 1, len(hashtags)):
            G.add_edge(hashtags[i], hashtags[j])

# Vẽ biểu đồ mạng
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Mối quan hệ giữa các Hashtags')
plt.show()

# Xác định các hashtag trung tâm
centrality = nx.degree_centrality(G)
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 hashtag trung tâm:")
for hashtag, score in sorted_centrality[:5]:
    print(f"{hashtag}: {score:.3f}")