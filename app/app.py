import numpy as np
from utils import DataUtils

net_edges_data=DataUtils.load_timing_features_data(
    file_name=f"8-RISCY-a-1-c2-u0.7-m1-p8-f0",
    data_type=f"net_edges"
)
net_edges=net_edges_data["net_edges"]

src = net_edges[:,0].astype(np.int64)
dst = net_edges[:,1].astype(np.int64)

# 중복 연결 제거
edges = np.unique(np.column_stack([src, dst]), axis=0)

# 각 Destination의 서로 다른 Source 개수
unique_dst, indegree = np.unique(
    edges[:, 1], return_counts=True
)

# Source이면서 Destination인 노드
both = np.intersect1d(
    np.unique(edges[:, 0]),
    np.unique(edges[:, 1])
)

print("여러 Source를 갖는 Sink 수:", np.sum(indegree > 1))
print("Source이면서 Sink인 노드 수:", len(both))