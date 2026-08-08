import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# CSVファイル読み込み
filename = "loc_movie.csv"
x, y, z = [], [], []

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i<40:
            continue
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(float(row[3]))

# numpy配列に変換
points = np.array([x, y, z]).T  # shape = (N, 3)

# セグメントを作る（N-1本の線分）
segments = np.array([points[i:i+2] for i in range(len(points)-1)])

# カラーグラデーションを作成（例：始点→終点で"viridis"カラーマップ）
from matplotlib import cm
from matplotlib.colors import Normalize

norm = Normalize(vmin=0, vmax=len(segments))
colors = cm.viridis(np.linspace(0, 1, len(segments)))

# 3Dプロット作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軌跡ライン（カラー付き）を追加
lc = Line3DCollection(segments, colors=colors, linewidths=2)
ax.add_collection3d(lc)

# 点も重ねておく（任意）
ax.scatter(x, y, z, c='gray', s=5)

# 軸ラベル
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 縮尺を揃える（前回のやつ）
x_range = max(x) - min(x)
y_range = max(y) - min(y)
z_range = max(z) - min(z)
max_range = max(x_range, y_range, z_range)

x_middle = (max(x) + min(x)) / 2
y_middle = (max(y) + min(y)) / 2
z_middle = (max(z) + min(z)) / 2

ax.set_xlim(x_middle - max_range/2, x_middle + max_range/2)
ax.set_ylim(y_middle - max_range/2, y_middle + max_range/2)
ax.set_zlim(z_middle - max_range/2, z_middle + max_range/2)

# 表示
plt.show()