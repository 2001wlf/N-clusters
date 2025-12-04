import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from util3 import compute_cluster_size_density_radius
from mytest3 import draw_zone_picture
def main():
    # 1. 读入 pkl 文件（按你的路径来）
    pkl_path = "C:/Users/10998/Desktop/N-clusters/range/mydata2/train/100.pkl"
    with open(pkl_path, "rb") as f:
        feat = pickle.load(f)

    # 2. 取出 node_feat，形状 (n_samples, n_nodes, 2)
    node_feat = feat["node_feat"]
    density_feat=feat["density_feat"]
    print("node_feat shape:", node_feat.shape)

    n_samples = node_feat.shape[0]

    # 3. 选出大概四五个样本，这里取 min(5, n_samples) 个
    num_show = min(5, n_samples)

    # 用前几个样本（如果你想随机就改成 np.random.choice）
    #idxs = np.arange(num_show)
    # 如果想随机抽样，请注释上一行，改用下面这一行：
    idxs = np.random.choice(n_samples, size=num_show, replace=False)

    samples = node_feat[idxs]   # 形状 (num_show, n_nodes, 2)
    density= density_feat[idxs]
    print("origin radius: ",density)
    # 4. 画图：每个样本一个子图
    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))

    # 兼容 num_show == 1 的情况
    if num_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        points = samples[i]      # (n_nodes, 2)
        Zone=density[i]
        draw_zone_picture(points,Zone,ax)
        x = points[:, 0]
        y = points[:, 1]
        ax.scatter(x, y, s=5)
        ax.set_title(f"Sample {idxs[i]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")   # 坐标比例 1:1，避免变形

    plt.tight_layout()

    # 5. 保存图像为文件（不会弹出窗口）
    output_path = "C:/Users/10998/Desktop/N-clusters/range/pkl_result.png"
    plt.savefig(output_path, dpi=200)
    print(f"图像已保存到: {output_path}")

if __name__ == "__main__":
    main()