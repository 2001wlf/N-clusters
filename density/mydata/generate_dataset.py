from sklearn.datasets import make_blobs, make_circles
import numpy as np
#笑脸
def generate_smile(n_samples=1000, factor=0.6, noise=0.05):
    # 生成数据集
    # 外圈
    outer_circle, _ = make_circles(n_samples=n_samples, factor=factor, noise=noise)

    # 眼睛 + 嘴（blob）
    eye1, _ = make_blobs(n_samples=int(n_samples/3), centers=[[0.3, 0.3]], cluster_std=0.05)
    eye2, _ = make_blobs(n_samples=int(n_samples/3), centers=[[-0.3, 0.3]], cluster_std=0.05)
    mouth, _ = make_blobs(n_samples=int(n_samples/3), centers=[[0.0, -0.4]], cluster_std=0.08)

    X1 = np.vstack((outer_circle, eye1, eye2, mouth))
    y1 = np.array(
        [0] * int(n_samples) +
        [1] * int(n_samples/3) +
        [2] * int(n_samples/3) +
        [3] * int(n_samples/3),
        dtype=int
    )
    return X1, y1

#双螺旋
def make_spiral(n_points=1000, noise=0.2):
    n = np.sqrt(np.random.rand(n_points)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points)*noise
    d1y = np.sin(n)*n + np.random.rand(n_points)*noise
    X3 = np.vstack((d1x, d1y)).T
    y3 = np.zeros(n_points, dtype=int)
    return X3, y3

#抛物线
def generate_parabola(n_samples=1000):
    x = np.random.uniform(0, 4, n_samples)
    y = x**2 / 8 + np.random.normal(0, 0.05, n_samples)
    curve = np.vstack((x, y)).T
    blob1, _ = make_blobs(n_samples=int(n_samples/5), centers=[[2, 0]], cluster_std=0.1)
    blob2, _ = make_blobs(n_samples=int(n_samples/5), centers=[[3, 0]], cluster_std=0.1)
    X4 = np.vstack((curve, blob1, blob2))
    y4 = np.array([0]*int(n_samples) + [1]*int(n_samples/5) + [2]*int(n_samples/5))
    return X4, y4

# 自定义生成嵌套方框结构
def make_nested_squares(n_samples=1000):
    outer = np.random.uniform(0, 1, size=(n_samples, 2))
    inner1 = np.random.uniform(0.3, 0.45, size=(int(n_samples/5), 2))
    inner2 = np.random.uniform(0.55, 0.7, size=(int(n_samples/5), 2))
    X2 = np.vstack([outer, inner1, inner2])
    y2 = np.array([0]*int(n_samples) + [1]*int(n_samples/5) + [2]*int(n_samples/5))
    return X2, y2

