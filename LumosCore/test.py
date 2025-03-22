import numpy as np
import timeit

from numpy.lib.stride_tricks import as_strided

# 假设原始数组为 a，形状为 (n, m), 新的数组 X 的形状为 (n-k+1, k*m)
m = 1000
n = 2000
k = 3
a = np.random.rand(n, m)


# 方法 1：使用 for 循环和 append
def method1():
    X = []
    for i in range(n - k + 1):
        X.append(a[i:i + k, :].flatten())
    return np.asarray(X)


# 方法 2：使用 NumPy 的切片和重塑
def method2():
    new_shape = (n - k + 1, k * m)
    result = np.zeros(new_shape)
    for i in range(k):
        result[:, i * m:(i + 1) * m] = a[i:n - k + i + 1, :]
    return result


def method3():
    # 创建一个视图，形状为 (n-k+1, k, m)
    view = as_strided(a, shape=(n - k + 1, k, m), strides=(a.strides[0], a.strides[0], a.strides[1]))
    # 将视图重塑成 (n-k+1, k*m)
    result = view.reshape(n - k + 1, k * m)
    return result


# 测试性能
print("Method 1 time:", timeit.timeit(method1, number=100))
print("Method 2 time:", timeit.timeit(method2, number=100))
print("Method 3 time:", timeit.timeit(method2, number=100))

print(np.array_equal(method3(), method2()))  # True
