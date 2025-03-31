import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# 生成堆
def build_heap(N: int, seed: int = None) -> np.ndarray:
    """
    输出：heap: 以一维数组形式存储的完整堆
    参数：
        N: 堆的层数
        seed: 随机数种子，默认为 None
    """
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    length = N * (N + 1) // 2  
    heap = np.random.rand(length)  
    return heap

# 找到最短路径
def find_shortest_path(heap: np.ndarray, N: int):
    """
    输出：
        shortest_path: 最短路径长度
        x_star: 最短路径的终点横坐标
    """
    length = N * (N + 1) // 2 
    min_path = np.zeros(length, dtype=float)  
    identical_path_total = []
    for i in range(length):
        identical_path_total.append([i])

    start_index = N * (N - 1) // 2  
    min_path[start_index:] = heap[start_index:]

    for i in range(start_index - 1, -1, -1): 

        n = [n for n in range(1, N + 1) if n * (n - 1) // 2 <= i < n * (n + 1) // 2]
        n_int = n[0] if n else 0  # 确定当前节点所在层数

        left_child = i + n_int
        right_child = i + n_int + 1

        if min_path[left_child] <= min_path[right_child]:
            min_path[i] = heap[i] + min_path[left_child]
            identical_path_total[i].extend(identical_path_total[left_child])
        else:
            min_path[i] = heap[i] + min_path[right_child]
            identical_path_total[i].extend(identical_path_total[right_child])

    shortest_path = min_path[0]  
    important_nodes = identical_path_total[0]
    x_star = important_nodes[-1]
    return shortest_path, x_star,important_nodes

if __name__ == "__main__":
    # 示例运行
    N = 5
    heap = build_heap(N, seed=42)
    shortest_path, x_star,important_nodes= find_shortest_path(heap, N)
    
    print("随机生成堆：\n", heap)
    print("堆的层数: ", N)
    print(f"最短路径长度 (p*): {shortest_path}")
    print(f"最短路径节点索引: {important_nodes}")
    print(f"最短路径终点横坐标 (x*): {x_star}")
    




