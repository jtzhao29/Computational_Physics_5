from B import build_heap,find_shortest_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_w(N: int,num:int) -> float:
    """"
    输入:N为堆的层数，num为对于相同层数的堆的重复次数
    输出$w(N)=\sqrt{\langle[x^\star(N)]^2\rangle-\langle x^\star(N)\rangle^2}$
    """
    x_star_list = []
    for i in range(num):
        heap = build_heap(N)
        _, x_star, _ = find_shortest_path(heap, N)
        x_star_list.append(float(x_star))

    x_star_array = np.array(x_star_list)
    # mean_x_star = np.mean(x_star_array)
    # variance_x_star = np.mean(x_star_array**2) - mean_x_star**2
    # w_N = np.sqrt(variance_x_star)
    w_N = np.std(x_star_array, ddof=0)
    return w_N

def plot_w_vs_N(N_values: np.ndarray, num:int) :
    """
    输入：N_values: 堆的层数范围，num: 对于相同层数的堆的重复次数
    """
    w_values = np.array([calculate_w(N, num) for N in N_values])

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, w_values, marker='o')
    plt.title(r"$w(N)$ vs $N$", fontsize=20)
    plt.xlabel("N", fontsize=20)
    plt.ylabel(r"$w(N)$", fontsize=20)
    plt.grid()
    path = f"./figure/w_vs_N_from_{min(N_values)}_to_{max(N_values)}_num_{num}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_w_vs_N_and_log(N_values: np.ndarray, num:int) :
    """
    输入：N_values: 堆的层数范围，num: 对于相同层数的堆的重复次数
    """
    w_values = np.array([calculate_w(N, num) for N in N_values])

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, w_values, marker='o', label=r"$w(N)$")
    plt.title(r"$log(w(N))$ vs $log(N)$", fontsize=20)
    plt.xlabel("log(N)", fontsize=20)
    plt.ylabel(r"$log(w(N))$", fontsize=20)
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    path = f"./figure/w_vs_N_from_{min(N_values)}_to_{max(N_values)}_num_{num}_with_log.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_w_vs_N_and_log_fit(N_values: np.ndarray, num: int):
    """
    输入：N_values: 堆的层数范围，num: 对于相同层数的堆的重复次数
    """
    # 计算 w(N)
    w_values = np.array([calculate_w(N, num) for N in N_values])

    # 转换为对数坐标
    log_N_values = np.log(N_values)
    log_w_values = np.log(w_values)

    # 拟合直线
    coefficients = np.polyfit(log_N_values, log_w_values, 1)  # 一次多项式拟合
    slope, intercept = coefficients
    print(f"拟合直线表达式: log(w(N)) = {slope:.4f} * log(N) + {intercept:.4f}")

    # 生成拟合直线的值
    fitted_log_w_values = slope * log_N_values + intercept

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(log_N_values, log_w_values, marker='o', label=r"$\log(w(N))$")
    plt.plot(log_N_values, fitted_log_w_values, label=fr"fit: $\log(w(N)) = {slope:.4f} \cdot \log(N) + {intercept:.4f}$", linestyle='--', color='red')
    plt.title(r"$\log(w(N))$ vs $\log(N)$", fontsize=20)
    plt.xlabel(r"$\log(N)$", fontsize=20)
    plt.ylabel(r"$\log(w(N))$", fontsize=20)
    plt.grid()
    plt.legend()
    plt.yscale('linear')
    plt.xscale('linear')

    # 保存图像
    path = f"./figure/w_vs_N_from_{min(N_values)}_to_{max(N_values)}_num_{num}_with_log_fit.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    N_values = np.arange(3, 41)  
    num = 10000
    plot_w_vs_N_and_log_fit(N_values, num)