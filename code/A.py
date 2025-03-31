import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def v(x:np.ndarray,phi:np.ndarray)->np.ndarray:
    """
    势能函数
    输入：x: 离散的位置坐标
    输出：势能值
    """
    # $$\psi(x_j,t+\Delta t/2)=\psi(x_j,t)\cdot\exp\left(i\frac{\Delta t}{2}\left(\frac{x^2}{2}-\frac{1}{2}|\psi(x_j,t)|^2\right)\right).$$
    return 0.5 * x**2 

def phi_origin(x:np.ndarray)->np.ndarray:
    """
    初始波函数
    输入：x: 离散的位置坐标
    输出：初始波函数值
    """
    # $$\psi(x_j,0)=\frac{1}{\sqrt{2\pi}}\exp(-x_j^2/2)\cdot\exp(i\cdot 0.5 \cdot x_j^2).$$     
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def calculate_phi_and_density(x:np.ndarray,t0:float,tf:float,dt:float,phi_0:np.ndarray,V)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    输入：
        x: 离散的位置坐标
        t0: 初始时间
        tf: 结束时间
        dt: 时间步长
        phi_0: 初始波函数
        V: 势能函数
    输出：
        phi: 波函数随时间演化的结果，是二维数组，
              第一维是时间步数，第二维是空间坐标点数
        t: 时间数组
    """
    
    dx = x[1] - x[0]  
    t = np.arange(t0, tf, dt)
    num_t = len(t)
    phi = np.zeros((num_t, len(x)), dtype=complex)
    density = np.zeros((num_t, len(x)), dtype=float)
    current_phi = phi_0
    k = 2*np.pi*np.fft.fftfreq(len(x), dx)  # 频率空间的波数
    for i in range(num_t):
        current_phi =current_phi*np.exp(-1j*dt*V(x,current_phi)/2)
        k_phi = np.fft.fft(current_phi)  # 傅里叶变换
        k_phi = k_phi*np.exp(-1j*dt*k**2/2)
        current_phi = np.fft.ifft(k_phi)
        current_phi = current_phi*np.exp(-1j*dt*V(x,current_phi)/2)
        phi[i,:] = current_phi
        density[i,:] = np.abs(current_phi)**2
    return phi, density, t

def plot_density(density:np.ndarray,t:np.ndarray,x:np.ndarray)->None:
    """
    输入：
        density: 波函数的密度随时间演化的结果，是二维数组，
                  第一维是时间步数，第二维是空间坐标点数
        t: 时间数组
        x: 离散的位置坐标
    输出：
        无返回值，直接绘制图形，画热图，横轴是时间，纵轴是空间坐标点数
        颜色表示密度值
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(t, x, density.T, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title("Density Evolution Over Time",fontsize=20) 
    ax.set_xlabel("Time",fontsize=20)
    ax.set_ylabel("Position",fontsize=20)
    
    path = f"./figure/density_evolution_with_x_from_{x[0]}_to_{x[-1]}_myself.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_w(density:np.ndarray,x:np.ndarray,t:np.ndarray)->np.ndarray:
    """
    计算波包宽度w(t)
    """
    dx= x[1]-x[0]
    num_t = len(t) 
    w = np.zeros(num_t,dtype=float)  
    for i in range(num_t):
        w[i] = np.sum(x**2 * density[i, :]* dx) 
    return w

def plot_w(w:np.ndarray,t:np.ndarray)->None:
    """
    输入：
        w: 波包宽度随时间演化的结果，是一维数组
        t: 时间数组
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, w, label=r"$w(t)$")
    ax.set_title(r"Wave Packet Width Evolution",fontsize=20)
    ax.set_xlabel("Time",fontsize=20)
    ax.set_ylabel(r"$w(t)$",fontsize=20)
    path = f"./figure/w_vs_t.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    t0 = 0
    tf = 20
    dt = 0.01
    N = 1000
    x = np.linspace(-5, 5, N)
    phi_0 = phi_origin(x)
    phi_for_times,density_for_times,t = calculate_phi_and_density(x, t0, tf, dt, phi_0, v)
    plot_density(density_for_times,t,x)
    w = calculate_w(density_for_times,x,t)
    plot_w(w,t)
