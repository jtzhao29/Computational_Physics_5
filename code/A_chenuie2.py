import numpy as np
import matplotlib.pyplot as plt
def V(x):
    v=0.5*x**2
    return v
def solveGP(psi0,V,dt,t,x):
    H=np.zeros(((len(x)),len(np.arange(0,t,dt))+1),dtype=np.complex128)
    H[:,0]=np.abs(psi0)**2
    x2=0
    for i in range(len(x)):
        x2+=psi0[i]*x[i]**2*psi0[i].conjugate()
    x2_=[x2]

    psi=psi0
    f=np.fft.fftfreq(len(x),d=x[1]-x[0])
    for i in np.arange(0,t,dt):
        psi=psi*np.exp(-1j*(V(x)+abs(psi**2)/2)*dt/2)
        psi_=np.fft.fft(psi)#傅里叶变换
        psi_=psi_*np.exp(-1j*(f**2)*(np.pi**2)*dt)
        psi=np.fft.ifft(psi_)#傅里叶逆变换
        psi=psi*np.exp(-1j*(V(x)+abs(psi**2)/2)*dt/2)
        H[:,int(i/dt)]=abs(psi)**2
        x2=0
        
        for i in range(len(x)):
            x2+=psi[i]*x[i]**2*psi[i].conjugate()
        x2_.append(x2)
    H = abs(H)
    return H,x2_
x=np.linspace(-3,3,100)
psi0=np.exp(-x**2/2)/np.sqrt(2*np.pi)

psi0=psi0.astype(np.complex128)

t=20
dt=0.001
t_span=np.arange(0,t+dt,dt)
X,Y=np.meshgrid(t_span,x)


h,x2_=solveGP(psi0,V,dt,t,x)

plt.imshow(h,extent=[0,t,x[0],x[-1]],aspect='auto',cmap='jet')
plt.xlabel('Time')
plt.ylabel('Position')
plt.colorbar()
plt.show()
plt.plot(t_span,x2_)
plt.xlabel('Time')
plt.ylabel('average x^2')
plt.show()



