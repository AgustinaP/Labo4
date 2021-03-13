import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# define the true objective function
def objective(f, r, l, c):
	return np.sqrt((4*f**2*l**2*np.pi**2*r**2)/(2500*r**2 + 40000*c**2*f**4*l**2*np.pi**4*r**2 + 4*f**2*l*np.pi**2*(-5000*c*r**2 + l*(50 + r)**2)))


#salida  = np.genfromtxt('C1_G1LIN.CSV', delimiter=',', skip_header=1)
#entrada = np.genfromtxt('C1_VLIN.CSV' , delimiter=',', skip_header=1)
simulacion = np.genfromtxt('Q15.dat', delimiter='\t',skip_header=1)
simulacion2 = np.genfromtxt('lineal.dat', delimiter='\t',skip_header=1)
simulacion3 = np.genfromtxt('Q15norm.dat', delimiter='\t',skip_header=1)


#aux1=aux2= np.array([[0,0],[0,0]])
'''
for i in range(0, len(entrada)):
	if ((i+1)%2)==0:
		aux1 = np.append(aux1, [entrada[i,:]],axis=0)

for i in range(0, len(salida)):
	if ((i+1)%2)==0:
		aux2 = np.append(aux2, [salida[i,:]],axis=0)

normalizado = aux2[:,1]/aux1[:,1]
'''

#p = plt.semilogx(aux2[3:-50,0]*59E6+1E6,20* np.log(normalizado[3:-50]),label='Experimental')
g = plt.semilogx(simulacion3[:,0],simulacion3[:,1],label='Simulación Q=15 normalizado')
g = plt.semilogx(simulacion[:,0],simulacion[:,1],label='Simulación Q=15')
g = plt.semilogx(simulacion2[:,0],simulacion2[:,1],label='Simulación Q=0,52')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Voltaje [dB]')
plt.legend(loc='lower right')
plt.show()

'''
popt, _ = curve_fit(objective,aux2[3:-50,0]*59E6+1E6,normalizado[3:-50] )
# summarize the parameter values
r,l,c  = popt
print(popt)
# plot input vs output
plt.semilogx(aux2[3:-50,0]*59E6+1E6,20*np.log(normalizado[3:-50]) )
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(np.min(aux2[3:-50,0]*59E6+1E6), np.max(aux2[3:-50,0]*59E6+1E6), 0.1E6)
# calculate the output for the range
y_line = objective(x_line, r,l,c)
# create a line plot for the mapping function
plt.semilogx(x_line, y_line, '--', color='red')
plt.show()
'''