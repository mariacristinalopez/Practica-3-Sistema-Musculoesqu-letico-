"""
Práctica 4: Sistema Musculoesqueletico

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Maria Cristina Lopez Lopez
Número de control: 22212259
Correo institucional: 22212259@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot


# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

X0,t0,tf,dt,w,h = 0,0,10,1E-3,10,5
n=round((tf-t0)/dt)+1
u=np.zeros(n); u[round(1/dt):round(2/dt)]=1
t = np.linspace(t0,tf,n)

def ME(Cs,R,Cp,a):
    num=[0,Cs*R,1-a]
    den=[0,R*(Cs+Cp),1]
    sys=ctrl.tf(num,den)
    return sys

#Funcion d transferencia: Control
Cs,R,Cp,a = 10e-6,100,100e-6,0.25
syscontrol=ME(Cs,R,Cp,a)
print(f'funcion de transferencia normotenso: {syscontrol}')

#Funcion d transferencia: Caso
Cs,R,Cp,a =10e-6,10e3,100e-6,0.25
syscaso=ME(Cs,R,Cp,a)
print(f'funcion de transferencia normotenso: {syscaso}')

       
_,Pp0 = ctrl.forced_response(syscontrol,t,u,X0)
_,Pp1 = ctrl.forced_response(syscaso,t,u,X0)

clr1= np.array([145,18,188])/255
clr2 = np.array([0,120,157])/255
clr6 = np.array([0,0,0])/255
clr3=np.array([120,157,188])/255
clr4=np.array([203,4,4])/255

fg1=plt.figure()
plt.plot(t,u,'-',linewidth=1,color=clr1,label='F(t)')
plt.plot(t,Pp0,'-',linewidth=1,color=clr1,label='F(t): Control')
plt.plot(t,Pp1,':',linewidth=1,color=clr2,label='F(t): Caso')
plt.xlabel('t [s]',fontsize = 11)
plt.ylabel('Pp(t) [V]',fontsize = 11)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(0,1.1);plt.yticks(np.arange(-.2,1.4,0.2))
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol=3,fontsize = 9,frameon = True)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('SistemaMELA.pdf',bbox_inches = 'tight',dpi=660)
fg1.savefig('SistemaMELA.png',bbox_inches = 'tight',dpi=660)

def controlador(kP,kI,sys):
    Cr=1E-6
    Re=1/(kI*Cr)
    Rr=kP*Re
    numPI=[Rr*Cr,1]
    denPI=[Re*Cr,0]
    Pi=ctrl.tf(numPI,denPI)
    X=ctrl.series(Pi,sys)
    sysPI=ctrl.feedback(X,1,sign=-1)
    return sysPI
CasoPI=controlador(10,28908.56,syscaso)

_,Pp3=ctrl.forced_response(CasoPI,t,Pp0,X0)
\
fg2=plt.figure()
plt.plot(t,u,'-',linewidth=1,color=clr1,label='F(t)')
plt.plot(t,Pp0,'-',linewidth=1,color=clr1,label='F(t): Control')
plt.plot(t,Pp1,':',linewidth=1,color=clr2,label='F(t): Caso')
plt.plot(t,Pp3,':',linewidth=3,color=clr3,label='Pp(t): Caso Tratamiento')
plt.xlabel('t [s]',fontsize = 11)
plt.ylabel('Pp(t) [V]',fontsize = 11)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(0,1.1);plt.yticks(np.arange(-0.1,1.1,0.1))
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol=4,fontsize = 9,frameon = True)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('SistemaMEPI.pdf',bbox_inches = 'tight',dpi=660)
fg2.savefig('SistemaMEPI.png',bbox_inches = 'tight',dpi=660)



