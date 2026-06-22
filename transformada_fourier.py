# ==========================================
# SIMULACIÓN Y ANÁLISIS DE SEÑALES
# TRANSFORMADA DE FOURIER
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ------------------------------------------
# PARÁMETROS GENERALES
# ------------------------------------------

Fs = 1000  # Frecuencia de muestreo
T = 1      # Duración de la señal
t = np.linspace(0, T, Fs, endpoint=False)

# ------------------------------------------
# 1. SEÑAL SENOIDAL
# ------------------------------------------

f = 10  # Frecuencia de la señal

senal_senoidal = np.sin(2*np.pi*f*t)

# Transformada de Fourier
fft_senoidal = np.fft.fft(senal_senoidal)
frecuencias = np.fft.fftfreq(len(t), 1/Fs)

# ------------------------------------------
# GRAFICAR SEÑAL SENOIDAL
# ------------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, senal_senoidal)
plt.title("Señal Senoidal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(1,2,2)
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_senoidal[:Fs//2]))
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 2. PULSO RECTANGULAR
# ------------------------------------------

pulso = signal.square(2*np.pi*5*t)

fft_pulso = np.fft.fft(pulso)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, pulso)
plt.title("Pulso Rectangular")
plt.xlabel("Tiempo (s)")

plt.subplot(1,2,2)
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_pulso[:Fs//2]))
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 3. FUNCIÓN ESCALÓN
# ------------------------------------------

escalon = np.heaviside(t-0.5, 1)

fft_escalon = np.fft.fft(escalon)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, escalon)
plt.title("Función Escalón")
plt.xlabel("Tiempo (s)")

plt.subplot(1,2,2)
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_escalon[:Fs//2]))
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")

plt.tight_layout()
plt.show()

# ------------------------------------------
# MAGNITUD Y FASE
# ------------------------------------------

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_senoidal[:Fs//2]))
plt.title("Magnitud")

plt.subplot(2,1,2)
plt.plot(frecuencias[:Fs//2],
         np.angle(fft_senoidal[:Fs//2]))
plt.title("Fase")

plt.tight_layout()
plt.show()

# ------------------------------------------
# PROPIEDAD DE LINEALIDAD
# ------------------------------------------

senal1 = np.sin(2*np.pi*10*t)
senal2 = np.sin(2*np.pi*20*t)

suma = senal1 + senal2

fft_suma = np.fft.fft(suma)

plt.figure(figsize=(10,4))
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_suma[:Fs//2]))
plt.title("Linealidad: Suma de dos señales")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()

# ------------------------------------------
# DESPLAZAMIENTO EN EL TIEMPO
# ------------------------------------------

senal_desplazada = np.roll(senal_senoidal, 100)

fft_desplazada = np.fft.fft(senal_desplazada)

plt.figure(figsize=(10,4))
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_desplazada[:Fs//2]))
plt.title("Desplazamiento Temporal")
plt.xlabel("Frecuencia (Hz)")
plt.show()

# ------------------------------------------
# ESCALAMIENTO EN FRECUENCIA
# ------------------------------------------

senal_escalada = np.sin(2*np.pi*30*t)

fft_escalada = np.fft.fft(senal_escalada)

plt.figure(figsize=(10,4))
plt.plot(frecuencias[:Fs//2],
         np.abs(fft_escalada[:Fs//2]))
plt.title("Escalamiento en Frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.show()

print("Análisis completado correctamente.")