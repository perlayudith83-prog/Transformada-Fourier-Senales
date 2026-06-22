# =====================================================
# SIMULACIÓN Y ANÁLISIS DE SEÑALES CON FOURIER
# Autor: Perla Delgadillo
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Parámetros generales
# ------------------------------------------

fs = 1000  # Frecuencia de muestreo (Hz)
T = 1      # Duración de la señal (s)

t = np.linspace(0, T, fs, endpoint=False)

# ------------------------------------------
# Señal 1: Función senoidal
# ------------------------------------------

f = 10  # frecuencia de la señal (Hz)

senal_seno = np.sin(2 * np.pi * f * t)

# ------------------------------------------
# Señal 2: Pulso rectangular
# ------------------------------------------

pulso = np.where((t >= 0.4) & (t <= 0.6), 1, 0)

# ------------------------------------------
# Señal 3: Función escalón
# ------------------------------------------

escalon = np.where(t >= 0.5, 1, 0)

# ------------------------------------------
# Función para calcular FFT
# ------------------------------------------

def calcular_fft(signal):
    N = len(signal)

    fft_signal = np.fft.fft(signal)

    frecuencia = np.fft.fftfreq(N, d=1/fs)

    magnitud = np.abs(fft_signal)

    fase = np.angle(fft_signal)

    return frecuencia, magnitud, fase

# FFT de las señales

freq_seno, mag_seno, fase_seno = calcular_fft(senal_seno)
freq_pulso, mag_pulso, fase_pulso = calcular_fft(pulso)
freq_escalon, mag_escalon, fase_escalon = calcular_fft(escalon)

# ------------------------------------------
# Gráficas
# ------------------------------------------

fig, ax = plt.subplots(3,3, figsize=(15,10))

# Señal Senoidal
ax[0,0].plot(t, senal_seno)
ax[0,0].set_title("Señal Senoidal")

ax[0,1].plot(freq_seno, mag_seno)
ax[0,1].set_title("Magnitud FFT")

ax[0,2].plot(freq_seno, fase_seno)
ax[0,2].set_title("Fase FFT")

# Pulso Rectangular
ax[1,0].plot(t, pulso)
ax[1,0].set_title("Pulso Rectangular")

ax[1,1].plot(freq_pulso, mag_pulso)
ax[1,1].set_title("Magnitud FFT")

ax[1,2].plot(freq_pulso, fase_pulso)
ax[1,2].set_title("Fase FFT")

# Escalón
ax[2,0].plot(t, escalon)
ax[2,0].set_title("Función Escalón")

ax[2,1].plot(freq_escalon, mag_escalon)
ax[2,1].set_title("Magnitud FFT")

ax[2,2].plot(freq_escalon, fase_escalon)
ax[2,2].set_title("Fase FFT")

plt.tight_layout()
plt.show()

# ------------------------------------------
# Verificación de Linealidad
# ------------------------------------------

senal_combinada = senal_seno + pulso

fft_combinada = np.fft.fft(senal_combinada)

fft_suma = np.fft.fft(senal_seno) + np.fft.fft(pulso)

print("Propiedad de Linealidad:")
print(np.allclose(fft_combinada, fft_suma))

# ------------------------------------------
# Desplazamiento en el tiempo
# ------------------------------------------

senal_desplazada = np.roll(senal_seno, 100)

fft_desplazada = np.fft.fft(senal_desplazada)

# ------------------------------------------
# Escalamiento en frecuencia
# ------------------------------------------

senal_frec2 = np.sin(2 * np.pi * 20 * t)

fft_frec2 = np.fft.fft(senal_frec2)

print("Análisis completado correctamente.")
