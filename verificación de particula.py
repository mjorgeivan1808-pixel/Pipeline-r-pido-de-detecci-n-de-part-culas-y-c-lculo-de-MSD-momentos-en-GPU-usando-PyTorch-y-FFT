# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:14:29 2026

@author: mjorg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

# --- Parámetros iniciales (ajústalos después de ver las imágenes) ---
lnoise = 2.0
lobject = 10.0

# --- Cargar un frame del video ---
ruta_video = r"C:\Users\mjorg\Desktop\memorias\M2\ivan17aa1_7.avi"
cap = cv2.VideoCapture(ruta_video)
ret, frame = cap.read()
cap.release()

if not ret:
    print("No se pudo leer el video. Revisa la ruta.")
    exit()

# Convertir a grises y float32
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

# --- Filtro paso‑banda (versión CPU, más fácil para debug) ---
def filtro_debug(img, lnoise, lobject):
    b = float(lnoise)
    w = int(np.round(max(lobject, 2*b)))
    N = 2*w + 1
    r = (np.arange(N) - w) / (2. * b)
    gx = np.exp(-r**2) / (2. * b * np.sqrt(np.pi))
    gy = gx.copy()
    bx = np.ones(N) / N
    by = bx.copy()
    g = convolve1d(convolve1d(img, gx, axis=1), gy, axis=0)
    a = convolve1d(convolve1d(img, bx, axis=1), by, axis=0)
    diferencia = g - a
    mascara = diferencia > 0
    return g, a, diferencia, mascara

g, a, dif, mask = filtro_debug(gray, lnoise, lobject)

# --- Visualización ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(gray, cmap='gray'); axes[0,0].set_title('Original (escala de grises)')
axes[0,1].imshow(g, cmap='gray'); axes[0,1].set_title('Gaussiano (pasa‑bajo)')
axes[0,2].imshow(a, cmap='gray'); axes[0,2].set_title('Caja (uniforme)')
axes[1,0].imshow(dif, cmap='gray'); axes[1,0].set_title('Diferencia g - a')
axes[1,1].imshow(mask, cmap='gray'); axes[1,1].set_title('Máscara >0')
# Mostrar histograma de la diferencia
axes[1,2].hist(dif.ravel(), bins=100)
axes[1,2].set_title('Histograma de la diferencia')
axes[1,2].axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# --- Información adicional ---
print(f"Valores de la diferencia: min={dif.min():.2f}, max={dif.max():.2f}")
print(f"Píxeles en la máscara (>0): {mask.sum()} de {mask.size}")
print("\nSi las partículas son oscuras sobre fondo claro, la diferencia será negativa en ellas.")
print("En ese caso, cambia la máscara a: diferencia < 0")