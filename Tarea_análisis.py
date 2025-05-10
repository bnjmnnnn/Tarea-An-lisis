import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import toeplitz
import cmath

def fft_custom(signal):
    """Transformada rápida de Fourier (FFT) mediante dividir y conquistar."""
    N = len(signal)
    if N <= 1:
        return signal
    pares = fft_custom(signal[::2])
    impares = fft_custom(signal[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * impares[k] for k in range(N // 2)]
    return [pares[k] + T[k] for k in range(N // 2)] + [pares[k] - T[k] for k in range(N // 2)]

def ifft_custom(espectro):
    """Transformada rápida de Fourier inversa (IFFT) mediante dividir y conquistar."""
    N = len(espectro)
    if N <= 1:
        return espectro
    pares = ifft_custom(espectro[::2])
    impares = ifft_custom(espectro[1::2])
    T = [cmath.exp(2j * cmath.pi * k / N) * impares[k] for k in range(N // 2)]
    return [(pares[k] + T[k]) / 2 for k in range(N // 2)] + [(pares[k] - T[k]) / 2 for k in range(N // 2)]

def scc_toeplitz(x, h):
    """
    Calcula la Suma de Convolución Circular (SCC) utilizando la matriz Toeplitz
    y bucles for.
    """
    len_x = len(x)
    len_h = len(h)
    N = len_x + len_h - 1  # Longitud de la salida

    # Rellenar las señales con ceros para que tengan longitud N
    x_rellenado = np.pad(x, (0, N - len_x), mode='constant')
    h_rellenado = np.pad(h, (0, N - len_h), mode='constant')

    # Construir la matriz Toeplitz 
    toeplitz_matrix = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            if i - j >= 0:
                toeplitz_matrix[i, j] = h_rellenado[i - j]

    # Realizar la multiplicación de la matriz Toeplitz con el vector x
    y = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            y[i] += toeplitz_matrix[i, j] * x_rellenado[j]

    return y

def evaluar_rendimiento():
  
    #Compara el rendimiento de SCC con Toeplitz y FFT.
  
    print("Evaluando eficiencia de SCC con Toeplitz y FFT:\n")
    tamaños = []
    tiempo_toeplitz = []
    tiempo_fft = []
    margen_error = []


    for p in range(1, 14):  # Para tamaños 2^1 a 2^13
        tam = 2**p
        x = np.random.uniform(-1, 1, tam).astype(np.complex128)
        h = np.random.uniform(-1, 1, tam).astype(np.complex128)
        # Extender las señales al tamaño actual (asegurando que no haya valores negativos)

        # SCC con Toeplitz
        ini = time.process_time()
        y_toeplitz = scc_toeplitz(x, h)
        #print(y_toeplitz)
        fin = time.process_time()
        tiempo_toeplitz.append(fin - ini)

        # SCC con FFT
        ini = time.process_time()
        X = fft_custom(x)
        H = fft_custom(h)
        Y = [X[k] * H[k] for k in range(len(X))]
        y_fft = ifft_custom(Y)
        y_fft = [val.real for val in y_fft]
        fin = time.process_time()
        tiempo_fft.append(fin - ini)

        # Calcular el error máximo absoluto
        # error_acum = np.max(np.abs(y_iterativa - y_fft))
        # margen_error.append(error_acum)

        tamaños.append(tam)
        print(f"Tamaño {tam}:  Toeplitz = {tiempo_toeplitz[-1]:.6f}s, FFT = {tiempo_fft[-1]:.6f}s") #, Error = {error_acum:.2e}")
        #Iterativa = {tiempo_iterativa[-1]:.6f}s,

    # Gráfico de comparación de tiempos de ejecución
    plt.figure(figsize=(10, 5))
    plt.plot(tamaños, tiempo_toeplitz, marker='s',color='red', label='SCC Toeplitz (O(N^2))')
    plt.plot(tamaños, tiempo_fft, marker='^',color='blue', label='FFT (O(N log N))')
    plt.xlabel('Tamaño de la señal (N)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de tiempos de ejecución')
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.xscale('log', base=2)  # Escala logarítmica para el eje x
    plt.ylim(0, max(max(tiempo_toeplitz), max(tiempo_fft)) * 1.1)  # Asegura que el eje y comience en 0
    plt.xlim(0, max(tamaños) * 1.1)  # Asegura que el eje x comience en 0
    plt.tight_layout()
    plt.show()

    # Gráfico de error máximo absoluto
    plt.figure(figsize=(10, 5))
    plt.plot(tamaños, margen_error, marker='o', color='red', label='Error máximo absoluto')
    plt.xlabel('Tamaño de la señal (N)')
    plt.ylabel('Error')
    plt.title('Error entre SCC iterativa y FFT')
    plt.legend()
    plt.grid()
    plt.xscale('log', base=2)  # Escala logarítmica para el eje x
    plt.ylim(0, max(margen_error) * 1.1)  # Asegura que el eje y comience en 0
    plt.xlim(0, max(tamaños) * 1.1)  # Asegura que el eje x comience en 0
    plt.show()

def menu():
    while True:
        print("\n--- Menú de Convolución ---")
        print("1. Comparar tiempos entre SCC iterativa, Toeplitz y FFT")
        print("2. Salir")
        opcion = input("Seleccione una opción (1-2): ")

        if opcion == "1":
            evaluar_rendimiento()
        elif opcion == "2":
            print("Programa finalizado.")
            break
        else:
            print("Opción inválida. Intente nuevamente.")

if __name__ == "__main__":
    menu()