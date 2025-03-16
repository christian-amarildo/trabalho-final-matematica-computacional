import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def funcao_rastrigin(lista_x):
    """
    Calcula o valor da função de Rastrigin para uma lista de entradas.

    Parâmetros:
        lista_x (list of float): Lista de valores de entrada para a função de Rastrigin.

    Retorna:
        float: O valor da função de Rastrigin calculado para a entrada fornecida.

    Observações:
        - Se a lista estiver vazia, a função retorna -1.
    """
    if not isinstance(lista_x, list):
        raise TypeError("A entrada deve ser uma lista de números.")

    if len(lista_x) == 0:
        return -1

    acumulador = 0
    for x in lista_x:
        if not isinstance(x, (int, float)):
            raise ValueError("Todos os elementos da lista devem ser números.")
        acumulador += x ** 2 - 10 * math.cos(2 * math.pi * x)

    y = 10 * len(lista_x) + acumulador
    return y

# Definir espaço de entrada
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)

# Calcular valores da função de Rastrigin
Z = np.array([[funcao_rastrigin([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])

# Criar gráfico 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plotar a superfície
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Personalizar o eixo Z
ax.set_zlim(np.min(Z), np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

# Adicionar barra de cores
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar o gráfico
plt.show()
