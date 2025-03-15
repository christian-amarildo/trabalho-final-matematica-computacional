import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def funcao_schaffer(lista_x):
    """
    Calcula o valor da função de Schaffer para uma lista de entradas.



    onde x e y são os valores da lista de entrada.

    Parâmetros:
        lista_x (list of float): Lista de dois valores de entrada [x, y].

    Retorna:
        float: O valor da função de Schaffer calculado para a entrada fornecida.

    Observações:
        - Se a lista não tiver exatamente dois elementos, a função gera um erro.
    """
    if not isinstance(lista_x, list):
        raise TypeError("A entrada deve ser uma lista de números.")

    if len(lista_x) != 2:
        raise ValueError("A lista deve conter exatamente dois elementos [x, y].")

    x, y = lista_x
    y_val = 0.5 + (math.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
    return y_val


# Definir espaço de entrada
x = np.linspace(-100, 100, 200)  # Reduzindo pontos para otimizar performance
y = np.linspace(-100, 100, 200)
X, Y = np.meshgrid(x, y)

# Calcular valores da função de Schaffer
Z = np.array([[funcao_schaffer([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])

# Criar gráfico 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
from matplotlib import cm

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



