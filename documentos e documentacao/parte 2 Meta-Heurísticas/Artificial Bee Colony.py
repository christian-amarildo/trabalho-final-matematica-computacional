import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Função de Rastrigin (a ser otimizada)
def rastrigin(x):
    n = len(x)
    return 10 * n + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

# Algoritmo ABC (Artificial Bee Colony)
class ABC:
    def __init__(self, cost_func, num_bees=30, max_iter=100, dim=2, lim_inf=-5.12, lim_sup=5.12):
        self.cost_func = cost_func
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.dim = dim
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup
        self.food_sources = np.random.uniform(lim_inf, lim_sup, (num_bees, dim))
        self.fitness_values = np.array([cost_func(self.food_sources[i]) for i in range(num_bees)])
        self.best_position = self.food_sources[np.argmin(self.fitness_values)]
        self.best_fitness = np.min(self.fitness_values)

    def optimize(self):
        # Loop de iteração
        for iteration in range(self.max_iter):
            # Fase de abelhas funcionárias
            for i in range(self.num_bees):
                # Escolhe uma nova posição para a abelha i
                new_position = self.food_sources[i] + np.random.uniform(-1, 1, self.dim) * (self.food_sources[i] - self.best_position)
                new_position = np.clip(new_position, self.lim_inf, self.lim_sup)
                
                # Avalia a nova posição
                new_fitness = self.cost_func(new_position)
                
                # Se a nova posição for melhor, atualiza
                if new_fitness < self.fitness_values[i]:
                    self.food_sources[i] = new_position
                    self.fitness_values[i] = new_fitness
            
            # Fase de abelhas observadoras
            for i in range(self.num_bees):
                # Escolhe a posição mais promissora
                prob = self.fitness_values / np.sum(self.fitness_values)
                selected_bee = np.random.choice(range(self.num_bees), p=prob)
                # Faz um movimento em direção à posição selecionada
                new_position = self.food_sources[selected_bee] + np.random.uniform(-1, 1, self.dim) * (self.food_sources[selected_bee] - self.best_position)
                new_position = np.clip(new_position, self.lim_inf, self.lim_sup)
                
                # Avalia a nova posição
                new_fitness = self.cost_func(new_position)
                
                # Se a nova posição for melhor, atualiza
                if new_fitness < self.fitness_values[selected_bee]:
                    self.food_sources[selected_bee] = new_position
                    self.fitness_values[selected_bee] = new_fitness
            
            # Atualiza a melhor posição encontrada
            if np.min(self.fitness_values) < self.best_fitness:
                self.best_fitness = np.min(self.fitness_values)
                self.best_position = self.food_sources[np.argmin(self.fitness_values)]
            
        return self.best_position, self.best_fitness

# Parâmetros para o algoritmo ABC
dim = 2
num_bees = 30
max_iter = 100
lim_inf = -5.12
lim_sup = 5.12

# Rodando o ABC para encontrar a solução da função Rastrigin
abc_optimizer = ABC(rastrigin, num_bees=num_bees, max_iter=max_iter, dim=dim, lim_inf=lim_inf, lim_sup=lim_sup)
solution_abc, fitness_abc = abc_optimizer.optimize()

# Exibindo o resultado
print("Melhor solução encontrada pelo ABC:", solution_abc)
print("Valor da função (fitness) na solução:", fitness_abc)

# Criar um meshgrid para visualização da função
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

# Criar um gráfico 3D da função de Rastrigin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plotar a solução encontrada pelo ABC
ax.scatter(solution_abc[0], solution_abc[1], fitness_abc, color='red', label='Solução do ABC')
ax.legend()

plt.show()
