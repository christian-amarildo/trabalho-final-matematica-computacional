import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de Rastrigin
def rastrigin(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Algoritmo PSO (Particle Swarm Optimization)
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Inicializa as partículas e velocidades
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Inicializa as melhores posições e valores de fitness
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Itera pelas iterações atualizando a posição e a velocidade das partículas
    for i in range(max_iter):
        # Atualiza as velocidades
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Atualiza as posições
        particles += velocities

        # Avalia o fitness de cada partícula
        fitness_values = np.array([cost_func(p) for p in particles])

        # Atualiza as melhores posições e valores de fitness
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Retorna a melhor solução encontrada pelo PSO
    return swarm_best_position, swarm_best_fitness

# Dimensão do problema
dim = 2

# Rodar o algoritmo PSO na função de Rastrigin
solution, fitness = pso(rastrigin, dim=dim)

# Imprimir a solução e o valor de fitness
print('Solução:', solution)
print('Fitness:', fitness)

# Criar um meshgrid para visualização
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

# Criar um gráfico 3D da função de Rastrigin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plotar a solução encontrada pelo PSO
ax.scatter(solution[0], solution[1], fitness, color='red', label='Melhor solução PSO')
ax.legend()

# Exibir o gráfico
plt.show()
