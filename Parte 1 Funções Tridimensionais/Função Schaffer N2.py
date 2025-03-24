import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
from PIL import Image


def funcao_schaffer_np(lista_x):
    if len(lista_x) != 2:
        raise ValueError("A entrada deve ter exatamente dois elementos [x, y].")
    x, y = lista_x
    numerador = (math.sin(x**2 - y**2))**2 - 0.5
    denominador = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerador / denominador

# ----------------------------
# Função para desenhar o gráfico com os indivíduos e salvar imagem
# ----------------------------
def plotar_particulas(particulas, funcao_objetivo, titulo, limites, nome_arquivo=None, mostrar=True):
    x = np.linspace(limites[0], limites[1], 200)
    y = np.linspace(limites[0], limites[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 + (np.sin(X**2 - Y**2)**2 - 0.5) / (1 + 0.001 * (X**2 + Y**2))**2

    fig = plt.figure(figsize=(12, 8))  # Aumenta o tamanho da figura
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.7)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.set_title(titulo, fontsize=14)
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.zaxis.set_major_locator(LinearLocator(10))

    for p in particulas:
        z = funcao_objetivo(p)
        ax.scatter(p[0], p[1], z, color='black', s=30)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    if nome_arquivo:
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')  # Aumenta a qualidade com dpi=300

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

# Algoritmo PSO - Particle Swarm Optimization

def pso(funcao_custo, dimensao=2, num_particulas=100, num_iteracoes=100, w=0.5, c1=1, c2=2,
        limites=(-100, 100), plot_interval=1, salvar_gif=True, mostrar_prints=True):

    limite_inferior, limite_superior = limites
    historico_melhor = []
    historico_pior = []
    imagens_geradas = []

    particulas = np.random.uniform(limite_inferior, limite_superior, (num_particulas, dimensao))
    velocidades = np.zeros((num_particulas, dimensao))
    melhores_posicoes = np.copy(particulas)
    melhores_fitness = np.array([funcao_custo(p) for p in particulas])
    indice_melhor = np.argmin(melhores_fitness)
    melhor_global = melhores_posicoes[indice_melhor]
    melhor_fitness_global = melhores_fitness[indice_melhor]

    for geracao in range(num_iteracoes):
        for i in range(num_particulas):
            for d in range(dimensao):
                r1 = np.random.rand()
                r2 = np.random.rand()
                debug = velocidades[i, d]
                debug = melhores_posicoes[i, d]
                debug = particulas[i, d]
                debug = melhor_global[d]
                velocidades[i, d] = (
                    w * velocidades[i, d]
                    + c1 * r1 * (melhores_posicoes[i, d] - particulas[i, d])
                    + c2 * r2 * (melhor_global[d] - particulas[i, d])
                )
                particulas[i, d] += velocidades[i, d]
                particulas[i, d] = np.clip(particulas[i, d], limite_inferior, limite_superior)

        fitness_atual = np.array([funcao_custo(p) for p in particulas])
        for i in range(num_particulas):
            if fitness_atual[i] < melhores_fitness[i]:
                melhores_posicoes[i] = particulas[i]
                melhores_fitness[i] = fitness_atual[i]

        indice_melhor = np.argmin(melhores_fitness)
        if melhores_fitness[indice_melhor] < melhor_fitness_global:
            melhor_global = melhores_posicoes[indice_melhor]
            melhor_fitness_global = melhores_fitness[indice_melhor]

        indice_pior = np.argmax(fitness_atual)
        pior_fitness = fitness_atual[indice_pior]

        historico_melhor.append(melhor_fitness_global)
        historico_pior.append(pior_fitness)

        if mostrar_prints:
            print(f"[Geração {geracao + 1}]")
            print(f"  Melhor indivíduo: posição = {melhor_global}, fitness = {melhor_fitness_global:.6f}")
            print(f"  Pior  indivíduo: posição = {particulas[indice_pior]}, fitness = {pior_fitness:.6f}")
            print("-" * 60)

        if (geracao + 1) % plot_interval == 0 or (geracao + 1) == num_iteracoes:
            nome_arquivo = f"frame_gen_{geracao + 1:03d}.png"
            plotar_particulas(particulas, funcao_custo,
                              titulo=f"Geração {geracao + 1}", limites=limites,
                              nome_arquivo=nome_arquivo, mostrar=False)
            imagens_geradas.append(nome_arquivo)

    # Criar GIF após todas as iterações
    if imagens_geradas:
        frames = []
        for filename in imagens_geradas:
            img = Image.open(filename)  # Abrir cada imagem gerada
            frames.append(img)

        # Salvar as imagens como um GIF com a duração e loop definidos
        frames[0].save('convergencia_pso.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Limpar imagens temporárias
        for filename in imagens_geradas:
            os.remove(filename)

        print("GIF salvo como 'convergencia_pso.gif'.")

    return melhor_global, melhor_fitness_global, historico_melhor, historico_pior

# ----------------------------
# Algoritmo ABC - Artificial Bee Colony
# ----------------------------
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

    def optimize(self, plot_interval=1):
        historico_melhor = []
        historico_pior = []
        imagens_geradas = []

        for iteration in range(self.max_iter):
            # Fase de abelhas funcionárias
            for i in range(self.num_bees):
                new_position = self.food_sources[i] + np.random.uniform(-1, 1, self.dim) * (self.food_sources[i] - self.best_position)
                new_position = np.clip(new_position, self.lim_inf, self.lim_sup)
                
                new_fitness = self.cost_func(new_position)
                if new_fitness < self.fitness_values[i]:
                    self.food_sources[i] = new_position
                    self.fitness_values[i] = new_fitness
            
            # Fase de abelhas observadoras
            for i in range(self.num_bees):
                prob = self.fitness_values / np.sum(self.fitness_values)
                selected_bee = np.random.choice(range(self.num_bees), p=prob)
                new_position = self.food_sources[selected_bee] + np.random.uniform(-1, 1, self.dim) * (self.food_sources[selected_bee] - self.best_position)
                new_position = np.clip(new_position, self.lim_inf, self.lim_sup)
                
                new_fitness = self.cost_func(new_position)
                if new_fitness < self.fitness_values[selected_bee]:
                    self.food_sources[selected_bee] = new_position
                    self.fitness_values[selected_bee] = new_fitness
            
            # Atualiza a melhor posição
            if np.min(self.fitness_values) < self.best_fitness:
                self.best_fitness = np.min(self.fitness_values)
                self.best_position = self.food_sources[np.argmin(self.fitness_values)]

            historico_melhor.append(self.best_fitness)
            historico_pior.append(np.max(self.fitness_values))

            # Plotar as partículas a cada plot_interval gerações
            if (iteration + 1) % plot_interval == 0 or (iteration + 1) == self.max_iter:
                nome_arquivo = f"frame_gen_abc_{iteration + 1:03d}.png"
                plotar_particulas(self.food_sources, self.cost_func,
                                  titulo=f"Geração ABC {iteration + 1}", limites=(self.lim_inf, self.lim_sup),
                                  nome_arquivo=nome_arquivo, mostrar=False)
                imagens_geradas.append(nome_arquivo)

        # Criar GIF após todas as iterações
        if imagens_geradas:
            frames = []
            for filename in imagens_geradas:
                img = Image.open(filename)  # Abrir cada imagem gerada
                frames.append(img)

            # Salvar as imagens como um GIF com a duração e loop definidos
            frames[0].save('convergencia_abc.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)

            # Limpar imagens temporárias
            for filename in imagens_geradas:
                os.remove(filename)

            print("GIF salvo como 'convergencia_abc.gif'.")

        return self.best_position, self.best_fitness, historico_melhor, historico_pior
# ----------------------------
# Função para escolher entre PSO e ABC
# ----------------------------
def otimizar_por_algoritmo(algoritmo='PSO'):
    if algoritmo == 'PSO':
        solucao, melhor_valor, historico_melhor, historico_pior = pso(
            funcao_custo=funcao_schaffer_np,
            num_iteracoes=100,
            plot_interval=1,
            salvar_gif=True,
            mostrar_prints=False
        )
    elif algoritmo == 'ABC':
        abc_optimizer = ABC(funcao_schaffer_np, num_bees=30, max_iter=100, dim=2, lim_inf=-100, lim_sup=100)
        solucao, melhor_valor, historico_melhor, historico_pior = abc_optimizer.optimize()
    else:
        raise ValueError("Algoritmo não reconhecido. Escolha 'PSO' ou 'ABC'.")

    return solucao, melhor_valor

# ----------------------------
# Rodar o algoritmo escolhido
# ----------------------------
algoritmo_escolhido = 'PSO'  # Alterar para 'ABC' para usar o ABC
solucao, melhor_valor = otimizar_por_algoritmo(algoritmo=algoritmo_escolhido)

print("\n== Resultado Final ==")
print("Melhor solução encontrada:", solucao)
print("Valor da função de Schaffer:", melhor_valor)
