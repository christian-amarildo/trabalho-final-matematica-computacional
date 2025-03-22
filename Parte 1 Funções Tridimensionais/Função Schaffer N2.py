import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
import imageio

# ----------------------------
# Função de Schaffer (entrada: lista [x, y])
# ----------------------------
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

    # Adiciona as partículas
    for p in particulas:
        z = funcao_objetivo(p)
        ax.scatter(p[0], p[1], z, color='black', s=30)

    # Adiciona barra de cores
    fig.colorbar(surf, shrink=0.5, aspect=8)

    # Salvar imagem com alta qualidade
    if nome_arquivo:
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')  # Aumenta a qualidade com dpi=300

    # Mostrar ou fechar
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# Algoritmo PSO - Particle Swarm Optimization
# ----------------------------
def pso(funcao_custo, dimensao=2, num_particulas=100, num_iteracoes=100, w=0.5, c1=1, c2=2,
        limites=(-100, 100), plot_interval=10, salvar_gif=True, mostrar_prints=True):

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
                              nome_arquivo=nome_arquivo, mostrar=True)
            imagens_geradas.append(nome_arquivo)

    # Criar GIF
    if salvar_gif and imagens_geradas:
        with imageio.get_writer("convergencia_pso.gif", mode="I", duration=2, loop=0) as writer:
            for filename in imagens_geradas:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Limpar imagens temporárias
        for filename in imagens_geradas:
            os.remove(filename)

        print("GIF salvo como 'convergencia_pso.gif'.")

    return melhor_global, melhor_fitness_global, historico_melhor, historico_pior

# ----------------------------
# Executar PSO
# ----------------------------
solucao, melhor_valor, historico_melhor, historico_pior = pso(
    funcao_custo=funcao_schaffer_np,
    num_iteracoes=100,
    plot_interval=10,
    salvar_gif=True,
    mostrar_prints=False  # ✅ Alterar para True se quiser ver os prints
)

print("\n== Resultado Final ==")
print("Melhor solução encontrada:", solucao)
print("Valor da função de Schaffer:", melhor_valor)

# ----------------------------
# Gráfico de convergência (melhor e pior indivíduo por geração)
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(historico_melhor, label="Melhor indivíduo", color="green")
plt.plot(historico_pior, label="Pior indivíduo", color="red")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do PSO - Função de Schaffer")
plt.legend()
plt.grid(True)
plt.show()
