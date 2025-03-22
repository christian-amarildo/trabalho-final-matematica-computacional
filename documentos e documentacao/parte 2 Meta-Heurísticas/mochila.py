# Passo 4 - Escolher o Problema
# O problema escolhido é:

# - Problema da Mochila (Knapsack Problem)
#   - Fácil de entender e implementar.
#   - Possui soluções heurísticas bem documentadas.

# Passo 5 - Escolher a Meta-heurística
# O problema escolhido é:

# - Algoritmo Genético (GA)
#   - Inspirado na seleção natural.
#   - Simples de codificar com operadores de mutação e crossover.

# Passo 6 - Implementação
# 1. Criar uma instância do problema da mochila com pesos e valores.
# 2. Implementar a solução usando o **Algoritmo Genético (GA)**.
# 3. Comparar a solução obtida com a solução exata (se possível).
# 4. Apresentar gráficos e visualizações.

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definição do problema
itens = [
    {"peso": 2, "valor": 3},
    {"peso": 3, "valor": 4},
    {"peso": 4, "valor": 5},
    {"peso": 5, "valor": 8},
    {"peso": 9, "valor": 10},
    {"peso": 6, "valor": 7},
    {"peso": 7, "valor": 9},
    {"peso": 8, "valor": 11},
    {"peso": 10, "valor": 13},
    {"peso": 12, "valor": 15}
]

capacidade_mochila = 30
# Populações maiores ajudam a manter diversidade e evitar soluções ruins.
tamanho_populacao = 100
# Se a mutação for alta, a busca fica mais exploratória e pode gerar mais variabilidade.
taxa_mutacao = 0.8
# Mais iterações permitem que o algoritmo refine melhor as soluções.
geracoes = 40

# Função para criar um indivíduo (solução)


def criar_individuo():
    return [random.randint(0, 1) for _ in range(len(itens))]

# Função para calcular o fitness


def calcular_fitness(individuo):
    peso_total = sum(itens[i]["peso"] * individuo[i]
                     for i in range(len(itens)))
    valor_total = sum(itens[i]["valor"] * individuo[i]
                      for i in range(len(itens)))

    if peso_total > capacidade_mochila:
        return 0  # Penalização para soluções inválidas

    return valor_total


historico_escolhas = np.zeros((geracoes, len(itens)))

# Inicialização da população
populacao = [criar_individuo() for _ in range(tamanho_populacao)]

# Evolução da população
melhores_fitness = []

for geracao in range(geracoes):
    # Avaliação do fitness da população
    fitness = [calcular_fitness(individuo) for individuo in populacao]

    # Armazena o melhor fitness da geração
    melhores_fitness.append(max(fitness))

    for i in range(len(itens)):
        historico_escolhas[geracao, i] = sum(
            individuo[i] for individuo in populacao)

    # Seleção por torneio
    nova_populacao = []
    for _ in range(tamanho_populacao // 2):
        candidatos = random.sample(populacao, 3)
        melhor_pai = max(candidatos, key=calcular_fitness)
        nova_populacao.append(melhor_pai)

    # Crossover (recombinação)
    filhos = []
    for i in range(0, len(nova_populacao) - 1, 2):
        ponto_corte = random.randint(1, len(itens) - 1)
        pai1, pai2 = nova_populacao[i], nova_populacao[i + 1]
        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
        filhos.extend([filho1, filho2])

    # Mutação
    for filho in filhos:
        if random.random() < taxa_mutacao:
            gene = random.randint(0, len(itens) - 1)
            filho[gene] = 1 - filho[gene]  # Alterna entre 0 e 1

    # Atualiza a população
    # populacao = filhos
    # Mantém os melhores indivíduos da geração anterior
    num_elitismo = 2  # Número de indivíduos mantidos
    melhores_individuos = sorted(populacao, key=calcular_fitness, reverse=True)[
        :num_elitismo]

    # Atualiza a população com os melhores + novos filhos
    populacao = melhores_individuos + filhos[:tamanho_populacao - num_elitismo]


# Melhor solução encontrada
melhor_individuo = max(populacao, key=calcular_fitness)
melhor_fitness = calcular_fitness(melhor_individuo)

print("Melhor solução encontrada:", melhor_individuo)
print("Valor total da mochila:", melhor_fitness)
print("Peso total da mochila:", sum(itens[i]["peso"] * melhor_individuo[i]
                                    for i in range(len(itens))))

# Gráfico da evolução do fitness
plt.plot(melhores_fitness)
plt.xlabel("Geração")
plt.ylabel("Melhor Fitness")
plt.title("Evolução do Algoritmo Genético")
plt.show()


# Criando Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(historico_escolhas.T, cmap="YlGnBu", xticklabels=range(
    geracoes), yticklabels=range(len(itens)))
plt.xlabel("Geração")
plt.ylabel("Índice do Item")
plt.title("Heatmap da Frequência de Escolha dos Itens")
plt.show()
