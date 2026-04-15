# Descrição da Base de Dados – Vertebral Column Dataset

A base de dados **Vertebral Column** é um conjunto clínico utilizado para classificar condições da coluna vertebral a partir de medições biomecânicas.

Cada amostra possui **6 atributos numéricos**:

- pelvic incidence
- pelvic tilt
- lumbar lordosis angle
- sacral slope
- pelvic radius
- degree spondylolisthesis

---

## Versões da Base

### Base 2C (2 classes)

A versão **2C** contém duas classes:

- **AB (Abnormal)** → pacientes com patologias
- **NO (Normal)** → pacientes saudáveis

Problema de classificação binária.

---

### Base 3C (3 classes)

A versão **3C** possui três classes:

- **DH (Disk Hernia)** → hérnia de disco
- **SL (Spondylolisthesis)** → deslizamento vertebral
- **NO (Normal)** → pacientes saudáveis

Problema de classificação multiclasse.

---

# Seleção do Melhor Par de Atributos

Para escolher o melhor par de atributos, foi utilizado um critério baseado em **análise discriminante**.

A ideia é maximizar a separação entre classes e minimizar a dispersão dentro das classes.

---

## Matrizes utilizadas

### Dispersão intra-classe

S_w = Σ Σ (x - μ_c)(x - μ_c)^T

---

### Dispersão entre classes

S_b = Σ n_c (μ_c - μ)(μ_c - μ)^T

---

## Critério de escolha

J = trace(S_w^{-1} S_b)

Quanto maior J → melhor separação

---

# Métodos de Classificação

Foram utilizados três algoritmos:

- k-NN
- LDA
- QDA

Todos avaliados com Leave-One-Out (LOOCV).

---

## k-NN (k-Nearest Neighbors)

d(x, x_i) = sqrt(Σ (x_j - x_ij)^2)

ŷ = argmax_c Σ 1(y_i = c)

---

## LDA (Linear Discriminant Analysis)

g_c(x) = x^T Σ^{-1} μ_c - 1/2 μ_c^T Σ^{-1} μ_c + log P(c)

---

## QDA (Quadratic Discriminant Analysis)

g_c(x) = -1/2 log|Σ_c| - 1/2 (x - μ_c)^T Σ_c^{-1}(x - μ_c) + log P(c)

---

# Validação

Leave-One-Out (LOOCV):

- treina com N-1 amostras
- testa na restante

---

## Métrica

Accuracy = acertos / total

---

# Conclusão

- Melhor par escolhido maximiza separação entre classes
- k-NN depende da distância
- LDA gera fronteiras lineares
- QDA gera fronteiras não lineares
