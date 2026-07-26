# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Eduardo Rodrigues Do Nascimento

### 1️⃣ Resumo da Arquitetura do Modelo

#### Arquitetura da CNN:

A Rede Neural Convolucional implementada em `train_model.py` segue uma estrutura de aprofundamento progressivo, dividida em quatro blocos convolucionais e uma etapa final de classificação:

- **Bloco 1:** `Conv2D` com 32 filtros (3x3), seguida de `BatchNormalization` para estabilizar as distribuições internas e acelerar a convergência, finalizado com `MaxPooling2D` para redução da dimensionalidade espacial.
- **Bloco 2:** `Conv2D` com 64 filtros, novamente estabilizada com `BatchNormalization` e reduzida por `MaxPooling2D`.
- **Bloco 3:** `Conv2D` com 128 filtros, acompanhada de `BatchNormalization` e `MaxPooling2D`.
- **Bloco 4:** `Conv2D` com 128 filtros, repetindo o padrão de normalização e pooling, atingindo o nível mais profundo de extração de características.
- **Transição e Classificação Final:** os mapas de características são achatados (`Flatten`) e alimentam um classificador totalmente conectado, iniciando com uma camada `Dense` de 128 neurônios (ReLU), seguida de `Dropout` (taxa de 0.5) para regularização contra overfitting, e finalizada pela camada de saída `Dense` com 10 neurônios e ativação `softmax`, correspondendo à distribuição de probabilidades das 10 classes de dígitos (0-9).

#### Estratégia de validação e early stopping:

A separação treino/validação foi feita via `validation_split=0.1` no próprio `model.fit()`, reservando 10% dos dados de treino para validação a cada época. O treinamento utilizou `EarlyStopping` monitorando a `val_loss`, com paciência de 3 épocas e restauração automática dos melhores pesos (`restore_best_weights=True`). Na execução, o treinamento convergiu e parou automaticamente antes de atingir o limite de 15 épocas, evidenciando que o modelo já havia estabilizado sem sinais de overfitting.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow / Keras** (versão 2.21.0): framework principal do projeto. O módulo `tensorflow.keras` foi utilizado para a construção da arquitetura convolucional, carregamento do dataset MNIST e configuração do treinamento (otimizador Adam e callback de `EarlyStopping`). O módulo `tensorflow.lite` foi utilizado para a conversão, quantização e execução do modelo em formato de edge (TFLite).
- **NumPy** (versão 2.5.1): utilizada na etapa de inferência para manipulação de arrays multidimensionais, redimensionamento das amostras (`np.expand_dims`) e extração da classe de maior probabilidade a partir da saída do modelo (`np.argmax`).

### 3️⃣ Técnica de Otimização do Modelo

A técnica escolhida foi a **Quantização de Faixa Dinâmica (Dynamic Range Quantization)**, habilitada através do parâmetro `tf.lite.Optimize.DEFAULT` no `TFLiteConverter`.

**Como funciona:** durante o treinamento, os pesos e vieses da rede são armazenados em ponto flutuante de 32 bits (`float32`). A quantização atua *pós-treinamento*, comprimindo essas matrizes de pesos estáticos de `float32` para inteiros de 8 bits (`int8`), enquanto as ativações permanecem em ponto flutuante durante a inferência. Por não exigir um dataset representativo de calibração, é uma técnica simples de aplicar e eficaz para reduzir o tamanho do modelo com impacto mínimo na acurácia.

**Benefícios observados:**
- **Redução drástica de tamanho:** a conversão de 32 para 8 bits reduziu o artefato final em mais de 90%, fundamental para armazenamento em dispositivos com memória restrita.
- **Eficiência computacional:** operações com inteiros exigem menos ciclos de CPU que operações em ponto flutuante, resultando em inferência mais rápida — relevante para cenários de Edge AI.
- **Manutenção da acurácia:** o modelo quantizado manteve 100% de acerto nas amostras testadas na etapa de inferência, confirmando que a degradação de precisão foi imperceptível nesse caso.

### 4️⃣ Resultados Obtidos

**Acurácia de validação:** o modelo atingiu **99.27%** de acurácia de validação final, com **99.16%** de acurácia no conjunto de teste — resultado consistente com o esperado para uma CNN de 4 blocos convolucionais aplicada ao MNIST.

**Comparativo de tamanho dos arquivos:**

| Arquivo | Tamanho |
|---|---|
| `model.h5` (Keras original) | 3700.6 KB (~3.6 MB) |
| `model.tflite` (otimizado para Edge) | 318.8 KB |
| **Redução obtida** | **91.4%** |

A quantização dos pesos de `float32` para `int8` cumpriu seu papel de compressão, tornando o modelo consideravelmente mais leve sem comprometer a capacidade preditiva observada nas inferências locais.

### 5️⃣ Comentários Adicionais

**Decisões técnicas importantes:** optou-se por uma arquitetura com 4 blocos convolucionais (no limite superior do intervalo sugerido de 3-4), já que o MNIST é um dataset relativamente simples e a rede convergiu rapidamente sem overfitting, mesmo com essa profundidade adicional. O uso de `BatchNormalization` após cada `Conv2D` contribuiu para uma convergência mais estável e rápida (early stopping ativado bem antes do limite de 15 épocas).

**Dificuldades encontradas:** a principal restrição foi o treinamento exclusivo em CPU, o que tornou cada época mais lenta que em ambiente com GPU — ainda assim, o tempo total de treinamento permaneceu na casa de poucos minutos, dado o tamanho reduzido das imagens (28x28, 1 canal) e a convergência rápida via early stopping.

**Aprendizados durante o desafio:** o principal aprendizado foi a compreensão prática do fluxo completo de Edge AI — construir e treinar a rede é apenas a primeira etapa; conversão e quantização exigem compromissos de engenharia entre tamanho, velocidade e precisão. A etapa de `TFLiteConverter` evidenciou de forma concreta como a quantização de pesos permite levar modelos robustos para dispositivos com recursos computacionais limitados, mantendo a integridade preditiva da rede.

### 6️⃣ Exemplo de Inferência

#### Saída do terminal:

```
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```
aa