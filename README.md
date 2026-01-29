# SAVI — Deteção e Classificação de Dígitos Manuscritos

# Pascoal Mandinge Gime Sumbo 
# 123190

Este projeto implementa um pipeline incremental de **Visão Computacional e Deep Learning** aplicado ao problema da **classificação e deteção de dígitos manuscritos**, evoluindo desde o dataset MNIST clássico até imagens sintéticas contendo múltiplos dígitos.

O trabalho está organizado em quatro tarefas principais:

- **Task 1**: Treino e avaliação de um classificador CNN no dataset MNIST completo, incluindo métricas avançadas como matriz de confusão, precisão, revocação e F1-score.
- **Task 2**: Geração de um dataset sintético de imagens contendo múltiplos dígitos, com respetivas *bounding boxes* (ground truth), inspirado em ferramentas de deteção de objetos.
- **Task 3**: Implementação de uma abordagem baseline de deteção baseada em *sliding window*, reutilizando o classificador treinado na Task 1.
- **Task 4**: Abordagem melhorada de deteção, recorrendo ao re-treino da rede com uma classe adicional de *background*, permitindo reduzir falsos positivos e melhorar a robustez do sistema.

## Task 1 — Classificação de Dígitos Manuscritos (MNIST)

A Task 1 teve como objetivo o desenvolvimento, treino e avaliação de um classificador de dígitos manuscritos utilizando o dataset MNIST. Esta tarefa constitui a base de todo o projeto, uma vez que o modelo aqui treinado é posteriormente reutilizado e adaptado nas tarefas de deteção.

---

### Dataset

Foi utilizado o dataset **MNIST**, amplamente adotado como benchmark em problemas de classificação de dígitos manuscritos. O dataset é composto por:

- 60 000 imagens para treino  
- 10 000 imagens para teste  
- imagens em tons de cinzento com resolução de 28×28 pixels  
- 10 classes correspondentes aos dígitos de 0 a 9  

O dataset é descarregado automaticamente através da biblioteca `torchvision`, não sendo incluído no repositório, o que garante a reprodutibilidade dos resultados.

---

### Arquitetura do Modelo

O classificador desenvolvido baseia-se numa **Rede Neuronal Convolucional (CNN)** otimizada para o reconhecimento de dígitos. A arquitetura inclui:

- camadas convolucionais com ativação ReLU  
- normalização por *Batch Normalization* para estabilização do treino  
- camadas de *Max Pooling* para redução da dimensionalidade espacial  
- *Dropout* para mitigação de *overfitting*  
- camadas totalmente ligadas para a classificação final  

Esta arquitetura foi escolhida de forma a equilibrar desempenho e custo computacional, permitindo uma boa generalização no dataset MNIST e servindo como classificador base para as tarefas seguintes.

---

### Treino

O treino do modelo foi realizado com as seguintes configurações principais:

- função de perda: **Cross-Entropy Loss**  
- otimizador: **Adam**  
- divisão do conjunto de treino em treino e validação  
- seleção automática do melhor modelo com base na *accuracy* de validação  

Os principais hiperparâmetros, como número de épocas, *learning rate*, *batch size* e *dropout*, são configuráveis através de argumentos de linha de comandos.

---

### Avaliação e Resultados

A avaliação do modelo foi realizada no conjunto de teste do MNIST, recorrendo a métricas quantitativas e qualitativas, nomeadamente:

- *Accuracy*  
- matriz de confusão  
- *Precision*, *Recall* e *F1-score* (médias macro)  

#### Resultados Quantitativos

| Métrica | Valor |
|------|------|
| Accuracy (teste) | 0.98 |
| Precision (macro) | 0.98 |
| Recall (macro) | 0.98 |
| F1-score (macro) | 0.98 |

Os resultados obtidos demonstram uma elevada capacidade de generalização do classificador, com erros residuais maioritariamente associados a dígitos visualmente semelhantes.

---

#### Matriz de Confusão

A figura seguinte apresenta a matriz de confusão obtida no conjunto de teste do MNIST. Observa-se uma forte concentração dos valores na diagonal principal, indicando um elevado desempenho do classificador em todas as classes.

![Matriz de Confusão — Task 1](assets/task1_confusion_matrix.png)

---

O modelo treinado nesta tarefa é reutilizado diretamente na **Task 3** como classificador base e serve como ponto de partida conceptual para a abordagem melhorada desenvolvida na **Task 4**.
