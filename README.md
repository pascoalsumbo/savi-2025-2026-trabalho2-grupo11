# SAVI — Deteção e Classificação de Dígitos Manuscritos

# Pascoal Mandinge Gime Sumbo 
# 123190

Este projeto implementa um pipeline incremental de **Visão Computacional e Deep Learning** aplicado ao problema da **classificação e deteção de dígitos manuscritos**, evoluindo desde o dataset MNIST clássico até imagens sintéticas contendo múltiplos dígitos.

O trabalho está organizado em quatro tarefas principais:

- **Task 1**: Treino e avaliação de um classificador CNN no dataset MNIST completo, incluindo métricas avançadas como matriz de confusão, precisão, revocação e F1-score.
- **Task 2**: Geração de um dataset sintético de imagens contendo múltiplos dígitos, com respetivas *bounding boxes* (ground truth), inspirado em ferramentas de deteção de objetos.
- **Task 3**: Implementação de uma abordagem baseline de deteção baseada em *sliding window*, reutilizando o classificador treinado na Task 1.
- **Task 4**: Abordagem melhorada de deteção, recorrendo ao re-treino da rede com uma classe adicional de *background*, permitindo reduzir falsos positivos e melhorar a robustez do sistema.
