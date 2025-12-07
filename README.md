# **_RecycleAI: Classificação Automática de Contentores de Lixo_**

## Descrição do Projeto

Este projeto tem como objetivo desenvolver um sistema automático de classificação de contentores de lixo a partir de imagens, distinguindo entre 7 tipos diferentes de contentores: indiferenciado, papel, plástico, vidro, orgânico, pilhas e óleo. Para isso, implementamos e avaliamos modelos de deep learning e técnicas de visão computacional, com foco em pré-processamento de dados e aumento de imagens ed forma a melhorar a classificação final.

O trabalho permite:
 - Reconhecer automaticamente diferentes tipos de contentores de lixo.
 - Avaliar o desempenho do classificador usando métricas como accuracy, matriz de confusão, precision, recall e F1-score.
 - Explorar técnicas de pré-processamento e aumento de dados para melhorar a generalização do modelo.

## 🗑️ Classes de Contentores

O sistema é treinado para classificar imagens em 7 categorias distintas de contentores, representadas da seguinte forma nos dados:

| Nome Interno | Tipo de Resíduo                   |
| :--- |:----------------------------------|
| `container_battery` | **Pilhas**                        |
| `container_biodegradable` | **Orgânico**                      |
| `container_blue` | **Papel**                 |
| `container_default` | **Indiferenciado**                |
| `container_green` | **Vidro**                         |
| `container_oil` | **Óleo**             |
| `container_yellow` | **Plástico** |

---

## 🤖 Modelos Avaliados

O projeto avaliou o desempenho de diferentes arquiteturas de Deep Learning para a tarefa de classificação, identificadas na seguinte tabela:

| Modelo | Arquitetura                        |
| :--- |:-----------------------------------|
| **ResNet50** | Residual Network                   |
| **MobileNetV2** | Otimizada para dispositivos móveis |
| **DenseNet121** | Dense Convolutional Network        |

---

## 🚀 Avaliação

Para **avaliar** um modelo treinado e/ou **testá-lo** para gerar previsões em um novo conjunto de dados:

1.  O modelo com o melhor desempenho de classificação (`.weights.h5`) para ser utilizado nas previsões e testes encontra-se em:
    
    `Teste_Modelo/modelo_contentor.weights.h5`
    
2.  Para realizar a avaliação (num conjunto de imagens que estejam divididas pelas suas respetivas pastas de classes), utilize o seguinte script:
    
    ```bash
    python Teste_Modelo/script_avaliacao.py
    ```

---

## Elaborado por:

![Universidade](https://img.shields.io/badge/Universidade%20da%20Beira%20Interior-1E90FF?style=for-the-badge)  
![Curso](https://img.shields.io/badge/Curso-Intelig%C3%AAncia%20Artificial%20e%20Ci%C3%AAncia%20de%20Dados-1E90FF?style=for-the-badge)  
![Disciplina](https://img.shields.io/badge/Disciplina-Processamento%20de%20Dados%20Audovisuais-1E90FF?style=for-the-badge)  
![Aluno](https://img.shields.io/badge/Tiago%20Miguel%20Fernandes%20Marques-51653-1E90FF?style=for-the-badge)

### 💻 Tecnologia Utilizada:

[![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=PyCharm&logoColor=white)](https://www.jetbrains.com/pycharm/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
