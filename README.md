<table>
<tr>
<td>
<a href= "https://www.sette.ag/"><img src="./artefatos/img/sette.jpg" alt="Sette" border="0" width="100%"></a>
</td>
<td><a href= "https://www.inteli.edu.br/"><img src="./inteli-logo.png" alt="Inteli - Instituto de Tecnologia e Liderança" border="0" width="30%"></a>
</td>
</tr>
</table>

# Projeto: Visão Computacional Aplicada na Segmentação Binária de Talhões em Áreas Agrícolas

# Grupo: T de Talhão

# Integrantes:

* André Junior <andre.junior@sou.inteli.edu.br>
* Arthur Reis <arthur.reis@sou.inteli.edu.br>
* Jonas Sales <jonas.sales@sou.inteli.edu.br>
* Mateus Rafael <mateus.silva@sou.inteli.edu.br>
* Melyssa Rojas <melyssa.rojas@sou.inteli.edu.br>
* Yasmin Vitória <yasmin.jesus@sou.inteli.edu.br>

# Descrição

O projeto desenvolvido trata-se de um modelo de visão computacional para identificar talhões na região sul do Brasil, mitigando a problemática da identificação irregular dessas áreas agrícolas. O modelo utiliza a tarefa de segmentação, treinando e testando com imagens sintéticas e do satélite Sentinel-2. A saída é uma máscara binária, focada nos limites das bordas em áreas agrícolas. O projeto é realizado em parceria com a Sette, uma startup de agro, que busca aumentar a precisão das segmentações através desse modelo e, assim, melhorar a assertividade de suas análises espaciais em áreas agrícolas.

# Configuração para desenvolvimento

Diante da perspectiva dos modelos, para se usufruir deste é necessário seguir os passos abaixo:

**1° passo:** Caso deseje executar o modelo que utiliza imagens sintéticas, vá até a pasta do modelo e carregue o arquivo "unet-imagens-sinteticas.ipynb":

```
\codigo\modelo\unet-imagens-sinteticas.ipynb
```

**2° passo:** Abra o modelo em qualquer plataforma da sua escolha, por exemplo, o Google Colab;

**3° passo:** Antes de rodar as células garanta que tenha estes arquivos:

O arquivo "requirements" dentro deste repositório:

```
\codigo\modelo\requirements.txt
```

Um dataset de imagens agrícolas de extensão .PNG com tamanho mínimo de 200x200 cada dentro de uma pasta denominada "sem_mask". 

Além disso, outro dataset de labels correspondentes com as imagens implantadas na pasta "sem_mask" de extensão .PNG dentro de uma pasta denominada "mask_borda".

Contudo, com estes arquivos, faça upload no seu próprio drive.

**4° passo:** Dentro dessas variáveis globais do modelo, modifique pelos caminhos que correspondem aos artefatos implementandos no passo 3:

```
REQUIREMENTS = "caminho_do_arquivo_requirements"
```

```
PATH_IMAGES_TRAIN_VAL = "caminho_da_pasta_sem_mask"
```

```
PATH_IMAGES_TRAIN_GROUND_TRUTH = "caminho_da_pasta_mask_borda"
```

**5° passo:** As células do modelo estão divididas em seções, com isso, rode as células seguindo a ordem imposta.

No entanto, como observação, caso houver a utilização do modelo "unet-imagens-sentinel2.ipynb" o modo como as variáveis globais vão ser tratadas vai ser diferente, as imagens utilizadas serão do satélite Sentinel-2 no tamanho mínimo de 200x200 com 12 bandas separadas entre máscaras e máscaras binárias. O preenchimento das váriaveis globais serão essas:

```
PATH_IMAGES_DIR = "caminho_das_mascaras"
```

```
PATH_LABELS_DIR = "caminho_das_mascaras_binarias"
```

Outrora, também será necessário colocar o arquivo "augmentation.py" (contém todos métodos necessários para o Data Augmentation de imagens por satélite), presente no caminho:

```
\codigo\modelo\augmentation.py
```

Portanto, o arquivo do modelo contém complementariedade das instruções de forma aprofundada e o artigo presente em "artefatos" orientações da metodologia implementada na construção do modelo com descrição de suas configurações.

# Tags

- [SPRINT 1](https://github.com/Inteli-College/2024-1B-T01-CC10-G03/releases/tag/Sprint1)

    - Introdução com definição de problema relevante e objetivo do trabalho;

    - Trabalhos Relacionados com análise da literatura existente;

    - Materiais e Métodos, contemplando a aquisição, processamento e manipulação de imagens para montagem da base de dados;

    - Descrição da fonte de dados com justificativa para escolha da base de dados;

    - Pipeline de processamento e preparação de dados de imagens para treinamento;

    - Base de dados de imagens processadas com uma análise exploratória sobre a base obtida.

- [SPRINT 2](https://github.com/Inteli-College/2024-1B-T01-CC10-G03/releases/tag/Sprint2)

    - Descrição dos modelos CNN no artigo com referências;

    - Justificativa da métrica utilizada no artigo;

    - Visualização da performance e desempenho no artigo;

    - Implementação de modelo CNN próprio;

    - Refinamento de modelo CNN pré-treinado;

- [SPRINT 3](https://github.com/Inteli-College/2024-1B-T01-CC10-G03/releases/tag/Sprint3)

    - Refinamento do Data Augmentation;
  
    - Refinamento do modelo CNN;
  
    - Descrição do processamento utilizado nas imagens no artigo;

    - Refinamento de todas as seções do artigo;

- [SPRINT 4](https://github.com/Inteli-College/2024-1B-T01-CC10-G03/releases/tag/Sprint4)
  
    - Refinamento do modelo CNN com técnicas de regularização;

    - Refinamento do artigo adicionando descrições das técnicas implementadas;

- [SPRINT 5](https://github.com/Inteli-College/2024-1B-T01-CC10-G03/releases/tag/Sprint5)

    - Experimentação de diferentes datasets no modelo CNN;
  
    - Conclusão no artigo;
  
    - Últimos ajustes no artigo;

# Licença

<img src="https://mirrors.creativecommons.org/presskit/icons/cc.large.png" alt="CC Logo" width="150"/><br>

<img src="https://mirrors.creativecommons.org/presskit/icons/by.large.png" alt="CC BY Logo" width="150"/>

[Application 4.0 International](https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1)



