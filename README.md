# Buscador de Imagens
Pedro Cunial

## Proposta

Presente em diversas aplicações do dia-a-dia moderno, a busca por similaridade de imagens tem se mostrado, cada vez mais, um tópico de grande interesse no campo da Visão Computacional.


Neste projeto, busca-se realizar a busca e pareamento de imagens a partir do método *Bag of Visual Words*. Análogo ao algoritmo para processamento de textos *Bag of Words*, o *Bag of Visual Words* busca *ranquear* imagens baseado na similaridade entre seus pontos de interesse. 


## Método

Como mencionado anteriormente, o projeto foi realizado com base no *Bag of Visual Words*. O algoritmo consiste de uma análise comparativa entre histogramas de ocorrência destes pontos de interesse com relação ao *dataset* como um todo.

Para a extração destes pontos de interesse, utilizou-se a biblioteca *OpenCV* para *Python*, com o detector e e descritor de pontos *SURF*. A partir dos pontos de interesse, foram extraídos os seus descritores: Dados que classificam (ou como o próprio nome já sugere, descrevem) os mesmos.

Além disso, foi utilizado o classificador *KMeans* da biblioteca *scikit learn* para a geração dos "pontos genéricos do *dataset*", utilizados como eixo X do histograma de ocorrência destes descritores.


## Uso

### Dependências

O programa foi desenvolvido com Python 3.6.4 em mente, o seu funcionamento por completo não é garantido em versões inferiores ou superiores ao mesmo.

Além disso, o programa utilizou as seguintes bibliotecas (além das padrões do Python):

- NumPy
- scikit-learn
- matplotlib
- Pickle
- argparse
- OpenCV - necessáriamente a versão contrib 3.4.0.12
(`pip install opencv-contrib-python==3.4.0.12`)


### Executando

O programa é separado em duas partes, o treinamento dos dados ([train_dataset.py](train_dataset.py)) e o teste de um novo dado ([test_data.py](test_data.py)).

O [train_dataset.py](train_dataset.py) aceita os seguinte argumentos (uma breve explicação sobre os mesmos pode ser chamada com `python train_dataset.py -h`):

- **`--dir` ou `-d`:** Define o diretório contendo pastas com imagens. O programa espera que o diretório seja tal como o do *dataset* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download), ou seja, uma pasta contendo diversas pastas com diferentes conteúdos dentro de cada uma. Exemplo de uso: `python train_dataset.py -d ./Caltech101/` (sendo que dentro da pasta `Caltech101` haveriam diversas pastas separadas por tópicos diferentes de imagens, como rostos, carros etc). Este parâmetro é **obrigatório**
- **`--max-items` ou `-mi`:** Define o máximo de itens (imagens) utilizados por pasta. Reduz consideravelmente o tempo de treinamento, no entanto, um número muito pequeno de imagens no treinamento impacta negativamente os resultados. Tem como valor padrão 10, não sendo obrigatório o seu uso. Exemplo: `python train_dataset.py -d ... -mi 5`.
- **`--max-dirs` ou `-md`:**  Assim como o `--max-items`, o `--max-dirs` define o número máximo de diretórios a serem explorados dentro do raíz. Novamente, quanto menor este valor, pior o resultado final. O seu valor padrão é 10. Exemplo: `python train_dataset.py -d ... -md 5`.
- **`--clusters` ou `-c`:** Define o número de clusters a serem utilizados na criação do vocabulário. Tem o valor padrão de 300. Exemplo: `python train_dataset.py -d ... -c 500`.

Além disso, é esperada a existência de um diretório [data](./data/), onde serão salvos os arquivos resultantes do treinamento (tanto os vocabulários gerados, quanto os histogramas). 

O [test_data.py](test_data.py) é um pouco mais simples, recebendo apenas dois parâmetros:

- **`--image` ou `--img` ou `-i`:** Imagem a ser testada. Argumento **obrigatório**. Exemplo de uso: `python test_data.py -i Caltech101/brontossaurus/image_0001.jpg`.
- ***`--datadir` ou `-d`:** Diretório dos dados de treinamento. Argumento não obrigatório, tendo como valor padrão o [data/](./data/). Exemplo de uso: `python test_data.py -i ... -d data/`.

O código foi separado em treinamento e teste, pois não existe a necessidade de retreinar os mesmos dados (que resultariam nos mesmos histogramas e vocabulários), mas que são a parte mais computacionalmente custosa do processo como um todo.
