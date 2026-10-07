# NLP - Análise de Sentimentos em Avaliações

Projeto de **Processamento de Linguagem Natural (NLP)** desenvolvido em Python para analisar avaliações de produtos e classificar seus sentimentos como **positivo** ou **negativo**.

O projeto explora diferentes técnicas de processamento e representação de textos, desde a transformação das palavras em dados numéricos até o treinamento e teste de um modelo de classificação.

## 📌 Sobre o projeto

O projeto utiliza um conjunto de avaliações de produtos contendo textos e seus respectivos sentimentos.

Ao longo do notebook são abordadas as seguintes etapas:

- Exploração dos dados textuais;
- Transformação de textos em dados numéricos;
- Classificação de sentimentos;
- Análise da frequência das palavras;
- Geração de nuvens de palavras;
- Tokenização;
- Remoção de stopwords;
- Remoção de pontuação;
- Remoção de acentos;
- Padronização para letras minúsculas;
- Stemming;
- Vetorização com TF-IDF;
- Utilização de n-grams;
- Testes com diferentes quantidades de features;
- Análise dos pesos das palavras no modelo;
- Salvamento e carregamento do modelo;
- Classificação de novas avaliações.

## 🧠 Tecnologias e bibliotecas

O projeto foi desenvolvido em **Python** utilizando:

- [Pandas](https://pandas.pydata.org/) - manipulação e análise dos dados;
- [Scikit-learn](https://scikit-learn.org/) - vetorização, divisão dos dados e modelo de classificação;
- [NLTK](https://www.nltk.org/) - tokenização, stopwords, frequência de palavras e stemming;
- [WordCloud](https://amueller.github.io/word_cloud/) - geração de nuvens de palavras;
- [Matplotlib](https://matplotlib.org/) - visualização dos dados;
- [Seaborn](https://seaborn.pydata.org/) - gráficos de frequência;
- [Unidecode](https://pypi.org/project/Unidecode/) - remoção de acentos;
- [Joblib](https://joblib.readthedocs.io/) - salvamento e carregamento dos modelos.

## 📂 Estrutura do projeto

Uma estrutura recomendada para publicar este projeto no GitHub é:

```text
nlp-analise-sentimentos/
│
├── Técnicas_NLP.ipynb
├── README.md
├── modelo_regressao_logistica.pkl
├── tfidf_vectorizer.pkl
└── requirements.txt
```

> Os arquivos `.pkl` são gerados pelo notebook durante a etapa de salvamento do modelo. O arquivo `requirements.txt` pode ser criado para facilitar a instalação das dependências.

## 📊 Dataset

O notebook utiliza o dataset de avaliações disponibilizado pelo curso da Alura:

```text
https://raw.githubusercontent.com/alura-cursos/nlp_analise_sentimento/refs/heads/main/Dados/dataset_avaliacoes.csv
```

O arquivo contém, entre outras informações utilizadas no projeto:

- `avaliacao` - texto da avaliação;
- `sentimento` - classificação do sentimento da avaliação.

## 🔎 1. Exploração dos dados

Inicialmente, o dataset é carregado com Pandas e são realizadas algumas verificações para conhecer os dados:

```python
import pandas as pd

df = pd.read_csv(
    'https://raw.githubusercontent.com/alura-cursos/nlp_analise_sentimento/refs/heads/main/Dados/dataset_avaliacoes.csv'
)

df.head()
df.shape
df.value_counts('sentimento')
```

Também são analisadas avaliações individuais para observar exemplos de textos positivos e negativos.

## 🔢 2. Transformação de texto em números

Modelos de Machine Learning não trabalham diretamente com textos. Por isso, as avaliações são transformadas em representações numéricas.

O primeiro método utilizado é o **Bag of Words**, através do `CountVectorizer`:

```python
from sklearn.feature_extraction.text import CountVectorizer

vetorizar = CountVectorizer()
bag_of_words = vetorizar.fit_transform(texto)
```

Também são realizados testes limitando a quantidade de características:

```python
CountVectorizer(lowercase=False, max_features=50)
```

## 🤖 3. Classificação de sentimentos

Os dados são divididos em conjuntos de treinamento e teste utilizando `train_test_split`.

O modelo utilizado para classificação é a **Regressão Logística**:

```python
from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_treino, y_treino)

acuracia = regressao_logistica.score(X_teste, y_teste)
```

A acurácia é utilizada para avaliar o desempenho do modelo no conjunto de teste.

## ☁️ 4. Análise das palavras mais frequentes

O projeto utiliza **WordCloud** para visualizar as palavras mais frequentes presentes nas avaliações.

Também são geradas nuvens de palavras separadas por sentimento, permitindo comparar os termos mais presentes em avaliações:

- Positivas;
- Negativas.

Além disso, o `NLTK` é utilizado para calcular a frequência das palavras e os resultados são apresentados em gráficos.

## 🧹 5. Limpeza e normalização dos textos

O projeto aplica diferentes etapas de pré-processamento.

### Stopwords

São removidas palavras consideradas pouco relevantes para a análise utilizando as stopwords em português do NLTK:

```python
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
```

### Remoção de pontuação

O `WordPunctTokenizer` é utilizado para separar os elementos do texto e manter apenas palavras relevantes:

```python
token_pontuacao = tokenize.WordPunctTokenizer()
```

### Remoção de acentos

A biblioteca `Unidecode` é utilizada para transformar palavras acentuadas em suas versões sem acento.

### Padronização

Os textos são convertidos para letras minúsculas para reduzir diferenças entre palavras que representam o mesmo termo.

### Stemming

O `RSLPStemmer`, disponível no NLTK, é utilizado para reduzir palavras às suas raízes:

```python
stemmer = nltk.RSLPStemmer()
```

O resultado dessas etapas é armazenado progressivamente nas colunas:

```text
tratamento_1
tratamento_2
tratamento_3
tratamento_4
tratamento_5
```

Isso permite comparar o efeito das diferentes etapas de tratamento sobre o modelo.

## 📐 6. TF-IDF

Além do Bag of Words, o projeto utiliza **TF-IDF (Term Frequency-Inverse Document Frequency)** para representar os textos.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=False,
    max_features=50
)
```

O TF-IDF atribui pesos às palavras de acordo com sua relevância dentro dos textos analisados.

O projeto compara o desempenho utilizando:

- Texto bruto;
- Texto tratado;
- Diferentes quantidades de features.

## 🔗 7. N-grams

Também são utilizados **n-grams**, permitindo considerar combinações de palavras e não apenas palavras individuais.

No projeto é utilizado:

```python
ngram_range=(1,2)
```

Isso significa que o modelo considera:

- Unigrams: palavras individuais;
- Bigrams: pares de palavras.

São realizados testes com diferentes quantidades de features:

```text
50 features
100 features
1000 features
Todas as features disponíveis
```

## ⚖️ 8. Análise dos pesos das palavras

Depois do treinamento da regressão logística, os coeficientes do modelo são analisados para identificar palavras com maior influência na classificação.

São observadas as palavras com:

- Maiores pesos;
- Menores pesos.

Essa análise ajuda a entender quais termos estão mais associados às diferentes classes de sentimento.

## 💾 9. Salvando o modelo

O projeto utiliza `joblib` para salvar o vetor TF-IDF e o modelo treinado:

```python
import joblib

joblib.dump(tfidf_1000, 'tfidf_vectorizer.pkl')
joblib.dump(
    regressao_logistica,
    'modelo_regressao_logistica.pkl'
)
```

Os arquivos podem posteriormente ser carregados:

```python
tfidf = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load(
    'modelo_regressao_logistica.pkl'
)
```

Isso permite utilizar o modelo sem precisar treiná-lo novamente.

## 🔮 10. Classificando novas avaliações

O projeto possui uma função chamada `processar_avaliacao()` responsável por aplicar o mesmo fluxo de tratamento utilizado durante o treinamento.

O processamento inclui:

1. Tokenização;
2. Remoção de stopwords;
3. Remoção de elementos que não são palavras;
4. Remoção de acentos;
5. Stemming.

Depois do processamento, as novas avaliações são transformadas com o vetor TF-IDF e enviadas para o modelo:

```python
novas_avaliacoes_tfidf = tfidf.transform(
    novas_avaliacoes_processadas
)

predicoes = regressao_logistica.predict(
    novas_avaliacoes_tfidf
)
```

O resultado final é organizado em um DataFrame contendo a avaliação original e o sentimento previsto pelo modelo.

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```

### 2. Instale as dependências

```bash
pip install pandas scikit-learn nltk wordcloud matplotlib seaborn unidecode joblib
```

### 3. Execute o notebook

Abra o arquivo:

```text
Técnicas_NLP.ipynb
```

Você pode executar o projeto utilizando:

- Jupyter Notebook;
- JupyterLab;
- Google Colab;
- Visual Studio Code com suporte a notebooks.

### 4. Execute as células em ordem

É importante executar as células sequencialmente, pois o notebook utiliza variáveis e objetos criados em etapas anteriores.

O notebook também realiza o download dos recursos do NLTK necessários para o processamento textual.

## 📁 Arquivos gerados

Durante a execução, o projeto pode gerar:

```text
tfidf_vectorizer.pkl
modelo_regressao_logistica.pkl
```

Esses arquivos representam, respectivamente:

- O vetor utilizado para transformar textos em características TF-IDF;
- O modelo de Regressão Logística treinado para classificação dos sentimentos.

## 🎯 Objetivo de aprendizado

Este projeto foi desenvolvido com foco no aprendizado prático de **NLP e Machine Learning aplicado a textos**.

A sequência permite acompanhar a evolução do processamento:

```text
Texto
  ↓
Tokenização
  ↓
Limpeza
  ↓
Normalização
  ↓
Stemming
  ↓
Vetorização
  ↓
TF-IDF + N-grams
  ↓
Regressão Logística
  ↓
Classificação de sentimento
```

## 🧪 Principais conceitos praticados

- Processamento de Linguagem Natural (NLP);
- Análise exploratória de textos;
- Tokenização;
- Stopwords;
- Stemming;
- Bag of Words;
- TF-IDF;
- N-grams;
- Classificação supervisionada;
- Regressão Logística;
- Treinamento e teste de modelos;
- Avaliação por acurácia;
- Análise de características;
- Persistência de modelos com Joblib.

## 📌 Observação

O notebook contém células organizadas como **Aula 1 até Aula 5**, acompanhando progressivamente as etapas de exploração, tratamento, representação e classificação dos textos.

O arquivo `Técnicas_NLP.ipynb` é o principal material do projeto e reúne todo o processo desenvolvido.

---

## 👨‍💻 Autor

Projeto desenvolvido como estudo prático de **NLP, Machine Learning e análise de sentimentos com Python**.
