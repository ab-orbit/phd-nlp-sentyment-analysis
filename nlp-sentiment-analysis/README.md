# Análise de Sentimentos - NLP

Projeto de classificação de sentimentos usando diferentes abordagens de NLP: SVM + BoW, SVM + Embeddings e BERT.

**Autor:** João Wellington Cunha (jwc)
**Lab:** lab2@

## Objetivos

- Coletar avaliações de produtos: texto e nota
- Treinar classificadores:
  1. SVM + Bag of Words (BoW)
  2. SVM + Word Embeddings
  3. BERT (Transformers)
- Bônus: utilizar in-context learning para classificação
- Apresentação reportando resultados (F1 e acurácia) e análises

## Notebooks

1. **svm_bow.ipynb** - Classificador SVM com Bag of Words
2. **svm_embeddings.ipynb** - Classificador SVM com Word2Vec Embeddings
3. **bert_approach.ipynb** - Classificador com BERT (Transformers)

## Setup do Ambiente

### Opção 1: Script Automatizado (RECOMENDADO)

```bash
# Executar script de setup
./setup_clean_env.sh

# Ativar ambiente
source venv/bin/activate

# Iniciar Jupyter
jupyter notebook
```

### Opção 2: Manual

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Registrar kernel Jupyter
python -m ipykernel install --user --name=nlp-sentiment
```

## Resolver Conflitos de Dependências

Se você encontrar o erro:
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**Solução Rápida:**
```bash
# Remover pacotes desnecessários
pip uninstall docling streamlit scrapy scrapegraph-py mem0ai instructor ollama pdftext -y

# Reinstalar apenas o necessário
pip install -r requirements.txt
```

**Consulte:** `FIX_DEPENDENCIES.md` para guia completo

## Estrutura do Projeto

```
nlp-sentiment-analysis/
├── dataset/
│   └── yelp_reviews.csv          # Dataset de avaliações
├── docs/
│   └── restaurantes_ativos_recife.csv
├── svm_bow.ipynb                 # Notebook 1: SVM + BoW
├── svm_embeddings.ipynb          # Notebook 2: SVM + Embeddings
├── bert_approach.ipynb           # Notebook 3: BERT
├── requirements.txt              # Dependências do projeto
├── setup_clean_env.sh           # Script de setup automatizado
├── FIX_DEPENDENCIES.md          # Guia de resolução de conflitos
└── README.md                    # Este arquivo
```

## Dependências Principais

- **Core**: pandas, numpy, scikit-learn
- **Visualização**: matplotlib, seaborn
- **NLP Clássico**: gensim (Word2Vec)
- **Deep Learning**: torch, transformers, datasets
- **Jupyter**: ipykernel, jupyter

## Dataset

O projeto usa avaliações do Yelp com classificação binária:
- Label 1 (0): Negativo
- Label 2 (1): Positivo

## Uso

1. Ativar ambiente: `source venv/bin/activate`
2. Iniciar Jupyter: `jupyter notebook`
3. Selecionar kernel: "Python (NLP Sentiment)"
4. Executar notebooks na ordem

## Comparação das Abordagens

| Abordagem | Acurácia Esperada | Tempo Treinamento | Recursos |
|-----------|------------------|-------------------|----------|
| SVM + BoW | ~85-88% | Minutos | CPU |
| SVM + Embeddings | ~88-90% | < 1 hora | CPU |
| BERT | ~92-95% | Horas | GPU recomendada |

## Troubleshooting

### Erros Comuns
- **ImportError: cannot import 'AdamW' from 'transformers'** - Ver `COMMON_ERRORS.md`
- **Conflitos de dependências** - Ver `FIX_DEPENDENCIES.md`
- **Out of Memory** - Reduzir batch_size ou usar CPU
- Consulte `COMMON_ERRORS.md` para lista completa de soluções

### Conflitos de Versão
- Ver `FIX_DEPENDENCIES.md`
- Executar `./setup_clean_env.sh`

### PyTorch no macOS
```bash
# Para Mac com Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio
```

### Kernel não aparece no Jupyter
```bash
python -m ipykernel install --user --name=nlp-sentiment --display-name "Python (NLP Sentiment)"
```

### Out of Memory (GPU)
- Reduzir batch_size no BERT (de 16 para 8 ou 4)
- Reduzir max_length (de 128 para 64)
- Usar CPU: `device = torch.device('cpu')`

## Licença

Projeto acadêmico - PhD Data Science
