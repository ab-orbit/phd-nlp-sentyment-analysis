# Análise de Sentimentos - NLP

Projeto de classificação de sentimentos comparando **4 abordagens de NLP** desde técnicas clássicas (2000s) até métodos modernos (2024), demonstrando a evolução do processamento de linguagem natural.

## **Autor:** Jefferson Wellington da Cunha (jwc@)
## Doutorado em Engenharia de Software - CESAR - Centro de Estudos e Sistemas Avançados do Recife
## **Sponsored by** Aeon Bridge Co.
## **Lab:** lab2@
## **Dataset:** Yelp Restaurant Reviews (38,000 avaliações)

---

## Objetivos

Este projeto implementa e compara 4 abordagens distintas para classificação de sentimentos:

1. **SVM + Bag of Words (BoW)** - Baseline clássica (2000s)
2. **SVM + Word Embeddings** - Representação semântica (2013-2017)
3. **BERT Fine-tuning** - Transformers com embeddings contextuais (2018-2022)
4. **In-Context Learning** - LLMs com zero/few-shot learning (2023+)

Cada abordagem é avaliada com métricas completas (Acurácia, Precision, Recall, F1-Score) e análise de trade-offs (performance vs custo vs latência).

---

## 📓 Notebooks

### Implementações Principais

1. **`svm_bow.ipynb`** - SVM + Bag of Words
   - Acurácia: **89.92%** 
   - Técnica: Vetores esparsos de contagem de palavras
   - Treino: ~10 minutos (CPU)
   - Melhor para: Produção em alta escala, baixo custo

2. **`svm_embeddings.ipynb`** - SVM + Word2Vec Embeddings
   - Acurácia: **90.67%** 
   - Técnica: Vetores densos com semântica
   - Treino: ~30-60 minutos (CPU)
   - Melhor para: Balanço entre performance e simplicidade

3. **`bert_approach.ipynb`** - BERT Fine-tuning
   - Acurácia: **94.04%** 
   - Técnica: Transformers com embeddings contextuais
   - Treino: ~2-4 horas (GPU recomendada)
   - Melhor para: Máxima performance
   - ✨ **Suporta Apple Silicon (MPS)** para Mac M1/M2/M3/M4

4. **`in_context_learning_approach.ipynb`** - LLMs (Zero/Few-shot)
   - Acurácia: **94.00%**  (Zero-Shot)
   - Técnica: Prompting de LLMs (GPT, Claude, Llama)
   - Treino: **Não requer** (zero-shot possível)
   - Melhor para: Prototipagem rápida, poucos dados
   - **Suporta LM Studio local** (gratuito)

### Análise e Consolidação

5. **`summary.ipynb`** - **Apresentação Consolidada**
   - Comparação completa das 4 abordagens
   - Visualizações de performance e trade-offs
   - Matriz de decisão por cenário de uso
   - Roadmap híbrido recomendado
   - Recomendações práticas
   - **Execute este notebook primeiro para visão geral do projeto!**

---

## Setup do Ambiente

### Opção 1: Usando UV (Recomendado)

```bash
# Instalar dependências
uv pip install transformers torch datasets accelerate scikit-learn matplotlib seaborn pandas numpy openai anthropic requests gensim ipywidgets

# Iniciar Jupyter
uv run jupyter lab
```

### Opção 2: Script Automatizado

```bash
# Executar script de setup
./setup_clean_env.sh

# Ativar ambiente
source .venv/bin/activate

# Iniciar Jupyter
jupyter lab
```

### Opção 3: Manual

```bash
# Criar ambiente virtual com uv
uv venv

# Ativar ambiente
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instalar dependências
uv pip install -r requirements.txt

# Registrar kernel Jupyter
python -m ipykernel install --user --name=nlp-sentiment
```

---

## 🗂️ Estrutura do Projeto

```
sentiment-analysis/
├── dataset/
│   └── yelp_reviews.csv              # 38k avaliações Yelp (binário)
├── docs/
│   ├── projeto-mod1.jpg
│   └── restaurantes_ativos_recife.csv
├── restaurant_reviews/               # Spider Scrapy (coleta de dados)
│   └── restaurant_reviews/
│       └── spiders/
├── svm_bow.ipynb                     # Abordagem 1: SVM + BoW (89.92%)
├── svm_embeddings.ipynb              # Abordagem 2: SVM + Embeddings (~91.5%)
├── bert_approach.ipynb               # Abordagem 3: BERT (~93.5%)
├── in_context_learning_approach.ipynb # Abordagem 4: ICL (~90%)
├── summary.ipynb                     # 📊 Apresentação consolidada
├── requirements.txt                  # Dependências pip
├── pyproject.toml                    # Configuração uv
├── uv.lock                          # Lock file uv
├── setup_clean_env.sh               # Script de setup
├── FIX_DEPENDENCIES.md              # Guia de troubleshooting
├── COMMON_ERRORS.md                 # Erros comuns e soluções
└── README.md                        # Este arquivo
```

---

## 📦 Dependências Principais

### Core ML/NLP
- **pandas**, **numpy** - Manipulação de dados
- **scikit-learn** - SVM, métricas, validação
- **gensim** - Word2Vec, embeddings

### Deep Learning
- **torch** - PyTorch (com suporte MPS para Mac)
- **transformers** - BERT, tokenizers (Hugging Face)
- **datasets**, **accelerate** - Utilidades Hugging Face

### LLMs (In-Context Learning)
- **openai** - API OpenAI (GPT-3.5/4) [opcional]
- **anthropic** - API Anthropic (Claude) [opcional]
- **requests** - Para LM Studio/Ollama local

### Visualização & Jupyter
- **matplotlib**, **seaborn** - Gráficos
- **ipywidgets** - Widgets interativos
- **jupyter**, **ipykernel** - Ambiente notebook

---

## Dataset

**Fonte:** Yelp Restaurant Reviews
**Tamanho:** 38,000 avaliações
**Distribuição:** Balanceada (19k negativas, 19k positivas)
**Formato:** CSV com 2 colunas (label, text)

### Labels
- **1** = Negativo (avaliação ruim)
- **2** = Positivo (avaliação boa)

### Características
- **Comprimento médio:** 133 palavras por avaliação
- **Range:** 4 a 5,093 caracteres
- **Idioma:** Inglês
- **Domínio:** Restaurantes (comida, serviço, atendimento)

---

## Uso

### Quick Start

1. **Clone o repositório** (se aplicável)
2. **Instale as dependências** (veja seção Setup)
3. **Inicie Jupyter Lab**: `uv run jupyter lab`
4. **Comece pelo Summary**: Execute `summary.ipynb` para visão geral
5. **Execute notebooks individuais** na ordem desejada

### Ordem Recomendada

**Para aprendizado:**
1. `svm_bow.ipynb` (mais simples)
2. `svm_embeddings.ipynb` (intermediário)
3. `bert_approach.ipynb` (avançado)
4. `in_context_learning_approach.ipynb` (moderno)
5. `summary.ipynb` (consolidação)

**Para decisão rápida:**
1. `summary.ipynb` (veja matriz de decisão)
2. Execute o notebook recomendado para seu caso

---

## Comparação das Abordagens

| Abordagem | Acurácia | Treino | Hardware | Latência | Custo/Inf | Quando Usar |
|-----------|----------|--------|----------|----------|-----------|-------------|
| **SVM + BoW** | 89.92% | 5-10 min | CPU | <1ms | Muito baixo | Produção em escala |
| **SVM + Embeddings** | 90.67% | 30-60 min | CPU | <10ms | Baixo | Balanço performance/custo |
| **BERT** | 94.04% | 2-4h (GPU) | GPU/MPS | 50-100ms | Médio | Máxima performance |
| **In-Context Learning** | 94.00% | **Não requer** | API/Local | 100-500ms | Alto | Prototipagem, poucos dados |

### Trade-offs Principais

**Performance:** BERT > SVM Embeddings > SVM BoW ≈ ICL
**Velocidade:** SVM BoW > SVM Embeddings > BERT > ICL
**Custo Operacional:** SVM BoW > SVM Embeddings > BERT > ICL
**Facilidade Setup:** ICL > SVM BoW > SVM Embeddings > BERT
**Flexibilidade:** ICL > BERT > SVM Embeddings > SVM BoW

---

## Configurações Especiais

### GPU no Mac (Apple Silicon)

Os notebooks BERT suportam **MPS (Metal Performance Shaders)** para aceleração em GPU Apple:

```python
# Detecção automática no notebook
if torch.backends.mps.is_available():
    device = torch.device('mps')  # Usa GPU do Mac
    print("✓ Usando Apple Silicon GPU")
```

**Performance:** 10-50x mais rápido que CPU em Mac M1/M2/M3/M4

### LM Studio para In-Context Learning

O notebook `in_context_learning_approach.ipynb` suporta **LM Studio** (gratuito, local):

1. Instale LM Studio: https://lmstudio.ai
2. Baixe um modelo (ex: Llama 3.1)
3. Inicie o servidor na porta 11434
4. O notebook detecta automaticamente!

**Vantagem:** Zero custo, sem APIs pagas, dados privados

---

## Troubleshooting

### Erros Comuns

#### 1. ImportError: cannot import 'AdamW' from 'transformers'

**Solução:** AdamW foi movido para `torch.optim`
```python
from torch.optim import AdamW  # Correto
# from transformers import AdamW  # Deprecado
```

#### 2. TypeError: Cannot convert MPS Tensor to float64

**Solução:** MPS não suporta float64, use float32
```python
# Use .float() ao invés de .double()
accuracy = correct_predictions.float() / total
```

#### 3. KeyError: 'label' ao ler CSV

**Solução:** Adicione `header=None` ao ler o CSV
```python
df = pd.read_csv('dataset/yelp_reviews.csv',
                 names=['label', 'text'],
                 header=None)  # Importante!
```

#### 4. Conflitos de dependências

**Solução:** Use ambiente limpo com uv
```bash
uv venv --force
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### 5. Kernel não aparece no Jupyter

**Solução:**
```bash
python -m ipykernel install --user --name=nlp-sentiment --display-name "Python (NLP Sentiment)"
# Reinicie Jupyter Lab
```

#### 6. Out of Memory (GPU/MPS)

**Soluções:**
- Reduzir `batch_size` (16 → 8 → 4)
- Reduzir `max_length` (128 → 64)
- Usar CPU: `device = torch.device('cpu')`

#### 7. LM Studio não detectado (In-Context Learning)

**Soluções:**
- Verifique se servidor está rodando (porta 11434)
- Teste: `curl http://localhost:11434/v1/models`
- Instale pacotes: `uv pip install openai requests`

**Consulte:** `COMMON_ERRORS.md` para lista completa

---

## Dicas e Recomendações

### Para Estudantes/Pesquisadores
- Execute todos os 4 notebooks para entender a evolução do NLP
- Compare resultados e analise trade-offs
- Experimente com diferentes parâmetros
- Use `summary.ipynb` para apresentações

### Para Profissionais/Produção
1. **MVP/Prototipagem:** Use In-Context Learning (setup em minutos)
2. **Produção em Escala:** Use SVM + BoW (custo mínimo, latência <1ms)
3. **Performance Crítica:** Use BERT fine-tuned (máxima acurácia)
4. **Híbrido:** SVM para casos fáceis (80%), BERT para difíceis (20%)

### Próximos Passos Sugeridos
- Implementar TF-IDF ao invés de BoW (+1-2% acurácia)
- Adicionar n-gramas (bigramas, trigramas)
- Ensemble: combinar SVM + BERT
- Deploy com FastAPI
- Monitoramento de drift

---

## Recursos Adicionais

### Papers Fundamentais
- Joachims (1998) - Text Categorization with SVM
- Mikolov et al. (2013) - Word2Vec
- Vaswani et al. (2017) - Attention Is All You Need
- Devlin et al. (2018) - BERT
- Brown et al. (2020) - GPT-3 (Few-Shot Learning)

### Documentação
- Scikit-learn: https://scikit-learn.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- PyTorch: https://pytorch.org/docs/
- Gensim: https://radimrehurek.com/gensim/

### Cursos Online
- Stanford CS224N: NLP with Deep Learning
- Fast.ai: Practical Deep Learning
- DeepLearning.AI: NLP Specialization

---

## Licença

Projeto acadêmico - PhD Data Science
Universidade/Instituição: CESAR


---

## 🤝 Contribuições

Para dúvidas, sugestões ou contribuições:
- Abra uma issue no repositório
- Entre em contato com o autor: Jefferson Wellington Cunha (jwc@)
- contact@aeonbridge.com

---

**Dica Final:** Comece executando `summary.ipynb` para ter uma visão completa do projeto e decidir qual abordagem explorar em detalhes!
