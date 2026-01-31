# Erros Comuns e Soluções

## 1. ImportError: cannot import name 'AdamW' from 'transformers'

### Problema
```python
from transformers import AdamW  # ERRO!
```

### Causa
A partir da versão 4.30+ do transformers, o otimizador `AdamW` foi removido da biblioteca transformers e agora deve ser importado diretamente do PyTorch.

### Solução
```python
# ✅ CORRETO - Versões novas
from torch.optim import AdamW

# ❌ ERRADO - Apenas em versões antigas
from transformers import AdamW
```

### Código Corrigido no Notebook
O notebook `bert_approach.ipynb` já está corrigido com:
```python
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
```

---

## 1.1 AttributeError: BertTokenizer has no attribute 'encode_plus'

### Problema
```python
encoded = tokenizer.encode_plus(...)  # ERRO!
# AttributeError: BertTokenizer has no attribute encode_plus
```

### Causa
O método `encode_plus` foi **deprecado** nas versões mais recentes do transformers. A API moderna usa a chamada direta ao tokenizer.

### Solução
```python
# ❌ ERRADO - API antiga (deprecada)
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# ✅ CORRETO - API moderna
encoded = tokenizer(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
```

### Mudanças no Código
A API é exatamente a mesma, apenas removendo `encode_plus`:
- `tokenizer.encode_plus(...)` → `tokenizer(...)`
- `tokenizer.batch_encode_plus(...)` → `tokenizer(...)`

### Código Corrigido
O notebook já está atualizado com:
```python
# Tokenizar
encoding = tokenizer(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)
```

---

## 2. OutOfMemoryError: CUDA out of memory

### Problema
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Soluções

#### Opção 1: Reduzir Batch Size
```python
# Trocar de 16 para 8 ou 4
BATCH_SIZE = 8  # ou 4 se ainda houver erro
```

#### Opção 2: Reduzir Max Length
```python
# Trocar de 128 para 64
max_length = 64
```

#### Opção 3: Usar CPU (mais lento)
```python
device = torch.device('cpu')
```

#### Opção 4: Usar Gradient Accumulation
```python
# Acumular gradientes por N steps
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = model(...).loss
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 3. ModuleNotFoundError: No module named 'transformers'

### Solução
```bash
pip install transformers torch datasets accelerate
```

Ou use o script de setup:
```bash
./setup_clean_env.sh
```

---

## 4. Tokenizer muito lento

### Problema
Tokenização demora muito tempo

### Solução
Usar fast tokenizers (implementados em Rust):
```python
# Automaticamente usa versão rápida se disponível
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Ou forçar versão rápida
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
```

---

## 5. Warning: Some weights were not initialized

### Mensagem
```
Some weights of BertForSequenceClassification were not initialized from the model checkpoint
```

### Explicação
Isso é **NORMAL** quando fazemos fine-tuning. A camada de classificação final é nova e será treinada do zero.

### Não é um erro!
O modelo base do BERT está pré-treinado, apenas a camada de classificação é inicializada aleatoriamente.

---

## 6. GridSearchCV muito lento / Travado por horas

### Problema
```python
grid_search.fit(X_train, y_train)  # Demora horas!
```

### Causa
SVM com **kernel RBF** em datasets grandes é **extremamente lento**:
- Complexidade: O(n² × m) onde n = amostras, m = features
- Com 38.000 amostras: ~1,4 bilhões de operações por fold
- GridSearch com cv=5: multiplica por 5x

### Soluções

#### Opção 1: Usar apenas Kernel Linear (RECOMENDADO para BoW)
```python
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear']  # Muito mais rápido!
}
```

#### Opção 2: Reduzir CV folds
```python
GridSearchCV(..., cv=3)  # Ao invés de cv=5
```

#### Opção 3: Reduzir grid de parâmetros
```python
param_grid = {
    'C': [1, 10]  # Apenas 2 valores ao invés de 3+
}
```

#### Opção 4: Pular GridSearch (usar modelo padrão)
```python
# Usar modelo já treinado
best_svm = SVC(kernel='linear', C=1.0)
best_svm.fit(X_train, y_train)
```

#### Opção 5: Interromper processo travado
- Jupyter: `Kernel > Interrupt`
- Terminal: `Ctrl+C`

### Por que Linear é melhor para BoW?
- BoW gera matrizes **esparsas** (>95% zeros)
- Kernel linear aproveita esparsidade: **muito rápido**
- Kernel RBF precisa calcular distâncias: **muito lento**
- Performance: linear geralmente igual ou melhor com BoW

### Tempo Estimado
| Configuração | Tempo Aproximado |
|--------------|------------------|
| Linear, cv=3, C=[0.1,1,10] | 5-10 min |
| Linear, cv=5, C=[0.1,1,10] | 10-15 min |
| RBF, cv=3, C=[0.1,1,10] | 2-4 HORAS |
| RBF, cv=5, C=[0.1,1,10] | 4-8 HORAS |

---

## 7. Kernel died / Jupyter crash

### Causas Comuns
1. Out of memory (GPU ou RAM)
2. Incompatibilidade de versões
3. Erro de segmentação
4. GridSearchCV muito longo (timeout)

### Soluções
```bash
# 1. Reiniciar kernel Jupyter
# Menu: Kernel > Restart

# 2. Limpar cache GPU (se usar CUDA)
import torch
torch.cuda.empty_cache()

# 3. Reduzir tamanho do modelo ou batch
BATCH_SIZE = 4

# 4. Verificar memória disponível
import psutil
print(f"RAM disponível: {psutil.virtual_memory().available / 1e9:.2f} GB")

# 5. Se travado em GridSearch, ver seção 6
```

---

## 7. Slow training / No progress bar

### Problema
Treinamento muito lento ou sem barra de progresso

### Solução
```python
# Verificar se tqdm está instalado
pip install tqdm

# Usar tqdm.auto para melhor compatibilidade
from tqdm.auto import tqdm

# Verificar se está usando GPU
print(f"Device: {device}")
print(f"GPU disponível: {torch.cuda.is_available()}")
```

---

## 8. FileNotFoundError: dataset/yelp_reviews.csv

### Problema
Dataset não encontrado

### Solução
```bash
# Verificar estrutura de diretórios
ls -la dataset/

# Criar diretório se não existir
mkdir -p dataset

# Baixar dataset (exemplo)
# wget URL_DO_DATASET -O dataset/yelp_reviews.csv
```

---

## 9. TypeError: 'DataLoader' object is not subscriptable

### Problema
```python
batch = train_loader[0]  # ERRO!
```

### Solução
```python
# ✅ CORRETO - Iterar sobre DataLoader
for batch in train_loader:
    # processar batch
    pass

# OU pegar primeiro batch
batch = next(iter(train_loader))
```

---

## 10. RuntimeError: Expected all tensors to be on same device

### Problema
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

### Solução
Garantir que modelo E dados estão no mesmo device:
```python
# Mover modelo para device
model = model.to(device)

# Mover dados para device no loop de treinamento
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)
```

---

## Prevenção de Erros

### Checklist antes de executar notebooks:

1. ✅ Ambiente virtual ativado
   ```bash
   source venv/bin/activate
   ```

2. ✅ Dependências instaladas
   ```bash
   pip check
   ```

3. ✅ GPU disponível (opcional)
   ```python
   torch.cuda.is_available()
   ```

4. ✅ Dataset presente
   ```bash
   ls dataset/yelp_reviews.csv
   ```

5. ✅ Memória suficiente
   ```python
   import psutil
   psutil.virtual_memory().available / 1e9  # GB
   ```

---

## Recursos Úteis

- **Transformers Docs**: https://huggingface.co/docs/transformers/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Common Issues**: https://github.com/huggingface/transformers/issues

---

## Reportar Novos Problemas

Se encontrar um erro não listado aqui:
1. Anotar mensagem de erro completa
2. Versões: `pip list | grep -E "torch|transformers"`
3. Sistema: `uname -a` (Linux/Mac) ou `ver` (Windows)
