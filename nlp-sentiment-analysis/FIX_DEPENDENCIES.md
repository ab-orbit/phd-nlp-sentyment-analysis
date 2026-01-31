# Guia de Resolução de Conflitos de Dependências

## Problema Identificado

Você tem conflitos entre:
- `docling 2.4.2` (requer pyarrow<17.0.0, pydantic-settings>=2.3.0)
- `streamlit 1.32.0` (requer versões específicas antigas)
- Versões mais recentes instaladas no ambiente

## Solução Recomendada

### Opção 1: Ambiente Virtual Limpo (RECOMENDADO)

Criar um ambiente virtual novo exclusivo para este projeto:

```bash
# Navegar para o diretório do projeto
cd /Users/jwcunha/Documents/repos/phd-datascience/nlp-sentiment-analysis

# Remover ambiente virtual antigo (se existir)
rm -rf venv .venv env

# Criar novo ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar apenas as dependências necessárias para NLP
pip install -r requirements.txt
```

### Opção 2: Resolver Conflitos Específicos

Se você precisa de `docling` E `streamlit` juntos:

```bash
# Criar arquivo requirements-compatible.txt
cat > requirements-compatible.txt << 'EOF'
# Versões compatíveis
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<1.0.0
gensim>=4.3.0,<5.0.0
torch>=2.0.0,<3.0.0
transformers>=4.30.0,<5.0.0
datasets>=2.14.0,<3.0.0
accelerate>=0.20.0,<1.0.0
tqdm>=4.65.0,<5.0.0

# Versões compatíveis com docling E streamlit
pyarrow>=16.1.0,<17.0.0
pydantic-settings>=2.3.0,<3.0.0
packaging>=16.8,<24.0
pillow>=7.1.0,<11.0
protobuf>=3.20,<5.0
tenacity>=8.1.0,<9.0

# Instalar com compatibilidade forçada
streamlit>=1.32.0,<1.33.0
docling>=2.4.0,<2.5.0
EOF

# Reinstalar com o arquivo compatível
pip install -r requirements-compatible.txt
```

### Opção 3: Remover Pacotes Conflitantes

Se você NÃO precisa de `docling` ou `streamlit` para este projeto:

```bash
# Remover pacotes que não são necessários
pip uninstall docling streamlit -y

# Instalar apenas o necessário
pip install -r requirements.txt
```

### Opção 4: Atualizar Streamlit e Docling

Tentar versões mais recentes que podem ser compatíveis:

```bash
# Atualizar para versões mais recentes
pip install --upgrade streamlit docling

# Se ainda houver conflitos, fixar versões específicas
pip install streamlit==1.40.0 docling==2.10.0
```

## Verificação

Após aplicar qualquer solução, verificar:

```bash
# Verificar dependências
pip check

# Listar pacotes instalados
pip list

# Verificar conflitos
pip install pipdeptree
pipdeptree --warn fail
```

## Para Este Projeto Específico (NLP Sentiment Analysis)

**Pacotes ESSENCIAIS:**
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- gensim (para Word2Vec)
- torch, transformers (para BERT)
- jupyter, ipykernel

**Pacotes NÃO necessários:**
- docling (não usado nos notebooks)
- streamlit (não usado nos notebooks)
- scrapy (apenas se for fazer web scraping)

## Comando Final Recomendado

```bash
# 1. Criar ambiente limpo
python3 -m venv venv
source venv/bin/activate

# 2. Atualizar pip
pip install --upgrade pip

# 3. Instalar APENAS o necessário
pip install pandas numpy scikit-learn matplotlib seaborn gensim \
            torch transformers datasets accelerate tqdm \
            ipykernel jupyter

# 4. Verificar
pip check
```

## Prevenção Futura

1. **Sempre use ambientes virtuais** para cada projeto
2. **Documente versões** no requirements.txt com ranges compatíveis
3. **Use pip-tools** para gerar requirements.txt com versões pinadas:
   ```bash
   pip install pip-tools
   pip-compile requirements.in -o requirements.txt
   ```
4. **Teste instalação limpa** regularmente em ambiente novo
