#!/bin/bash

# Script para criar ambiente virtual limpo para o projeto NLP

set -e  # Parar se houver erro

echo "=================================="
echo "Setup Ambiente NLP - Análise de Sentimentos"
echo "=================================="

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Diretório do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "\n${YELLOW}1. Limpando ambiente virtual antigo...${NC}"
if [ -d "venv" ]; then
    rm -rf venv
    echo -e "${GREEN}   Removido: venv/${NC}"
fi
if [ -d ".venv" ]; then
    rm -rf .venv
    echo -e "${GREEN}   Removido: .venv/${NC}"
fi

echo -e "\n${YELLOW}2. Criando novo ambiente virtual...${NC}"
python3 -m venv venv
echo -e "${GREEN}   Ambiente virtual criado!${NC}"

echo -e "\n${YELLOW}3. Ativando ambiente virtual...${NC}"
source venv/bin/activate
echo -e "${GREEN}   Ambiente ativado!${NC}"

echo -e "\n${YELLOW}4. Atualizando pip, setuptools e wheel...${NC}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}   Ferramentas atualizadas!${NC}"

echo -e "\n${YELLOW}5. Instalando dependências do projeto...${NC}"
echo -e "   ${GREEN}Core packages...${NC}"
pip install pandas numpy scikit-learn --quiet

echo -e "   ${GREEN}Visualization...${NC}"
pip install matplotlib seaborn --quiet

echo -e "   ${GREEN}NLP - Word2Vec...${NC}"
pip install gensim --quiet

echo -e "   ${GREEN}NLP - BERT (PyTorch + Transformers)...${NC}"
pip install torch torchvision torchaudio --quiet
pip install transformers datasets accelerate --quiet

echo -e "   ${GREEN}Jupyter & Utils...${NC}"
pip install jupyter ipykernel tqdm --quiet

echo -e "\n${YELLOW}6. Registrando kernel do Jupyter...${NC}"
python -m ipykernel install --user --name=nlp-sentiment --display-name "Python (NLP Sentiment)"
echo -e "${GREEN}   Kernel registrado: NLP Sentiment${NC}"

echo -e "\n${YELLOW}7. Verificando instalação...${NC}"
pip check
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   Sem conflitos de dependências!${NC}"
else
    echo -e "${RED}   Atenção: alguns conflitos detectados (pode ser ignorável)${NC}"
fi

echo -e "\n${YELLOW}8. Salvando dependências instaladas...${NC}"
pip freeze > requirements-installed.txt
echo -e "${GREEN}   Salvo em: requirements-installed.txt${NC}"

echo -e "\n${GREEN}=================================="
echo "Setup concluído com sucesso!"
echo "=================================="
echo -e "\nPara usar o ambiente:${NC}"
echo "  1. Ativar: source venv/bin/activate"
echo "  2. Jupyter: jupyter notebook"
echo "  3. Selecionar kernel: 'Python (NLP Sentiment)'"
echo ""
echo -e "${YELLOW}Para desativar: deactivate${NC}"
