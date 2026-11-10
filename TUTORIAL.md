# 📖 Guia Prático: Paperless-ngx RAG Assistant & Classificador Inteligente

Este tutorial orienta o processo de implantação, configuração e execução do módulo de Retrieval-Augmented Generation (RAG) integrado ao Paperless-ngx.

---

## 📋 Pré-requisitos

1. **Python 3.10+** instalado.
2. Servidor do **Paperless-ngx** em execução (localmente ou via rede/Docker).
3. Chave de API da **OpenAI** com créditos válidos.
4. Token de autenticação da API do Paperless-ngx gerado no painel administrativo.

---

## 🚀 Passo a Passo de Execução

### 1. Instalar Dependências
```bash
cd paperless-rag
pip install requests langchain-community langchain-openai chromadb python-dotenv langchain-core langchain-classic
```

### 2. Configurar Variáveis de Ambiente
Crie ou edite o arquivo `.env`:
```env
PAPERLESS_URL=http://localhost:8000
PAPERLESS_TOKEN=seu_token_api_do_paperless
OPENAI_API_KEY=sk-proj-sua_chave_openai
```

### 3. Executar o Assistente RAG
```bash
python paperless_rag.py
```

### 4. Executar os Testes Unitários
```bash
python -m unittest test_paperless_rag.py
```
