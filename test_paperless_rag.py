import os
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import paperless_rag

class TestPaperlessRAG(unittest.TestCase):

    def setUp(self):
        # Backup environment variables
        self.orig_paperless_token = os.environ.get("PAPERLESS_TOKEN")
        self.orig_openai_key = os.environ.get("OPENAI_API_KEY")
        self.orig_paperless_url = os.environ.get("PAPERLESS_URL")

    def tearDown(self):
        # Restore environment variables
        if self.orig_paperless_token is not None:
            os.environ["PAPERLESS_TOKEN"] = self.orig_paperless_token
        else:
            os.environ.pop("PAPERLESS_TOKEN", None)

        if self.orig_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = self.orig_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        if self.orig_paperless_url is not None:
            os.environ["PAPERLESS_URL"] = self.orig_paperless_url
        else:
            os.environ.pop("PAPERLESS_URL", None)

    def test_validate_environment_success(self):
        os.environ["PAPERLESS_TOKEN"] = "valid_token"
        os.environ["OPENAI_API_KEY"] = "sk-validkey"
        # Should execute without throwing ValueError
        paperless_rag.validate_environment()

    def test_validate_environment_missing_token(self):
        os.environ["PAPERLESS_TOKEN"] = ""
        os.environ["OPENAI_API_KEY"] = "sk-validkey"
        with self.assertRaises(ValueError) as ctx:
            paperless_rag.validate_environment()
        self.assertIn("PAPERLESS_TOKEN", str(ctx.exception))

    def test_validate_environment_missing_openai_key(self):
        os.environ["PAPERLESS_TOKEN"] = "valid_token"
        os.environ["OPENAI_API_KEY"] = ""
        with self.assertRaises(ValueError) as ctx:
            paperless_rag.validate_environment()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_get_auth_headers(self):
        os.environ["PAPERLESS_TOKEN"] = "test_secret_token"
        headers = paperless_rag.get_auth_headers()
        self.assertEqual(headers["Authorization"], "Token test_secret_token")
        self.assertEqual(headers["Accept"], "application/json")

    @patch('paperless_rag.requests.get')
    def test_fetch_paperless_documents_single_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Conta de Luz",
                    "content": "Valor a pagar: R$ 150,00. Vencimento: 10/10/2026",
                    "created": "2026-10-01"
                },
                {
                    "id": 2,
                    "title": "Documento Vazio",
                    "content": "   ",
                    "created": "2026-10-02"
                }
            ]
        }
        mock_get.return_value = mock_response

        docs = paperless_rag.fetch_paperless_documents(paperless_url="http://localhost:8000", token="test_token")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source_id"], 1)
        self.assertEqual(docs[0].metadata["title"], "Conta de Luz")
        self.assertEqual(docs[0].page_content, "Valor a pagar: R$ 150,00. Vencimento: 10/10/2026")
        mock_get.assert_called_once()

    @patch('paperless_rag.requests.get')
    def test_fetch_paperless_documents_pagination(self, mock_get):
        # Page 1 mock response with next page URL
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {
            "next": "http://localhost:8000/api/documents/?page=2",
            "results": [
                {"id": 101, "title": "Fatura Internet", "content": "Valor: R$ 100", "created": "2026-09-01"}
            ]
        }

        # Page 2 mock response without next page
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {
            "next": None,
            "results": [
                {"id": 102, "title": "Nota Fiscal", "content": "Serviços de TI R$ 500", "created": "2026-09-02"}
            ]
        }

        mock_get.side_effect = [mock_resp1, mock_resp2]

        docs = paperless_rag.fetch_paperless_documents(paperless_url="http://localhost:8000", token="test_token")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["source_id"], 101)
        self.assertEqual(docs[1].metadata["source_id"], 102)
        self.assertEqual(mock_get.call_count, 2)

    @patch('paperless_rag.requests.get')
    def test_fetch_paperless_documents_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized Token"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            paperless_rag.fetch_paperless_documents(token="bad_token")

        self.assertIn("Erro ao conectar com Paperless-ngx: 401", str(context.exception))

    @patch('paperless_rag.requests.get')
    def test_fetch_paperless_documents_invalid_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            paperless_rag.fetch_paperless_documents()

        self.assertIn("Resposta inválida recebida do Paperless-ngx", str(context.exception))

    @patch('paperless_rag.requests.get')
    def test_fetch_paperless_documents_connection_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection timed out")

        with self.assertRaises(Exception) as context:
            paperless_rag.fetch_paperless_documents()

        self.assertIn("Erro de conexão com Paperless-ngx", str(context.exception))

    def test_build_vector_store_empty_docs(self):
        result = paperless_rag.build_vector_store([])
        self.assertIsNone(result)

    @patch('paperless_rag.Chroma.from_documents')
    @patch('paperless_rag.OpenAIEmbeddings')
    def test_build_vector_store_with_docs(self, mock_embeddings, mock_chroma):
        docs = [
            Document(page_content="Texto de teste 1", metadata={"source_id": 1, "title": "Doc 1"}),
            Document(page_content="Texto de teste 2", metadata={"source_id": 2, "title": "Doc 2"})
        ]

        paperless_rag.build_vector_store(docs, persist_directory="./test_db")

        mock_chroma.assert_called_once()
        kwargs = mock_chroma.call_args[1]
        self.assertEqual(kwargs["persist_directory"], "./test_db")
        self.assertIn("ids", kwargs)
        self.assertEqual(len(kwargs["ids"]), 2)
        self.assertTrue(kwargs["ids"][0].startswith("doc_1_chunk_"))

    @patch('paperless_rag.ChatOpenAI')
    @patch('paperless_rag.create_retrieval_chain')
    @patch('paperless_rag.create_stuff_documents_chain')
    def test_ask_question(self, mock_stuff, mock_retrieval, mock_chat_openai):
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_chain_instance = MagicMock()
        mock_chain_instance.invoke.return_value = {"answer": "O valor total é R$ 150,00."}
        mock_retrieval.return_value = mock_chain_instance

        response = paperless_rag.ask_question(mock_vectorstore, "Qual o valor total?")

        self.assertEqual(response["answer"], "O valor total é R$ 150,00.")
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
        mock_chain_instance.invoke.assert_called_once_with({"input": "Qual o valor total?"})

if __name__ == '__main__':
    unittest.main()
