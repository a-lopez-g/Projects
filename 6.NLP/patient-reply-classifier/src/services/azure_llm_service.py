import openai
class AzureLLMService:
    def __init__(self) -> None:
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = ""

    def compute_embeddings(self, text, embedding_model = "test-embeddings"):
        response = openai.Embedding.create(
            input=text,
            engine=embedding_model,
            deployment_id = embedding_model
        )
        return response['data'][0]['embedding']
