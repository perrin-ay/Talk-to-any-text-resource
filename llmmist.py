import os
from langchain_mistralai import ChatMistralAI
os.environ['MISTRAL_API_KEY'] = os.getenv('mist')

# per mistral docs models that support tool calling are
#Mistral Small, Mistral Large, Mixtral 8x22B

llmMist = ChatMistralAI(model="mistral-large-latest", temperature =0)