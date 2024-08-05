from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import re
from sentence_transformers import SentenceTransformer
from llmmist import llmMist


import os
import time
import sqlite3
import re
from pytz import timezone
import pytz
import datetime
import json
from operator import itemgetter
import sys
import pandas as pd


from langchain_mistralai import ChatMistralAI

from langchain import hub
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.callbacks.tracers import ConsoleCallbackHandler

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)


#### HF secret ####

os.environ['HUGGINGFACEHUB_API_TOKEN']  = os.getenv('HF_TOKEN')

####### embeddings model #########

#model_name = "Alibaba-NLP/gte-large-en-v1.5"
model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}


model_name = "gte-large-en-v1.5"
hf = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)


############# load embeddings ########
persist_directory = 'fulldb'
vectordb_v1 = Chroma(persist_directory=persist_directory, embedding_function=hf)


os.environ["COHERE_API_KEY"] = os.getenv('coherepass')


prompt_official_mist = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {query}""")


#v1 retriver
retriever_v1 = vectordb_v1.as_retriever(search_type="similarity", search_kwargs={"k": 10})

### cohere reranker

compressor = CohereRerank(model="rerank-english-v3.0",top_n=7) 
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever_v1
)


def return_only_answer_source_docs_mist(response, citations_only= True):
  cites = []
  docs= response["context"]
  for doc in docs:
    cites.append(doc.metadata['source'])

  if citations_only:
    respstr = response["answer"] + "\n\nSources:\n\n" + "\n".join(cites)
  else:
    respstr = response
  return respstr


rag_chain_from_docs_mist = (
    prompt_official_mist
    | llmMist
    | StrOutputParser()
)


setupandret = RunnableParallel({"context": compression_retriever, "query": RunnablePassthrough()})
rag_chain_with_source = setupandret.assign(answer=rag_chain_from_docs_mist)
RESTapi_chain = rag_chain_with_source | RunnableLambda(return_only_answer_source_docs_mist)

print("restRAG import done!\n")
