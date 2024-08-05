import re
from llmmist import llmMist

import os
import glob
import time
import tempfile
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
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


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

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

################# gcloud storage for aw db #############################


##### Authenticate for gcloud to pull aw db####
def get_credentials():
    creds_json_str = os.getenv('gcloud_json') 
    if creds_json_str is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        temp.write(creds_json_str) # write in json format
        temp_filename = temp.name 

    return temp_filename


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= get_credentials()

### end of authentication####

###### gcs helper classes #####
class gcsLoad(object):
  def __init__(self, project_name = None, bucket_name=None):
      try:
          from google.cloud import storage
      except ImportError:
          raise ImportError(
              "Could not import google-cloud-storage python package. "
              "Please install it with `pip install google-cloud-storage`."
          )
      if project_name and bucket_name:
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.client = storage.Client(project=self.project_name)
        self.bucket = self.client.get_bucket(bucket_name)
      else:
        raise ValueError('Enter project and bucket name')

  def bucket_metadata(self):
    """Prints out a bucket's metadata."""

    print(f"ID: {self.bucket.id}")
    print(f"Name: {self.bucket.name}")
    print(f"Storage Class: {self.bucket.storage_class}")
    print(f"Location: {self.bucket.location}")
    print(f"Location Type: {self.bucket.location_type}")
    print(f"Retention Effective Time: {self.bucket.retention_policy_effective_time}")
    print(f"Retention Period: {self.bucket.retention_period}")
    print(f"Retention Policy Locked: {self.bucket.retention_policy_locked}")
    print(f"Object Retention Mode: {self.bucket.object_retention_mode}")
    print(f"Requester Pays: {self.bucket.requester_pays}")
    print(f"Self Link: {self.bucket.self_link}")
    print(f"Time Created: {self.bucket.time_created}")
    print(f"Labels: {self.bucket.labels}")

  def bucketContents(self, onlyfiles = False):
    self.listing = []
    for blob in self.bucket.list_blobs():
      if onlyfiles:
        if blob.name.endswith("/"):
          pass
        else:
          self.listing.append(blob.name)
      else:
        self.listing.append(blob.name)
    return self.listing

  def foldercontents(self, folder=''):
    """Returns blobs and not blob.name
    each blob element in the list looks like this
    [<Blob: lab_data_1, userguides/Alteon CLI User Guide.pdf, 1719864729164443>]
    """
    self.docs = []
    if folder == '':
      folder = None
    for blob in self.bucket.list_blobs(prefix=folder): # prefix filters to that folder within bucket
      if blob.name.endswith("/"):
          pass
      else:
        self.docs.append(blob)
    return self.docs

  def downloadFilesMemory(self, folder = None, files = None):
    """Downloads as bytes. Print only works if its text."""
    download = {}
    if folder != None:
      blobsls = self.foldercontents(folder=folder)
    elif files != None and isinstance(files, list):
      blobsls = [self.bucket.blob(i) for i in files]
    else:
      raise ValueError('Provide folder or file names')

    for blob in blobsls:
      contents = blob.download_as_bytes()
      download[blob.name] = contents
    return download

  def downloadFilesLocal(self, folder = None, files = None, destinationFolder ='/content/'):
      filenames = []
      if folder != None:
        blobsls = self.foldercontents(folder=folder)
      elif files != None and isinstance(files, list):
        blobsls = [self.bucket.blob(i) for i in files]
      else:
        raise ValueError('Provide folder or file names')
#      assert destinationFolder != '', 'Provide destination path for local download'

      for blob in blobsls:
        fname = destinationFolder+blob.name.replace('/','_')
        filenames.append(fname)
        blob.download_to_filename(fname) # cannot write to colab because of /
      return filenames

  def uploadFilesToGCS(self, source_file_name = '', gcsFileName = ''):
      """Placeholder for upload. NOT tested
      """
      assert gcsFileName != '', 'Provide destination file name in GCS bucket'
      assert source_file_name != '', 'Provide source file name'
      blob = self.bucket.blob(gcsFileName)
      generation_match_precondition = 0
      blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)
      print(
          f"File {source_file_name} uploaded to {gcsFileName}."
      )


  def langchainLoader(self, gcsfiles = None, maketempfiles = True, loader_func=UnstructuredFileLoader, custom_metadata={}):
    """takes gcs file names and create langchain document objects.
    """

    docs = []
    if gcsfiles != None and isinstance(gcsfiles, list):
      blobsls = [self.bucket.blob(i) for i in gcsfiles]
    if maketempfiles:
      with tempfile.TemporaryDirectory() as temp_dir:
        for blob in blobsls:
          file_path = f"{temp_dir}/{blob.name.replace('/','_')}"
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          blob.download_to_filename(file_path)
          loader = loader_func(file_path)
          docs.extend(loader.load())
          if custom_metadata:
            for doc in docs:
              doc.metadata.update(custom_metadata)
          else:
            for doc in docs:
                if "source" in doc.metadata:
                    doc.metadata["source"] = f"gs://{self.bucket_name}/{blob.name}"
      return docs

#### end of gcs helper class #########

### instantiate helper class , download in memory##
gcs = gcsLoad(project_name = "My Project", bucket_name="mybucket")
filesdownloaded = gcs.downloadFilesLocal(folder='myfolder',destinationFolder ='')
print ("Downloaded filename", filesdownloaded)

#################################################################################################################



def epochtotzone(ep,tzo= None):
  if not tzo:
    tzo = "Asia/Jerusalem"
  tzn=timezone(tzo)
  try:
      t=datetime.datetime.utcfromtimestamp(ep).replace(tzinfo=pytz.utc)
      x= t.astimezone(tzn)
      return t.strftime("%Y-%m-%dT%H:%M:%S")
  except Exception as e:
      print ('datetime conversion issue')
      print( e)
      print ('ep is:', ep)

def aw_to_pd(f,ftype='security',tzone=None):

  jsonlist=[]
  
  if 'security' in ftype:
      #colls=['DateTime','TargetID','TargetType','TargetPort','TargetIP','TransID','TunnelID','VHostID','VDID','IsPassiveMode','Title','ParamName','ParamValue','Parameters','ParamType','ServerID','URI','Description','Geo']
      colls= ['DateTime','TargetPort','TargetIP','TransID','TunnelID','IsPassiveMode','Title','URI','Description']
  elif 'system' in ftype:
      colls=['DateTime','Title','Description']

  con = sqlite3.connect(f)
  con.text_factory = lambda b: b.decode(errors = 'ignore') # added to ignore utf-8 decode failure
  cursor = con.cursor()
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
  qu='SELECT * from %s'%'Events'
  df=pd.read_sql(qu,con)
  print ("original df cols", df.columns.to_list())
  #df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%dT%H:%M:%S')
  # seletcing only df which dont have nan in datetime col
  df=df[~df['DateTime'].isna()]
  df['DateTime']=df['DateTime'].apply(epochtotzone,args=(tzone,))
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df=df[colls]
  df['Date']= df['DateTime']
  df = df.drop('DateTime', axis=1, inplace=False)
  df['Date'] = df['Date'].astype(str)
  #df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%dT%H:%M:%S')
  if 'security' in ftype:
      df['TransID'] = df['TransID'].astype(str)
  return df

df_aw = aw_to_pd(filesdownloaded[0]) # point to downloaded file from gcs

##### delete downloaded file ####

for f in filesdownloaded:
  for filename in glob.glob(f):
      print ("deleting file", filename)
      try:
          os.remove(filename)
      except Exception as e:
          print ("exception while deleting file", e)
      try:
          os.remove(filename+"-shm")
      except Exception as e:
          print ("exception while deleting file", e)    
      try:    
          os.remove(filename+"-wal")
      except Exception as e:
          print ("exception while deleting file", e)  

#############################################3

python_repl_appwall = PythonREPL()
python_repl_appwall.globals['df'] = df_aw


# Permanently changes the pandas settings
python_repl_appwall.run(pd.set_option('display.max_rows', None))
python_repl_appwall.run(pd.set_option('display.max_columns', None))
python_repl_appwall.run(pd.set_option('display.width', None))
python_repl_appwall.run(pd.set_option('display.max_colwidth', 500))


name_appwall = "python_repl_for_pandas_dataframes_appwall_database"

description_appwall_2= """
A Python shell to execute pandas commands on appwall dataframe 'df' which contains appwall database (db).
The column names in this pandas dataframe df can be found in this list = ['Date','TargetPort','TargetIP','TransID','TunnelID','IsPassiveMode','Title','URI','Description']
The dataframe df contains entries in increasing order of time,
such that the latest entry is at the end and earliest entry is at the beginning.
The final answer should always be enclosed in a print() function
"""

repl_tool_appwall = Tool(
    name=name_appwall,
    description=description_appwall_2,
    func=python_repl_appwall.run

)


appwall_llm = llmMist.bind_tools([repl_tool_appwall])

sys_appwall= """
Only if user query is talking about 'appwall' or 'WAF' or 'AW' or 'CWAF' or 'database' or 'db',
do they have questions about the data stored in the pandas dataframe df for appwall database.
"""

prompt_appwall = ChatPromptTemplate.from_messages([
        ("system", sys_appwall),
        ("human", "{query}"),
        ])

setupinputs = {"query": RunnablePassthrough()}


def check_print(result: str) -> str:
  if 'print' in result:
    return result
  else:
    return f"print({result})"

def out_parser_appwall(result: str) -> str:

  """Parses the output of python repl and returns the final formatted output"""

  no_answer = "I couldnt locate what you are asking for in the appwall db."
  attributeerr = "I dont have an answer to your query. Please try to revise your question and ask again. "
  if not result:
    return no_answer
  if 'Empty DataFrame' in result:
    return no_answer
  if 'AttributeError' in result:
    return attributeerr
  return result
  
def toolingfunc_appwall(x: str) -> str:
  try:
    return x.tool_calls[0]["args"]['__arg1']
  except:
    return "print('I dont have an answer to your query. Kindly try rephrasing your question or adding more specificity to it.')"


def notebook_rate_print(data: str) -> str:
#  data_size = sys.getsizeof(data)
  if not data:
    return data
  data_size = len(data)
  if data_size > 999000:
    return "Response data size exceeds display limit and will only be partially displayed. Please refine your query and ask again!\n\n" +data[:999000]
  else:
    return data


appwall_chain = setupinputs| prompt_appwall | appwall_llm | RunnableLambda(toolingfunc_appwall) | RunnableLambda(check_print) | python_repl_appwall.run | RunnableLambda(notebook_rate_print) | RunnableLambda(out_parser_appwall)
print("appwallchain import done!\n")




