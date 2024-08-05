import re
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

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

with open("alteon_syslogs.txt") as fd:
  txtdocuments = fd.read()

txts = txtdocuments.splitlines()
txts = [i[i.find(",2024")+1:] for i in txts]
dii = {"syslogs":txts}
df_syslogs = pd.DataFrame(dii)


##### prepending each entry with Month-day-year-time #################

from datetime import datetime

def prependdt(row):
  # Regular expression pattern to match the datetime format
  pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\+\-]\d{2}:\d{2}"
  time_str = re.findall(pattern, row)
  if len(time_str) > 0:
    time_str= time_str[0]
    #time_str = "2024-02-08T15:16:40+05:30"

    # Parse the string with format specifier considering timezone offset
    datetime_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S%z")

    # Format the datetime object as desired (Month-day-year-time)
    formatted_time = datetime_obj.strftime("%B-%d-%Y-%H:%M:%S")

    return formatted_time + " " + row
  else:
    print ("emty timestr: ", time_str, row)
    return row

df_syslogs['syslogs'] = df_syslogs['syslogs'].apply(prependdt)


########## convert to string lower for all data in syslog ###########

df_syslogs['syslogs'] = df_syslogs['syslogs'].str.lower()

############################################################################

python_repl_syslogs = PythonREPL()
python_repl_syslogs.globals['df'] = df_syslogs

# Permanently changes the pandas settings
python_repl_syslogs.run(pd.set_option('display.max_rows', None))
python_repl_syslogs.run(pd.set_option('display.max_columns', None))
python_repl_syslogs.run(pd.set_option('display.width', None))
#python_repl.run(pd.set_option('display.max_colwidth', -1))

name_syslogs = "python_repl_for_pandas_dataframes_alteon_syslogs"

description_syslogs= """
A Python shell to execute pandas commands on dataframe 'df' which contains alteon syslogs.
There is only one column in this df called 'syslogs'. 
The alteon syslogs are stored in df['syslogs'] in increasing order of time, 
such that the latest syslog is at the end and earliest syslog is at the beginning.
Output should be only the pandas commands converted to dict using to_dict() and no other words. 
The final answer should always be enclosed in a print() function
"""

repl_tool_syslogs = Tool(
    name=name_syslogs,
    description=description_syslogs,
    func=python_repl_syslogs.run

)

syslogs_llm = llmMist.bind_tools([repl_tool_syslogs])

sys_syslogs= """
When the user query is talking about 'syslogs' or 'alteon syslogs' or 'logs' or 'alteon logs', 
they have questions about the data stored in pandas dataframe df.
"""

prompt_syslogs = ChatPromptTemplate.from_messages([
        ("system", sys_syslogs),
        ("human", "{query}"),
        ])


def check_print(result: str) -> str:
  if 'print' in result:
    return result
  else:
    return f"print({result})"

def out_parser_syslogs(result: str) -> str:

  """Parses the output of python repl and returns the final formatted output"""
  no_answer = "I couldnt locate what you are asking for in the alteon syslogs"
  if not result:
    return no_answer

  finalls = []
  outtype = None
  
  if result[0] =="[":
    outtype = "list"
  elif result[0] == "{":
    outtype = "dict"
  else:
    outtype = "string"

  if outtype =="list":

    ls = list(eval(result))
    if isinstance(ls[0], dict):
      for i in ls:
        finalls.append("".join(i.values()))
      if finalls:  
        return "\n".join(finalls)
      else:
        return no_answer
    else:
      for i in ls:
        finalls.append(i)
      if finalls:
        return "\n".join(finalls)
      else:
        return no_answer

  elif outtype =="dict":

    ls = dict(eval(result))
    for i in ls:
      if isinstance(ls[i], dict):
        for j in ls[i]:
          finalls.append(ls[i][j])
        if finalls:
          return "\n".join(finalls)
        else:
          return no_answer
      else:
        for i in ls:
          finalls.append(ls[i])
        if finalls:
          return "\n".join(finalls)
        else:
          return no_answer

  elif outtype == "string":
    return result 

def toolingfunc_syslogs(x: str) -> str:
  try:
    return x.tool_calls[0]["args"]['__arg1']
  except:

    return "print('I dont have an answer to your query. Kindly try rephrasing your question or adding more specificity to it.')"

def notebook_rate_print(data: str) -> str:

  if not data:
    return data
  data_size = len(data)
  if data_size > 999000:
    return "Response data size exceeds display limit and will only be partially displayed. Please refine your query and ask again!\n\n" +data[:999000]
  else:
    return data



setupinputs = {"query": RunnablePassthrough()}

syslogs_chain = setupinputs| prompt_syslogs | syslogs_llm | RunnableLambda(toolingfunc_syslogs) | RunnableLambda(check_print) | python_repl_syslogs.run | RunnableLambda(out_parser_syslogs) |RunnableLambda(notebook_rate_print)
print("syslogchain import done!\n")








