import gradio as gr
from huggingface_hub import InferenceClient
import llmmist
import appwallchain
import syslogchain
import restRAG
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from appwallchain import appwall_chain
from syslogchain import syslogs_chain
from restRAG import RESTapi_chain


def query_classifier(query):
    
  q = query.lower()
  cls = {'class':'','query':query}
    
  if 'rest' in q or 'api call' in q or 'ui' in q or 'gui' in q or 'api' in q or 'url' in q:
    cls['class'] = 'RESTapi'
    return cls  

  elif 'appwall' in q or 'cwaf' in q or 'waf' in q or 'database' in q or 'db' in q or 'aw' in q:
    cls['class'] = 'appwall'
    return cls

  elif 'syslog' in q or 'alteon syslog' in q or 'alteon log' in q:
    cls['class'] = 'syslogs'
    return cls

  else:
    cls['class'] = 'Other'
    return cls

def chain_selector(clsdict):

  otherans = """Hello! I am a chat service to talk to Alteon syslogs, or Appwall db. You can also ask me how to make REST calls to the Alteon.
Please phrase your query indicating which service you are interested in and ask again.

For example if you want to talk to Alteon syslogs , you can trying asking:

Are there any cpu events in the alteon syslogs ?
  """
  print ("Class selected : ",clsdict['class'] ) 

  if clsdict['class'] == 'RESTapi':
    return RESTapi_chain.invoke(clsdict['query']) 
      
  elif clsdict['class'] == 'appwall':
    return appwall_chain.invoke({'query':clsdict['query']})
      
  elif clsdict['class'] == 'syslogs':
    return syslogs_chain.invoke({'query':clsdict['query'].lower()})

  else:
    return otherans

query = "show me the latest MP CPU event in the syslogs ?"
query = "In aw db , how many entries under Description column contain AllowList?"
query = "how is apples made ?"
full_chain = RunnableLambda(query_classifier) | RunnableLambda(chain_selector)
#print ("query: ", query)
#resp =  full_chain.invoke(query)
#print (resp)

def chained_qa(message, history):
  return full_chain.invoke(message)

if __name__ == "__main__":
    gr.ChatInterface(
        chained_qa,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Talk to Alteon syslogs, WAF events or ask me how to make Alteon REST API calls", container=False, scale=7),
        title="Radware Services",
#        description="Alteon syslogs, WAF events & Alteon REST API",
        theme="soft",
        examples= ["In AW DB how many entries under Description column contain AllowList", 
                   "Are there any CPU events in the alteon syslogs ?",
                   "How to check Alteon syslog table via rest call?"],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch(share= True)




