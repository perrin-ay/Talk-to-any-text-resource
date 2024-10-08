## Try it on huggingface spaces

https://huggingface.co/spaces/bridge4/syslogs

---


- Multi service - Talk to appwall security events database,  alteon syslogs and Alteon REST API guide
- Deterministic routing to select service chain with keyword matching in user questions
- Talk to database: Converts database to pandas dataframe and uses LLM to generate pandas commands from natural language, in context of your dataframe, which is then sent to a python repl to execute generated commands and fetch response from the dataframe.
- Similarly talk to syslogs or any text based resources which can be converted to pandas dataframe

**Advantage of conversing with dataframes**

- Avoids RAG
- Overcomes limitations of RAG based question answering when dealing with data in csv , excel, db formats or where textual data contains events or logs.
- When dealing with aforementioned data and data sources , the types of questions asked are typically different from regular RAG approach of talking to a document
- For example , one may want to know : "Are there any cpu logs at time 10:55:14 ?", or "In appwall db show me the latest entries where URI contains string favicon" or "Show me the different type of appwall events under URI?" and so on..
- Typical RAG approaches struggle with such questions, but the approach utilizing dataframes produces great results.  


  
**For RAG**

- The third service to talk to REST API userguide is done using RAG
- Hybrid search with ensemble retriever: BM25 and semanctic search retrievers
- Rerank with Cohere
- Langchain LCEL chaining used

**User-interface** 

- Chat interface with Gradio
