## Overview

Star Trek Search takes a directory you specify which can be in any file system readable by you and makes it accessible to the LLM is a tangential way.

It starts a local server (a FastAPI) on port 8000 which communicates with OpenWebUI. When you add the address to the tools setting then OpenWebUI will get from the API its abilities and inform the language model of them.

When you ask the model a question which make the tools useful, it will invoke them and wait for the response from the tool. When it gets a response it evaluates the response and replies as best answers the question asked. The only data added to the ongoing context is the question, the tool response, and the model's answer.

## Researcher

The tool spawns a 'researcher' agent which has as its task to go through and find relevant document snippets. It is directed not to try and answer the question but to rate the relevance of the document and if high enough to extract the parts which will help answer the question. 

After all the documents have been scored then information is collated and sent back to the conversation model instance.

## Data In, Data Out

The data that get ingested are everything in the directory specified in config.py, including any sub-directories. It will try and parse everything and will gracefully (or loudly) fail on things it can't read, but it won't do anything but try and read them.

The data that goes out from the tool is a relevance score, the relevant information from the document, and the file metadata.
 
 ## Disclaimer

There is no warranty or support for this tool. Use at your own risk. LLMs are unpredictable, etc etc, it might become conscious and turn you into a paperclip in order to maximize its reward function. 
