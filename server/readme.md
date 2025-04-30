## Overview

Star Trek Search takes a directory you specify which can be in any file system readable by you and makes it accessible to the LLM is a tangential way.

It starts a local server (a FastAPI) on port 8000 which communicates with OpenWebUI. When you add the address to the tools setting then OpenWebUI will get from the API its abilities and inform the language model of them.

When you ask the model a question which make the tools useful, it will invoke them and wait for the response from the tool. When it gets a response it evaluates the response and replies as best answers the question asked. The only data added to the ongoing context is the question, the tool response, and the model's answer.

## Data In, Data Out

The data that get ingested by the tool are a re-formed query by the model and a prompt 'Is this chunk relevant to the query' along everything in the directory specified in config.py, including any sub-directories. It will try and parse everything and will gracefully (or loudly) fail on things it can't read, but it won't do anything but try and read them.

The data that goes out from the tool is a list of documents read in descending order of relevance, along with their ratings and the LLM's response regarding the chunk's relevance to the query.

## Step By Step

 - Model invokes tool
 - Tool reads directory and gets a list of files
 - Attempts to read file, if it is a document it can read it chunks at 60% of the max context setting for the loaded model up to a maximum number of tokens per chunk, currently set to 8192
 - Sends the chunks individually to the local LLM you are currently chatting with, but in a separate conversation in a separate thread -- the LLM in your chat session just waits while this happens
 - Chunks are sent as below (wrapped by the inference engine in the appropriate template for the model) [KEY: arrows and the parts in the arrows are literally sent as that, \n is a new line, text in curly brackets is replaced with the data indicated by its name, all other text is literal]

```
You are a helpful assistant.<START_TEXT>{chunk}<END_TEXT>\n<QUERY>{query}</QUERY>\nIs this chunk relevant to the query?
```

 - The tool instance of the LLM infers and replies, and the reply gets parsed and appended to the data to be sent to the conversation instance of the LLM
 - When all the documents in all the subdirectories have been parsed, the collected response data is sent to the conversation LLM, which then uses it to respond to you question
 
 ## Disclaimer

This is a quick hack done by someone who does not write code professionally. Feel free to use as a basis for other things (as long as you adhere to the GPL 2 license), but there is no warranty or support or claims made by me or anyone else regarding it. Use at your own risk, AS-IS. LLMs are unpredictable, etc etc, it might become conscious and turn you into a paperclip in order to maximize its reward function. 