import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import requests
import pathlib
import extractous

from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
from requests.exceptions import RequestException

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from chunker_regex import chunk_regex
from config import ALLOWED_DIRECTORIES

app = FastAPI(
    title="Sart Rekt Search",
    version="0.0.1",
    description="Document searching by sending chunks of files to the LLM.",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_path(requested_path: str) -> pathlib.Path:
    requested = pathlib.Path(os.path.expanduser(requested_path)).resolve()
    for allowed in ALLOWED_DIRECTORIES:
        if str(requested).lower().startswith(allowed.lower()): # Case-insensitive check
            return requested
    raise HTTPException(
        status_code=403,
        detail={
            "error": "Access Denied",
            "requested_path": str(requested),
            "message": "Requested path is outside allowed directories.",
            "allowed_directories": ALLOWED_DIRECTORIES,
        },
    )

class SearchRequest(BaseModel):
    search_query: str = Field(..., description="What are we looking for.")
    recursive: bool = Field(
        default=True, description="Whether to search recursively in subdirectories."
    )
    prompt: str = Field(
        default="""Is this chunk relevant to the query? Rate the relevance from 0 being completely irrelevant to 10 being an exact answer. If it is above 5 use the chunk to remark with a brief summary the relevant data and attempt to answer the query with it or relate the important details. If it is below 5, remark ONLY that it is irrelevant. Respond in a JSON object like so ```json {'Relevance': int, 'Remark': str}```""", description="Prompt"
    )

class SearchResult(BaseModel):
    file_path: str
    chunk_index: int
    content: str
    score: float
    metadata: Dict

class SearchResponse(BaseModel):
    results: List[SearchResult]
    metadata: Dict
    
class ChunkingProcessor:
    """ Handles splitting content into manageable chunks using natural breaks """
    def __init__(self, api_url: str, 
                 max_chunk_length: int,
                 api_password: Optional[str] = None,
                 max_total_chunks: int = 1000):

        if max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be positive")
        self.api_url = api_url
        self.max_chunk = max_chunk_length
        self.max_total_chunks = max_total_chunks
        self.api_password = api_password
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
        self.api_max_context = self._get_max_context_length()
        if self.max_chunk > self.api_max_context // 2:
            print(f"Warning: Reducing chunk size to fit model context window")
            self.max_chunk = self.api_max_context // 2
        
    def _get_max_context_length(self) -> int:
        """ Get the maximum context length  """
        try:
            response = requests.get(f"{self.api_url}/props")
            if response.status_code == 200:
                max_context = int((response.json())["default_generation_settings"].get("n_ctx", 8192))
                print(f"Model has maximum context length of: {max_context}")
                return max_context
            else:
                print(f"Warning: Could not get max context length. Defaulting to 8192")
                return 8192
        except Exception as e:
            print(f"Error getting max context length: {str(e)}. Defaulting to 8192")
            return 8192
            
    def count_tokens(self, text: str) -> int:
        """ Count tokens in the provided text """
        try:
            payload = {
                "content": text,
                "add_special": False,
                "with_pieces": False
            }
            response = requests.post(
                f"{self.api_url}/tokenize",
                json=payload,
                headers=self.headers
            )
            if response.status_code == 200:
                return len(response.json().get("tokens", []))
            else:
                # Fallback estimation
                return len(text.split()) 
        except Exception as e:
            print(f"Error counting tokens: {str(e)}. Using word count as estimate.")
            return len(text.split())
    
    def chunk_text(self, content: str) -> List[Tuple[str, int]]:
        """ Split content into chunks using natural breakpoints
        """
        if not content:
            return []
        chunks = []
        remaining = content
        chunk_num = 0
        
        while remaining and chunk_num < self.max_total_chunks:
            current_section = remaining[:45000]
            remaining = remaining[45000:]
            chunk = self._get_chunk(current_section)
            chunk_len = len(chunk)
            
            if chunk_len == 0:
                continue
            chunk_tokens = self.count_tokens(chunk)
            chunks.append((chunk, chunk_tokens))
            
            # Update remaining with what wasn't included in this chunk
            remaining = current_section[len(chunk):].strip() + remaining
            chunk_num += 1
            print(f"Created chunk {chunk_num}: {chunk_tokens} tokens")
            
        if remaining and chunk_num >= self.max_total_chunks:
            raise ValueError(f"Text exceeded maximum of {self.max_total_chunks} chunks")
        return chunks

    def _get_chunk(self, content: str) -> str:
        """ Get appropriately sized chunk using natural breaks
        """
        total_tokens = self.count_tokens(content)
        if total_tokens < self.max_chunk:
            return content

        # chunk_regex is designed to break at natural language points
        # to preserve context and readability
        matches = chunk_regex.finditer(content)
        current_size = 0
        chunks = []
        
        for match in matches:
            chunk = match.group(0)
            chunk_size = self.count_tokens(chunk)
            if current_size + chunk_size > self.max_chunk:
                if not chunks:
                    chunks.append(chunk)
                break
            chunks.append(chunk)
            current_size += chunk_size
        return ''.join(chunks)

    def chunk_file(self, file_path) -> Tuple[List[Tuple[str, int]], Dict]:
        """ Chunk text from file
        """
        extractor = extractous.Extractor()
        extractor = extractor.set_extract_string_max_length(100000000)
        
        try:
            content, metadata = extractor.extract_file_to_string(str(file_path))
            chunks = self.chunk_text(content)
            return chunks, metadata
        except Exception as e:
            print(f"Error extracting file: {str(e)}")
            return [], {"error": str(e)}


class SSEProcessingClient:
    """ Client for processing chunks with OpenAI-compatible endpoints via SSE streaming """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        self.api_url = api_url
        self.api_password = api_password
        
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
    
    def _create_payload(self, instruction: str, content: str, 
                       max_tokens: int = 2048,
                       temperature: float = 0.2, 
                       top_p: float = 1.0,
                       top_k: int = 0,
                       rep_pen: float = 1.0,
                       min_p: float = 0.05) -> Dict:
        
        system_content = "You are a helpful assistant."
        combined_content = f"<START_TEXT>{content}<END_TEXT>\n{instruction}"
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": rep_pen,
            "min_p": min_p,
            "stream": True
        }
    
    async def process_chunk(self, instruction: str, content: str,
                         max_tokens: int = 2048,
                         temperature: float = 0.2) -> str:
        """ Process a single chunk with streaming output
        """
        payload = self._create_payload(
            instruction=instruction,
            content=content,
            max_tokens=max_tokens,
            temperature=temperature
        )
        result = []
        partial_line = ""
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                stream=True
            )
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove the "data: " prefix and decode
                line_text = line.decode('utf-8')
                
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                # Handle the "[DONE]" message
                if line_text == '[DONE]':
                    break
                    
                # Parse the JSON from the SSE stream
                try:
                    data = json.loads(line_text)
                    
                    # Extract the delta content from the received data
                    if 'choices' in data and len(data['choices']) > 0:
                        if 'delta' in data['choices'][0]:
                            if 'content' in data['choices'][0]['delta']:
                                token = data['choices'][0]['delta']['content']
                                # Handle any newlines in the token
                                if token.endswith('\n'):
                                    partial_line += token[:-1]
                                    print(partial_line)
                                    partial_line = ""
                                elif '\n' in token:
                                    parts = token.split('\n')
                                    for i, part in enumerate(parts):
                                        if i < len(parts) - 1:
                                            print(partial_line + part)
                                            partial_line = ""
                                        else:
                                            partial_line += part
                                else:
                                    partial_line += token
                                
                                result.append(token)
                except json.JSONDecodeError:
                    continue
            if partial_line:
                print(partial_line)
            return ''.join(result)   
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return ""

class TextProcessor:
    def __init__(self, api_url: str, 
                 max_chunk_size: int = 4096,
                 api_password: Optional[str] = None):
        self.api_url = api_url
        self.api_password = api_password
        self.chunker = ChunkingProcessor(
            api_url=api_url,
            max_chunk_length=max_chunk_size,
            api_password=api_password
        )
        self.processor = SSEProcessingClient(
            api_url=api_url,
            api_password=api_password
        )
    async def process_text(self, file_path, prompt):
        self.chunker.max_chunk = int(self.chunker.api_max_context * 0.6)
        chunks, metadata = self.chunker.chunk_file(file_path)
        
        print(f"\nProcessing {len(chunks)} chunks...")
        results = []
        
        for i, (chunk, tokens) in enumerate(chunks, 1):
            print(f"\nChunk {i}/{len(chunks)} ({tokens} tokens):")
            try:
                result = await self.processor.process_chunk(
                    instruction=prompt,
                    content=chunk
                )
                results.append(result)
            except Exception as e:
                print(f"\nError processing chunk {i}: {e}")
                results.append(f"[Error processing chunk {i}]")
        metadata.update({
            'processing_time': datetime.now().isoformat(),
            'chunks_processed': len(chunks),
        })
        return results, metadata

def write_output(output_path: str, results: List[str], metadata: Dict) -> None:
    """ Write processing results to a file
    Args:
        output_path: Path to output file
        results: List of processed text chunks
        metadata: Metadata about the processing
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"File: {metadata.get('resourceName', 'Unknown')}\n")
            f.write(f"Type: {metadata.get('Content-Type', 'Unknown')}\n")
            f.write(f"Processed: {metadata.get('processing_time', 'Unknown')}\n")
            f.write(f"Chunks: {metadata.get('chunks_processed', 0)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"--- Chunk {i} ---\n\n")
                f.write(f"{result}\n\n")   
        print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

async def process_file(api_url: str, input_path: Path, 
                      output_path: Optional[str] = None,
                      max_chunk_size: int = 4096,
                      api_password: Optional[str] = None,
                      prompt: Optional[str] = None,
                      output_to_console: bool = False) -> List[Dict]:
    """ Process a text file and save results
    """
    if not output_path and not output_to_console:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_processed.txt"

    try:
        processor = TextProcessor(
            api_url=api_url,
            max_chunk_size=max_chunk_size,
            api_password=api_password
        )
        results, metadata = await processor.process_text(
            file_path=input_path, 
            prompt=prompt
        )
        
        # Only write to file if output_to_console is False
        if not output_to_console and output_path:
            write_output(output_path, results, metadata)
            print("\nProcessing complete.")
        
        # Format results for return
        formatted_results = []
        for i, result in enumerate(results):
            # Extract a score from the result text
            score_match = re.search(r"relevance:?\s*(\d+(?:\.\d+)?)/10", result.lower())
            if score_match:
                score = float(score_match.group(1)) / 10.0
            else:
                # Look for any number that might represent a score
                score_match = re.search(r"(\d+(?:\.\d+)?)/10", result.lower())
                if score_match:
                    score = float(score_match.group(1)) / 10.0
                else:
                    # Default score
                    if "relevant" in result.lower():
                        score = 0.8
                    else:
                        score = 0.0
            
            formatted_results.append({
                "file_path": str(input_path),
                "chunk_index": i,
                "content": result,
                "score": score,
                "metadata": metadata.copy()
            })
        return formatted_results
    except Exception as e:
        print(f"Error: {e}")
        return []


async def process_directory(api_url: str, directory_path: Path,
                           max_chunk_size: int = 4096,
                           api_password: Optional[str] = None,
                           prompt: Optional[str] = None,
                           recursive: bool = True) -> List[Dict]:
    """     
    Returns a list of results with metadata
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory")
        return []
        
    processor = TextProcessor(
        api_url=api_url,
        max_chunk_size=max_chunk_size,
        api_password=api_password
    )
    
    # Get all files in directory (and subdirectories if recursive)
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    else:
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
    if not files:
        print(f"No files found in '{directory_path}'")
        return []
        
    print(f"Found {len(files)} files in '{directory_path}'")
    all_results = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path}")
        try:
            file_results = await process_file(
                api_url=api_url,
                input_path=file_path,
                max_chunk_size=max_chunk_size,
                api_password=api_password,
                prompt=prompt,
                output_to_console=True
            )
            all_results.extend(file_results)
        except Exception as e:
            print(f"  Error processing file {file_path}: {e}")
    
    # Sort results by score in descending order
    all_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nProcessed {len(all_results)} chunks across {len(files)} files.")
    return all_results

@app.post("/star_trek_search", response_model=SearchResponse, summary="Go through all data like Data") 
async def star_trek_search(data: SearchRequest = Body(...)):   
    """ Perform the search """
    api_url = "http://localhost:5002"
    max_chunk_size = 8192
    api_password = ""
    prompt = "\n<QUERY>" + data.search_query + "</QUERY>\n" + data.prompt
    base_path = ALLOWED_DIRECTORIES[0] #Only the first directory in the list is used

    results = await process_directory(
        api_url=api_url,
        directory_path=base_path,
        max_chunk_size=max_chunk_size,
        api_password=api_password,
        prompt=prompt,
        recursive=data.recursive
    )

    search_results = []
    for result in results:
        search_results.append(SearchResult(
            file_path=result["file_path"],
            chunk_index=result["chunk_index"],
            content=result["content"],
            score=result["score"],
            metadata=result["metadata"]
        ))
        
    metadata = {
        "total_results": len(search_results),
        "search_query": data.search_query,
        "search_path": base_path,
        "timestamp": datetime.now().isoformat()
    }
    return SearchResponse(results=search_results, metadata=metadata)
        
@app.get("/list_allowed_directories", summary="List access-permitted directories")
async def list_allowed_directories():
    """
    Show all directories this server can access.
    """
    return {"allowed_directories": ALLOWED_DIRECTORIES}
