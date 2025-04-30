import argparse
import json
import os
import random
import re
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
import pathlib
import extractous

from fix_busted_json import first_json
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from chunker_regex import chunk_regex
from config import ALLOWED_DIRECTORIES


INSTRUCTION = """Review this chunk and determine if has information which will help answer the query. The goal is not to answer the query, but to return information to be cited as an athoritative source.

There are two parts to this task:
1. Evaluate the revelance of the data in answering the query. Score revelance on a scale of 0 to 10 and return only the number
2. If the score is above 5, return the all information that is relevant to the query as items in an array. ONLY the text in the document is to be returned as this will act as the source of the information which another agent will use to answer the question.

Respond with ONLY a JSON object as follows: {relevance_score: int, source_text: [str]}"""

# Configuration with defaults, can be overridden by environment variables
DEFAULT_CONFIG = {
    "api_url": os.getenv("API_URL", "http://localhost:5002"),
    "api_password": os.getenv("API_PASSWORD", ""),
    "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "5")),
    "max_parallel_requests": int(os.getenv("MAX_PARALLEL_REQUESTS", "3"))
}

app = FastAPI(
    title="Real-time Document Evaluation",
    version="0.0.1",
    description="Sends the full text of all documents to be examined for potential usefulness by the LLM.",
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

class InfoRequest(BaseModel):
    question: str = Field(..., description="The specific question we are trying to answer")
    keywords: str = Field(..., description="Comma separated keywords that will let us find the relevant documents")
    details: str = Field(..., description="Any details that are important")
    
class InfoResult(BaseModel):
    file_path: str
    chunk_index: int
    information: List[str]
    score: float
    metadata: Dict

class InfoResponse(BaseModel):
    results: List[InfoResult]
    metadata: Dict
    
class ChunkingProcessor:
    """ Handles splitting content into manageable chunks using natural breaks """
    def __init__(self, api_url: str, 
                 api_password: Optional[str] = None,
                 max_total_chunks: int = 1000):

        self.api_url = api_url
        self.max_total_chunks = max_total_chunks
        self.api_password = api_password
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
        self.api_max_context = self._get_max_context_length()
        self.max_chunk = int(self.api_max_context * 0.7)
        
    def _get_max_context_length(self) -> int:
        """ Get the maximum context length  """
        try:
            response = requests.get(f"{self.api_url}/props", headers=self.headers)
            if response.status_code == 200:
                max_context = int((response.json())["default_generation_settings"].get("n_ctx", 8192))
                #print(f"Model has maximum context length of: {max_context}")
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


class ProcessingClient:
    """ Client for processing chunks with OpenAI-compatible endpoints """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        self.api_url = api_url
        self.api_password = api_password
        
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
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
        
        combined_content = f"<CHUNK>{content}</CHUNK>{instruction}"
        return {
            "messages": [
                {"role": "user", "content": combined_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": rep_pen,
            "min_p": min_p,
            "stream": False
        }
    
    def process_chunk(self, instruction: str, content: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.2) -> str:
        """ Process a single chunk
            Returns the complete response as a string.
        """
        payload = self._create_payload(
            instruction=instruction,
            content=content,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
            )
            
            if response.status_code != 200:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Extract content from response
            if 'choices' in data and len(data['choices']) > 0:
                if 'message' in data['choices'][0]:
                    return data['choices'][0]['message'].get('content', '')
                
            return ""
            
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            raise

class TextProcessor:
    def __init__(self, api_url: str, 
                 api_password: Optional[str] = None):
        self.api_url = api_url
        self.api_password = api_password
        self.chunker = ChunkingProcessor(
            api_url=api_url,
            api_password=api_password
        )
        self.processor = ProcessingClient(
            api_url=api_url,
            api_password=api_password
        )
        
    def process_text(self, file_path, prompt):
        chunks, metadata = self.chunker.chunk_file(file_path)
        
        print(f"Processing {len(chunks)} chunks...")
        results = []
        
        for i, (chunk, tokens) in enumerate(chunks, 1):
            print(f"Chunk {i}/{len(chunks)} ({tokens} tokens)")
            try:
                result = self.processor.process_chunk(
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
    
def process_file(api_url: str, input_path: Path, 
                 max_chunk_size: int = 4096,
                 api_password: Optional[str] = None,
                 prompt: Optional[str] = None):
    """ Process a text file and save results
    """

    try:
        processor = TextProcessor(
            api_url=api_url,
            api_password=api_password
        )
        results, metadata = processor.process_text(
            file_path=input_path, 
            prompt=prompt
        )
        
        formatted_results = []
        for i, result in enumerate(results):
            try:
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(first_json(result))
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing JSON from chunk {i}: {e}")
                        continue
                else:
                    parsed_result = result
                
                # Only include results with relevance score > 5
                if parsed_result.get("relevance_score", 0) > 5:
                    formatted_results.append({
                        "file_path": str(input_path),
                        "chunk_index": i,
                        "information": parsed_result.get("source_text", []),
                        "score": parsed_result.get("relevance_score", 0),
                        "metadata": metadata.copy()
                    })
            except Exception as e:
                print(f"Error processing result from chunk {i}: {e}")
                continue
                
        return formatted_results
    except Exception as e:
        print(f"Error: {e}")
        return []


def process_directory(api_url: str, directory_path: Path,
                      api_password: Optional[str] = None,
                      prompt: Optional[str] = None,
                      recursive: bool = True,
                      max_batch_size: int = 5):
    """     
    Process all files in a directory with batching
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory")
        return []
        
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
    
    for batch_start in range(0, len(files), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(files))
        batch = files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start // max_batch_size + 1} ({batch_start + 1}-{batch_end} of {len(files)})")
        
        for file_path in batch:
            print(f"\nProcessing file: {file_path}")
            try:
                results = process_file(
                    api_url=api_url,
                    input_path=file_path,
                    api_password=api_password,
                    prompt=prompt,
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing file {file_path}: {e}")
        
        if batch_end < len(files):
            time.sleep(1)
    
    # Sort results by score in descending order
    all_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nProcessed {len(all_results)} chunks across {len(files)} files.")
    return all_results

def get_config():
    return DEFAULT_CONFIG

@app.post("/realtime_data_eval", response_model=InfoResponse, summary="Chunk all documents and evaluate for relevance") 
def realtime_data_eval(
    data: InfoRequest = Body(...),
    config: Dict = Depends(get_config)
): 
    """ Perform the search using document evaluation """
    api_url = config["api_url"]
    api_password = config["api_password"]
    max_batch_size = config["max_batch_size"]
    
    prompt = f"<QUESTION>{data.question}</QUESTION><DETAILS>{data.details}</DETAILS><KEYWORDS>{data.keywords}</KEYWORDS>\n\n{INSTRUCTION}"
    
    base_paths = ALLOWED_DIRECTORIES
    all_results = []
    
    for base_path in base_paths:
        results = process_directory(
            api_url=api_url,
            directory_path=base_path,
            api_password=api_password,
            prompt=prompt,
            recursive=True,
            max_batch_size=max_batch_size
        )
        all_results.extend(results)
    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    info_results = []
    for result in all_results:
        info_results.append(InfoResult(
            file_path=result["file_path"],
            chunk_index=result["chunk_index"],
            information=result["information"],
            score=result["score"],
            metadata=result["metadata"]
        ))
        
    metadata = {
        "total_results": len(info_results),
        "question": data.question,
        "details": data.details,
        "keywords": data.keywords,
        "data_paths": base_paths,
        "timestamp": datetime.now().isoformat()
    }
    return InfoResponse(results=info_results, metadata=metadata)
        
@app.get("/list_allowed_directories", summary="List access-permitted directories")
def list_allowed_directories():
    """
    Show all directories this server can access.
    """
    return {"allowed_directories": ALLOWED_DIRECTORIES}
