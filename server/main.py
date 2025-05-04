import argparse
import json
import os
import re
import sys
import time
import requests
import base64
import fitz

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
import pathlib
import extractous

from fix_busted_json import first_json
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from chunker_regex import chunk_regex
from config import ALLOWED_DIRECTORIES


# Configuration with defaults, can be overridden by environment variables
DEFAULT_CONFIG = {
    "api_url": os.getenv("API_URL", "http://localhost:5002"),
    "api_password": os.getenv("API_PASSWORD", ""),
    "max_parallel_requests": int(os.getenv("MAX_PARALLEL_REQUESTS", "3"))
}

app = FastAPI(
    title="Real-time Document Evaluation",
    version="0.1.0",
    description="Creates a research assistant LLM to find relevant sources and return the scored and distilled data to the conversation LLM.",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_INSTRUCTION = """Review this chunk and determine if it has information which will help answer the query. The goal is not to answer the query, but to return information to be cited as an authoritative source.

There are two parts to this task:
1. Evaluate the relevance of the data in answering the query. Score relevance on a scale of 0 to 10 and return only the number
2. If the score is above 5, return the information that is relevant to the query as items in an array. ONLY text that is in the document is to be returned as this will act as the source of the information which another agent will use to answer the question

Respond with ONLY a JSON object as follows: {relevance_score: int, source_text: [str]}"""

IMAGE_INSTRUCTION = """Analyze this image and determine if it contains information which will help answer the query. The goal is not to answer the query, but to return information visible in the image that can be cited.

There are two parts to this task:
1. Evaluate the relevance of the image in answering the query. Score relevance on a scale of 0 to 10 and return only the number
2. If the score is above 5, describe in detail what is visible in the image that relates to the query, including any text, which should be returned verbatim. Be specific and factual about what you can see.

Respond with ONLY a JSON object as follows: {relevance_score: int, source_text: [str]} where source_text contains descriptions of relevant visual elements along with any image text."""

class InfoRequest(BaseModel):
    """ Request model for information retrieval """
    question: str = Field(..., description="The question we are trying to answer")
    keywords: str = Field(..., description="Comma separated keywords that will let us find the relevant documents")
    details: str = Field(..., description="Details which will be used to recognize relevant data which would otherwise not be obvious")

class ChunkResult(BaseModel):
    """ Model for individual chunk processing results """
    content: str
    tokens: int

class DocumentMetadata(BaseModel):
    """ Model for document metadata """
    document_type: Optional[str] = None
    page_count: Optional[int] = None
    processing_time: Optional[str] = None
    chunks_processed: Optional[int] = None
    error: Optional[str] = None
    
    class Config:
        extra = "allow"

class RelevanceResult(BaseModel):
    """ Model for relevance assessment results """
    relevance_score: int
    source_text: List[str] = []

class ProcessedChunk(BaseModel):
    """ Model for a processed chunk with relevance information """
    file_path: str
    chunk_index: int
    information: List[str]
    score: float
    
    class Config:
        arbitrary_types_allowed = True

class InfoResponse(BaseModel):
    """ Response model for information retrieval """
    results: List[ProcessedChunk]
    metadata: Dict

class ExtractedContent:
    """ Container for extracted document content, supporting both text and images """
    def __init__(self, content_type='text'):
        self.content_type = content_type  # 'text', 'image', or 'image_set'
        self.text_content = ""
        self.image_content = None  # Single base64 image
        self.image_set = []  # List of base64 images (for multi-page PDFs)
        self.page_count = 1

class ApiClient:
    """ Base class for API interactions """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_password = api_password
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
            
    def get_max_context_length(self) -> int:
        """ Get the maximum context length from the API """
        try:
            response = requests.get(f"{self.api_url}/props", headers=self.headers)
            if response.status_code == 200:
                max_context = int((response.json())["default_generation_settings"].get("n_ctx", 8192))
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

class DocumentChunker(ApiClient):
    """ Handles document retrieval and chunking """
    
    SUPPORTED_DOCUMENT_FORMATS = {
        'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'rtf', 'odt', 'ods', 'odp',
        'pdf', 'csv', 'tsv', 'html', 'xml', 'txt', 'md', 'eml', 'msg', 'mbox', 'pst'
    }
    
    SUPPORTED_IMAGE_FORMATS = {
        'png', 'bmp', 'jpeg', 'jpg'
    }
    
    def __init__(self, api_url: str, api_password: Optional[str] = None, max_total_chunks: int = 1000):
        super().__init__(api_url, api_password)
        self.max_total_chunks = max_total_chunks
        self.api_max_context = self.get_max_context_length()
        self.max_chunk = int(self.api_max_context * 0.75)  # Conservative chunk size
    
    def get_file_extension(self, file_path: Union[str, Path]) -> str:
        """ Get the lowercase file extension without the dot """
        return Path(file_path).suffix.lower().lstrip('.')
    
    def is_supported_format(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """ Check if the file format is supported and return the type """
        extension = self.get_file_extension(file_path)
        
        if extension in self.SUPPORTED_DOCUMENT_FORMATS:
            return True, "document"
        elif extension in self.SUPPORTED_IMAGE_FORMATS:
            return True, "image"
        else:
            return False, "unsupported"
    
    def _process_direct_image(self, file_path: Union[str, Path]) -> Tuple[ExtractedContent, DocumentMetadata]:
        """ Process direct image files """
        try:
            with open(file_path, "rb") as image_file:
                image_bytes = image_file.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                extracted_content = ExtractedContent(content_type="image")
                extracted_content.image_content = base64_image
                
                metadata = DocumentMetadata(
                    document_type="image",
                    processing_time=datetime.now().isoformat()
                )
                
                return extracted_content, metadata
                
        except Exception as e:
            error_msg = f"Error processing image file: {str(e)}"
            print(error_msg)
            return ExtractedContent(), DocumentMetadata(
                document_type="image",
                error=error_msg,
                processing_time=datetime.now().isoformat()
            )
    
    def _process_pdf_as_images(self, file_path) -> Tuple[ExtractedContent, DocumentMetadata]:
        """ Extract images from a PDF and process them """
        try:
            doc = fitz.open(file_path)
            extracted_content = ExtractedContent(content_type="image_set")
            extracted_content.page_count = len(doc)
            
            for page in doc:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("jpeg")
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                extracted_content.image_set.append(base64_image)
                
            metadata = DocumentMetadata(
                document_type="image_set",
                page_count=len(doc),
                processing_time=datetime.now().isoformat()
            )
            return extracted_content, metadata
     
        except Exception as e:
            error_msg = f"Error processing PDF file as images: {str(e)}"
            print(error_msg)
            return ExtractedContent(), DocumentMetadata(
                document_type="image_set",
                error=error_msg,
                processing_time=datetime.now().isoformat()
            )
            
    def _is_pdf_image_based(self, metadata, extracted_text_length):
        """
        Determines if a PDF is primarily image-based or text-based.
        Only call this method for actual PDF files.
        """
        if not metadata.get("Content-Type", [""])[0].lower().endswith("/pdf"):
            print("Warning: Non-PDF file sent to PDF image detection")
            return False
            
        pages = int(metadata.get("xmpTPg:NPages", [0])[0]) if "xmpTPg:NPages" in metadata else 0
        chars_per_page = metadata.get("pdf:charsPerPage", [])
        total_chars_from_metadata = sum(int(chars) for chars in chars_per_page) if chars_per_page else 0
        avg_chars_per_page = extracted_text_length / pages if pages > 0 else extracted_text_length
        text_length_matches_metadata = abs(total_chars_from_metadata - extracted_text_length) < 20
        
        content_length = int(metadata.get("Content-Length", [0])[0]) if "Content-Length" in metadata else 0
        bytes_per_char = content_length / extracted_text_length if extracted_text_length > 0 else 0
        
        is_image_based = False
        confidence = 0
        reasons = []
        
        if pages == 0:
            return False
            
        if avg_chars_per_page < 100:
            is_image_based = True
            confidence += 40
            reasons.append("Few characters per page")
        
        if bytes_per_char > 1000:
            is_image_based = True
            confidence += 30
            reasons.append("High bytes per character ratio")
        
        if text_length_matches_metadata and total_chars_from_metadata < 100 * pages:
            is_image_based = True
            confidence += 30
            reasons.append("Few total characters in metadata")
        
        if confidence < 50:
            is_image_based = False
        
        if is_image_based:
            print(f"PDF is probably composed of images (confidence: {confidence}%)")
        
        return is_image_based
    
    def extract_document(self, file_path: Union[str, Path]) -> Tuple[ExtractedContent, DocumentMetadata]:
        """ Extract content and metadata from a document """
        is_supported, file_type = self.is_supported_format(file_path)
        extension = self.get_file_extension(file_path)
        
        if not is_supported:
            error_msg = f"Unsupported file format: {extension}"
            print(error_msg)
            return ExtractedContent(), DocumentMetadata(error=error_msg)
        
        # Check if it is an image
        if file_type == "image":
            return self._process_direct_image(file_path)
        
        extracted_content = ExtractedContent()
        extractor = extractous.Extractor()
        extractor = extractor.set_extract_string_max_length(100000000)
        
        try:
            content, metadata_dict = extractor.extract_file_to_string(str(file_path))
            
            # Check if it is a PDF and if so if it is composed of images
            if extension == "pdf" and self._is_pdf_image_based(metadata_dict, len(content)):
                print(f"Detected image-based PDF: {file_path}")
                return self._process_pdf_as_images(file_path)
            
            extracted_content.content_type = "text"
            extracted_content.text_content = content
            
            metadata = DocumentMetadata(
                document_type="document",
                processing_time=datetime.now().isoformat()
            )
            return extracted_content, metadata
        
        except Exception as e:
            error_msg = f"Error extracting file: {str(e)}"
            print(error_msg)
            return ExtractedContent(), DocumentMetadata(error=error_msg)
    
    def chunk_text(self, content: str) -> List[ChunkResult]:
        """ Split content into chunks using natural breakpoints """
        if not content:
            return []
            
        chunks = []
        remaining = content
        chunk_num = 0
        
        while remaining and chunk_num < self.max_total_chunks:
            current_section = remaining[:45000]  # Process a manageable section
            remaining = remaining[45000:]
            
            chunk = self._get_chunk(current_section)
            chunk_len = len(chunk)
            
            if chunk_len == 0:
                continue
                
            chunk_tokens = self.count_tokens(chunk)
            chunks.append(ChunkResult(content=chunk, tokens=chunk_tokens))
            
            # Update remaining with what wasn't included in this chunk
            remaining = current_section[len(chunk):].strip() + remaining
            chunk_num += 1
            
        if remaining and chunk_num >= self.max_total_chunks:
            raise ValueError(f"Text exceeded maximum of {self.max_total_chunks} chunks")
            
        return chunks
         
    def _get_chunk(self, content: str) -> str:
        """ Get appropriately sized chunk using natural breaks """
        total_tokens = self.count_tokens(content)
        if total_tokens < self.max_chunk:
            return content

        # chunk_regex is designed to break at natural language points
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
    
    def process_file(self, file_path: Union[str, Path]) -> Tuple[List[Union[ChunkResult, str]], DocumentMetadata, str]:
        """ Process a single file into chunks or images with metadata """
        extracted_content, metadata = self.extract_document(file_path)
        
        if metadata.error:
            return [], metadata, "error"
        
        if extracted_content.content_type == "text":
            chunks = self.chunk_text(extracted_content.text_content)
            
            metadata_dict = metadata.dict()
            metadata_dict.update({
                'chunks_processed': len(chunks),
                'processing_time': datetime.now().isoformat()
            })
            
            return chunks, DocumentMetadata(**metadata_dict), "text"
            
        elif extracted_content.content_type == "image":
            return [extracted_content.image_content], metadata, "image"
            
        elif extracted_content.content_type == "image_set":
            return extracted_content.image_set, metadata, "image_set"
        
        return [], metadata, "unknown"

class ChunkProcessor(ApiClient):
    """ Handles processing and scoring of document chunks """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        super().__init__(api_url, api_password)
        
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url}/v1/chat/completions"
    
    def format_prompt(self, question: str, details: str, keywords: str, is_image: bool = False) -> str:
        """ Format the evaluation prompt with query details """
        instruction = IMAGE_INSTRUCTION if is_image else DEFAULT_INSTRUCTION
        return f"<QUESTION>{question}</QUESTION><DETAILS>{details}</DETAILS><KEYWORDS>{keywords}</KEYWORDS>\n\n{instruction}"
    
    def process_chunk(self, chunk_content: str, instruction: str,
                      max_tokens: int = 2048, temperature: float = 0.2,
                      is_image: bool = False) -> str:
        """ Process a single chunk with the LLM """
        
        # Image content
        if is_image:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{chunk_content}"
                            }
                        }
                    ]
                }
            ]
        else:
            # Text content
            combined_content = f"<CHUNK>{chunk_content}</CHUNK>{instruction}"
            messages = [{"role": "user", "content": combined_content}]
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "min_p": 0.05,
            "stream": False
        }
        
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
    
    def parse_response(self, response_text: str) -> Optional[RelevanceResult]:
        """ Parse the LLM response into a structured format """
        try:
            # Extract JSON from the response text
            json_content = first_json(response_text) if isinstance(response_text, str) else response_text
            result = json.loads(json_content)
            return RelevanceResult(**result)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return None

class DocumentEvaluator:
    """ Main coordinator for document evaluation process """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        self.chunker = DocumentChunker(api_url, api_password)
        self.processor = ChunkProcessor(api_url, api_password)
        self.relevance_threshold = 5  # Default threshold for relevance
    
    def process_document(self, file_path: Union[str, Path], instruction: str) -> List[ProcessedChunk]:
        """ Process a single document and return relevant chunks """
        is_supported, file_type = self.chunker.is_supported_format(file_path)
        
        if not is_supported:
            print(f"Skipping unsupported file: {file_path}")
            return []
        
        chunks_or_images, metadata, content_type = self.chunker.process_file(file_path)
        
        if not chunks_or_images or metadata.error:
            print(f"Error or no content from file {file_path}: {metadata.error or 'No content extracted'}")
            return []
        
        results = []
        
        is_image = content_type in ["image", "image_set"]
        
        for i, item in enumerate(chunks_or_images):
            try:
                if is_image:
                    # For images, use the image content and instruction
                    content = item  # Base64 encoded image
                    image_instruction = instruction.replace(DEFAULT_INSTRUCTION, IMAGE_INSTRUCTION)
                    response = self.processor.process_chunk(
                        chunk_content=content,
                        instruction=image_instruction,
                        is_image=True
                    )
                else:
                    # For text chunks, use the text content and default instruction
                    content = item.content  # ChunkResult object
                    response = self.processor.process_chunk(
                        chunk_content=content,
                        instruction=instruction,
                        is_image=False
                    )
                
                relevance = self.processor.parse_response(response)
                
                if relevance and relevance.relevance_score > self.relevance_threshold:
                    print(f"Item {i} from {file_path} - Relevance Score: {relevance.relevance_score}")
                    
                    results.append(ProcessedChunk(
                        file_path=str(file_path),
                        chunk_index=i,
                        information=relevance.source_text,
                        score=relevance.relevance_score
                    ))
                else:
                    print(f"Item {i} from {file_path} scored below threshold")
                    
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                
        return results
    
    def process_directory(self, directory: Union[str, Path], instruction: str, recursive: bool = True) -> List[ProcessedChunk]:
        """ Process all files in a directory """
        directory_path = Path(directory)
        if not directory_path.is_dir():
            print(f"Error: '{directory_path}' is not a directory")
            return []
            
        files = []
        if recursive:
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    files.append(Path(root) / filename)
        else:
            files = [directory_path / f for f in os.listdir(directory_path) 
                    if (directory_path / f).is_file()]
                    
        if not files:
            print(f"No files found in '{directory_path}'")
            return []
        
        supported_files = []
        unsupported_count = 0
        document_count = 0
        image_count = 0
        
        for file_path in files:
            is_supported, file_type = self.chunker.is_supported_format(file_path)
            if is_supported:
                supported_files.append(file_path)
                if file_type == "document":
                    document_count += 1
                elif file_type == "image":
                    image_count += 1
            else:
                unsupported_count += 1
                
        print(f"Found {len(supported_files)} supported files in '{directory_path}'")
        print(f"  - Documents: {document_count}")
        print(f"  - Images: {image_count}")
        print(f"  - Skipping {unsupported_count} unsupported files")
        
        if not supported_files:
            print("No supported files to process")
            return []
            
        all_results = []
        for file_path in supported_files:
            print(f"\nProcessing file: {file_path}")
            try:
                results = self.process_document(
                    file_path=file_path,
                    instruction=instruction
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing file {file_path}: {e}")
        
        return all_results

def normalize_path(requested_path: str) -> pathlib.Path:
    """ Ensure the requested path is within allowed directories """
    requested = pathlib.Path(os.path.expanduser(requested_path)).resolve()
    for allowed in ALLOWED_DIRECTORIES:
        if str(requested).lower().startswith(allowed.lower()):  # Case-insensitive check
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
    
def get_config():
    """ Get application configuration """
    return DEFAULT_CONFIG

@app.post("/realtime_data_eval", response_model=InfoResponse, summary="Use passages and data from the available documents to inform an answer to the query") 
def realtime_data_eval(
    data: InfoRequest = Body(...),
    config: Dict = Depends(get_config)
): 
    """ Perform the search using document evaluation """
    api_url = config["api_url"]
    api_password = config["api_password"]
    
    evaluator = DocumentEvaluator(
        api_url=api_url,
        api_password=api_password,
    )
    
    prompt = evaluator.processor.format_prompt(
        question=data.question,
        details=data.details,
        keywords=data.keywords
    )
    
    all_results = []
    
    for base_path in ALLOWED_DIRECTORIES:
        results = evaluator.process_directory(
            directory=base_path,
            instruction=prompt,
            recursive=True
        )
        all_results.extend(results)
    
    # Sort results by relevance score
    all_results.sort(key=lambda x: x.score, reverse=True)
    
    metadata = {
        "total_results": len(all_results),
        "question": data.question,
        "details": data.details,
        "keywords": data.keywords,
        "data_paths": ALLOWED_DIRECTORIES,
        "timestamp": datetime.now().isoformat()
    }
    
    return InfoResponse(results=all_results, metadata=metadata)
        
@app.get("/list_allowed_directories", summary="List access-permitted directories")
def list_allowed_directories():
    """
    Show all directories this server can access.
    """
    return {"allowed_directories": ALLOWED_DIRECTORIES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
