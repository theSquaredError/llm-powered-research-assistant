import logging
import time
from pathlib import Path
from typing import List, Any
import tempfile
import os
from io import BytesIO
import uuid
from sentence_transformers import SentenceTransformer
import requests


from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION = "papers"


class DocumentProcessor:
    def __init__(self):
        # Configure pipeline options for PDF processing
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        
        # Initialize converter once in __init__
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        
    def process_pdf(self, file_bytes: BytesIO, filename:str) -> dict:
        temp_dir = tempfile.mkdtemp()
        try:
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes.getvalue())

            result = self.converter.convert(temp_file_path)
            markdown_content = result.document.export_to_markdown()

            return {
                'markdown': markdown_content,
                'doc': result.document,
                'filename': filename
            }
        
        except Exception as e:
            _log.error(f"Error processing PDF {filename}: {str(e)}")
            raise
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                _log.warning(f"Could not clean up temp directory: {str(e)}")

    def process_urls(self, urls):
        markdown_results = ""
        docs = []

        for url in urls:
            try:
                response = requests.get(url, timeout = 30)
                response.raise_for_status()

                filename = url.split('/')[-1] or "document"

                if url.lower().endswith('.pdf') or 'application/pdf' in response.headers.get('content-type', ''):
                    file_bytes = BytesIO(response.content)
                    doc = self.process_pdf(file_bytes, filename)
                    markdown_results +=f"\n\n## {filename}\n{doc['markdown']}"
                    docs.append({
                        'markdown':doc['markdown'],
                        'doc':doc['doc'],
                        'filename':filename
                    })
                    _log.info(f"successfully processed PDF from URI: {url}")
                
                else:
                    _log.warning(f"unsupported file type for URL: {url}")
                
            except requests.exceptions.Timeout:
                error_msg = f"Timeout downloading {url}"
                _log.error(error_msg)
                markdown_results += f"\n\n**Error:** {error_msg}"
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error downloading {url}: {str(e)}"
                _log.error(error_msg)
                markdown_results += f"\n\n**Error:** {error_msg}"
            except Exception as e:
                error_msg = f"Error processing URL {url}: {str(e)}"
                _log.error(error_msg)
                markdown_results += f"\n\n**Error:** {error_msg}"
        
        return markdown_results, docs


    def process_uploaded_files(self, uploaded_files) -> tuple[str, List[Any], List[Any]]:
        """
        Process uploaded files and convert them to markdown and Docling documents.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Tuple of (markdown_content, docling_documents, metadata)
        """
        markdown_contents = []
        docling_docs = []
        temp_dir = tempfile.mkdtemp()

        try:
            for uploaded_file in uploaded_files:
                _log.info(f"Processing {uploaded_file.name}...")

                # Save uploaded file to temporary location
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the document with Docling
                try:
                    result = self.converter.convert(temp_file_path)

                    # Export to markdown
                    markdown_content = result.document.export_to_markdown()
                    markdown_contents.append(markdown_content)

                    # Store the Docling document for structure and indexing
                    docling_docs.append({
                        'filename': uploaded_file.name,
                        'doc': result.document,
                        'markdown': markdown_content
                    })

                    _log.info(f"Successfully processed {uploaded_file.name}")

                except Exception as e:
                    _log.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue

        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                _log.warning(f"Could not clean up temp directory: {str(e)}")

        _log.info(f"Processed {len(docling_docs)} documents successfully")
        return "\n".join(markdown_contents), docling_docs


class QdrantIndexer:
    def __init__(self, collection_name: str, host="localhost", port=6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.emb_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create collection if not exists
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.emb_dim,
                    distance=Distance.COSINE
                )
            )
            _log.info(f"Created collection: {collection_name}")
    
    def clear_collection(self):
        """Delete and recreate the collection to remove old data"""

        try:
            self.client.delete_collection(collection_name=self.collection_name)
            _log.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            _log.warning(f"Could not delete collection: {str(e)}")

        
        # recreate empty collection

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.emb_dim,
                distance=Distance.COSINE
            )
        )
        _log.info(f"Recreated collection: {self.collection_name}")


    def index_document(self, doc_obj, source_name="document"):
        """
        Index a Docling using HierarchicalChunker
        """
        points = []
        chunker = HierarchicalChunker(
            max_tokens = 500,
            overlap = 50
        )
        chunks = list(chunker.chunk(doc_obj))
    
        for chunk in chunks:
            text = chunk.text.strip()

            # skip very small chunks
            if len(text) < 200:
                continue

            section_path = chunk.meta.headings or []
            enriched_text = " > ".join(section_path) + "\n\n" + text
            vector = self.embedder.encode(enriched_text).tolist()

            payload = {
                "type": "text",
                "content": text,
                "section_path": section_path,
                "source_name": source_name
            }

            points.append(
                PointStruct(
                    id = int(uuid.uuid4().int % (2**32)),
                    vector = vector,
                    payload=payload
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points = points
            )
            _log.info(f"Indexed {len(points)} chunks from {source_name}")
            
        return len(points)
        
    def retrieve(self, query: str, limit: int = 5, filter_type: str = None) -> List[dict]:
        """Retrieve relevant documents based on semantic similarity"""
        query_vec = self.embedder.encode(query).tolist()

        try:
            # Use query_points (available in your qdrant-client version)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=limit,
                with_payload=True
            )

            # Normalize results
            retrieved = []
            for r in results.points:
                retrieved.append({
                    "id": r.id,
                    "payload": r.payload or {},
                    "score": r.score
                })
            return retrieved

        except Exception as e:
            _log.error(f"Error during retrieve: {e}")
            return []


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../data"
    input_doc_path = data_folder / "arxiv_papers/2024.emnlp-main.268.pdf"
    output_dir = Path("scratch")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown variants
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    end_time = time.time() - start_time

    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()