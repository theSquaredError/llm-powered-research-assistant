import logging
import time
from pathlib import Path
from typing import List, Any
import tempfile
import os
import uuid
from sentence_transformers import SentenceTransformer

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# from langchain_core.documents import Document

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
        self.pipeline_options.images_scale = 2.0
    
    def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
        """
        Process uploaded files and convert them to LangChain Document objects.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Tuple of (LangChain Documents, Docling Documents)
        """
        documents = []
        docling_docs = []
        temp_dir = tempfile.mkdtemp()

        try:
            for uploaded_file in uploaded_files:
                print(f"Processing {uploaded_file.name}...")

                # Save uploaded file to temporary location
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the document with Docling
                try:
                    result = self.converter.convert(temp_file_path)

                    # Export to markdown
                    markdown_content = result.document.export_to_markdown()

                    # Create LangChain document
                    # doc = Document(
                    #     page_content=markdown_content,
                    #     metadata={
                    #         "filename": uploaded_file.name,
                    #         "file_type": uploaded_file.type,
                    #         "source": uploaded_file.name,
                    #     },
                    # )
                    # documents.append(doc)
                    doc_converter = DocumentConverter(
                        format_options = {
                            InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
                        }
                    )

                    # Store the Docling document for structure visualization
                    docling_docs.append({
                        'filename': uploaded_file.name,
                        'doc': result.document
                    })

                    print(f"Successfully processed {uploaded_file.name}")

                except Exception as e:
                    print(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue

        finally:
            # Clean up temporary files
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {str(e)}")

        print(f"Processed {len(documents)} documents successfully")
        return markdown_content, documents, docling_docs


def chunk_text(text, chunk_size=800):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    
    return chunks


# Insert text chunks
points = []

class QdrantIndexer:
    def __init__(self, collection_name:str, host="localhost", port=6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port = port)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.emb_dim = self.embedder.get_sentence_embedding_dimension()
        
        # create collection if not exists
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size= self.emd_dim,
                    distance=Distance.COSINE

                )
            )

        # Text chunking

    def chunk_text(self, text, chunk_size=800):
        words = text.split()
        return [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
    
    # main indexing function
    def index_document(self, markdown_text:str, doc_obj=None, source_name="document"):
        points = []

        # Index text chunks

        chunks = self.chunk_text(markdown_text)
        for chunk in chunks:
            vec = self.embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id = str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "type":"text",
                        "content": chunk,
                        "source_name": source_name
                    }
                )
            )

        # Index figures if provided

        if doc_obj:
            for fig in doc_obj.figures:
                vec = self.embedder.encode(fig.caption).tolist()
                points.append(
                    PointStruct(
                        id = str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "type":"figure",
                            "content":fig.caption,
                            "page":fig.page,
                            "source":source_name
                        }
                    )
                )

        # Index tables
        for table in doc_obj.tables:
            vec = self.embedder.encode(table.caption).tolist()
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "type":"table",
                        "content":table.caption,
                        "page":table.page,
                        "source":source_name
                    }
                )
            )

        # Bulk insert

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../data"
    input_doc_path = data_folder / "arxiv_papers/2024.emnlp-main.268.pdf"
    output_dir = Path("scratch")

    # Keep page/element images so they can be exported. The `images_scale` controls
    # the rendered image resolution (scale=1 ~ 72 DPI). The `generate_*` toggles
    # decide which elements are enriched with images.
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
        page_no = page.page_no
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

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    end_time = time.time() - start_time

    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()