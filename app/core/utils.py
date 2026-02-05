# from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import re
import logging
import time
from pathlib import Path
from typing import List, Any
import tempfile
import os

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

class DocumentProcessor:
    def __init__(self):
        # configure pipeline options for PDF processing
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = 2.0
    
    def process_file(self, pdf_path, output_dir):
        doc_converter = DocumentConverter(
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        conv_res = doc_converter.convert(pdf_path)
        doc_filename = conv_res.input_file.stem
        # save page images
        for page_no, page in conv_res.document.pages.items():
            page_no = page.page_no
            page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")
        
        # save images of figures and tables
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
        

        # save markdown with embedded pictures
        md_filename = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

        # Save markdown with externally referenced pictures
        md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

        # Save HTML with externally referenced pictures
        html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
        conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
