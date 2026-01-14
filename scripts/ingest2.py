import fitz
import pdfplumber
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from math import ceil


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load caption model
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# ).to(DEVICE)

def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chunks.append({
                "type": "text",
                "page": i+1,
                "content": text
            })
    return chunks

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                tables.append({
                    "type": "table",
                    "page": i+1,
                    "content": str(table)  # later store CSV
                })
    return tables

def extract_figures(pdf_path, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    figures = []

    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            img_path = f"{out_dir}/p{i+1}_{img_index}.png"
            with open(img_path, "wb") as f:
                f.write(base["image"])

            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(out[0], skip_special_tokens=True)

            figures.append({
                "type": "figure",
                "page": i+1,
                "content": caption,
                "image_path": img_path
            })
    return figures

def save_preview(figures, out_path="figs/preview.png"):
    images = [Image.open(f["image_path"]) for f in figures]
    if not images:
        return

    thumb_size = (256, 256)
    images = [img.resize(thumb_size) for img in images]

    cols = 4
    rows = ceil(len(images) / cols)

    grid = Image.new("RGB", (cols * 256, rows * 256), "white")

    for idx, img in enumerate(images):
        grid.paste(img, ((idx % cols) * 256, (idx // cols) * 256))

    grid.save(out_path)
    print(f"Preview saved to {out_path}")





pdf_path = "/Users/vikas/Documents/llm-powered-research-assistant/data/arxiv_papers/2024.emnlp-main.268.pdf"

import fitz # imports the PyMuPDF library as fitz
import os

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    doc = fitz.open(pdf_path)
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_count = 0
    # Use a set to track unique image Xrefs and avoid duplicates
    image_xrefs = set()

    for page_num in range(doc.page_count):
        # Get list of images on the page
        image_list = doc.get_page_images(page_num)
        for img in image_list:
            xref = img[0]
            if xref not in image_xrefs:
                image_xrefs.add(xref)
                # Extract image details
                image_bytes = doc.extract_image(xref)
                if image_bytes:
                    image_data = image_bytes["image"]
                    image_ext = image_bytes["ext"]
                    
                    # Save the image file
                    image_filename = os.path.join(output_folder, f"image_p{page_num+1}_xref{xref}.{image_ext}")
                    with open(image_filename, "wb") as f:
                        f.write(image_data)
                    image_count += 1
                    print(f"Saved {image_filename}")

    doc.close()
    print(f"\nExtraction complete. Total unique images saved: {image_count}")



extract_images_from_pdf(pdf_path)





# chunks = []
# chunks += extract_text_chunks(pdf_path)
# chunks += extract_tables(pdf_path)
# figures = extract_figures(pdf_path)
# save_preview(figures)
# print(chunks)
