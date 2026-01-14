# Challenge 001 — Image Captioning with BLIP

## Context
Explored using the BLIP model to generate captions for research paper figures as part of multimodal retrieval enhancement.

## Problem
BLIP produced inconsistent captions on scientific figures, especially charts and equations. Captions were often generic and failed to preserve technical meaning.

## Investigation
- Tested BLIP-base and BLIP-large checkpoints  
- Tried higher resolution inputs  
- Evaluated on a sample set of arXiv paper figures  

## Outcome
Caption quality was insufficient for reliable retrieval indexing in scientific contexts.

## Decision
Dropped BLIP-based captioning for now. Will revisit with domain-adapted vision-language models or OCR-assisted pipelines.# Challenge 001 — Image Captioning with BLIP


