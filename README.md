# Validation-Aware Retrieval Augmented Generation for Reliable Document Question Answering using Domain Interaction Transformer

## Overview
This project implements a modality-aware RAG system for document question answering across text and image content.

## Architecture
- Separate vector collections per modality (text, image)
- Text embeddings: all-mpnet-base-v2 (768-d)
- Image embeddings: CLIP ViT-B/32 (512-d)
- Vector database: Qdrant (HNSW indexing)

## Dataset
MMDocRAG (train/dev/eval splits)

## Setup

```bash
conda env create -f environment.yml
conda activate mmrag

## Start Qdrant
```bash
docker run -d --name qdrant_mmrag -p 6333:6333 qdrant/qdrant 


```bash
python scripts/init_qdrant.py