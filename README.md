# Multimodal RAG for Document Question Answering

A comprehensive system with 6 progressive RAG architectures for multimodal document QA, featuring a novel Context Decision Engine (CDE) validation layer.

## Architectures

1. **Naive Multimodal RAG** - Pure dense vector search (baseline)
2. **Hybrid RAG + RRF** - BM25 + dense vectors with Reciprocal Rank Fusion
3. **Metadata-Filtered RAG** - Entity extraction + metadata pre-filtering
4. **Late Fusion RAG** - Weighted score fusion between modalities
5. **Query Expansion RAG** - Multi-query expansion with RRF
6. **Full Proposed (CDE)** - Complete system with Context Decision Engine

## Setup

```bash
# Clone repository
git clone <your-repo>
cd multimodal-rag

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker-compose up -d

# Start LM Studio with Qwen2.5-VL-7B
# Download from: https://lmstudio.ai/
# Load model and start server on port 1234