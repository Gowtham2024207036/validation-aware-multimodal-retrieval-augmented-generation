from pipeline import MMDocRAG

rag = MMDocRAG()

pdf = "sample.pdf"

rag.index_document(pdf)

question = "What does the graph show?"

texts, images = rag.retrieve(question)

print("Text Results:")
for t in texts:
    print(t.payload)

print("Image Results:")
for i in images:
    print(i.payload)