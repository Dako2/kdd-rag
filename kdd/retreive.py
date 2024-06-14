import requests
from readability import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, __version__
print(__version__)  # This prints the version of the sentence_transformers library

"""
from sentence_transformers import SentenceTransformer
# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode([
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
])
similarities = model.similarity(embeddings, embeddings)
"""

def single_url(url = 'https://en.wikipedia.org/wiki/London'):
    # Fetch the webpage content
    response = requests.get(url)
    html_content = response.text
    doc = Document(html_content)

    print(doc.title())
    main_content = doc.summary()
    soup = BeautifulSoup(main_content, 'html.parser')
    final_text = soup.get_text().replace("\n", " ").strip()
    print(final_text)
    
    return final_text
    
final_text = single_url()
model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

query_embedding = model.encode("How big is London")
passage_embeddings = model.encode([
    final_text,
])

similarity = model.similarity(query_embedding, passage_embeddings)
