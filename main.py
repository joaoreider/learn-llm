from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

dataset = load_dataset("multi_news", split="test")
df = dataset.to_pandas().sample(2000, random_state=42)

model = SentenceTransformer("all-MiniLM-L6-v2")

# convert summaries to embeddings, with a single vector representing the meaning of the summary
embeddings = list(model.encode(df["summary"].tolist(), show_progress_bar=True))
def find_relevant_articles(query):
    # Transform the query to a query embedding using the same model
    query_embedding = model.encode(query, show_progress_bar=True)
    # How close in meaning is the query to each sentence
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = torch.topk(similarities.flatten(), 3).indices
    top_relevant_passages = [df.iloc[x.item()]['summary'][:200]+ "..." for x in top_indices]
    return top_relevant_passages

find_relevant_articles("Nature")

