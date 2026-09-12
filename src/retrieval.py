"""
Lightweight retrieval: TF-IDF + cosine similarity over historical resolved
AmericanAir customer->reply pairs. Returns the top-k most similar historical
cases for a new customer message, to ground reply drafting.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_PATH = "data/processed/retrieval_corpus.csv"


class Retriever:
    def __init__(self, corpus_path=CORPUS_PATH):
        self.df = pd.read_csv(corpus_path)
        self.df = self.df.dropna(subset=["customer_text", "americanair_reply"])
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["customer_text"])

    def retrieve(self, query_text, k=3):
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_idx = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append({
                "customer_text": self.df.iloc[idx]["customer_text"],
                "americanair_reply": self.df.iloc[idx]["americanair_reply"],
                "similarity": float(similarities[idx]),
            })
        return results


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "My flight AA1234 has been delayed 3 hours and nobody will tell me why"
    print(f"Query: {test_query}\n")
    results = retriever.retrieve(test_query, k=3)
    for i, r in enumerate(results, 1):
        print(f"--- Match {i} (similarity: {r['similarity']:.3f}) ---")
        print(f"Customer: {r['customer_text']}")
        print(f"AA reply: {r['americanair_reply']}\n")
        