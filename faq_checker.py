#a simple faq checker using chat models 

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

#FAQ bank
faqs = [
    "How can I reset my password?",
    "What are your support hours?",
    "How do I cancel my subscription?",
    "Do you offer student discounts?",
    "How do I change my email address?"
]

answers = [
    "Go to Settings → Account → Reset Password and follow the email link.",
    "Our support team is available Mon–Fri, 9:00–17:00 UTC.",
    "Visit Billing → Subscriptions and click 'Cancel subscription'.",
    "Yes — we offer a 30% student discount with a valid student ID.",
    "Go to Profile → Edit and change your email; we'll send a verification link."
]

query = "How do i change password?"

faq_vectors = embedding.embed_documents(faqs)   #2d vector of shape (no. of texts, embedding_dim)
query_vector = embedding.embed_query(query)     #1d vector of shape (embedding_dim,)
# print(faq_vectors)
# print(query_vector)

scores = cosine_similarity([query_vector],faq_vectors)[0]    #change shape of query vector to 2d: (1,emb_dim)
print(scores)

# print(list(enumerate(scores)))
# print(sorted(list(enumerate(scores)),key = lambda x:x[1]))
# idx, score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]   #O(nlogn)

idx = int(np.argmax(scores))   #O(n)
score = float(scores[idx])    

print(query)
print("FAQ: ", faqs[idx])
print("Answer: ", answers[idx])
print("Similarity Score: ",score)
