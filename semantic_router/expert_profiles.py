"""
semantic_router/expert_profiles.py
----------------------------------
Semantic profiles and keyword descriptors for all active expert domains in IntelliMoE.
These profiles are embedded to calculate cosine similarity against incoming queries.
"""

from typing import Dict

EXPERT_PROFILES: Dict[str, str] = {
    "coding": (
        "Programming, Algorithms, Data Structures, Debugging, Python, Java, C++, "
        "Software Engineering, REST APIs, LeetCode, writing code, code syntax, "
        "compilers, interpreters, git version control, object oriented programming"
    ),
    "math": (
        "Calculus, Algebra, Linear Algebra, Statistics, Probability, Derivations, "
        "Optimization, Proofs, mathematical equations, matrices, vectors, integration, "
        "derivatives, geometry, trigonometry, number theory"
    ),
    "ml": (
        "Regression, Classification, Random Forest, XGBoost, Feature Engineering, "
        "Scikit-learn, machine learning algorithms, cross validation, hyperparameter tuning, "
        "precision, recall, training model, supervised learning, unsupervised clustering"
    ),
    "deep_learning": (
        "CNN, RNN, LSTM, Transformers, Attention, PyTorch, TensorFlow, Vision Transformers, "
        "deep neural networks, backpropagation, autoencoders, GANs, diffusion models, dropout"
    ),
    "genai": (
        "Prompt Engineering, RAG, Agents, LLMs, LangChain, Vector Databases, Fine-tuning, "
        "Embeddings, generative AI models, ChatGPT, GPT-4, Llama, prompt engineering design, "
        "retrieval augmented generation"
    ),
    "research": (
        "Research Papers, Literature Review, Methodology, Survey, Academic Writing, "
        "Summaries, scientific studies, arXiv reviews, citation search, benchmark datasets, "
        "state-of-the-art results, peer reviews"
    ),
    "system_design": (
        "Distributed Systems, Cloud, Scalability, Load Balancing, Microservices, Caching, "
        "Databases, system architecture design, message queues like Kafka, CAP theorem, "
        "sharding, database replication, latency and throughput optimization"
    ),
    "news": (
        "Current Events, Sports, Cricket, Football, Politics, Finance, Breaking News, "
        "Stock Market, Live Updates, current events, global news stories, weather, matches, "
        "cricket match scores, financial movements, today's stock price"
    ),
    "vision": (
        "Images, OCR, Object Detection, Image Captioning, Visual Question Answering, "
        "describing image contents, charts and plots explanation, diagram understanding, "
        "processing screenshot images, identifying objects in picture"
    )
}
