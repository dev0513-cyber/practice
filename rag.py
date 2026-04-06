from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

CRYPTO_KNOWLEDGE = [
    "Bitcoin is a decentralized digital currency that operates without a central bank. It uses blockchain technology to record transactions.",
    "The Bitcoin halving event reduces the block reward by 50% approximately every four years. This historically has been followed by significant price increases.",
    "Market sentiment plays a large role in Bitcoin price movements. Fear and greed index values above 75 indicate extreme greed, which often precedes corrections.",
    "Dollar-cost averaging (DCA) is a strategy where investors buy fixed amounts of Bitcoin at regular intervals, reducing the impact of volatility.",
    "Bitcoin's 200-day moving average is considered a key long-term trend indicator. Price above it is generally bullish; price below is bearish.",
    "RSI above 70 indicates Bitcoin may be overbought and a correction could follow. RSI below 30 suggests oversold conditions and a potential rebound.",
    "On-chain metrics like active addresses, transaction volume, and miner flows can provide insight into Bitcoin network health.",
    "Bitcoin has historically performed well after periods of high accumulation by long-term holders and low exchange reserves.",
    "Institutional adoption of Bitcoin via ETFs and corporate treasury allocations has increased its mainstream legitimacy.",
    "Volatility is inherent to Bitcoin. It can swing 10-20% in a single day. Risk management and position sizing are critical.",
    "Bitcoin's total supply is capped at 21 million coins. Scarcity is a core part of its value proposition.",
    "Macroeconomic factors such as inflation, interest rate decisions by the Fed, and global risk appetite strongly influence Bitcoin prices.",
    "When investing in Bitcoin, never invest more than you can afford to lose. It remains a high-risk, high-reward asset.",
    "Diversification across asset classes reduces overall portfolio risk. Bitcoin should typically form only a portion of a diversified portfolio.",
    "The concept of support and resistance levels in Bitcoin trading refers to price zones where buying or selling pressure has historically been strong.",
]


def build_vector_store():
    docs = [Document(page_content=text) for text in CRYPTO_KNOWLEDGE]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


def retrieve_context(vector_store, query, k=4):
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])
