import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma


# --- Configuration ---
# Load OpenAI API key from environment variable, fallback to constants.py
try:
    import constants

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", constants.APIKEY)
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY env variable or add it to constants.py."
    )
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Set to True to persist the vectorstore index to disk
# and reuse it on subsequent runs. Requires more disk space.
PERSIST = False


# Query to be answered by the bot
query = None


# Read question from command line argument if provided
if len(sys.argv) > 1:
    query = sys.argv[1]


# --- Data Loading and Indexing ---
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(
        persist_directory="persist", embedding_function=OpenAIEmbeddings()
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader(
    #     "knowledge_base.txt", encoding="utf-8"
    # )
    # Use this line if you want to query all data dumped in the data/ folder.
    # It can be any type of file.
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(
            embedding=OpenAIEmbeddings(),
            vectorstore_kwargs={"persist_directory": "persist"},
        ).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings()).from_loaders(
            [loader]
        )


# --- Chain Setup ---
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)


# --- Conversation Loop ---
chat_history = []
chat_history_tuples = []
for message in chat_history:
    chat_history_tuples.append((message[0], message[1]))


def main():
    """
    Main conversational loop for AI-Assistant bot.
    Type your question and get answers from the knowledge base.
    Type 'quit', 'q', or 'exit' to end the session.
    """
    global query
    while True:
        if not query:
            user_input = input("How can I assist you?: ")
        else:
            user_input = query
        if user_input in ["quit", "q", "exit"]:
            sys.exit()
        result = chain(
            {
                "question": user_input,
                "chat_history": chat_history_tuples,
            }
        )
        print(result["answer"])

        chat_history.append((user_input, result["answer"]))
        # Reset query for next loop
        query = None


if __name__ == "__main__":
    main()
