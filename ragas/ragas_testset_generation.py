from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.testset.generator import TestsetGenerator
from llama_index.core import SimpleDirectoryReader

print("Loading documents...")
documents = SimpleDirectoryReader("./data").load_data()

print("Initializing generator and critic models...")
generator_llm = Ollama(model="phi3:latest")

critic_llm = Ollama(model="phi3:latest")  # OpenAI(model="gpt-4")

print("Initializing embeddings...")
embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("Initializing testset generator...")
generator = TestsetGenerator.from_llama_index(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

print("Generating testset...")
# generate testset
# CANNOT GET THIS WORKING LOOP STUCK AT 14%
testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=5,
    # distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    # COMMENTING ABOVE LINE RESULTED IN THE LOOP GETTING STUCK AT 78%
)

print("writing testset...")
testset.to_pandas().to_csv("testset.csv", index=False)
