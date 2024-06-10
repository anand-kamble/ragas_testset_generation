from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.testset.generator import TestsetGenerator

def main():
    """
    Main function to load documents, initialize models and embeddings, and generate a test set.
    The generated test set is saved as a CSV file.
    """
    print("Loading documents...")
    # Load documents from the specified directory
    documents = SimpleDirectoryReader("./data").load_data()

    print("Initializing generator and critic models...")
    # Initialize the generator and critic models using the Ollama LLM
    generator_llm = Ollama(model="phi3:latest")
    critic_llm = Ollama(model="phi3:latest")  # Alternatively, OpenAI(model="gpt-4")

    print("Initializing embeddings...")
    # Initialize embeddings using the HuggingFace embedding model
    embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Initializing testset generator...")
    # Create a testset generator using the initialized models and embeddings
    generator = TestsetGenerator.from_llama_index(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )

    print("Generating testset...")
    # Generate the test set from the loaded documents
    # Note: This process might take some time; be patient if it seems to be stuck at certain percentages
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=1,
        # distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
        # Uncommenting the above line might cause the loop to get stuck at 78%
    )

    print("Writing testset to CSV...")
    # Save the generated test set as a CSV file
    testset.to_pandas().to_csv("testset.csv", index=False)
    print("Testset generation completed successfully.")

if __name__ == "__main__":
    main()
