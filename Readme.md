# Synthetic Testset generation using Ragas


## Project Overview

This project is designed to generate a test set using LlamaIndex, Ollama, and HuggingFaceEmbedding. The generated test set can be used for various NLP tasks, including testing and evaluating models.

## Requirements

The project uses **Poetry** for package management. Ensure that Poetry is installed on your system. You can install Poetry by following the instructions on the [official website](https://python-poetry.org/docs/#installation).

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:anand-kamble/ragas_testset_generation.git
cd ragas_testset_generation
```

### 2. Install Dependencies

Install the required packages using Poetry:

```bash
poetry install
```

### 3. Add Your Data

Place your data files in the `./data` directory. The script will load and process all data files located in this directory.

## Running the Script

To run the script and generate the test set, execute the `run.sh` script:

```bash
bash run.sh
```

### Script Breakdown

1. **Loading Documents**
   - The script starts by loading documents from the `./data` directory using `SimpleDirectoryReader`.

2. **Initializing Generator and Critic Models**
   - Two instances of the Ollama model (`phi3:latest`) are initialized for generating and critiquing the test set.

3. **Initializing Embeddings**
   - The script uses `HuggingFaceEmbedding` with the model `BAAI/bge-small-en-v1.5` for generating embeddings.

4. **Initializing Testset Generator**
   - A `TestsetGenerator` is created using the generator and critic models, along with the embeddings.

5. **Generating Testset**
   - The test set is generated from the documents. Note that the current configuration may have issues with the loop getting stuck at certain percentages.

6. **Writing Testset to CSV**
   - The generated test set is saved as a CSV file named `testset.csv`.

## Troubleshooting

### Common Issues

- **Loop Stuck During Testset Generation**:   
    The generation process can be time-consuming and may appear to be stuck. However, it is normal for one iteration to take upwards of 5 minutes, depending on the size and complexity of the input documents. Please be patient and allow the script to complete its execution. If the process takes significantly longer, ensure that your data is correctly formatted and try reducing the test_size or adjusting the distributions settings in the generate_with_llamaindex_docs function.

## Project Structure

- **data/**: Directory containing input data files.
- **run.sh**: Bash script to start the Python script.
- **testset.csv**: Output file containing the generated test set.
- **pyproject.toml**: Poetry configuration file.

