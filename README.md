# Mitigating Spurious Correlations with Knowledge Graph Representations

This repository contains the code, datasets, and results for the paper **"Data Augmentation via Knowledge Graph Representation as a Way of Mitigation of Spurious Correlations"**. The project explores the use of knowledge graph representations to mitigate spurious correlations in datasets used for fine-tuning large language models (LLMs).

## Overview

Machine learning models often suffer from biased predictions due to spurious correlations in the training data. This project introduces a novel approach to mitigate these biases by augmenting data with knowledge graph representations. The approach is evaluated on text classification tasks using datasets such as Amazon Shoes, IMDB, Yelp, and CeBaB.

Key findings:
- Knowledge graph representations consistently reduce bias (measured via Bias@C) while maintaining high accuracy.
- This approach outperforms traditional mitigation techniques such as downsampling and upsampling in balancing bias reduction and model performance.

## Project Structure

- **`knowledge-graph-creation.ipynb`**: Code for generating knowledge graph representations of sentences using the Mistral-7B model via the Ollama framework.
- **`knowledge-graph-training.ipynb`**: Code for fine-tuning the DistilBERT model using the augmented dataset with knowledge graph representations.
- **`original-dataset-training.ipynb`**: Code for replicating the Maryland experiment using the original and biased datasets without graph augmentation.
- **`Spurious_Correlations.pdf`**: The detailed paper documenting the methodology, experiments, results, and findings.

## Datasets

The project uses the following datasets:
1. **Amazon Shoes**: Product reviews with concepts like size, color, and style.
2. **IMDB**: Movie reviews with concepts such as acting, comedy, and music.
3. **Yelp**: Restaurant reviews focusing on food, price, and service.
4. **CeBaB**: Customer experience reviews analyzing service, ambiance, and food.

Each dataset was preprocessed to include knowledge graph representations, combining graph data with original sentences and labels.

## Methodology

1. **Knowledge Graph Creation**:
   - Graphs are generated for each sentence using the Mistral-7B model with a custom prompt to extract terms and relationships.
   - The generated graphs are serialized as JSON and integrated with the original datasets.

2. **Fine-Tuning**:
   - The datasets (original, biased, and graph-augmented) are used to fine-tune the DistilBERT model.
   - Performance is evaluated based on accuracy and bias metrics (Bias@C, Acc@NoC, Acc@C).

3. **Comparison**:
   - Results are compared with traditional bias mitigation techniques such as downsampling and upsampling.

## Results

- **Graph-Augmented Dataset**:
  - Significant reduction in spurious correlations across all datasets.
  - Improved balance between bias reduction and accuracy, particularly for concepts like "size" and "service."
  
- **Traditional Techniques**:
  - Downsampling and upsampling performed well in some cases but lacked consistency across datasets.

Detailed results and analysis are available in the paper (*Spurious_Correlations.pdf*).

## Usage

### Prerequisites
- Python (>=3.8)
- Required Python libraries (see `requirements.txt`)

### Steps to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spurious-correlation-mitigation.git
   cd spurious-correlation-mitigation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate knowledge graphs:
   Run the notebook `knowledge-graph-creation.ipynb` to preprocess the datasets.

4. Train the model:
   Use the notebook `knowledge-graph-training.ipynb` for fine-tuning DistilBERT on the augmented dataset.

5. Evaluate:
   Compare results with those from `original-dataset-training.ipynb` to assess the impact of the knowledge graph augmentation.

## Citation

If you use this code or dataset, please cite our work:
```
@article{Hancharova2024KnowledgeGraphs,
  title={Data Augmentation via Knowledge Graph Representation as a Way of Mitigation of Spurious Correlations},
  author={Alina Hancharova and Ali Zhunis},
  journal={University of Tübingen},
  year={2024}
}
```