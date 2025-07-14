# Generating Data Engineering Code Using LLMs

This repository accompanies the paper **"Generating Data Engineering Code Using LLMs"**. It contains datasets and notebooks used to evaluate large language models on multi-step data transformation tasks in Python/Pandas.

## Repository Structure

- `arcade_dataset_new.ipynb`, `arcade_existing_dataset.ipynb`, `arcade_experiments.ipynb` – notebooks for processing and evaluating on the ARCADE benchmark.
- `spider2-intents/` – manually curated notebooks derived from Spider 2.0 used in our study.
- `spider2.single_intent.jsonl` – single–intent tasks extracted from Spider 2.0-lite.
- `spider2_utils.py` – helper functions for loading datasets and evaluating Pandas outputs.
- `experiments.ipynb`, `model_eval.ipynb` – example evaluation pipelines.

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The notebooks rely on `pandas`, `numpy` and JupyterLab. LLM APIs are accessed through provider SDKs.

2. **Datasets**
   - `datasets/llm_etl_spider2.csv` contains metadata for Spider 2.0 tables.
   - Additional CSV dumps used in the study are referenced in the notebooks.

3. **Running Experiments**
   Use the provided notebooks to reproduce the evaluations. Each notebook describes the expected LLM prompts and shows how execution feedback is collected for iterative refinement.

## Citation

If you use this repository or the Spider2-intents dataset in your research, please cite:

```
@article{anonymous2025generating,
  title={Generating Data Engineering Code Using LLMs},
  year={2025},
  note={Anonymous submission}
}
```

## License

This code is released for research purposes only.
