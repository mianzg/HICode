# HICode

Code repository for EMNLP 2025 long paper HICode: Hierarchical Inductive Coding with LLMs \[[Arxiv](https://arxiv.org/abs/2509.17946)\]

---

## Prerequisites

- [OpenAI API credentials](https://platform.openai.com/api-keys) (for GPT-based models)
- Install dependencies (see below)

### Installation

```bash
pip install requirements.txt
```

## Quickstart

See [`demo.ipynb`](https://github.com/mianzg/HICode/blob/main/demo.ipynb) for a step-by-step example of how to use HICode for identifying the query types asked by astronomers.

## Configuration

The pipeline is configured through a plain `config` dict (see the demo notebook). Optional keys:

| Key | Default | Description |
|---|---|---|
| `max_concurrency` | `8` | Number of concurrent API requests in generation and clustering. Independent calls run in a thread pool, so large corpora finish roughly `max_concurrency`× faster. |
| `checkpoint_path` | `None` | If set, generation appends one JSONL line per completed segment. Re-running with the same path resumes and skips completed segments (delete the file to start fresh). |
| `max_retries` | `5` | Retries with exponential backoff on rate limits (429) and transient errors, handled by the OpenAI client. |
| `cluster_batch_size` | `100` | Number of labels packed into each clustering prompt. |
| `base_url` | `None` | Point the OpenAI client at any OpenAI-compatible endpoint (e.g. vLLM, Ollama, Together) to use open-source or other proprietary models. |


## Cite the work

If you use HICode in your research, please cite:

```{}
@misc{zhong2025hicodehierarchicalinductivecoding,
      title={HICode: Hierarchical Inductive Coding with LLMs}, 
      author={Mian Zhong and Pristina Wang and Anjalie Field},
      year={2025},
      eprint={2509.17946},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.17946}, 
}
```

---
For more details, see the documentation or contact the authors.
