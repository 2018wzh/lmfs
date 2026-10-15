# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## BPE

The BPE interface lives in `bpe.py`. Its Rust extension lives in `bpe-rs`.
The course adapter calls `bpe.py` for training and tokenization.

```sh
uv sync
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
uv run python train_bpe.py
```

The training command uses the course OWT paths and W&B settings in `train_bpe.py`.
It saves one tokenizer JSON file under `outputs` and prints the path prefix.
For task 2, `BPE.encode_numpy(text)` returns `uint16` IDs, and
`BPE.encode_to_file(input_path, output_path)` writes an `.npy` file for NumPy mmap.
Pass `num_threads=N` to `BPE` to set the Rust worker count. The default uses the available CPU count.
