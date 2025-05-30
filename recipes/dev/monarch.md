Instructions on how to run Cabernet with Monarch

# Set up the environment

Current steps:
- Install torch 2.7.0
    - haven't had time to debug this yet, but trainer hangs on distributed tensor setup
- Install torchtune / dependencies
- Install vLLM / dependencies manually
    - not sure the exact reason yet, but we need to use vllm 0.8.4 (latest is 0.9.0), otherwise we run into an error on stateless_init_process_group
- Install Monarch


```
conda create -n monarch_tune python=3.10
conda activate monarch_tune

# TorchTune setup
cd ../
git clone -b monarch_dev https://github.com/allenwang28/torchtune
cd torchtune
pip install torchvision==0.22.0 torchao
pip install torch==2.7.0
pip install -e .[async_rl]

# vLLM setup - we need 0.8.4, but we need torch 2.7 (otherwise trainer hangs on distributed tensor setup)

git clone -b v0.8.4 https://github.com/vllm-project/vllm
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
conda install ccache
CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .

# Monarch setup (these are basically taken from the Monarch README)
git clone https://github.com/pytorch-labs/monarch
cd monarch
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly
# Install non-python dependencies
conda install python=3.10
conda install libunwind -y

# needs cuda-toolkit-12-0 as that is the version that matches the /usr/local/cuda/ on devservers

sudo dnf install cuda-toolkit-12-0 cuda-12-0 libnccl-devel clang-devel

# install build dependencies
pip install setuptools-rust

# install core deps, see pyproject.toml for latest
pip install pyzmq requests numpy pyre-extensions cloudpickle

# Install test dependencies
pip install pytest pytest-timeout pytest-asyncio

# install the package
python setup.py install

# run unit tests. consider -s for more verbose output
# tests may take awhile, simulator tests can fail but that might be ok..
pytest python/tests/ -v -m "not oss_skip"
```


# Running the workload

```
tune download Qwen/Qwen2.5-3B --output-dir /tmp/Qwen2.5-3B --ignore-patterns "original/consolidated.00.pth"

tune run dev/async_grpo_monarch --config dev/monarch_qwen3B_async_grpo
```
