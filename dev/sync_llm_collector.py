"""Exploring the question: Can I run vLLM in Monarch?"""

import asyncio
import time
from typing import List

import torch

# import torchtune
from monarch.proc_mesh import proc_mesh
from monarch.service import Actor, endpoint
from torchtune.dev.rl.utils import stateless_init_process_group
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker


class vLLMActor(Actor):
    def __init__(self):
        print("initializing vllm actor")
        self.llm = LLM(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
        )

    @endpoint
    async def test(self, prompts: List[str]) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.llm.generate(prompts)


async def main():
    mesh = await proc_mesh(
        gpus=1,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    print("mesh created")
    actor = await mesh.spawn("vllm_actor", vLLMActor)
    print("actor spawned")
    results = await actor.test.call(["The meaning of life is"])
    print("results: ", results)
    time.sleep(4)


asyncio.run(main())
