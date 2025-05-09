"""Exploring the question: Can I run vLLM in Monarch?"""

import asyncio

import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch

# import torchtune
from monarch.proc_mesh import proc_mesh
from monarch.service import Actor, endpoint

from omegaconf import DictConfig, ListConfig

from ray.util.queue import Full as QueueFull
from tensordict import lazy_stack, NonTensorStack, TensorDictBase
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.collectors import (
    SyncDataCollector,
    WeightUpdateReceiverBase,
    WeightUpdateSenderBase,
)

from torchtune import utils
from torchtune.dev.rl.datatypes import Trajectory
from torchtune.dev.rl.utils import stateless_init_process_group
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

log = utils.get_logger()


class VLLMWorkerWrapper(Worker):
    """
    vLLM worker for Ray.

    vLLMParameterServer will always take rank 0 in the stateless process group
    initialized by this worker. And the tp ranks associated with the LLM class
    will be in the range [1, tp_size].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset

        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

        self.version = torch.tensor([0], device="cuda")

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # src=0 because fsdp worker 0 has been assigned as "0" in this process group
        self._model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_policy_version(self):
        self._model_update_group.broadcast(
            self.version, src=0, stream=torch.cuda.current_stream()
        )
        self.policy_version = self.version
        torch.cuda.synchronize()


class SyncLLMActor(Actor):
    def __init__(self):
        print("initializing sync LLM actor")
        self.llm = LLM(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            worker_cls=VLLMWorkerWrapper,
        )

    @endpoint
    async def run(self):
        i = 0
        while True:
            await self.rollout(i)
            if i % self.cfg.inference.steps_before_weight_Sync == 0:
                log.info(f"{self.worker_id} about to update weights")
                self.update_policy_weights_()
            i += 1

    async def rollout(self, idx):
        pass

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
    actor = await mesh.spawn("vllm_actor", SyncLLMActor)
    print("actor spawned")
    results = await actor.test(prompts=["The meaning of life is"]).call()
    print("results: ", results)
    time.sleep(4)


asyncio.run(main())
