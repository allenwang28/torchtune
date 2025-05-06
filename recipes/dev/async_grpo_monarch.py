import asyncio
import functools
import os
import time

from typing import Any, Dict

import torch
import torch.distributed

from monarch.proc_mesh import proc_mesh
from monarch.service import Actor, endpoint

from omegaconf import DictConfig, OmegaConf

from ray.util.queue import Queue
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right

from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, utils
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory
from torchtune.dev.rl.monarch_actors import SyncLLMCollector
from torchtune.recipe_interfaces import OrchestrationRecipeInterface
from vllm import SamplingParams

from vllm.utils import get_ip, get_open_port

log = utils.get_logger("DEBUG")


class MonarchGRPORecipe(OrchestrationRecipeInterface):
    async def setup(self, cfg: DictConfig) -> None:
        print("initializing w/ config: ", cfg)
        self.cfg = cfg
        # Store worker counts as instance variables
        self.num_inference_workers = cfg.orchestration.num_inference_workers
        self.num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
        self.num_training_workers = cfg.orchestration.num_training_workers

        # Creating ProcMesh
        print("creating collector mesh with gpus: ", self.num_inference_workers)
        self.collector_mesh = await proc_mesh(
            gpus=self.num_inference_workers,
            env={},
        )

    async def run(self):
        print("start run")

    async def cleanup(self):
        print("start cleanup")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    if cfg.get("enable_expandable_segments", True):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    recipe = MonarchGRPORecipe()
    asyncio.run(recipe.setup(cfg))
    asyncio.run(recipe.run())
    asyncio.run(recipe.cleanup())


if __name__ == "__main__":
    recipe_main()
