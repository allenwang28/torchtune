# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.import asyncio
import asyncio
import os
import random
import socket
import time
from dataclasses import dataclass
from functools import partial

from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import numpy as np

import torch
import torch.distributed
import torchtune.training as training

from monarch.proc_mesh import proc_mesh
from monarch.service import Actor, current_rank, current_size, endpoint
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensordict import lazy_stack, NonTensorStack, TensorDict, TensorDictBase
from tensordict.utils import expand_as_right
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from torchrl.collectors import (
    SyncDataCollector,
    WeightUpdateReceiverBase,
    WeightUpdateSenderBase,
)

from torchrl.data import LazyStackStorage, ReplayBuffer
from torchtune import config, generation, rlhf, utils
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory
from torchtune.dev.rl.monarch_actors import get_logger, MetricsLoggerActor
from torchtune.dev.rl.rewards_async import batched_rewards
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory

from torchtune.dev.rl.utils import stateless_init_process_group

from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf
from torchtune.modules.transformer import TransformerSelfAttentionLayer
from torchtune.recipe_interfaces import OrchestrationRecipeInterface

from torchtune.training import disable_dropout, DummyProfiler, PROFILER_KEY
from vllm.outputs import RequestOutput as vllmRequestOutput
from vllm.utils import get_open_port
from vllm.worker.worker import Worker

T = TypeVar("T")


# ========= Common functions =========
def get_ip() -> str:
    """Returns an IP address."""

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't actually establish a connection
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def set_seed(seed: int):
    """Sets a seed for torch, numpy and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ========= Generic data structures + functionality =========
def get_device_index(
    entity: str, local_rank: int, global_rank: int, cfg: DictConfig
) -> int:
    """Returns the device index for the given entity and rank.

    This function maps logical entity ranks to physical GPU indices based on entity type and configuration.

    Args:
        entity: The type of entity ("param_store", "training", "rollout", or "postprocessing")
        local_rank: The local rank within the entity's process group
        global_rank: The global rank of the entity (used for multiple workers of same type)
        cfg: The configuration object containing orchestration settings

    Returns:
        int: The physical GPU device index to use

    Note:
        The device assignment follows this pattern:
        - Parameter server: Uses 1 GPU at the beginning
        - Training workers: Use num_training_workers GPUs after the parameter server
        - Rollout workers: Each worker uses tensor_parallel_dim GPUs, with multiple workers
        - Postprocessing workers: Each worker uses tensor_parallel_dim GPUs, with multiple workers

    This is a placeholder implementation for now, and would need to change in a multi-host setting.
    """
    trainer_world_size = cfg.orchestration.num_training_workers
    trainer_offset = trainer_world_size

    param_store_world_size = 1
    param_store_offset = param_store_world_size

    rollout_world_size = cfg.inference.tensor_parallel_dim
    num_rollout_workers = cfg.orchestration.num_inference_workers
    rollout_offset = rollout_world_size * num_rollout_workers

    postprocessing_world_size = cfg.postprocessing.tensor_parallel_dim
    num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
    postprocessing_offset = postprocessing_world_size * num_postprocessing_workers

    entity_world_size = -1

    if entity == "param_store":
        entity_world_size = param_store_world_size
        offset = 0
    elif entity == "training":
        entity_world_size = trainer_world_size
        offset = param_store_offset
    elif entity == "rollout":
        entity_world_size = rollout_world_size
        offset = trainer_offset + param_store_offset
    elif entity == "postprocessing":
        entity_world_size = postprocessing_world_size
        offset = trainer_offset + param_store_offset + rollout_offset
    else:
        raise KeyError(f"Unknown entity: {entity}")
    return offset + global_rank * entity_world_size + local_rank


class QueueActor(Actor, Generic[T]):
    """A simple async queue, implemented as a Monarch actor."""

    def __init__(self):
        self._q: asyncio.Queue[T] = asyncio.Queue()

    @endpoint
    async def put(self, item: T) -> None:
        await self._q.put(item)

    @endpoint
    async def get(self) -> T:
        return await self._q.get()

    @endpoint
    async def qsize(self) -> int:
        return self._q.qsize()

    @endpoint
    async def is_empty(self) -> bool:
        return self._q.empty()


class ReplayBufferActor(Actor):
    """A replay buffer, implemented as a Monarch actor."""

    def __init__(self, cfg: DictConfig):
        self.rb = ReplayBuffer(
            storage=partial(
                LazyStackStorage, max_size=cfg.orchestration.replay_buffer_size
            ),
            batch_size=cfg.training.batch_size,
        )

    @endpoint
    async def extend(self, sample: Trajectory):
        self.rb.extend(sample)

    @endpoint
    async def sample(self) -> torch.Tensor:
        return self.rb.sample()

    @endpoint
    async def len(self) -> int:
        return len(self.rb)

    @endpoint
    async def is_empty(self) -> bool:
        return len(self.rb) == 0


# ========= Cabernet actors + data structures=========
class ParameterStore(Actor):
    """Base class for parameter storage and synchronization between trainers and generators.

    This abstract class defines the interface for parameter stores that manage model weights
    in distributed training scenarios. Parameter stores act as central repositories that
    facilitate weight sharing between trainers (which update weights) and generators
    (which use weights for inference).

    The parameter store maintains version tracking to ensure consistency across the
    distributed system and provides synchronization mechanisms for weight updates.

    Implementations should handle efficient weight transfer, format conversions if needed,
    and proper synchronization to prevent race conditions during updates.
    """

    @endpoint
    async def get(self, generator: Actor | int) -> int:
        """Transmits weights from the parameter store to the generator.

        This method sends the current model weights to the specified generator,
        typically using an efficient communication protocol like NCCL.

        Args:
            generator (Generator): The generator (or its index) requesting the weights

        Returns:
            int: The current policy version number
        """
        raise NotImplementedError

    @endpoint
    async def put(self, version: int):
        """Transmits weights from the trainer to the parameter store.

        This method receives updated weights from a trainer and updates
        the internal state of the parameter store.

        Args:
            version (int): The new version number for these weights
        """
        raise NotImplementedError

    @endpoint
    async def get_version(self) -> int:
        """Returns the current version of the model weights.

        Returns:
            int: The current policy version number
        """
        raise NotImplementedError

    @endpoint
    async def get_generator_model_metadata(
        self,
    ) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Returns metadata about the model's state dictionary.

        This method provides information about tensor shapes and dtypes,
        which helps generators prepare memory for receiving weights.

        Returns:
            dict: Mapping of parameter names to tuples of (dtype, shape)
        """
        raise NotImplementedError


@dataclass
class ActorSet:
    """A dataclass that conveniently packages all actors."""

    param_store: ParameterStore
    metric_logger: MetricsLoggerActor
    trainer: Actor
    generators: list[Actor]


@dataclass
class DistributedInfo:
    """A representation for torch.distributed world size."""

    address: str
    port: int
    world_size: int


class HFVLLMParameterServer(ParameterStore):
    """A parameter server that manages model weights for HuggingFace models used with vLLM.

    This server acts as a central repository for model weights, facilitating weight sharing
    between trainers and generators. It handles the conversion between TorchTune's
    internal model format and HuggingFace's format required by vLLM inference workers.

    Communications are facilitated primarily over NCCL, using torch.distributed. As such,
    synchronization mechanisms are required to ensure consistency across reads and writes.

    The server maintains version tracking to ensure consistency across the distributed system
    and provides synchronization mechanisms to safely update and distribute weights.

    Attributes:
        cfg (DictConfig): Configuration for the parameter server
        state_dict (dict): The model's state dictionary in TorchTune format
        hf_state_dict (dict): The model's state dictionary converted to HuggingFace format
        lock (asyncio.Lock): Lock to prevent concurrent weight updates
        version (int): Current version of the model weights
        _model_metadata (dict): Metadata about model tensors (shapes and dtypes)
        _generator_channel_map (dict): Mapping between generators and communication channels
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @endpoint
    async def initialize(
        self,
        actor_set: ActorSet,
        dist_info_map: dict[Actor, DistributedInfo],
    ):
        """Initializes the parameter server and establish connections with trainers and generators.

        This method:
        1. Sets up the device for the parameter server
        2. Establishes distributed connections with trainers and generators
        3. Loads the initial model checkpoint
        4. Converts weights to HuggingFace format for vLLM compatibility
        5. Creates model metadata for efficient weight synchronization

        Args:
            actor_set (ActorSet): Set of actors in the system (trainers, generators, etc.)
            dist_info_map (dict): Mapping of actors to their distributed communication info
        """
        self.device_index = get_device_index("param_store", 0, 0, self.cfg)
        # Note - the logger isn't being pickled through Monarch which is why we're
        # calling it many times.
        logger = get_logger()
        logger.info("assigned device index: %d", self.device_index)
        self.device = torch.device("cuda:{}".format(self.device_index))
        torch.cuda.set_device(self.device)

        # A map between the generator and connection channel
        self.generators = actor_set.generators
        self._generator_channel_map = {}

        self._connect_to_trainer(dist_info=dist_info_map[actor_set.trainer])
        for generator in actor_set.generators:
            self._connect_to_generator(
                generator=generator, dist_info=dist_info_map[generator]
            )

        logger.info(
            "Parameter server is initialized, with all components connected. Loading model metadata..."
        )
        # Load model information
        checkpointer = config.instantiate(
            self.cfg.training.checkpointer, resume_from_checkpoint=False
        )
        self.train_state_dict = checkpointer.load_checkpoint()[training.MODEL_KEY]
        for k, v in self.train_state_dict.items():
            # Move to GPU so NCCL communication actually works
            self.train_state_dict[k] = v.to(self.device)
        logger.info("checkpoint loaded")

        # aiorwlock may be better here, it just doesn't play well with Monarch for some reason...
        # self.lock = aiorwlock.RWLock()
        self.lock = asyncio.Lock()
        self.version = 0

        # Create model metadata
        self.hf_state_dict = self._prepare_hf_weights()
        self._model_metadata = {
            k: (v.dtype, v.shape) for k, v in self.hf_state_dict.items()
        }
        logger.info("Done with initialization!")

    @endpoint
    async def get_generator_model_metadata(
        self,
    ) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Gets metadata about the model's tensors to prepare for weight synchronization.

        Returns:
            dict: Mapping of parameter names to tuples of (dtype, shape)
        """
        return self._model_metadata

    def _prepare_hf_weights(self) -> dict[str, torch.Tensor]:
        """Converts TorchTune model weights to HuggingFace format.

        This method transforms the internal state dictionary to the format expected
        by HuggingFace models used in vLLM, handling architecture-specific conversions.

        Returns:
            dict: The converted state dictionary in HuggingFace format
        """
        sd = qwen2_tune_to_hf(
            self.train_state_dict, num_heads=16, num_kv_heads=2, dim=2048
        )
        for k, v in sd.items():
            sd[k] = v.to(self.device)
        return sd

    def _connect_to_trainer(self, dist_info: DistributedInfo):
        """Establishes a distributed connection with the trainer.

        Sets up the NCCL communication channel with the trainer for weight updates.

        Args:
            dist_info (DistributedInfo): Connection information for the trainer
        """
        logger = get_logger()
        logger.info("Connecting to trainer...")
        os.environ["WORLD_SIZE"] = str(dist_info.world_size)
        os.environ["MASTER_ADDR"] = str(dist_info.address)
        os.environ["MASTER_PORT"] = str(dist_info.port)

        assert (
            not torch.distributed.is_initialized()
        ), "Parameter server's distributed world was initialized twice, which is unexpected."

        logger.info("Initializing torch distributed...")
        torch.distributed.init_process_group(backend="nccl", rank=0)
        logger.info("Done connecting to trainer")

    def _connect_to_generator(self, generator: Actor, dist_info: DistributedInfo):
        """Establishes a distributed connection with a generator.

        Sets up the NCCL communication channel with a generator for weight distribution.

        Args:
            generator (Actor): The generator actor to connect to
            dist_info (DistributedInfo): Connection information for the generator
        """
        logger = get_logger()
        logger.info("Connecting to generator {}".format(generator))
        assert (
            generator not in self._generator_channel_map
        ), "Connection was already created between parameter serverand generator {}".format(
            generator
        )
        self._generator_channel_map[generator] = stateless_init_process_group(
            master_address=dist_info.address,
            master_port=dist_info.port,
            # we assume that the parameter server is always rank 0 in server<>worker communications.
            rank=0,
            world_size=dist_info.world_size,
            device=self.device,
        )
        logger.info("Done connecting to generator {}".format(generator))

    @endpoint
    async def get(self, generator: Actor | int) -> int:
        """Sends model weights to a generator.

        Broadcasts the current HuggingFace-formatted weights to the specified generator
        using the established communication channel.

        Args:
            generator (Generator): The generator (or its index) requesting weights

        Returns:
            int: The current version number of the weights
        """
        logger = get_logger()
        if isinstance(generator, int):
            generator = self.generators[generator]
        logger.info("Sending weights to generator {}".format(generator))
        await self.lock.acquire()
        channel = self._generator_channel_map[generator]
        for k in sorted(self.hf_state_dict.keys()):
            channel.broadcast(
                self.hf_state_dict[k], src=0, stream=torch.cuda.current_stream()
            )
        torch.cuda.synchronize()
        version = self.version
        self.lock.release()
        return version

    @endpoint
    async def put(self, version: int):
        """Receives updated weights from the trainer.

        Updates the internal state dictionary with weights from the trainer and
        converts them to HuggingFace format for future distribution to generators.

        Args:
            version (int): The new version number for these weights
        """
        logger = get_logger()
        logger.info("Receiving weights from trainer...")
        await self.lock.acquire()
        for k in sorted(self.train_state_dict.keys()):
            v = self.train_state_dict[k]
            torch.distributed.recv(v, src=1)

        torch.distributed.barrier()
        self.hf_state_dict = self._prepare_hf_weights()
        logger.info("Done receiving weights from trainer.")
        self.version = version
        self.lock.release()

    @endpoint
    async def get_version(self) -> int:
        """Gets the current version of the model weights.

        Returns:
            int: The current version number
        """
        return self.version

    def __repr__(self) -> str:
        return "HFVLLMParameterServer"


# TODO - consider consolidating Rollout, without SyncDataCollector
class VLLMHFWeightUpdateReceiver(WeightUpdateReceiverBase):
    """A weight update receiver for vLLM / HuggingFace."""

    def __init__(
        self,
        dist_info: DistributedInfo,
        param_store: ParameterStore,
        worker_idx: int,
    ):
        self._dist_info = dist_info
        self.param_store = param_store
        self.worker_idx = worker_idx
        self.model_metadata = None

    def _get_server_weights(self):
        return None

    def _get_local_weights(self):
        # We don't implement this because we let vLLM's update_weights API handle everything for now
        return None

    def _maybe_map_weights(self, server_weights, local_weights):
        # vLLM update_weights function handles the mapping from huggingface
        # so we don't implement this for now
        return None

    async def connect_with_parameter_store(self):
        logger = get_logger()
        logger.info("Connecting with parameter store...")
        logger.info("Got model metadata")
        self.collector.inference_server.collective_rpc(
            "init_weight_update_group",
            args=(
                self._dist_info.address,
                self._dist_info.port,
                1,
                self._dist_info.world_size,
            ),
        )

    async def update_weights(self):
        logger = get_logger()
        logger.info("updating weights")
        if not self.model_metadata:
            self.model_metadata = (
                await self.param_store.get_generator_model_metadata.call_one()
            )
        fut = self.param_store.get.call_one(self.worker_idx)

        for k in sorted(self.model_metadata.keys()):
            dtype, shape = self.model_metadata[k]
            # logger.info("broadcast: {}".format(k))
            self.collector.inference_server.collective_rpc(
                "update_weight", args=(k, dtype, shape)
            )

        logger.info(
            "worker {} done with broadcasts, waiting for version".format(
                self.worker_idx
            )
        )
        version = await fut
        logger.info("worker {} done with update".format(self.worker_idx))
        return version


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
        torch.cuda.set_device(self.device)
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self._model_update_group = stateless_init_process_group(
            master_address=master_address,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            device=self.device,
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


class SyncLLMCollector(SyncDataCollector):
    """A synchronous data collector for LLM inference and trajectory collection.

    This collector:
    1. Sets up a dataloader to provide prompts
    2. Creates an LLMEnv environment to manage conversation state
    3. Initializes a vLLM inference server for text generation
    4. Collects trajectories by running inference on prompts and tracking conversation history

    The collector handles batched inference, maintains conversation state across turns,
    and properly formats inputs/outputs between the environment and inference server.
    """

    def __init__(
        self,
        cfg: DictConfig,
        local_rank: int,
        global_rank: int,
        reset_at_each_iter: bool = False,
        total_dialog_turns: int = -1,
        dialog_turns_per_batch: int = 1,
        weight_update_receiver: (
            WeightUpdateReceiverBase | Callable[[], WeightUpdateReceiverBase] | None
        ) = None,
        weight_update_sender: (
            WeightUpdateSenderBase | Callable[[], WeightUpdateSenderBase] | None
        ) = None,
    ):
        self.cfg = cfg
        self.reset_at_each_iter = reset_at_each_iter
        self.dialog_turns_per_batch = dialog_turns_per_batch
        self.total_dialog_turns = total_dialog_turns
        self.local_rank = local_rank
        self.global_rank = global_rank
        device_index = get_device_index("rollout", local_rank, global_rank, cfg)

        logger = get_logger()
        logger.info("device index: {}".format(device_index))
        device = torch.device("cuda:{}".format(device_index))
        torch.cuda.set_device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

        from torchtune import config

        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        # Create data loader
        dataloader = self._setup_data(
            self.cfg.dataset,
            self.cfg.get("shuffle", True),
            self.cfg.inference.batch_size,
            self.cfg.get("collate_fn", "torchtune.dev.rl.data.padded_collate_rl"),
            dataloader_state_dict=None,
        )
        # Create env
        from torchrl.envs import LLMEnv

        self._sequence_counter = 0

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            tokenizer=None,
            from_text=True,
            batch_size=self.cfg.inference.batch_size,
            repeats=self.cfg.inference.group_size,
        )
        from vllm import LLM

        self.inference_server = LLM(
            model=self.cfg.inference.model,
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            worker_cls=VLLMWorkerWrapper,
            tensor_parallel_size=self.cfg.inference.tensor_parallel_dim,
            device=device,
            **self.cfg.inference.get("engine_args", {}),
        )
        self.generation_time = 0
        super().__init__(
            create_env_fn=env,
            policy=self.policy_fn,
            frames_per_batch=self.dialog_turns_per_batch,
            total_frames=self.total_dialog_turns,
            weight_update_receiver=weight_update_receiver,
            weight_update_sender=weight_update_sender,
            reset_at_each_iter=self.reset_at_each_iter,
            use_buffers=False,
            device=device,
            # This argument allows a non-TensorDictModule policy to be assumed
            # to be compatible with the collector
            trust_policy=True,
        )

    def policy_fn(
        self, data: TensorDictBase, pad_outputs: bool = True
    ) -> TensorDictBase:
        """Generates responses using a vLLM inference server.

        Takes input text from the TensorDict, runs inference through vLLM,
        and returns a TensorDict with the padded generated responses and log probabilities.

        Args:
            data: TensorDict containing 'text' key with input prompts
            pad_outputs: Whether or not to pad the generated responses

        Returns:
            TensorDict with generated responses including:
                - tokens_response: token IDs of generated text
                - text_response: generated text strings
                - log_probs: log probabilities of generated tokens
        """
        _TOK_RESPONSE_KEY = "tokens_response"
        _TEXT_RESPONSE_KEY = "text_response"
        _LOG_PROBS_KEY = "log_probs"

        logger = get_logger()

        from vllm import SamplingParams

        with self.device:
            start = time.perf_counter()
            text_input = data.get("text")
            if not isinstance(text_input, (list, str)):
                text_input = text_input.tolist()
            token_outputs: List[vllmRequestOutput] = self.inference_server.generate(
                text_input,
                sampling_params=SamplingParams(
                    n=1,
                    max_tokens=self.cfg.inference.max_generated_tokens,
                    temperature=self.cfg.inference.temperature,
                    detokenize=True,
                    prompt_logprobs=False,
                    logprobs=True,
                ),
                use_tqdm=False,
            )
            # convert the vllmRequestOutput to a TensorDict
            outputs: RequestOutput = RequestOutput.from_request_output(token_outputs)
            response = outputs.outputs._tensordict.select(
                "text", "token_ids", "logprobs", strict=False
            )
            # replace with correct keys
            response.rename_key_("token_ids", _TOK_RESPONSE_KEY)
            response.rename_key_("text", _TEXT_RESPONSE_KEY)
            response.rename_key_("logprobs", _LOG_PROBS_KEY)

            if pad_outputs:
                padding = self._tokenizer.pad_id
                response = response.densify(layout=torch.strided).to_padded_tensor(
                    padding=padding,
                )
                padded_values = response[_TOK_RESPONSE_KEY] == padding
                if padded_values.any():
                    lps = response[_LOG_PROBS_KEY]
                    lps = torch.where(expand_as_right(~padded_values, lps), lps, 1.0)
                    response[_LOG_PROBS_KEY] = lps

            assert set(response.keys()) == set(
                [_TOK_RESPONSE_KEY, _TEXT_RESPONSE_KEY, _LOG_PROBS_KEY],
            ), "got keys {}".format(response.keys())

            # clone the input tensordict to preserve stateless transforms from breaking
            action = data.clone()
            action.update(response, keys_to_update=list(response.keys()))
            self.generation_time += time.perf_counter() - start
            return action

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_str: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """Sets up all data-related components.

        Note - this recipe currently only supports the DistributedSamplers with
        Map-style Datasets which fit into memory. Other samplers, iterable datasets
        and streaming datasets are not supported.

        Args:
            cfg_dataset: Configuration for the dataset
            shuffle: Whether to shuffle the dataset
            batch_size: Batch size for the dataloader
            collate_str: String path to the collate function
            dataloader_state_dict: Optional state dict to restore dataloader state

        Returns:
            A StatefulDataLoader instance configured with the dataset and sampler

        """
        # Not importing here and doing these imports globally will cause VLLM worker
        # to have no cuda devices during cuda lazy init for some reason?? Even when
        # this method is not actually called...
        from torchtune import config
        from torchtune.config._utils import _get_component_from_path
        from torchtune.datasets import ConcatDataset

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
        sampler = StatefulDistributedSampler(
            ds,
            # FIXME: hardcoding num_replicas and rank for now
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            # FIXME: set seed?
            # seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    _get_component_from_path(collate_str),
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            raise NotImplementedError()
        return dataloader

    def rollout(self) -> TensorDictBase:
        """Collect a batch of trajectories from the environment.

        This method:
        1. Gets prompts from the dataloader (via environment reset or continuing from previous state)
        2. For each prompt:
            a. Runs vLLM inference to generate text responses
            b. Passes responses to the LLMEnv which concatenates them with the original prompts
            c. Maintains conversation history and metadata across turns
        3. Collects the resulting trajectories until we have dialog_turns_per_batch steps

        The LLMEnv environment handles the state transitions by concatenating generated
        responses with previous context, creating an ongoing conversation history.

        Returns:
            TensorDictBase: Tensor dictionary containing all collected trajectories and
                other runtime metrics.
        """
        data = self.env.reset()
        trajectories = []
        collected_frames = 0
        while collected_frames < self.dialog_turns_per_batch:
            action = self.policy(data)

            env_output, env_next_output = self.env.step_and_maybe_reset(action)

            # Preserve collector metadata across environment transitions:
            # 1. Copy collector data from current state
            # 2. Apply it to the new environment state
            # 3. Update tracking variables and append to trajectory list
            collector_data = self._shuttle.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            self._shuttle.set("collector", collector_data)
            self._update_traj_ids(env_output)
            data = self._shuttle
            trajectories.append(data)
            collected_frames += data.numel()

        results_td: TensorDict = lazy_stack(trajectories, -1)
        return results_td

    def rollout_step(self, policy_version: int) -> tuple[Trajectory, dict[str, float]]:
        """Executes a rollout and processes the results into a standardized format.

        This method extends the base rollout functionality by:
        1. Converting raw TensorDict trajectories into a structured Trajectory object
        2. Computing sequence lengths and generating unique sequence IDs
        3. Tracking performance metrics (generation time, memory usage, etc.)

        Returns:
            Tuple containing:
            - Trajectory: Structured representation of collected trajectories with metadata
            - Dict[str, float]: Runtime metrics including generation time and memory usage
        """
        # TODO - replace perf counter with CUDA events
        start = time.perf_counter()
        # Convert raw trajectories into our Trajectory representation
        rollout_td = self.rollout().squeeze()
        query_responses = torch.cat(
            [rollout_td["tokens"], rollout_td["tokens_response"]], dim=-1
        )
        response_tokens = rollout_td["tokens_response"]
        logprobs = rollout_td["log_probs"]
        query_response_padding_masks = torch.ne(query_responses, self._tokenizer.pad_id)
        answers = rollout_td["answers"]

        response_padding_masks = torch.eq(response_tokens, self._tokenizer.pad_id)
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        del response_padding_masks

        # Generate unique sequence IDs for the batch
        # FIXME: it outputs a List[List[str]] when sampling from replay buffer, with shape num_samples X 16.
        # It should have shape num_samplesX1, so we can log a single sequence_id per sequence.
        batch_size = query_responses.shape[0]
        sequence_ids = NonTensorStack(
            *[
                f"worker{self.global_rank}_{self.local_rank}_{self._sequence_counter + i}"
                for i in range(batch_size)
            ]
        )
        total_generated_tokens = seq_lens.sum().item()

        trajectory = Trajectory(
            query_responses=query_responses.to("cpu"),
            responses=response_tokens.to("cpu"),
            logprobs=logprobs.to("cpu"),
            ref_logprobs=None,
            query_response_padding_masks=query_response_padding_masks.to("cpu"),
            seq_lens=seq_lens.to("cpu"),
            answers=answers,
            policy_version=policy_version,
            rewards=None,
            advantages=None,
            successes=None,
            reward_metadata=None,
            sequence_ids=sequence_ids,
        )

        # compute metrics
        rollout_time = time.perf_counter() - start
        generation_time = self.generation_time
        self.generation_time = 0

        pct_time_model_running = (
            (generation_time / rollout_time) * 100 if rollout_time > 0 else 0
        )
        tokens_per_second = (
            total_generated_tokens / generation_time if generation_time > 0 else 0
        )
        div_gib = 1024**3
        gpu_memory_peak_allocated_gib = (
            torch.cuda.max_memory_allocated(device=self.device) / div_gib
        )
        memory_reserved_gib = (
            torch.cuda.max_memory_reserved(device=self.device) / div_gib
        )
        memory_active_gib = (
            torch.cuda.memory_stats(device=self.device).get("active_bytes.all.peak", 0)
            / div_gib
        )
        runtime_metrics = {
            "datacollector_worker_performance/total_rollout_time (s)": rollout_time,
            "datacollector_worker_performance/pct_time_model_running (%)": pct_time_model_running,
            "datacollector_worker_performance/tokens_per_second": tokens_per_second,
            "datacollector_worker_performance/gpu_memory_peak_allocated (GiB)": gpu_memory_peak_allocated_gib,
            "datacollector_worker_performance/gpu_memory_peak_reserved (GiB)": memory_reserved_gib,
            "datacollector_worker_performance/gpu_memory_peak_active (GiB)": memory_active_gib,
        }
        return trajectory, runtime_metrics


class RolloutActor(Actor):
    """Data collector for LLM inference."""

    def __init__(
        self,
        global_rank: int,
        cfg: DictConfig,
        metric_actor: MetricsLoggerActor,
        rollout_queue_actor: QueueActor,
        param_store: ParameterStore,
        dist_info: DistributedInfo,
        reset_at_each_iter: bool = True,
        dialog_turns_per_batch: int = 1,
    ):
        self.cfg = cfg
        self.reset_at_each_iter = reset_at_each_iter
        self._shuttle = None
        self.dialog_turns_per_batch = dialog_turns_per_batch
        self._metric_actor = metric_actor
        self._rollout_queue_actor = rollout_queue_actor
        self._param_store = param_store
        self._dist_info = dist_info

        # local_rank = parallelism rank within a distributed group
        self.local_rank = current_rank()["gpus"]
        # global_rank = index of this rollout actor out of all rollout actors
        self.global_rank = global_rank

    @endpoint
    async def initialize(self):
        """Initializes the rollout actor's components.

        Notes:
        - RolloutActor is intentionally kept separate from SyncLLMCollector so that
          the latter can be kept defined synchronously (actor endpoints must be async)
        - Initialize is kept separate from __init__ so that things like rank, global rank,
          and errors can be propagated.

        """
        logger = get_logger()
        logger.info("Initializing rollout actor...")
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        set_seed(self.cfg.seed)
        weight_update_receiver = VLLMHFWeightUpdateReceiver(
            dist_info=self._dist_info,
            param_store=self._param_store,
            worker_idx=self.global_rank,
        )
        self.collector = SyncLLMCollector(
            cfg=self.cfg,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            reset_at_each_iter=self.reset_at_each_iter,
            dialog_turns_per_batch=self.dialog_turns_per_batch,
            weight_update_receiver=weight_update_receiver,
        )

        await self.collector.weight_update_receiver.connect_with_parameter_store()
        logger.info("Done with init!")

    @endpoint
    async def run(self):
        """Runs the rollout loop."""
        logger = get_logger()
        logger.info("Running rollout actor...")

        i = 0
        policy_version = 0
        while True:
            ps_version = await self._param_store.get_version.call_one()
            if ps_version != policy_version:
                logger.info(
                    "Updating weights from {} to {}...".format(
                        policy_version, ps_version
                    )
                )
                policy_version = (
                    await self.collector.weight_update_receiver.update_weights()
                )

            logger.info(f"starting rollout for step {i} (policy {policy_version})")
            trajectories, runtime_metrics = self.collector.rollout_step(policy_version)
            await self._metric_actor.log_dict.call(runtime_metrics, step=i)
            # TODO - the first rollout step triggers vLLM initialization which should not be necessary.
            # We should be able to avoid this, but needs further investigation.
            # TODO - time the push to queue time?
            await self._rollout_queue_actor.put.call(trajectories)
            i += 1

    def __repr__(self) -> str:
        return f"RolloutActor(global={self.global_rank}/{self.cfg.orchestration.num_inference_workers})[local={self.local_rank}/{self.cfg.inference.tensor_parallel_dim})]"


# TODO - is RewardActor possibly a better name for this?
class PostProcessingActor(Actor):
    """An actor responsible for post-processing rollout trajectories."""

    def __init__(
        self,
        global_rank: int,
        cfg: DictConfig,
        metric_actor: MetricsLoggerActor,
        rollout_queue_actor: QueueActor,
        replay_buffer: ReplayBufferActor,
    ):
        self.cfg = cfg
        self.rollout_queue_actor = rollout_queue_actor
        self.replay_buffer = replay_buffer
        self.metric_actor = metric_actor
        self.global_rank = global_rank
        self.local_rank = current_rank()["gpus"]
        self._is_actor_zero = self.local_rank == 0

    @endpoint
    async def initialize(self):
        """Initializes the post-processing actor's components.

        This method sets up the tokenizer, device, and other necessary components
        for processing trajectories and computing rewards.
        """
        logger = get_logger()
        logger.info("initializing")
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        device_index = get_device_index(
            "postprocessing", self.local_rank, self.global_rank, self.cfg
        )
        logger.info("device_index: {}".format(device_index))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        self._device = torch.device("cuda:{}".format(device_index))
        torch.cuda.set_device(self._device)
        set_seed(self.cfg.seed)
        self._dtype = training.get_dtype("bf16", device=self._device)
        self._ref_model = self._build_reference_model()
        self._temperature = self.cfg.inference.temperature
        self.group_size = self.cfg.inference.group_size
        self.vllm_batch_size = self.cfg.inference.batch_size
        self._log_peak_memory_stats = self.cfg.metric_logger.get(
            "log_peak_memory_stats", True
        )
        if self._is_actor_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        self.STOP_TOKENS_TENSOR = torch.tensor(
            self._tokenizer.stop_tokens, device=self._device
        )
        logger.info("done with init!")

    def _build_reference_model(self) -> torch.nn.Module:
        """Builds and returns the reference model for reward computation.

        This method initializes the reference model, loads its state, and prepares it
        for evaluation.

        Returns:
            torch.nn.Module: The initialized reference model
        """
        ref_checkpointer = config.instantiate(
            self.cfg.postprocessing.ref_checkpointer, resume_from_checkpoint=False
        )
        state_dict = ref_checkpointer.load_checkpoint()[training.MODEL_KEY]
        for k, v in state_dict.items():
            state_dict[k] = v.to(self._device)

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            ref_model = config.instantiate(self.cfg.model)

        with training.set_default_dtype(self._dtype), self._device:
            for m in ref_model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        ref_model.load_state_dict(state_dict, assign=True, strict=True)

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(ref_model)
        disable_dropout(ref_model)
        return ref_model

    async def _log_metrics(
        self,
        step_idx: int,
        time_total_ref_step: float,
        time_model_running: float,
        time_waiting_buffer: float,
        # full_queue_data_discard: int,
        rollout_queue_size: int,
        rewards_mean: torch.Tensor,
        successes_mean: torch.Tensor,
        rewards_mean_per_func: torch.Tensor,
        successes_mean_per_func: torch.Tensor,
        reward_metadata: Dict[str, List[str]],
    ):
        """Logs metrics for the post-processing actor.

        This method logs various performance and reward metrics to the metric actor.

        Args:
            step_idx (int): Current step index
            time_total_ref_step (float): Total time taken for the reference step
            time_model_running (float): Time spent running the model
            time_waiting_buffer (float): Time spent waiting for the buffer
            rollout_queue_size (int): Size of the rollout queue
            rewards_mean (torch.Tensor): Mean of the rewards
            successes_mean (torch.Tensor): Mean of the successes
            rewards_mean_per_func (torch.Tensor): Mean rewards per function
            successes_mean_per_func (torch.Tensor): Mean successes per function
            reward_metadata (Dict[str, List[str]]): Metadata about the rewards
        """
        if not self._is_actor_zero:
            return

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"postprocessing_worker_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        pct_time_model_running = (
            (time_model_running / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )
        pct_time_waiting_buffer = (
            (time_waiting_buffer / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )

        log_dict.update(
            {
                "postprocessing_worker_performance/time_total_ref_step (s)": time_total_ref_step,
                "postprocessing_worker_performance/time_model_running (s)": time_model_running,
                "postprocessing_worker_performance/pct_time_model_running (%)": pct_time_model_running,
                "postprocessing_worker_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "postprocessing_worker_performance/pct_time_waiting_buffer (%)": pct_time_waiting_buffer,
                # "queues/postprocessing_worker_full_queue_data_discard": full_queue_data_discard,
                "queues/rollout_queue_size": rollout_queue_size,
            }
        )

        log_dict.update(
            {
                "postprocessing_worker_rewards/rewards_mean": rewards_mean.item(),
                "postprocessing_worker_rewards/successes_mean": successes_mean.item(),
            }
        )

        # TODO we should encode this in the dataclass instead of keeping a dict
        # otherwise we end up with a list of identical dicts
        # assert all(
        #     metadata["func_names"] == reward_metadata[0]["func_names"]
        #     for metadata in reward_metadata
        # ), "Function names in reward_metadata are not consistent across all entries"
        # function_names = reward_metadata[0]["func_names"]

        function_names = reward_metadata["func_names"]

        # Per-function rewards and successes
        for func_name, func_mean in zip(function_names, rewards_mean_per_func):
            log_dict[f"postprocessing_worker_rewards/rewards_func_{func_name}_mean"] = (
                func_mean.item()
            )
        for func_name, func_mean in zip(function_names, successes_mean_per_func):
            log_dict[
                f"postprocessing_worker_rewards/successes_func_{func_name}_mean"
            ] = func_mean.item()
        await self.metric_actor.log_dict.call(log_dict, step=step_idx)

    @endpoint
    async def run(self):
        """Runs the post-processing loop.

        This method continuously processes trajectories from the rollout queue,
        computes rewards, updates the replay buffer, and logs metrics.
        """
        logger = get_logger()
        logger.info("running postprocessor")

        idx = 0
        with self._device:
            while True:
                # Start measuring total step time
                time_step_start = time.perf_counter()
                trajectory = None
                while trajectory is None:
                    if self._is_actor_zero:
                        logger.info("Getting from rollout_queue queue.")
                    # TODO - revisit this to check on failure conditions
                    trajectory = await self.rollout_queue_actor.get.call()
                    trajectory = trajectory.to(self._device)
                time_wait_end = time.perf_counter()
                time_waiting_buffer = time_wait_end - time_step_start

                context_length = (
                    trajectory.query_responses.shape[1] - trajectory.responses.shape[1]
                )

                masks = generation.get_causal_mask_from_padding_mask(
                    trajectory.query_response_padding_masks
                )
                position_ids = generation.get_position_ids_from_padding_mask(
                    trajectory.query_response_padding_masks
                )

                # Reset GPU memory stats before model_running
                torch.cuda.reset_peak_memory_stats()

                time_grpo_steps_start = time.perf_counter()
                with torch.no_grad():
                    ref_logits = self._ref_model(
                        trajectory.query_responses, input_pos=position_ids, mask=masks
                    )
                time_model_running = time.perf_counter() - time_grpo_steps_start

                ref_logits = rlhf.truncate_sequence_for_logprobs(
                    ref_logits, context_length
                )
                ref_logprobs = rlhf.batched_logits_to_logprobs(
                    ref_logits, trajectory.responses, self._temperature
                )

                group_size = self.cfg.inference.group_size  # G
                batch_size = self.cfg.inference.batch_size  # B

                del ref_logits, position_ids, masks
                # masking of ref_logprobs is done in grpo_step

                # Extract components from raw trajectory: these have size [B * G, T]
                query_responses = trajectory.query_responses
                responses = trajectory.responses
                query_response_padding_masks = trajectory.query_response_padding_masks
                answers = trajectory.answers  # list[str] of len (B * G)
                answers = [
                    answers[i : i + self.group_size]
                    for i in range(0, len(answers), self.group_size)
                ]  # list[list[str]] of len [B, G]. Basically a reshape

                # Truncate sequences at first stop token
                (
                    response_padding_masks,
                    responses,
                ) = rlhf.truncate_sequence_at_first_stop_token(
                    responses,
                    self.STOP_TOKENS_TENSOR.to(self._device),
                    self._tokenizer.pad_id,
                )

                # Generate masks and position IDs
                masks = generation.get_causal_mask_from_padding_mask(
                    query_response_padding_masks
                )
                position_ids = generation.get_position_ids_from_padding_mask(
                    query_response_padding_masks
                )
                context_length = query_responses.shape[1] - responses.shape[1]
                del query_response_padding_masks

                # Compute rewards
                responses = responses.reshape(batch_size, group_size, -1)
                rewards_by_fn, successes_by_fn, reward_metadata = await batched_rewards(
                    self._tokenizer, responses, answers, device=self._device
                )  # These are (B, G, num_funcs)

                # Compute advantages: B, G, num_funcs -> B, G
                group_rewards = rewards_by_fn.sum(-1)

                # To compute advantage, subtract the mean of the group rewards from each group reward
                group_advantages = (
                    group_rewards - group_rewards.mean(1, keepdim=True)
                ) / (
                    group_rewards.std(1, keepdim=True) + 1e-4
                )  # (B, G)

                # Repack trajectory with policy_version

                trajectory = Trajectory(
                    query_responses=trajectory.query_responses,
                    responses=trajectory.responses,
                    logprobs=trajectory.logprobs,
                    ref_logprobs=ref_logprobs,
                    query_response_padding_masks=trajectory.query_response_padding_masks,
                    seq_lens=trajectory.seq_lens,
                    answers=trajectory.answers,
                    policy_version=trajectory.policy_version,
                    rewards=rewards_by_fn.reshape(
                        batch_size * group_size, -1
                    ),  # (B, G, num_funcs)
                    advantages=group_advantages.reshape(
                        batch_size * group_size
                    ),  # (B, G)
                    successes=successes_by_fn.reshape(
                        batch_size * group_size, -1
                    ),  # (B, G, num_funcs)
                    reward_metadata=reward_metadata,
                    batch_size=batch_size * group_size,
                    sequence_ids=trajectory.sequence_ids,
                )

                logger.info(f"Constructed trajectory: {trajectory}")
                # Move tensors to CPU before putting into the queue
                trajectory = trajectory.cpu()

                # Update circular queue
                logger.info("extending replay buffer")
                await self.replay_buffer.extend.call(trajectory)

                # End of step timing
                time_total_ref_step = time.perf_counter() - time_step_start

                # Calculate mean rewards and successes for logging
                rewards_mean_per_func = rewards_by_fn.mean(dim=(0, 1)).cpu()
                successes_mean_per_func = successes_by_fn.mean(dim=(0, 1)).cpu()
                rewards_mean = rewards_mean_per_func.mean()
                successes_mean = successes_mean_per_func.mean()
                # log metrics
                if self._is_actor_zero:
                    await self._log_metrics(
                        step_idx=idx,
                        time_total_ref_step=time_total_ref_step,
                        time_model_running=time_model_running,
                        time_waiting_buffer=time_waiting_buffer,
                        # TODO: what should we do with this? We can log the total number of elements written in the buffer instead
                        # full_queue_data_discard=full_queue_data_discard,
                        rollout_queue_size=await self.rollout_queue_actor.qsize.call(),
                        rewards_mean=rewards_mean,
                        successes_mean=successes_mean,
                        rewards_mean_per_func=rewards_mean_per_func,
                        successes_mean_per_func=successes_mean_per_func,
                        reward_metadata=reward_metadata,
                    )

                torch.cuda.empty_cache()

                idx += 1

    def __repr__(self) -> str:
        return f"PostProcessingActor(global={self.global_rank}/{self.cfg.orchestration.num_postprocessing_workers})[local={self.local_rank}/{self.cfg.postprocessing.tensor_parallel_dim})]"


class TrainingActor(Actor):
    """An actor responsible for training models."""

    def __init__(
        self,
        cfg: DictConfig,
        metric_actor: MetricsLoggerActor,
        replay_buffer: ReplayBufferActor,
        param_store: ParameterStore,
        dist_info: DistributedInfo,
    ):
        self.cfg = cfg
        self.local_rank = current_rank()["gpus"]
        self.metric_actor = metric_actor
        self.replay_buffer = replay_buffer
        self.param_store = param_store
        self._dist_info = dist_info

    @endpoint
    async def initialize(self):
        """Initializes the training actor's components.

        This method sets up the distributed environment, model, optimizer, and other
        necessary components for training. It also initializes the process group for
        distributed training and sets the device for computation.
        """
        logger = get_logger()
        logger.info("initializing trainer...")

        # Distributed training setup: Simulate torchrun environment
        # Note - we allocate an extra GPU for the parameter server.
        # 1 index reserved by parameter server
        # TODO - replace `get_device` with `get_device_idx`
        device_index = get_device_index(
            entity="training", local_rank=self.local_rank, global_rank=0, cfg=self.cfg
        )
        set_seed(self.cfg.seed)
        logger.info("device index: {}".format(device_index))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        os.environ["RANK"] = str(self.local_rank + 1)
        os.environ["WORLD_SIZE"] = str(self._dist_info.world_size)
        os.environ["MASTER_ADDR"] = str(self._dist_info.address)
        os.environ["MASTER_PORT"] = str(self._dist_info.port)

        self._output_dir = self.cfg.output_dir
        self._log_every_n_steps = self.cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = self.cfg.get("log_peak_memory_stats", True)

        self.fsdp_cpu_offload = self.cfg.training.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            "cuda", offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        self._device = torch.device("cuda:{}".format(device_index))
        torch.cuda.set_device(self._device)

        logger.info(
            f"distributed init: rank: {os.environ['RANK']}, world_size: {os.environ['WORLD_SIZE']}, addr: {os.environ['MASTER_ADDR']}, port: {os.environ['MASTER_PORT']}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}, backend: {self.distributed_backend}"
        )
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.distributed_backend, rank=self.local_rank + 1
            )

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        ranks = list(range(1, self.world_size))
        self.fsdp_group = torch.distributed.new_group(
            ranks=ranks,
            use_local_synchronization=True,
        )
        self.device_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(
            self.fsdp_group, device_type="cuda"
        )

        self._is_rank_zero = self.local_rank == 0

        # Training configuration
        self._clip_grad_norm = self.cfg.training.get("clip_grad_norm", None)

        # Activation checkpointing and offloading
        self._enable_activation_checkpointing = self.cfg.training.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = self.cfg.training.get(
            "enable_activation_offloading", False
        )
        if (
            self._enable_activation_offloading
            and not self._enable_activation_checkpointing
        ):
            raise RuntimeError(
                "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
            )

        self._dtype = training.get_dtype("bf16", device=self._device)
        # Recipe state
        self.global_step = 0
        self._steps_run = 0
        self._total_dialog_turns = self.cfg.orchestration.num_steps

        # RL parameters
        self.save_every_n_steps = self.cfg.training.save_every_n_steps
        self._ppo_epochs = self.cfg.training.ppo_epochs

        # Model and optimizer setup
        self._checkpointer = config.instantiate(
            self.cfg.training.checkpointer, resume_from_checkpoint=False
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        self._compile = self.cfg.training.get("compile", False)
        self._model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=self.cfg.training.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=self.cfg.training.optimizer
        )
        self._loss_fn = config.instantiate(self.cfg.training.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        # The loss may handle the output projection. If true, the model should skip it.
        self.linear_loss = getattr(self._loss_fn, "linear_loss", False)
        self._model.skip_linear_projection = self.linear_loss

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        # FIXME: need to get _steps_per_epoch when dataloader is no longer per fsdp worker but instead wrapped in vLLM
        self._lr_scheduler = None

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if self.cfg is missing profiler key or if `self.cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(self.cfg.get(PROFILER_KEY, None))
        self._steps_before_sync = self.cfg.training.steps_before_weight_sync

        # Initialize policy version for tracking age of trajectories
        self.policy_version = 0
        self.metric_logger = None  # Placeholder for the logger

        # Debugging configuration
        self.debug_logging_enabled = self.cfg.get("debug_logging_enabled", True)
        self.debug_num_samples_per_step = self.cfg.get("debug_num_samples_per_step", 2)
        logger.info("done with init")

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> torch.profiler.profile | DummyProfiler:
        """Set up the profiler based on the configuration. Returns DummyProfiler if not enabled."""
        logger = get_logger()
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)
        if self._is_rank_zero:
            logger.info(f"Profiler config after instantiation: {profiler_cfg}")
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
        return profiler

    # FIXME: do we need this?
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self._model(*args, **kwargs)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """
        logger = get_logger()
        logger.info("FSDP is enabled. Instantiating model and loading checkpoint...")
        time_setup_start = time.perf_counter()
        logger.info("instantiating model")
        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            logger.info("compiling model")
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            logger.info("enabling activation checkpointing")
            training.set_activation_checkpointing(
                model, auto_wrap_policy={TransformerSelfAttentionLayer}
            )

        fsdp_shard_conditions = [
            partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)
        ]
        logger.info("sharding model")
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=True,
            dp_mesh=self.device_mesh,
        )

        logger.info("setting dtypes and such")
        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        logger.info("loading full state dict")
        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        if self._is_rank_zero:
            logger.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - time_setup_start:.2f} secs"
            )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )
        logger.info("creating activations handling ctx")
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero and self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        # synchronize before training begins
        logger.info("barrier before training begins")
        torch.distributed.barrier(group=self.fsdp_group)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict=None
    ) -> torch.optim.Optimizer:
        """Initialize the optimizer."""
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        logger = get_logger()
        if self._is_rank_zero:
            logger.info("Optimizer is initialized.")
        return optimizer

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (GRPOTrajectory): a batch of trajectories
            context_length (int): the length of the context window

        Raises:
            NotImplementedError: If the loss is not a LinearGRPOLoss.

        Returns:
            GRPOStats: Instance of :class:`~torchtune.rlhf.GRPOStats`
        """
        # Create an output mask to avoid computing model.output on tokens we won't train
        # FIXME: when bsz>1, don't we have multiple context_length?
        # FIXME: because of chunked CE, the outout of pi_logits is a chunked list, so masking after the fact is
        # more annoying. Masking before the chunking is easier, but we have to figure out masking for bsz>1
        output_mask = torch.zeros_like(
            trajectory.query_responses, dtype=torch.bool, device=self._device
        )
        output_mask[:, context_length - 1 : -1] = True

        # call model
        with self.activations_handling_ctx:
            outputs = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
            )
        bsz, _, dim = outputs.shape
        outputs = outputs[output_mask]
        outputs = outputs.reshape(bsz, -1, dim)
        targets = trajectory.query_responses[:, context_length:]

        if self.linear_loss:
            weight = self._model.linear_projection_weight
            # Compute GRPO loss
            loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs = self._loss_fn(
                # pi_logits=pi_logits,
                weight=weight,
                outputs=outputs,
                targets=targets,
                ref_logprobs=trajectory.ref_logprobs,
                advantages=trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )
        else:
            raise NotImplementedError(
                "We currently only support linear losses. Please use LinearGRPOLoss."
            )

        with torch.no_grad():
            mask = ~trajectory.response_padding_masks  # True for non-padded tokens
            approx_policy_kls = (
                0.5 * ((pi_logprobs - trajectory.logprobs)[mask].pow(2)).mean()
            )

        # Handle trajectory return based on debug mode
        metadata = {}
        if self.debug_logging_enabled:
            metadata["pi_logprobs"] = pi_logprobs.detach()

        stats = GRPOStats(
            loss=loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            ratios=ratios,
            clipfrac=clipfrac,
            approx_policy_kls=approx_policy_kls,
            metadata=metadata,
        )

        del outputs, pi_logprobs
        torch.cuda.empty_cache()  # TODO: Test if this is needed
        loss.backward()

        return stats

    def cleanup_after_step(
        self, trajectory: GRPOTrajectory, l_grpo_stats: List[GRPOStats]
    ) -> None:
        """Clean up memory after a training step."""
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats

    async def _log_metrics(
        self,
        step_idx,
        trajectory,
        grpo_stats,
        total_step_time,
        time_grpo_steps,
        time_waiting_buffer,
        time_weight_sync,
        time_weight_gather,
        number_of_tokens,
        padded_tokens_percentage,
        policy_age,
        train_replay_buffer_size,
    ):
        """Log training metrics, only on rank zero."""
        if not self._is_rank_zero:
            return

        # Stack list[GRPOStats]
        tensor_fields = [
            "loss",
            "policy_loss",
            "kl_loss",
            "ratios",
            "clipfrac",
            "approx_policy_kls",
        ]
        grpo_stats_stacked = GRPOStats(
            **{
                field: torch.stack([getattr(stats, field) for stats in grpo_stats])
                for field in tensor_fields
            }
        )

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"train_worker_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        log_dict.update(
            {
                "train_worker_training/loss": grpo_stats_stacked.loss.mean().item(),
                "train_worker_training/policy_loss": grpo_stats_stacked.policy_loss.mean().item(),
                "train_worker_training/num_stop_tokens": trajectory.response_padding_masks.any(
                    -1
                )
                .sum()
                .item(),
                "train_worker_training/kl_loss": grpo_stats_stacked.kl_loss.mean().item(),
                "train_worker_training/ratios": grpo_stats_stacked.ratios.mean().item(),
                "train_worker_training/clipfrac": grpo_stats_stacked.clipfrac.mean().item(),
                "train_worker_training/approx_policy_kls": grpo_stats_stacked.approx_policy_kls.mean().item(),
                "train_worker_training/response_lengths": trajectory.seq_lens.float()
                .mean()
                .item(),
            }
        )

        log_dict.update(
            {
                "train_worker_performance/total_step_time (s)": total_step_time,
                "train_worker_performance/time_grpo_steps (s)": time_grpo_steps,
                "train_worker_performance/pct_time_grpo_steps (%)": (
                    time_grpo_steps / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_worker_performance/tokens_per_second": (
                    number_of_tokens / total_step_time if total_step_time > 0 else 0
                ),
                "train_worker_performance/time_weight_sync (s)": time_weight_sync,
                "train_worker_performance/pct_time_weight_sync (%)": (
                    time_weight_sync / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_worker_performance/padded_tokens_percentage (%)": padded_tokens_percentage,
                "train_worker_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "train_worker_performance/pct_time_waiting_buffer (%)": (
                    time_waiting_buffer / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_worker_performance/time_weight_gather (s)": time_weight_gather,
                "train_worker_performance/pct_time_weight_gather (%)": (
                    time_weight_gather / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "queues/train_worker_policy_age_mean": policy_age,
                "queues/train_replay_buffer_size": train_replay_buffer_size,
            }
        )

        await self.metric_actor.log_dict.call(log_dict, step=step_idx)

    async def _log_debug_table(
        self,
        grpo_trajectory: GRPOTrajectory,
        grpo_stats: GRPOStats,
        metadata: Dict[str, Any],
        context_length: int,
    ) -> None:
        """
        Log debugging tables to WandB with per-token and per-sample features using dictionaries.

        ATTENTION:
        - To see multiple tables in the logs check https://github.com/wandb/wandb/issues/6286#issuecomment-2734616342
        - To visualize the columns in wandb, click on 'Columns' in the bottom right, then add them to the graph."

        Args:
            grpo_trajectory (GRPOTrajectory): Object containing sequence data (query_responses, logprobs, etc.).
            grpo_stats (GRPOStats): Object with GRPO-related statistics (loss, policy_loss, etc.).
            metadata (Dict[str, Any]): Dictionary containing rewards, successes, policy_version, etc.
            context_length (int): Integer length of the prompt context.
        """

        async def _log_table(data: list, table_name: str) -> None:
            """Helper function to log table data to WandB."""
            if data:
                logger.info(f"Logging {table_name} for step {self._steps_run}")
                columns = list(data[0].keys())
                table_data = []
                for row in data:
                    table_data.append([row[col] for col in columns])

                await self.metric_actor.log_table.call(
                    table_data, columns, table_name, step=self._steps_run
                )
            else:
                logger.info(f"Failed to log {table_name} for step {self._steps_run}")

        # Determine the number of samples to log
        num_samples = min(
            self.debug_num_samples_per_step, grpo_trajectory.query_responses.size(0)
        )

        # Extract response tokens
        targets = grpo_trajectory.query_responses[:, context_length:]
        per_sample_table_data = []
        per_token_table_data = []

        # Iterate over each sample
        for idx in range(num_samples):
            func_names = metadata["reward_metadata"][idx]["func_names"]
            sequence_id = metadata["sequence_ids"][idx]
            seq_len = grpo_trajectory.seq_lens[idx].item()

            prompt_tokens = grpo_trajectory.query_responses[
                idx, :context_length
            ].tolist()

            response_tokens = grpo_trajectory.query_responses[
                idx, context_length:
            ].tolist()

            prompt = self._tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            response = self._tokenizer.decode(
                response_tokens, skip_special_tokens=False
            )
            decoded_tokens = [
                self._tokenizer.decode([token], skip_special_tokens=False)
                for token in response_tokens
            ]

            # Per-Sample Data
            per_sample_dict = {}
            per_sample_dict["Sequence ID"] = sequence_id
            per_sample_dict["prompt"] = prompt
            per_sample_dict["response"] = response
            per_sample_dict["answers"] = grpo_trajectory.answers[idx]
            per_sample_dict["policy_version"] = metadata["policy_version"][idx]

            # Add rewards dynamically based on func_names
            rewards = metadata["rewards"][idx].tolist()
            for func_name, reward in zip(func_names, rewards):
                per_sample_dict[f"reward_{func_name}"] = reward

            # Add successes dynamically based on func_names
            successes = metadata["successes"][idx].tolist()
            for func_name, success in zip(func_names, successes):
                per_sample_dict[f"success_{func_name}"] = success

            # Add GRPO statistics, handling per-sample vs. scalar cases
            # TODO: currently has one scalar per batch. We should enable a scalar per sentence.
            # Need to refactor loss reduction to enable that.
            stat_attrs = [
                "loss",
                "policy_loss",
                "kl_loss",
                "ratios",
                "clipfrac",
                "approx_policy_kls",
            ]
            for attr_name in stat_attrs:
                stat = getattr(grpo_stats, attr_name)
                per_sample_dict[attr_name] = (
                    stat[idx].item() if stat.dim() > 0 else stat.item()
                )

            # Add advantages
            per_sample_dict["advantages"] = grpo_trajectory.advantages[idx].item()

            # Add sequence metrics
            per_sample_dict["response_length"] = seq_len
            per_sample_dict["context_length"] = context_length
            per_sample_dict["has_stop_token"] = (
                grpo_trajectory.response_padding_masks[idx].any().item()
            )

            # Check if prompt tokens are included in loss (should be 0)
            per_sample_dict["prompt_masking_is_positive (should be 0)"] = (
                grpo_trajectory.response_padding_masks[idx, :context_length]
                .sum()
                .item()
            )

            # Check if tokens beyond seq_len are included in loss (should be 0)
            if context_length + seq_len < grpo_trajectory.query_responses.shape[1]:
                beyond_seq_len = (
                    grpo_trajectory.response_padding_masks[
                        idx, context_length + seq_len :
                    ]
                    .sum()
                    .item()
                )
            else:
                beyond_seq_len = 0
            per_sample_dict["beyond_seq_len_masking_is_positive (should be 0)"] = (
                beyond_seq_len
            )

            per_sample_dict["num_tokens_response"] = seq_len
            per_sample_dict["step"] = self._steps_run

            # Append the dictionary to the per-sample table data
            per_sample_table_data.append(per_sample_dict)

            # Per-Token Data
            for pos in range(seq_len):
                per_token_dict = {}
                per_token_dict["Sequence ID"] = sequence_id
                per_token_dict["Token Position"] = pos  # TODO: maybe remove?
                per_token_dict["Token ID"] = targets[idx, pos].item()
                per_token_dict["Decoded Token"] = decoded_tokens[pos]
                per_token_dict["generated_logprob"] = grpo_trajectory.logprobs[
                    idx, pos
                ].item()
                per_token_dict["ref_logprob"] = grpo_trajectory.ref_logprobs[
                    idx, pos
                ].item()
                per_token_dict["pi_logprob"] = (
                    grpo_stats.metadata["pi_logprobs"][idx, pos].item()
                    if grpo_stats.metadata
                    else None
                )
                per_token_dict["abs_diff_pi_ref_logprob"] = abs(
                    per_token_dict["pi_logprob"] - per_token_dict["ref_logprob"]
                )
                per_token_dict["abs_diff_pi_generated_logprob"] = abs(
                    per_token_dict["pi_logprob"] - per_token_dict["generated_logprob"]
                )
                per_token_dict["mask"] = int(
                    ~grpo_trajectory.response_padding_masks[idx, pos]
                )
                per_token_dict["step"] = self._steps_run

                # Append the dictionary
                per_token_table_data.append(per_token_dict)

        # Log tables to WandB
        await _log_table(per_sample_table_data, "per_sample_debug_table")
        await _log_table(per_token_table_data, "per_token_debug_table")

    @endpoint
    async def run(self):
        """Executes the GRPO training loop."""
        with self._device:
            logger = get_logger()
            logger.info("Starting GRPO training loop...")
            training.cleanup_before_training()
            self._optimizer.zero_grad()
            self._profiler.start()

            while self._steps_run < self._total_dialog_turns:
                # Memory profiling start
                if (
                    self._is_rank_zero
                    and self.profiler_profile_memory
                    and self._steps_run
                    == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                time_step_start = time.perf_counter()

                # Fetch trajectory from queue
                time_waiting_buffer_start = time.perf_counter()
                train_replay_buffer_size = None
                if self._is_rank_zero:
                    train_replay_buffer_size = await self.replay_buffer.len.call()

                num_waits = 0
                while await self.replay_buffer.is_empty.call():
                    if self._is_rank_zero and num_waits % 10 == 0:
                        logger.info("waiting for replay buffer...")
                    await asyncio.sleep(1)
                    num_waits += 1

                # TODO - batching?
                trajectory = await self.replay_buffer.sample.call()
                trajectory = trajectory.to(self._device)
                time_waiting_buffer = time.perf_counter() - time_waiting_buffer_start
                if self._is_rank_zero:
                    logger.info(f"{self.local_rank=} got from queue traj {trajectory}")

                # Prepare trajectory for optimization
                prepared_trajectory, context_length, metadata = (
                    self._prepare_trajectory(trajectory)
                )

                # Perform GRPO optimization
                time_grpo_steps_start = time.perf_counter()
                grpo_stats: list[GRPOStats] = []
                for _ in range(self._ppo_epochs):
                    # step
                    step_stats = self.grpo_step(
                        prepared_trajectory,
                        context_length,
                    )
                    grpo_stats.append(step_stats)
                    # grad norm
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )

                    # optimizer step
                    torch.distributed.barrier(group=self.fsdp_group)
                    self._optimizer.step()
                    torch.distributed.barrier(group=self.fsdp_group)
                    self._optimizer.zero_grad(set_to_none=True)

                    # scheduler
                    self.global_step += 1
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                logger.info(f"{self.local_rank=} finished step {self._steps_run}")
                time_grpo_steps = time.perf_counter() - time_grpo_steps_start
                self._steps_run += 1

                # Log debug table if enabled, using pi_logprobs from the first epoch
                if (
                    self.debug_logging_enabled
                    and self._is_rank_zero
                    and self._steps_run % self._log_every_n_steps == 0
                ):
                    await self._log_debug_table(
                        prepared_trajectory, grpo_stats[0], metadata, context_length
                    )

                # Synchronize weights
                time_weight_sync = time_weight_gather = 0
                gathered_sd = None
                if self._steps_run % self._steps_before_sync == 0:
                    torch.distributed.barrier(group=self.fsdp_group)
                    time_weight_gather_start = time.perf_counter()
                    gathered_sd = {
                        k: v.full_tensor() for k, v in self._model.state_dict().items()
                    }
                    torch.cuda.synchronize()
                    time_weight_gather = time.perf_counter() - time_weight_gather_start
                    if self._is_rank_zero:
                        logger.info(f"Done gather in {time_weight_gather}")
                    time_sync_start = time.perf_counter()
                    await self.sync_weights(gathered_sd)
                    time_weight_sync = time.perf_counter() - time_sync_start
                    if self._is_rank_zero:
                        logger.info(f"Done sync in {time_weight_sync}")

                # Log metrics
                total_step_time = time.perf_counter() - time_step_start
                if (
                    self._is_rank_zero
                    and self._steps_run % self._log_every_n_steps == 0
                ):
                    await self._log_metrics(
                        step_idx=self.global_step,
                        trajectory=prepared_trajectory,
                        grpo_stats=grpo_stats,
                        total_step_time=total_step_time,
                        time_grpo_steps=time_grpo_steps,
                        time_waiting_buffer=time_waiting_buffer,
                        time_weight_sync=0,
                        time_weight_gather=0,
                        number_of_tokens=metadata["number_of_tokens"],
                        padded_tokens_percentage=metadata["padded_tokens_percentage"],
                        policy_age=metadata["avg_policy_age"],
                        train_replay_buffer_size=train_replay_buffer_size,
                    )

                self.cleanup_after_step(trajectory, grpo_stats)

                # Save a copy of the weights
                if self._steps_run % self.save_every_n_steps == 0:
                    if gathered_sd is None:
                        gathered_sd = {
                            k: v.full_tensor()
                            for k, v in self._model.state_dict().items()
                        }
                    self._checkpointer.save_checkpoint(
                        state_dict={training.MODEL_KEY: gathered_sd},
                        epoch=0,
                        step=self._steps_run,
                    )
                del gathered_sd

                # Memory profiling stop
                self._profiler.step()
                if (
                    self._is_rank_zero
                    and self.profiler_profile_memory
                    and self._steps_run
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                torch.distributed.barrier(group=self.fsdp_group)

            self._profiler.stop()

    async def sync_weights(self, new_sd):
        self.policy_version += 1
        logger = get_logger()
        logger.info("syncing weights w/ PS at {}".format(self.policy_version))
        # TODO - replace w/ rdmabuffer
        if self._is_rank_zero:
            h = self.param_store.put.call(version=self.policy_version)
            for k in sorted(new_sd.keys()):
                v = new_sd[k]
                # dst is global rank, can switch to group_dst arg if not 2.5.1
                torch.distributed.send(v, dst=0)
            torch.distributed.barrier()
            await h
        else:
            torch.distributed.barrier()
        logger.info("done weight syncs w/ PS")

    def _prepare_trajectory(
        self, raw_trajectory: Trajectory
    ) -> tuple[GRPOTrajectory, int, dict[str, Any]]:
        """Processes raw trajectory, compute rewards, and prepare for optimization.

        Args:
            raw_trajectory (Trajectory): The trajectory sampled from the replay buffer.

        Returns:
            Tuple[trajectory, context_length, metadata]
        """
        # Extract components from raw trajectory
        query_responses = raw_trajectory.query_responses
        responses = raw_trajectory.responses
        logprobs = raw_trajectory.logprobs
        ref_logprobs = raw_trajectory.ref_logprobs
        query_response_padding_masks = raw_trajectory.query_response_padding_masks
        seq_lens = raw_trajectory.seq_lens
        advantages = raw_trajectory.advantages
        answers = raw_trajectory.answers

        # Compute padded tokens percentage
        total_tokens = query_responses.numel()
        padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
        padded_tokens_percentage = (
            (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        )
        number_of_tokens = seq_lens.sum().item()

        # Truncate sequences at first stop token
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses,
            torch.tensor(self._tokenizer.stop_tokens, device=self._device),
            self._tokenizer.pad_id,
        )

        # Generate masks and position IDs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        context_length = query_responses.shape[1] - responses.shape[1]
        del query_response_padding_masks

        # Create GRPOTrajectory
        prepared_trajectory = GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            answers=answers,
        )

        # Metadata for logging
        if isinstance(raw_trajectory.policy_version, list):
            avg_policy_age = self.policy_version - (
                sum(raw_trajectory.policy_version) / len(raw_trajectory.policy_version)
            )
        else:
            avg_policy_age = self.policy_version - raw_trajectory.policy_version

        metadata = {
            "padded_tokens_percentage": padded_tokens_percentage,
            "number_of_tokens": number_of_tokens,
            "avg_policy_age": avg_policy_age,
            "sequence_ids": raw_trajectory.sequence_ids,
            "policy_version": raw_trajectory.policy_version,
            "rewards": raw_trajectory.rewards,
            "successes": raw_trajectory.successes,
            "reward_metadata": raw_trajectory.reward_metadata,
            "query_response_padding_masks": raw_trajectory.query_response_padding_masks,
        }
        return prepared_trajectory, context_length, metadata

    def __repr__(self) -> str:
        return f"TrainingActor([{self.local_rank}/{self.cfg.orchestration.num_training_workers})]"


# ========= Recipe =========
class MonarchGRPORecipe(OrchestrationRecipeInterface):
    async def setup(self, cfg: DictConfig) -> None:
        logger = get_logger()
        self.cfg = cfg
        # Set seed if not already set...
        if not self.cfg.seed:
            self.cfg.seed = random.randint(0, 2**32 - 1)

        logger.info("Initializing w/ config: ", cfg)

        self.num_inference_workers = cfg.orchestration.num_inference_workers
        self.num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
        self.num_training_workers = cfg.orchestration.num_training_workers

        env = {}

        # Create addresses and ports for trainer and worker distirbuted worlds.
        addresses = [get_ip() for _ in range(self.num_inference_workers + 1)]
        ports = [get_open_port() for _ in range(self.num_inference_workers + 1)]
        logger.info("Using addresses: {}".format(addresses))
        logger.info("Using ports: {}".format(ports))

        trainer_address = addresses[0]
        trainer_port = ports[0]
        vllm_addresses = addresses[1:]
        vllm_ports = ports[1:]

        # singleton_proc_mesh is the designated proc where
        # global entities (metric logger, queue) are spawned.
        # If any of these components cause stalls, we can move them out to their own procs.
        self.singleton_proc_mesh = await proc_mesh(
            gpus=1,
            env=env,
        )
        logger.info("spawning metric actor")
        self.metric_actor = await self.singleton_proc_mesh.spawn(
            "metrics", MetricsLoggerActor, cfg=cfg
        )
        logger.info("spawning rollout actor")
        self.rollout_queue_actor = await self.singleton_proc_mesh.spawn(
            "queue", QueueActor
        )
        logger.info("spawning replay buffer actor")
        self.replay_buffer_actor = await self.singleton_proc_mesh.spawn(
            "replay_buffer", ReplayBufferActor, cfg=cfg
        )
        logger.info("spawning param server actor")

        self.param_store_proc_mesh = await proc_mesh(
            gpus=1,
            env=env,
        )
        self.param_store_actor = await self.param_store_proc_mesh.spawn(
            "param_store",
            HFVLLMParameterServer,
            cfg=cfg,
        )

        self.dist_info_map = {}

        # Create rollout actors and meshes
        self.rollout_proc_meshes = []
        self.rollout_actor_meshes = []
        self.weight_updater_actors = []
        inference_tp = cfg.inference.tensor_parallel_dim

        logger.info(
            f"[rollout] Creating {self.num_inference_workers} meshes of size {inference_tp}..."
        )
        for i in range(self.num_inference_workers):
            self.rollout_proc_meshes.append(
                await proc_mesh(
                    gpus=inference_tp,
                    env=env,
                )
            )
            dist_info = DistributedInfo(
                address=vllm_addresses[i],
                port=vllm_ports[i],
                world_size=inference_tp + 1,
            )
            generator = await self.rollout_proc_meshes[i].spawn(
                "rollout_actors",
                RolloutActor,
                global_rank=i,
                cfg=cfg,
                dist_info=dist_info,
                metric_actor=self.metric_actor,
                rollout_queue_actor=self.rollout_queue_actor,
                param_store=self.param_store_actor,
            )
            self.rollout_actor_meshes.append(generator)
            self.dist_info_map[generator] = dist_info

        # Create postprocess actors and meshes
        self.postprocess_proc_meshes = []
        self.postprocess_actor_meshes = []

        postprocess_tp = self.cfg.postprocessing.tensor_parallel_dim
        logger.info(
            f"[postprocess] Creating {self.num_postprocessing_workers} meshes of size {postprocess_tp}..."
        )
        for i in range(self.num_postprocessing_workers):
            self.postprocess_proc_meshes.append(
                await proc_mesh(
                    gpus=postprocess_tp,
                    env=env,
                )
            )
            self.postprocess_actor_meshes.append(
                await self.postprocess_proc_meshes[i].spawn(
                    "postprocess_actors",
                    PostProcessingActor,
                    global_rank=i,
                    cfg=cfg,
                    metric_actor=self.metric_actor,
                    rollout_queue_actor=self.rollout_queue_actor,
                    replay_buffer=self.replay_buffer_actor,
                )
            )

        # Create training actors and meshes
        training_shards = self.cfg.orchestration.num_training_workers
        logger.info(f"[training] Creating mesh of size {training_shards}...")
        self.training_mesh = await proc_mesh(
            gpus=training_shards,
            env=env,
        )

        train_dist_info = DistributedInfo(
            address=trainer_address,
            port=trainer_port,
            world_size=cfg.orchestration.num_training_workers + 1,
        )

        self.training_actor = await self.training_mesh.spawn(
            "training",
            TrainingActor,
            cfg=cfg,
            metric_actor=self.metric_actor,
            replay_buffer=self.replay_buffer_actor,
            param_store=self.param_store_actor,
            dist_info=train_dist_info,
        )
        self.dist_info_map[self.training_actor] = train_dist_info
        self.actor_set = ActorSet(
            param_store=self.param_store_actor,
            metric_logger=self.metric_actor,
            trainer=self.training_actor,
            generators=self.rollout_actor_meshes,
        )

        self.all_actors = (
            self.rollout_actor_meshes
            + self.postprocess_actor_meshes
            + [self.training_actor]
        )

    async def run(self):
        logger = get_logger()
        logger.info("initializing actors")
        # TODO - can we log which actors have and have not finished initializing somehow?
        await asyncio.gather(
            *[a.initialize.call() for a in self.all_actors]
            + [
                self.param_store_actor.initialize.call(
                    actor_set=self.actor_set, dist_info_map=self.dist_info_map
                )
            ]
        )
        logger.info("done initializing, starting to run actors")
        await asyncio.gather(*[a.run.call() for a in self.all_actors])

    async def cleanup(self):
        logger = get_logger()
        logger.info("cleaning up")

    def __repr__(self) -> str:
        return "MonarchGRPORecipeRunner"


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
