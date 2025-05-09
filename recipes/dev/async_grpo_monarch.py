import asyncio
import functools
import inspect
import logging
import os
import time

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed

from monarch.proc_mesh import proc_mesh
from monarch.service import Actor, current_rank, current_size, endpoint
from omegaconf import DictConfig, ListConfig, OmegaConf

from tensordict import lazy_stack, NonTensorStack, TensorDict, TensorDictBase
from tensordict.utils import expand_as_right
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.collectors import SyncDataCollector

from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, utils
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory

from torchtune.dev.rl.utils import stateless_init_process_group
from torchtune.recipe_interfaces import OrchestrationRecipeInterface
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput as vllmRequestOutput

from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker


# ========= Constants =========
_TOK_RESPONSE_KEY = "tokens_response"
_TEXT_RESPONSE_KEY = "text_response"
_LOG_PROBS_KEY = "log_probs"


# ========= Logging related components =========
class MetricLoggerActor(Actor):
    """Metric logger for all actors."""

    def __init__(self, cfg):
        self.logger = config.instantiate(cfg.metric_logger)
        self.logger.log_config(cfg)

    @endpoint
    async def log_dict(self, log_dict, step=None):
        # allowing actors to use their own step counters
        self.logger.log_dict(log_dict, step=step)

    @endpoint
    async def log_table(self, table_data, columns, table_name, step=None):
        """Log a table to WandB."""
        import wandb

        table = wandb.Table(columns=columns, data=table_data)
        self.logger.log_dict({table_name: table}, step=step)

    @endpoint
    async def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()


class MonarchLogger(logging.Logger):
    """A custom logger class that adds the caller representation to the log message."""

    # ANSI color codes
    BLUE = "\033[94m"
    RESET = "\033[0m"

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s - %(message)s", "%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        caller_frame = inspect.stack()[2]
        caller_self = caller_frame.frame.f_locals.get("self")
        caller_function = caller_frame.function

        if caller_self:
            if caller_self.__class__.__repr__ is object.__repr__:
                class_name = caller_self.__class__.__name__
                try:
                    caller_repr = f"{class_name}-({current_rank()}/{current_size()})"
                except Exception as e:
                    caller_repr = class_name
            else:
                caller_repr = str(caller_self)
            msg = f"{self.BLUE}[Monarch::{caller_repr}::{caller_function}]{self.RESET} {msg}"

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


def get_logger() -> logging.Logger:
    logging.setLoggerClass(MonarchLogger)
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


# ========= Generic data structures =========


# ========= Cabernet actors =========
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
    ):
        self.cfg = cfg
        self.reset_at_each_iter = reset_at_each_iter
        self.dialog_turns_per_batch = dialog_turns_per_batch
        self.total_dialog_turns = total_dialog_turns
        self.logger = get_logger()
        self.local_rank = local_rank
        self.global_rank = global_rank
        # Create data loader
        from torchtune import config

        self._tokenizer = config.instantiate(self.cfg.tokenizer)
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
        self.inference_server = LLM(
            model=self.cfg.inference.model,
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            # worker_cls=VLLMWorkerWrapper,
            tensor_parallel_size=self.cfg.inference.tensor_parallel_dim,
            **self.cfg.inference.get("engine_args", {}),
        )
        self.generation_time = 0
        super().__init__(
            create_env_fn=env,
            policy=self.policy_fn,  # TODO - is this needed at all?
            frames_per_batch=self.dialog_turns_per_batch,
            total_frames=self.total_dialog_turns,
            weight_update_receiver=None,
            weight_update_sender=None,
            reset_at_each_iter=self.reset_at_each_iter,
            use_buffers=False,
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
                functools.partial(
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
        if self.reset_at_each_iter or self._shuttle is None:
            data = self.env.reset()
        else:
            data = self._shuttle

        trajectories = []
        collected_frames = 0
        while collected_frames < self.dialog_turns_per_batch:
            action = self.policy_fn(data)

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

        Args:
            policy_version: Version identifier for the policy being used

        Returns:
            Tuple containing:
            - Trajectory: Structured representation of collected trajectories with metadata
            - Dict[str, float]: Runtime metrics including generation time and memory usage
        """
        # local import to avoid vLLM no CUDA GPUs available error
        from torchtune import training

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
            query_responses=query_responses,
            responses=response_tokens,
            logprobs=logprobs,
            ref_logprobs=None,
            query_response_padding_masks=query_response_padding_masks,
            seq_lens=seq_lens,
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
            torch.cuda.max_memory_allocated(device="cuda:0") / div_gib
        )
        memory_reserved_gib = torch.cuda.max_memory_reserved(device="cuda:0") / div_gib
        memory_active_gib = (
            torch.cuda.memory_stats(device="cuda:0").get("active_bytes.all.peak", 0)
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
        reset_at_each_iter: bool = False,
        dialog_turns_per_batch: int = 1,
    ):
        self.cfg = cfg
        self.logger = get_logger()
        self.reset_at_each_iter = reset_at_each_iter
        self._shuttle = None
        self.dialog_turns_per_batch = dialog_turns_per_batch

        # local_rank = parallelism rank within a distributed group
        self.local_rank = current_rank()
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
        self.logger.info("Initializing rollout actor...")
        self.collector = SyncLLMCollector(
            cfg=self.cfg,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            reset_at_each_iter=self.reset_at_each_iter,
            dialog_turns_per_batch=self.dialog_turns_per_batch,
        )
        self.logger.info("done with init!")

    @endpoint
    async def run(self):
        self.logger.info("Running rollout actor...")

        num_steps = 1
        for i in range(num_steps):
            self.logger.info(f"starting rollout for step {i}")
            trajectories, runtime_metrics = self.collector.rollout_step(
                policy_version=0
            )
            self.logger.info("trajectories: %s", trajectories)
            self.logger.info("metrics: %s", runtime_metrics)
            # self.metric_logger.

    def __repr__(self) -> str:
        return f"RolloutActor[local_rank=({current_rank()},{current_size()}),global_rank={self.global_rank}]"


# ========= Recipe =========
class MonarchGRPORecipe(OrchestrationRecipeInterface):
    async def setup(self, cfg: DictConfig) -> None:
        self.logger = get_logger()
        self.logger.info("initializing w/ config: ", cfg)
        self.cfg = cfg

        self.num_shards_per_inference_worker = (
            cfg.orchestration.num_shards_per_inference_worker
        )
        self.num_inference_workers = cfg.orchestration.num_inference_workers
        self.num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
        self.num_training_workers = cfg.orchestration.num_training_workers

        # Create Proc meshes
        self.rollout_proc_meshes = []
        self.rollout_actor_meshes = []
        print(
            f"Creating {self.num_inference_workers} meshes of size {self.num_shards_per_inference_worker}..."
        )
        for i in range(self.num_inference_workers):
            self.rollout_proc_meshes.append(
                await proc_mesh(
                    gpus=self.num_shards_per_inference_worker,
                    env={},
                )
            )
            self.rollout_actor_meshes.append(
                await self.rollout_proc_meshes[i].spawn(
                    "rollout_actors",
                    RolloutActor,
                    global_rank=i,
                    cfg=cfg,
                )
            )

    async def run(self):
        self.logger.info("initializing actors")
        [await a.initialize().broadcast_and_wait() for a in self.rollout_actor_meshes]
        self.logger.info("running rollout actors")
        [await a.run().broadcast_and_wait() for a in self.rollout_actor_meshes]

    async def cleanup(self):
        self.logger.info("cleaning up")

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
