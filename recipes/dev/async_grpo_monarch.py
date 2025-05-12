import asyncio
import functools
import inspect
import itertools
import logging
import os
import time

from typing import Any, Dict, Generic, List, Optional, TypeVar

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
from torchrl.collectors import SyncDataCollector

from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, generation, rlhf, utils
from torchtune.dev.rl.datatypes import RequestOutput, Trajectory
from torchtune.dev.rl.rewards import batched_rewards

from torchtune.dev.rl.utils import stateless_init_process_group
from torchtune.recipe_interfaces import OrchestrationRecipeInterface
from torchtune.training import disable_dropout
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput as vllmRequestOutput

from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

T = TypeVar("T")


# ========= Constants =========
_TOK_RESPONSE_KEY = "tokens_response"
_TEXT_RESPONSE_KEY = "text_response"
_LOG_PROBS_KEY = "log_probs"


# ========= Logging related components =========
class DisabledMetricsLoggerActor(Actor):
    """For developing quickly (skip the wandb init overhead)"""

    def __init__(self, cfg):
        pass

    @endpoint
    async def log_dict(self, log_dict, step=None):
        logger = get_logger()
        logger.info("logging %s at step %s", log_dict, step)

    @endpoint
    async def log_table(self, table_data, columns, table_name, step=None):
        pass

    @endpoint
    async def close(self):
        pass


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

        # Get current timestamp
        timestamp = time.strftime("%m-%d %H:%M:%S")
        level_name = logging.getLevelName(level)

        try:
            if caller_self:
                if caller_self.__class__.__repr__ is object.__repr__:
                    class_name = caller_self.__class__.__name__
                    try:
                        gpu_rank = current_rank()["gpus"]
                        num_gpus = current_size()["gpus"]
                        caller_repr = f"{class_name}-({gpu_rank}{num_gpus})"
                    except Exception as e:
                        caller_repr = class_name
                else:
                    caller_repr = str(caller_self)
                msg = f"{self.BLUE}{level_name} {timestamp} - [Monarch::{caller_repr}::{caller_function}]{self.RESET} {msg}"
        except Exception:
            msg = f"{self.BLUE}{level_name} {timestamp} - [Monarch::{caller_function}]{self.RESET} {msg}"
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
class QueueActor(Actor, Generic[T]):
    def __init__(self):
        self.logger = get_logger()
        self._q: asyncio.Queue[T] = asyncio.Queue()

    @endpoint
    async def put(self, item: T) -> None:
        self.logger.info("putting %s", item)
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


def get_device(
    entity: str, local_rank: int, global_rank: int, cfg: DictConfig
) -> torch.device:
    """Returns the torch.device for the given entity/rank/config.

    Maps logical entity ranks to physical GPU indices based on entity type and configuration.
    For rollout workers, GPUs are assigned starting at index 0.
    For postprocessing workers, GPUs are assigned after all rollout worker GPUs.

    This is a placeholder implementation for now, and would need to change in a multi-host setting.
    """
    entity_world_size = -1
    if entity == "rollout":
        entity_world_size = cfg.inference.tensor_parallel_dim
        offset = 0
    elif entity == "postprocessing":
        entity_world_size = cfg.postprocessing.tensor_parallel_dim
        offset = (
            cfg.inference.tensor_parallel_dim * cfg.orchestration.num_inference_workers
        )
    else:
        raise KeyError(f"Unknown entity: {entity}")

    device_idx = offset + global_rank * entity_world_size + local_rank
    return torch.device(f"cuda:{device_idx}")


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

        device = get_device("rollout", local_rank, global_rank, cfg)
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
            device=device,
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
        metric_actor: MetricLoggerActor,
        rollout_queue_actor: QueueActor,
        reset_at_each_iter: bool = False,
        dialog_turns_per_batch: int = 1,
    ):
        self.cfg = cfg
        self.logger = get_logger()
        self.reset_at_each_iter = reset_at_each_iter
        self._shuttle = None
        self.dialog_turns_per_batch = dialog_turns_per_batch
        self._metric_actor = metric_actor
        self._rollout_queue_actor = rollout_queue_actor

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
        # Set CUDA visible devices
        # TODO - some checking here?
        vllm_world_size = self.cfg.inference.tensor_parallel_dim
        gpu_indices = list(
            range(
                self.global_rank * vllm_world_size,
                (self.global_rank + 1) * vllm_world_size,
            )
        )
        # The following env variables help guarantee GPU isolation
        gpu_indices = ",".join(str(idx) for idx in gpu_indices)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
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

        num_steps = 10
        for i in range(num_steps):
            self.logger.info(f"starting rollout for step {i}")
            trajectories, runtime_metrics = self.collector.rollout_step(
                policy_version=0
            )
            # push to metrics logger
            await self._metric_actor.log_dict(runtime_metrics).call()
            # push to queue
            self.logger.info("pushing to queue.")
            await self._rollout_queue_actor.put(trajectories).call()
            self.logger.info("done pushing to queue.")

    def __repr__(self) -> str:
        return f"RolloutActor(global={self.global_rank}/{self.cfg.orchestration.num_inference_workers})[local={self.local_rank}/{self.cfg.inference.tensor_parallel_dim})]"


# is RewardActor possibly a better name for this?
class PostProcessingActor(Actor):
    def __init__(
        self,
        global_rank: int,
        cfg: DictConfig,
        metric_actor: MetricLoggerActor,
        rollout_queue_actor: QueueActor,
        replay_buffer=None,
    ):
        self.cfg = cfg
        self.rollout_queue_actor = rollout_queue_actor
        self.replay_buffer = replay_buffer
        self.metric_actor = metric_actor
        self.global_rank = global_rank
        self.logger = get_logger()
        self.local_rank = current_rank()["gpus"]
        self._is_actor_zero = self.local_rank == 0

    @endpoint
    async def initialize(self):
        self.logger.info("initializing")
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self._device = get_device(
            "postprocessing", self.local_rank, self.global_rank, self.cfg
        )
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
        self.logger.info("done with init!")

    def _build_reference_model(self) -> torch.nn.Module:
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
        self.logger.info("done setting up ref model")
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
        """Log metrics for the RefActor, only on actor zero."""
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
        await self.metric_actor.log_dict(log_dict, step=step_idx).call()

    @endpoint
    async def run(self):
        self.logger.info("running postprocessor")

        idx = 0
        with self._device:
            while True:
                # Start measuring total step time
                time_step_start = time.perf_counter()
                trajectory = None
                while trajectory is None:
                    if self._is_actor_zero:
                        self.logger.info("Getting from rollout_queue queue.")
                    # TODO - revisit this to check on failure conditions
                    trajectory = await self.rollout_queue_actor.get().call()
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
                rewards_by_fn, successes_by_fn, reward_metadata = batched_rewards(
                    self._tokenizer, responses, answers, device=self._device
                )  # These are (B, G, num_funcs)

                # Compute advantages: B, G, num_funcs -> B, G
                self.logger.info("computing batched rewards")
                group_rewards = rewards_by_fn.sum(-1)
                self.logger.info("computed batched rewards: {}".format(group_rewards))

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

                self.logger.info(f"Constructed trajectory: {trajectory}")
                # Move tensors to CPU before putting into the queue
                trajectory = trajectory.cpu()

                # Update circular queue
                # self.replay_buffer.extend(trajectory)

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
                        rollout_queue_size=await self.rollout_queue_actor.qsize().call(),
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


# ========= Recipe =========
class MonarchGRPORecipe(OrchestrationRecipeInterface):
    async def setup(self, cfg: DictConfig) -> None:
        self.logger = get_logger()
        self.logger.info("initializing w/ config: ", cfg)
        self.cfg = cfg

        self.num_inference_workers = cfg.orchestration.num_inference_workers
        self.num_postprocessing_workers = cfg.orchestration.num_postprocessing_workers
        self.num_training_workers = cfg.orchestration.num_training_workers

        # singleton_proc_mesh is the designated proc where
        # global entities (metric logger, queue) are spawned.
        self.singleton_proc_mesh = await proc_mesh(gpus=1, env={})
        self.metric_actor = await self.singleton_proc_mesh.spawn(
            "metrics", DisabledMetricsLoggerActor, cfg=cfg
        )
        self.rollout_queue_actor = await self.singleton_proc_mesh.spawn(
            "queue", QueueActor
        )

        # Create rollout actors and meshes
        self.rollout_proc_meshes = []
        self.rollout_actor_meshes = []

        inference_tp = cfg.inference.tensor_parallel_dim
        self.logger.info(
            f"Creating {self.num_inference_workers} meshes of size {inference_tp}..."
        )
        for i in range(self.num_inference_workers):
            self.rollout_proc_meshes.append(
                await proc_mesh(
                    gpus=inference_tp,
                    env={},
                )
            )
            self.rollout_actor_meshes.append(
                await self.rollout_proc_meshes[i].spawn(
                    "rollout_actors",
                    RolloutActor,
                    global_rank=i,
                    cfg=cfg,
                    metric_actor=self.metric_actor,
                    rollout_queue_actor=self.rollout_queue_actor,
                )
            )

        # Create postprocess actors and meshes
        self.postprocess_proc_meshes = []
        self.postprocess_actor_meshes = []

        postprocess_tp = self.cfg.postprocessing.tensor_parallel_dim
        self.logger.info(
            f"Creating {self.num_postprocessing_workers} meshes of size {postprocess_tp}..."
        )
        for i in range(self.num_postprocessing_workers):
            self.postprocess_proc_meshes.append(
                await proc_mesh(
                    gpus=postprocess_tp,
                    env={},
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
                )
            )

        self.all_actors = list(
            itertools.chain(self.rollout_actor_meshes, self.postprocess_actor_meshes)
        )
        # self.all_actors = self.rollout_actor_meshes
        # self.all_actors = self.postprocess_actor_meshes

    async def run(self):
        self.logger.info("initializing actors")
        await asyncio.gather(
            *[a.initialize().broadcast_and_wait() for a in self.all_actors]
        )
        self.logger.info("running actors")
        await asyncio.gather(*[a.run().broadcast_and_wait() for a in self.all_actors])

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
