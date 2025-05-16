# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import logging
import time

from monarch.service import Actor, current_rank, current_size, endpoint
from torchtune import config


class MetricsLoggerActor(Actor):
    """Metrics logger for all actors."""

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
    """Returns the Monarch-integrated logger."""
    logging.setLoggerClass(MonarchLogger)
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(message)s"
    )  # Simplified formatter since MonarchLogger adds its own formatting
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger
