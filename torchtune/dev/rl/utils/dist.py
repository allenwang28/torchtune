# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    print(
        "CABERNET! PROCESS GROUP INITIATED (host={}, port={}, rank={}, world_size={})".format(
            master_address, master_port, rank, world_size
        )
    )
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    print("CABERNET! PROCESS GROUP CREATED: ", pg)
    print("CABERNET! CREATING COMMUNICATOR! (pg={}, device={})".format(pg, device))
    pynccl = PyNcclCommunicator(pg, device=device)
    print("CABERNET! COMMUNICATOR DONE!", pynccl)
    return pynccl
