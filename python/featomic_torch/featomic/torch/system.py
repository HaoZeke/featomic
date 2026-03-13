import warnings
from typing import List, Optional, Sequence, overload

import numpy as np
import torch
from metatomic.torch import System

import featomic


@overload
def systems_to_torch(
    systems: featomic.systems.IntoSystem,
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> System:
    pass


@overload
def systems_to_torch(
    systems: Sequence[featomic.systems.IntoSystem],
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> List[System]:
    pass


def systems_to_torch(
    systems,
    positions_requires_grad=None,
    cell_requires_grad=None,
    dtype=None,
    device=None,
) -> List[System]:
    """
    Convert a arbitrary system to :py:class:`metatomic.torch.System`, putting all the
    data in :py:class:`torch.Tensor` and making the overall object compatible with
    TorchScript.

    :param system: any system supported by featomic. If this is an iterable of system,
        this function converts them all and returns a list of converted systems.

    :param positions_requires_grad: The value of ``requires_grad`` on the output
        ``positions``. If ``None`` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to ``False``.

    :param cell_requires_grad: The value of ``requires_grad`` on the output ``cell``. If
        ``None`` and the positions of the input is already a :py:class:`torch.Tensor`,
        ``requires_grad`` is kept the same. Otherwise it is initialized to ``False``.

    :param dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param device: The device of the output tensors. If ``None``, the default
        device is used.
    """

    try:
        return _system_to_torch(systems, positions_requires_grad, cell_requires_grad, dtype, device)
    except TypeError:
        # try iterating over the systems
        return [
            _system_to_torch(system, positions_requires_grad, cell_requires_grad, dtype, device)
            for system in systems
        ]


def _system_to_torch(system, positions_requires_grad, cell_requires_grad, dtype, device):
    if not _is_torch_system(system):
        system = featomic.systems.wrap_system(system)
        
        # Get cell and determine PBC
        cell = np.array(system.cell())
        
        # Determine PBC based on whether cell vectors are non-zero
        cell_norms = np.linalg.norm(cell, axis=1)
        pbc = cell_norms > 1e-9
        
        # Validate PBC consistency with cell vectors (stricter validation)
        if np.all(~pbc):
            # Non-periodic system: cell should be all zeros
            if np.any(cell != 0.0):
                warnings.warn(
                    "A conversion to `System` was requested for a system with non-zero "
                    "cell vectors but where all periodic boundary conditions are "
                    "disabled. The cell vectors will be set to zero.",
                    stacklevel=3,
                )
                cell = np.zeros((3, 3), dtype=cell.dtype)
        elif np.any(~pbc):
            # Mixed PBC: some directions periodic, some not
            # This is not fully supported, warn and zero out non-periodic directions
            warnings.warn(
                "A conversion to `System` was requested with mixed periodic boundary "
                "conditions. Only fully periodic or fully non-periodic systems are "
                "fully supported. Setting non-periodic cell vectors to zero.",
                stacklevel=3,
            )
            cell[~pbc] = 0.0
        
        system = System(
            types=torch.tensor(system.types(), dtype=dtype, device=device),
            positions=torch.tensor(system.positions(), dtype=dtype, device=device),
            cell=torch.tensor(cell, dtype=dtype, device=device),
            pbc=torch.tensor(pbc, device=device),
        )

    if positions_requires_grad is not None:
        system.positions.requires_grad_(positions_requires_grad)

    if cell_requires_grad is not None:
        system.cell.requires_grad_(cell_requires_grad)

    return system


def _is_torch_system(system):
    if not isinstance(system, torch.ScriptObject):
        return False

    return system._type().name() == "System"
