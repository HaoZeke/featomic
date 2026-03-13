from typing import List, Optional, Sequence, overload

import numpy as np
import torch
from metatomic.torch import System

import featomic


@overload
def systems_to_torch(
    systems: "featomic.systems.IntoSystem",
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> System:
    pass


@overload
def systems_to_torch(
    systems: Sequence["featomic.systems.IntoSystem"],
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
    Convert an arbitrary system to :py:class:`metatomic.torch.System`, putting all the
    data in :py:class:`torch.Tensor` and making the overall object compatible with
    TorchScript.

    This function uses :py:func:`featomic.systems.wrap_system` to support a wide range
    of input types (ASE, chemfiles, pyscf, etc.), then delegates to
    :py:func:`metatomic.torch.systems_to_torch` for the actual conversion.

    :param systems: any system supported by featomic. If this is an iterable of system,
        this function converts them all and returns a list of converted systems.

    :param positions_requires_grad: The value of ``requires_grad`` on the output
        ``positions``. If ``None`` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to ``False``.

    :param cell_requires_grad: The value of ``requires_grad`` on the output ``cell``. If
        ``None`` and the cell of the input is already a :py:class:`torch.Tensor`,
        ``requires_grad`` is kept the same. Otherwise it is initialized to ``False``.

    :param dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.

    :param device: The device of the output tensors. If ``None``, the default
        device is used.
    """

    try:
        return _system_to_torch(
            systems, positions_requires_grad, cell_requires_grad, dtype, device
        )
    except TypeError:
        # try iterating over the systems
        return [
            _system_to_torch(
                system, positions_requires_grad, cell_requires_grad, dtype, device
            )
            for system in systems
        ]


def _system_to_torch(
    system, positions_requires_grad, cell_requires_grad, dtype, device
):
    if not _is_torch_system(system):
        # Use featomic's wrap_system to support various input types (chemfiles, pyscf, etc.)
        wrapped = featomic.systems.wrap_system(system)

        # Create System directly from wrapped data
        # This avoids the ASE round-trip while still using metatomic's System class
        # Preserve input dtype if not explicitly specified
        positions_data = wrapped.positions()
        cell_data = wrapped.cell()
        
        # Convert numpy dtypes to torch dtypes if needed
        if dtype is None:
            positions_dtype = torch.from_numpy(positions_data).dtype
            cell_dtype = torch.from_numpy(cell_data).dtype
        else:
            positions_dtype = dtype
            cell_dtype = dtype
        
        system = System(
            types=torch.tensor(wrapped.types(), dtype=torch.int32, device=device),
            positions=torch.tensor(positions_data, dtype=positions_dtype, device=device),
            cell=torch.tensor(cell_data, dtype=cell_dtype, device=device),
            pbc=(
                torch.tensor([False, False, False], device=device)
                if np.all(cell_data == 0.0)
                else torch.tensor([True, True, True], device=device)
            ),
        )

    # Apply requires_grad if requested
    if positions_requires_grad is not None:
        system.positions.requires_grad_(positions_requires_grad)

    if cell_requires_grad is not None:
        system.cell.requires_grad_(cell_requires_grad)

    return system


def _is_torch_system(system):
    if not isinstance(system, torch.ScriptObject):
        return False

    return system._type().name() == "System"
