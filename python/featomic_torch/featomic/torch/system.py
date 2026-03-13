from typing import TYPE_CHECKING, List, Optional, Sequence, overload

import torch

from metatomic.torch import System, systems_to_torch as _metatomic_systems_to_torch

if TYPE_CHECKING:
    from featomic.systems import IntoSystem



@overload
def systems_to_torch(
    systems: "IntoSystem",
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> System:
    pass


@overload
def systems_to_torch(
    systems: Sequence["IntoSystem"],
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

    This is a wrapper around :py:func:`metatomic.torch.systems_to_torch` that provides
    type hints specific to featomic's :py:class:`featomic.systems.IntoSystem`.

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


def _system_to_torch(system, positions_requires_grad, cell_requires_grad, dtype, device):
    if _is_torch_system(system):
        # Already a torch System, just apply requires_grad if requested
        result = system
        if positions_requires_grad is not None:
            result.positions.requires_grad_(positions_requires_grad)
        if cell_requires_grad is not None:
            result.cell.requires_grad_(cell_requires_grad)
        return result
    else:
        # Delegate to metatomic for all other types
        return _metatomic_systems_to_torch(
            system,
            dtype=dtype,
            device=device,
            positions_requires_grad=positions_requires_grad,
            cell_requires_grad=cell_requires_grad,
        )


def _is_torch_system(system):
    if not isinstance(system, torch.ScriptObject):
        return False

    return system._type().name() == "System"
