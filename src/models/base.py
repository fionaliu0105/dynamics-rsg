"""The shared latent interface every model must expose.

Contract (AGENTS.md, "Shared latent interface")::

    outputs, states = model.forward(inputs)

    inputs  : [trials, time, n_in]    the drive (Ready/Set pulses + context channels)
    outputs : [trials, time]          the readout z (ramp -> threshold at Go)
    states  : [trials, time, units]   the comparison activity r = tanh(x)

``states`` is the rate-like activity (bounded, comparable to neural firing rates),
NOT the raw pre-activation x. RSA and iDSA consume ``states``; the store keeps
``states`` and ``inputs`` together because iDSA needs inputs aligned to states.

This module is import-light **on purpose**: it must not ``import torch`` at module
load, so ``src.conditions``, ``src.training.config``, tests, and the store stay
usable in an environment without torch. Concrete models (``bptt_rnn``, ``pc_rnn``)
are the only files that require torch.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # for type checkers / docs only; not executed at runtime
    import torch


class Model(abc.ABC):
    """Abstract base every RNN variant implements.

    Subclasses (BPTT, PC) provide identical ``forward`` dynamics and differ only in
    how they are trained. Keeping the interface here — and the training loop out of
    it — is what lets one learning rule swap for another without touching the
    comparison code.
    """

    #: Number of input channels: [Ready/Set pulse, prior-context, effector-context].
    N_IN: int = 3

    @abc.abstractmethod
    def forward(
        self,
        inputs: "torch.Tensor",
        *,
        noise: bool = True,
        return_states: bool = True,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Roll the network forward over the input drive.

        Args:
            inputs: ``[trials, time, N_IN]`` external drive.
            noise: inject per-unit process noise (disable for deterministic checks,
                e.g. PC-A gradient validation and encoding-axis estimation).
            return_states: if True, also return ``states`` ``[trials, time, units]``.

        Returns:
            ``(outputs, states)`` — ``outputs`` is ``[trials, time]``; ``states`` is
            ``[trials, time, units]`` (the activity ``r = tanh(x)``). Implementations
            must return ``states`` on the SAME time base as ``outputs``.
        """
        raise NotImplementedError


def check_interface(outputs, states, n_trials: int, n_time: int) -> None:
    """Assert a model's return shapes obey the contract. Use in model unit tests.

    Kept here (not in a test file) so every model track shares one definition of
    "conforms to the interface". Works with any array-like exposing ``.shape``.
    """
    assert tuple(outputs.shape) == (n_trials, n_time), (
        f"outputs must be [trials, time] = {(n_trials, n_time)}, got {tuple(outputs.shape)}"
    )
    assert len(states.shape) == 3 and tuple(states.shape[:2]) == (n_trials, n_time), (
        f"states must be [trials, time, units] with leading {(n_trials, n_time)}, "
        f"got {tuple(states.shape)}"
    )
