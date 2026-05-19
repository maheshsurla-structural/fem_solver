"""Abstract base class for uniaxial (1-D) materials.

A uniaxial material is the constitutive object held by a single fiber.
Its sole job is to map a strain to (stress, tangent) and to support a
trial/commit lifecycle so that it can participate in a Newton-Raphson
loop without losing its history when an iteration fails.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod


class UniaxialMaterial(ABC):
    """1-D constitutive law: strain -> (stress, tangent modulus).

    Subclasses implement :meth:`get_response`. Stateless models (e.g.
    :class:`UniaxialElastic`) leave the lifecycle hooks as no-ops; models
    with internal state (plasticity, damage) override them to roll
    history forward / back.
    """

    @abstractmethod
    def get_response(self, eps: float) -> tuple[float, float]:
        """Return ``(sigma, Et)`` at the trial strain ``eps``.

        Side effect: any internal *trial* state is updated. The committed
        state is unchanged until :meth:`commit_state` is called.
        """

    # ------------------------------------------------------------- state
    def commit_state(self) -> None:
        """Persist the current trial state. Default: no-op (stateless)."""
        return None

    def revert_state(self) -> None:
        """Discard the current trial state. Default: no-op (stateless)."""
        return None

    # ------------------------------------------------------------- clone
    def clone(self) -> "UniaxialMaterial":
        """Return a deep copy with independent state. Each fiber should
        carry its own copy so plasticity in one fiber does not pollute
        another."""
        return copy.deepcopy(self)
