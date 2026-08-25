"""Resolved I/O policy for NC/HDF5 binary output.

This module converts the user-facing ``program.use_h5`` selection and
runtime library availability into an immutable :class:`IOPolicy`.

Consumers should depend on :class:`IOPolicy`, not on raw config values,
module globals, or ``Optional[bool]`` tri-states.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum

from omegaconf import DictConfig, OmegaConf
from utils import log_init

from tcm import _constants, schema

lf = log_init.LoggingStyleAdapter(__name__)


class H5Mode(Enum):
    """Resolved runtime outcome after checking library availability."""

    ON = "binary I/O enabled"
    OFF_WARN = "binary I/O disabled; skipped operations are logged"
    OFF_SILENT = "binary I/O unavailable; skips are silent"

    def __bool__(self) -> bool:
        """Return ``True`` iff NC/HDF5 I/O may proceed."""
        return self is H5Mode.ON


class H5DisabledError(RuntimeError):
    """NC/HDF5 I/O is disabled by configuration."""


@dataclass(frozen=True, slots=True)
class IOPolicy:
    """Immutable resolved I/O policy.

    Stores both the user request and runtime availability so that
    diagnostics can distinguish configuration choice from missing libs.
    """

    request: schema.UseH5
    available: bool

    @property
    def h5(self) -> bool:
        """Return ``True`` iff NC/HDF5 I/O may proceed."""
        return self.available and self.request is not schema.UseH5.OFF

    def __bool__(self) -> bool:
        return self.h5

    @property
    def missing_dependency(self) -> bool:
        """Return ``True`` iff HDF5 libraries are unavailable."""
        return not self.available

    @property
    def reason(self) -> str:
        """Human-readable cause of the resolved state."""
        if self.request is schema.UseH5.OFF:
            return "program.use_h5=off"
        if not self.available:
            return "h5py/netCDF4 unavailable"
        return "binary I/O enabled"

    @property
    def warn_at_resolve(self) -> bool:
        """Return ``True`` iff resolution should emit a warning."""
        return self.request is schema.UseH5.PREFER and not self.available

    @property
    def skip_log_level(self) -> int | None:
        """Log level for skipped NC operations, or ``None`` for silent."""
        if self.h5:
            return None

        match self.request:
            case schema.UseH5.OFF:
                return logging.INFO      # or WARNING if contract requires
            case schema.UseH5.PREFER:
                return logging.DEBUG     # already warned at resolve
            case _:
                return None

    @classmethod
    def resolve(cls, cfg: DictConfig, /) -> IOPolicy:
        """Resolve ``program.use_h5`` against runtime availability."""
        raw = OmegaConf.select(cfg, "program.use_h5", default=schema.UseH5.AUTO)
        request = schema.UseH5(raw)
        available = _constants.H5_AVAILABLE

        if request is schema.UseH5.REQUIRE and not available:
            raise ImportError(
                "program.use_h5=require but h5py/netCDF4 are unavailable; "
                "install HDF5 dependencies or set program.use_h5=auto/off"
            )

        policy = cls(request, available)

        if policy.warn_at_resolve:
            lf.warning(
                "program.use_h5=prefer but h5py/netCDF4 are unavailable — "
                "falling back to TSV-only"
            )
        return policy

    def allow_nc(self, operation: str, /) -> bool:
        """Return ``True`` iff NC/HDF5 I/O may proceed; log skips by policy."""
        if self.h5:
            return True
        if (level := self.skip_log_level) is not None:
            lf.log(level, "{:s} — NC/HDF5 skipped: {:s}", operation, self.reason)
        return False

    def require_nc(self, operation: str, /) -> None:
        """Raise if NC/HDF5 I/O is required but unavailable or disabled."""
        if self.h5:
            return
        if self.missing_dependency:
            raise ImportError(
                f"{operation}: h5py/netCDF4 are unavailable; "
                "install HDF5 dependencies or use TSV output"
            )
        raise H5DisabledError(f"{operation}: NC/HDF5 output is disabled ({self.reason})")


_DEFAULT_POLICY = IOPolicy(request=schema.UseH5.AUTO, available=_constants.H5_AVAILABLE)

_io: ContextVar[IOPolicy] = ContextVar("io", default=_DEFAULT_POLICY)


def init_io(cfg: DictConfig, /) -> None:
    _io.set(IOPolicy.resolve(cfg))


def io() -> IOPolicy:
    return _io.get()
