import numpy.typing as npt
from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias

_VMLAccuracyMode: TypeAlias = Literal[0, 1, 2, 3]

MAX_THREADS: Final[int] = ...
__BLOCK_SIZE1__: Final[int] = ...

#ifdef USE_VML
def _get_vml_version() -> str: ...
def _set_vml_accuracy_mode(mode_in: _VMLAccuracyMode, /) -> _VMLAccuracyMode: ...
def _set_vml_num_threads(max_num_threads: int, /) -> None: ...
def _get_vml_num_threads() -> int: ...
#endif
def _get_num_threads() -> int: ...
def _set_num_threads(num_threads: int, /) -> int: ...

allaxes: Final = 255
funccodes: Final[dict[bytes, int]] = ...
maxdims: Final[int] = ...
opcodes: Final[dict[bytes, int]] = ...
use_vml: Final[bool] = ...

class NumExpr:
    signature: Final[bytes]
    constsig: Final[bytes]
    tempsig: Final[bytes]
    fullsig: Final[bytes]

    program: Final[bytes]
    constants: Final[Sequence[Any]]
    input_names: Final[Sequence[str]]

    def __init__(
        self,
        signature: bytes,
        tempsig: bytes,
        program: bytes,
        constants: Sequence[Any] = ...,
        input_names: Sequence[str] | None = None,
    ) -> None: ...
    def run(
        self,
        *args: Any,
        casting: str = ...,
        order: str = ...,
        ex_uses_vml: bool = ...,
        out: npt.NDArray[Any] = ...,
    ) -> Any: ...
    __call__ = run
