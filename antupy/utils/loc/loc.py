from typing import Protocol, runtime_checkable


@runtime_checkable
class Location(Protocol):
    """Protocol for location classes."""
    value: str | int | tuple[float, float]

    def __str__(self) -> str:
        ...