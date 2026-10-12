"""What arraybridge calls of ``jax.dlpack``."""

from jax import Array

def from_dlpack(external_array: object) -> Array: ...
