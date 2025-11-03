"""Tests for arraybridge.framework_config module."""

import pytest

from arraybridge.framework_config import _FRAMEWORK_CONFIG
from arraybridge.types import MemoryType


class TestFrameworkConfig:
    """Tests for framework configuration."""

    def test_all_memory_types_have_config(self):
        """Test that all memory types have configuration."""
        for mem_type in MemoryType:
            assert mem_type in _FRAMEWORK_CONFIG
            config = _FRAMEWORK_CONFIG[mem_type]
            assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Test that all configs have required keys."""
        required_keys = [
            'import_name', 'display_name', 'is_gpu', 'scaling_ops',
            'conversion_ops', 'supports_dlpack', 'lazy_getter'
        ]

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            for key in required_keys:
                assert key in config, f"Missing {key} in {mem_type.value} config"

    def test_numpy_config(self):
        """Test numpy-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.NUMPY]

        assert config['import_name'] == 'numpy'
        assert config['display_name'] == 'NumPy'
        assert config['is_gpu'] is False
        assert config['has_oom_recovery'] is False
        assert config['oom_string_patterns'] == ['cannot allocate memory', 'memory exhausted']

    def test_torch_config(self):
        """Test torch-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.TORCH]

        assert config['import_name'] == 'torch'
        assert config['display_name'] == 'PyTorch'
        assert config['is_gpu'] is True
        assert config['has_oom_recovery'] is True
        assert config['oom_exception_types'] == ['{mod}.cuda.OutOfMemoryError']
        assert 'out of memory' in config['oom_string_patterns']

    def test_cupy_config(self):
        """Test cupy-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.CUPY]

        assert config['import_name'] == 'cupy'
        assert config['display_name'] == 'CuPy'
        assert config['is_gpu'] is True
        assert config['has_oom_recovery'] is True

    def test_tensorflow_config(self):
        """Test tensorflow-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.TENSORFLOW]

        assert config['import_name'] == 'tensorflow'
        assert config['display_name'] == 'TensorFlow'
        assert config['is_gpu'] is True
        assert config['has_oom_recovery'] is True

    def test_jax_config(self):
        """Test jax-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.JAX]

        assert config['import_name'] == 'jax'
        assert config['display_name'] == 'JAX'
        assert config['is_gpu'] is True
        assert config['has_oom_recovery'] is True

    def test_pyclesperanto_config(self):
        """Test pyclesperanto-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.PYCLESPERANTO]

        assert config['import_name'] == 'pyclesperanto'
        assert config['display_name'] == 'pyclesperanto'
        assert config['is_gpu'] is True
        assert config['has_oom_recovery'] is True

    def test_scaling_ops_structure(self):
        """Test that scaling_ops have required structure."""
        required_scaling_keys = ['min', 'max', 'astype', 'check_float', 'check_int']

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            scaling_ops = config['scaling_ops']

            # Skip frameworks with custom scaling (like pyclesperanto)
            if scaling_ops is None:
                continue

            for key in required_scaling_keys:
                assert key in scaling_ops, f"Missing {key} in {mem_type.value} scaling_ops"

    def test_conversion_ops_structure(self):
        """Test that conversion_ops have required structure."""
        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            conversion_ops = config['conversion_ops']

            # All should have to_numpy
            assert 'to_numpy' in conversion_ops

            # GPU frameworks should have from_numpy
            if config['is_gpu']:
                assert 'from_numpy' in conversion_ops

    def test_dlpack_support(self):
        """Test DLPack support configuration."""
        # Frameworks that support DLPack
        dlpack_supported = [MemoryType.CUPY, MemoryType.TORCH, MemoryType.TENSORFLOW, MemoryType.JAX]

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            if mem_type in dlpack_supported:
                assert config['supports_dlpack'] is True
            else:
                assert config['supports_dlpack'] is False

    def test_gpu_frameworks_have_cleanup(self):
        """Test that GPU frameworks have cleanup operations."""
        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            if config['is_gpu']:
                # GPU frameworks should have cleanup_ops (may be None for some)
                assert 'cleanup_ops' in config
            else:
                # CPU frameworks should have None cleanup
                assert config['cleanup_ops'] is None

    def test_oom_recovery_config(self):
        """Test OOM recovery configuration."""
        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]

            # All configs should have OOM-related keys
            assert 'has_oom_recovery' in config
            assert 'oom_exception_types' in config
            assert 'oom_string_patterns' in config
            assert 'oom_clear_cache' in config

            # CPU frameworks shouldn't have OOM recovery
            if not config['is_gpu']:
                assert config['has_oom_recovery'] is False
                assert config['oom_exception_types'] == []
            else:
                assert config['has_oom_recovery'] is True
                assert isinstance(config['oom_exception_types'], list)
                assert isinstance(config['oom_string_patterns'], list)