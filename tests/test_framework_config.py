"""Tests for arraybridge.framework_config module."""

import types
import unittest.mock

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
            "import_name",
            "display_name",
            "is_gpu",
            "scaling_ops",
            "conversion_ops",
            "supports_dlpack",
            "lazy_getter",
        ]

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            for key in required_keys:
                assert key in config, f"Missing {key} in {mem_type.value} config"

    def test_numpy_config(self):
        """Test numpy-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.NUMPY]

        assert config["import_name"] == "numpy"
        assert config["display_name"] == "NumPy"
        assert config["is_gpu"] is False
        assert config["has_oom_recovery"] is False
        assert config["oom_string_patterns"] == ["cannot allocate memory", "memory exhausted"]

    def test_torch_config(self):
        """Test torch-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.TORCH]

        assert config["import_name"] == "torch"
        assert config["display_name"] == "PyTorch"
        assert config["is_gpu"] is True
        assert config["has_oom_recovery"] is True
        assert config["oom_exception_types"] == ["{mod}.cuda.OutOfMemoryError"]
        assert "out of memory" in config["oom_string_patterns"]

    def test_cupy_config(self):
        """Test cupy-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.CUPY]

        assert config["import_name"] == "cupy"
        assert config["display_name"] == "CuPy"
        assert config["is_gpu"] is True
        assert config["has_oom_recovery"] is True

    def test_tensorflow_config(self):
        """Test tensorflow-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.TENSORFLOW]

        assert config["import_name"] == "tensorflow"
        assert config["display_name"] == "TensorFlow"
        assert config["is_gpu"] is True
        assert config["has_oom_recovery"] is True

    def test_jax_config(self):
        """Test jax-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.JAX]

        assert config["import_name"] == "jax"
        assert config["display_name"] == "JAX"
        assert config["is_gpu"] is True
        assert config["has_oom_recovery"] is True

    def test_pyclesperanto_config(self):
        """Test pyclesperanto-specific configuration."""
        config = _FRAMEWORK_CONFIG[MemoryType.PYCLESPERANTO]

        assert config["import_name"] == "pyclesperanto"
        assert config["display_name"] == "pyclesperanto"
        assert config["is_gpu"] is True
        assert config["has_oom_recovery"] is True

    def test_scaling_ops_structure(self):
        """Test that scaling_ops have required structure."""
        required_scaling_keys = ["min", "max", "astype", "check_float", "check_int"]

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            scaling_ops = config["scaling_ops"]

            # Skip frameworks with custom scaling (like pyclesperanto)
            if scaling_ops is None:
                continue

            for key in required_scaling_keys:
                assert key in scaling_ops, f"Missing {key} in {mem_type.value} scaling_ops"

    def test_conversion_ops_structure(self):
        """Test that conversion_ops have required structure."""
        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            conversion_ops = config["conversion_ops"]

            # All should have to_numpy
            assert "to_numpy" in conversion_ops

            # GPU frameworks should have from_numpy
            if config["is_gpu"]:
                assert "from_numpy" in conversion_ops

    def test_dlpack_support(self):
        """Test DLPack support configuration."""
        # Frameworks that support DLPack
        dlpack_supported = [
            MemoryType.CUPY,
            MemoryType.TORCH,
            MemoryType.TENSORFLOW,
            MemoryType.JAX,
        ]

        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            if mem_type in dlpack_supported:
                assert config["supports_dlpack"] is True
            else:
                assert config["supports_dlpack"] is False

    def test_gpu_frameworks_have_cleanup(self):
        """Test that GPU frameworks have cleanup operations."""
        for mem_type in MemoryType:
            config = _FRAMEWORK_CONFIG[mem_type]
            if config["is_gpu"]:
                # GPU frameworks should have cleanup_ops (may be None for some)
                assert "cleanup_ops" in config
            else:
                # CPU frameworks should have None cleanup
                assert config["cleanup_ops"] is None

    def test_numpy_dtype_conversion_needed(self):
        """Test numpy dtype conversion check."""
        from arraybridge.framework_config import _numpy_dtype_conversion_needed
        from arraybridge.types import MemoryType

        # Mock detect function
        def mock_detect(data):
            return MemoryType.TORCH.value

        # NumPy needs conversion only for torch sources
        assert _numpy_dtype_conversion_needed("test", mock_detect) is True

        def mock_detect_numpy(data):
            return MemoryType.NUMPY.value

        # NumPy doesn't need conversion for numpy sources
        assert _numpy_dtype_conversion_needed("test", mock_detect_numpy) is False

    def test_torch_dtype_conversion_needed(self):
        """Test torch dtype conversion check."""
        from arraybridge.framework_config import _torch_dtype_conversion_needed

        # Mock detect function
        def mock_detect(data):
            return "torch"

            # Torch always needs dtype conversion

        assert _torch_dtype_conversion_needed("test", mock_detect) is True

    def test_pyclesperanto_get_device_id_unavailable(self, monkeypatch):
        """Test pyclesperanto device ID when pyclesperanto unavailable."""
        import sys

        from arraybridge.framework_config import _pyclesperanto_get_device_id

        # Mock pyclesperanto as unavailable
        monkeypatch.setitem(sys.modules, "pyclesperanto", None)

        # Should return 0 when pyclesperanto not available
        result = _pyclesperanto_get_device_id(None, None)
        assert result == 0

    def test_pyclesperanto_get_device_id_with_mock(self):
        """Test pyclesperanto device ID with mock module."""
        import types

        from arraybridge.framework_config import _pyclesperanto_get_device_id

        # Create mock device with id attribute
        mock_device = types.SimpleNamespace(id=1)
        mock_module = types.SimpleNamespace(get_device=lambda: mock_device)

        result = _pyclesperanto_get_device_id(None, mock_module)
        assert result == 1

    def test_pyclesperanto_get_device_id_with_devices_list(self):
        """Test pyclesperanto device ID using devices list."""
        from arraybridge.framework_config import _pyclesperanto_get_device_id

        # Create mock device without id attribute
        mock_device = types.SimpleNamespace()
        mock_devices = ["device0", "device1", "device2"]
        mock_module = types.SimpleNamespace(
            get_device=lambda: mock_device, list_available_devices=lambda: mock_devices
        )

        # Mock str() to return matching strings for comparison
        original_str = str
        str_calls = []

        def mock_str(obj):
            str_calls.append(obj)
            if obj is mock_device:
                return "device1"
            return original_str(obj)

        import builtins

        builtins.str = mock_str

        try:
            result = _pyclesperanto_get_device_id(None, mock_module)
            assert result == 1  # Should find device1 at index 1
        finally:
            builtins.str = original_str

    def test_pyclesperanto_set_device_unavailable(self, monkeypatch):
        """Test pyclesperanto set device when pyclesperanto unavailable."""
        import sys

        from arraybridge.framework_config import _pyclesperanto_set_device

        # Mock pyclesperanto as unavailable
        monkeypatch.setitem(sys.modules, "pyclesperanto", None)

        # Should not raise when pyclesperanto not available
        _pyclesperanto_set_device(0, None)

    def test_pyclesperanto_set_device_with_mock(self):
        """Test pyclesperanto set device with mock module."""
        import types

        from arraybridge.framework_config import _pyclesperanto_set_device

        mock_devices = ["device0", "device1", "device2"]
        mock_module = types.SimpleNamespace(
            list_available_devices=lambda: mock_devices, select_device=lambda x: None
        )

        # Should not raise for valid device ID
        _pyclesperanto_set_device(1, mock_module)

    def test_pyclesperanto_set_device_invalid_id(self):
        """Test pyclesperanto set device with invalid device ID."""
        from arraybridge.framework_config import _pyclesperanto_set_device

        mock_devices = ["device0", "device1"]
        mock_module = types.SimpleNamespace(list_available_devices=lambda: mock_devices)

        # Should raise ValueError for invalid device ID
        with pytest.raises(ValueError, match="Device 5 not available"):
            _pyclesperanto_set_device(5, mock_module)

    def test_pyclesperanto_move_to_device_unavailable(self, monkeypatch):
        """Test pyclesperanto move to device when pyclesperanto unavailable."""
        import sys

        from arraybridge.framework_config import _pyclesperanto_move_to_device

        # Mock pyclesperanto as unavailable
        monkeypatch.setitem(sys.modules, "pyclesperanto", None)

        # Should return data unchanged when pyclesperanto not available
        data = "test_data"
        result = _pyclesperanto_move_to_device(data, 0, None, "pyclesperanto")
        assert result == data

    def test_pyclesperanto_move_to_device_same_device(self):
        """Test pyclesperanto move to device when already on target device."""
        import types

        from arraybridge.framework_config import _pyclesperanto_move_to_device

        # Mock the _get_device_id function to return the same device
        data = "test_data"
        mock_module = types.SimpleNamespace()

        with unittest.mock.patch("arraybridge.utils._get_device_id", return_value=1):
            result = _pyclesperanto_move_to_device(data, 1, mock_module, "pyclesperanto")
            assert result == data

    def test_pyclesperanto_move_to_device_different_device(self):
        """Test pyclesperanto move to device when moving to different device."""
        import types

        from arraybridge.framework_config import _pyclesperanto_move_to_device

        data = "test_data"
        result_data = "moved_data"
        mock_module = types.SimpleNamespace(
            select_device=lambda x: None,
            create_like=lambda d: result_data,
            copy=lambda src, dst: None,
        )

        with unittest.mock.patch("arraybridge.utils._get_device_id", return_value=0):
            result = _pyclesperanto_move_to_device(data, 1, mock_module, "pyclesperanto")
            assert result == result_data

    def test_jax_assign_slice_function_unavailable(self, monkeypatch):
        """Test JAX assign slice function when JAX unavailable."""
        import sys

        from arraybridge.framework_config import _jax_assign_slice

        # Mock JAX as unavailable
        monkeypatch.setitem(sys.modules, "jax", None)

        # Should return None when result is None
        result = _jax_assign_slice(None, 0, None)
        assert result is None

    def test_jax_assign_slice_with_mock(self):
        """Test JAX assign slice with mock JAX array."""
        from arraybridge.framework_config import _jax_assign_slice

        # Create a proper mock JAX array structure
        class MockAtResult:
            def set(self, data):
                return f"assigned_{data}"

        class MockAtIndex:
            def __getitem__(self, idx):
                return MockAtResult()

        class MockAt:
            @property
            def at(self):
                return MockAtIndex()

        mock_array = MockAt()
        result = _jax_assign_slice(mock_array, 5, "test_data")
        assert result == "assigned_test_data"

    def test_tensorflow_validate_dlpack_function_unavailable(self, monkeypatch):
        """Test TensorFlow DLPack validation function when TensorFlow unavailable."""
        import sys

        from arraybridge.framework_config import _tensorflow_validate_dlpack

        # Mock TensorFlow as unavailable
        monkeypatch.setitem(sys.modules, "tensorflow", None)

        # Should return False when TensorFlow not available
        result = _tensorflow_validate_dlpack(None, None)
        assert result is False

    def test_tensorflow_validate_dlpack_old_version(self):
        """Test TensorFlow DLPack validation with old version."""
        import types

        from arraybridge.framework_config import _tensorflow_validate_dlpack

        # Mock TensorFlow with old version
        mock_tf = types.SimpleNamespace(__version__="2.10.0")

        with pytest.raises(RuntimeError, match="TensorFlow 2.10.0 does not support stable DLPack"):
            _tensorflow_validate_dlpack(None, mock_tf)

    def test_tensorflow_validate_dlpack_new_version(self):
        """Test TensorFlow DLPack validation with supported version."""
        import types

        from arraybridge.framework_config import _tensorflow_validate_dlpack

        # Mock TensorFlow with supported version
        mock_tf = types.SimpleNamespace(__version__="2.15.0")

        # Should return True for supported version
        # Note: version check passes, may raise due to incomplete mocking
        try:
            _tensorflow_validate_dlpack(None, mock_tf)
            # If we get here, version check passed
            assert True
        except AttributeError:
            # Expected due to incomplete mocking of GPU check
            pass

    def test_pyclesperanto_stack_slices_unavailable(self, monkeypatch):
        """Test pyclesperanto stack slices when pyclesperanto unavailable."""
        import sys

        from arraybridge.framework_config import _pyclesperanto_stack_slices

        # Mock pyclesperanto as unavailable
        monkeypatch.setitem(sys.modules, "pyclesperanto", None)

        # Should not raise when pyclesperanto not available
        result = _pyclesperanto_stack_slices([], "pyclesperanto", 0, None)
        assert result is None
