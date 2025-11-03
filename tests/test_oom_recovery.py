"""Tests for arraybridge.oom_recovery module."""

import pytest


class TestOOMRecovery:
    """Tests for OOM recovery functions."""

    def test_is_oom_error_none_memory_type(self):
        """Test _is_oom_error with None/unknown memory type."""
        from arraybridge.oom_recovery import _is_oom_error

        e = Exception("some error")
        assert _is_oom_error(e, "unknown_type") is False

    def test_is_oom_error_generic_exception(self):
        """Test _is_oom_error with generic exception."""
        from arraybridge.oom_recovery import _is_oom_error

        e = Exception("some random error")
        assert _is_oom_error(e, "torch") is False

    def test_is_oom_error_memory_error(self):
        """Test _is_oom_error with MemoryError."""
        from arraybridge.oom_recovery import _is_oom_error

        e = MemoryError("out of memory")
        # Should detect based on string patterns
        assert _is_oom_error(e, "torch") is True

    def test_is_oom_error_string_patterns(self):
        """Test _is_oom_error with various OOM string patterns."""
        from arraybridge.oom_recovery import _is_oom_error

        # Test torch patterns
        torch_oom_messages = [
            "out of memory",
            "cuda_error_out_of_memory",
        ]
        for msg in torch_oom_messages:
            e = Exception(msg)
            assert _is_oom_error(e, "torch") is True, f"Failed to detect OOM in torch: {msg}"

        # Test numpy patterns
        numpy_oom_messages = ["memory exhausted", "cannot allocate memory"]
        for msg in numpy_oom_messages:
            e = Exception(msg)
            assert _is_oom_error(e, "numpy") is True, f"Failed to detect OOM in numpy: {msg}"

    def test_clear_cache_for_memory_type_unknown(self):
        """Test _clear_cache_for_memory_type with unknown memory type."""
        from arraybridge.oom_recovery import _clear_cache_for_memory_type

        # Should not raise, just log warning and do gc.collect()
        _clear_cache_for_memory_type("unknown_type")

    def test_clear_cache_for_memory_type_numpy(self):
        """Test _clear_cache_for_memory_type with numpy (CPU)."""
        from arraybridge.oom_recovery import _clear_cache_for_memory_type

        # Should just do gc.collect()
        _clear_cache_for_memory_type("numpy")

    def test_execute_with_oom_recovery_no_oom(self):
        """Test _execute_with_oom_recovery when no OOM occurs."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        def successful_func():
            return "success"

        result = _execute_with_oom_recovery(successful_func, "torch", max_retries=2)
        assert result == "success"

    def test_execute_with_oom_recovery_oom_retry_success(self):
        """Test _execute_with_oom_recovery with OOM that succeeds on retry."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        call_count = {"count": 0}

        def failing_then_success_func():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise MemoryError("out of memory")
            return "success"

        result = _execute_with_oom_recovery(failing_then_success_func, "torch", max_retries=2)
        assert result == "success"
        assert call_count["count"] == 2

    def test_execute_with_oom_recovery_oom_exhausted_retries(self):
        """Test _execute_with_oom_recovery with OOM that exhausts retries."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        def always_fails_func():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError) as exc_info:
            _execute_with_oom_recovery(always_fails_func, "torch", max_retries=2)
        assert "out of memory" in str(exc_info.value)

    def test_execute_with_oom_recovery_non_oom_exception(self):
        """Test _execute_with_oom_recovery with non-OOM exception (should not retry)."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        def raises_value_error():
            raise ValueError("not an OOM error")

        with pytest.raises(ValueError) as exc_info:
            _execute_with_oom_recovery(raises_value_error, "torch", max_retries=2)
        assert "not an OOM error" in str(exc_info.value)

    def test_execute_with_oom_recovery_max_retries_zero(self):
        """Test _execute_with_oom_recovery with max_retries=0."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        def always_fails_func():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            _execute_with_oom_recovery(always_fails_func, "torch", max_retries=0)

    @pytest.mark.parametrize("memory_type", ["torch", "cupy", "tensorflow"])
    def test_execute_with_oom_recovery_different_frameworks(self, memory_type):
        """Test _execute_with_oom_recovery with different GPU frameworks."""
        from arraybridge.oom_recovery import _execute_with_oom_recovery

        call_count = {"count": 0}

        def failing_then_success_func():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise MemoryError("out of memory")
            return f"success_{memory_type}"

        result = _execute_with_oom_recovery(failing_then_success_func, memory_type, max_retries=1)
        assert result == f"success_{memory_type}"
        assert call_count["count"] == 2


def test_torch_cache_clear_mock():
    """Test torch cache clear with mocked torch."""
