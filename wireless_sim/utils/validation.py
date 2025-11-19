"""
Parameter Validation Utilities for Orbiter-2

Provides reusable validation functions for simulation parameters with
detailed error messages and type checking.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
from typing import Any, Tuple, List, Dict, Optional


def validate_modulation_order(order: int) -> Tuple[bool, str]:
    """Validate that modulation order is a power of 2.

    Args:
        order: Modulation order to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.

    Examples:
        >>> validate_modulation_order(16)
        (True, '')
        >>> validate_modulation_order(15)
        (False, 'Modulation order must be a power of 2, got 15')
    """
    if not isinstance(order, int):
        return False, f"Modulation order must be an integer, got {type(order).__name__}"

    if order <= 0:
        return False, f"Modulation order must be positive, got {order}"

    # Check if power of 2
    if (order & (order - 1)) != 0:
        return False, f"Modulation order must be a power of 2, got {order}"

    # Reasonable range check
    if order < 2 or order > 1024:
        return False, f"Modulation order must be between 2 and 1024, got {order}"

    return True, ""


def validate_snr(snr_db: float, min_snr: float = -20.0, max_snr: float = 50.0) -> Tuple[bool, str]:
    """Validate Signal-to-Noise Ratio value.

    Args:
        snr_db: SNR in decibels
        min_snr: Minimum allowed SNR (default: -20 dB)
        max_snr: Maximum allowed SNR (default: 50 dB)

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_snr(10.0)
        (True, '')
        >>> validate_snr(100.0)
        (False, 'SNR must be between -20.0 and 50.0 dB, got 100.0')
    """
    if not isinstance(snr_db, (int, float)):
        return False, f"SNR must be numeric, got {type(snr_db).__name__}"

    if snr_db < min_snr or snr_db > max_snr:
        return False, f"SNR must be between {min_snr} and {max_snr} dB, got {snr_db}"

    return True, ""


def validate_positive_integer(value: Any, name: str, max_value: Optional[int] = None) -> Tuple[bool, str]:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        max_value: Optional maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_positive_integer(100, "num_symbols")
        (True, '')
        >>> validate_positive_integer(-5, "num_symbols")
        (False, 'num_symbols must be positive, got -5')
    """
    if not isinstance(value, int):
        return False, f"{name} must be an integer, got {type(value).__name__}"

    if value <= 0:
        return False, f"{name} must be positive, got {value}"

    if max_value is not None and value > max_value:
        return False, f"{name} must be <= {max_value}, got {value}"

    return True, ""


def validate_positive_float(value: Any, name: str, min_value: float = 0.0, max_value: Optional[float] = None) -> Tuple[bool, str]:
    """Validate that a value is a positive float.

    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (default: 0.0)
        max_value: Optional maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_positive_float(1.5, "bandwidth", min_value=1.0)
        (True, '')
        >>> validate_positive_float(0.5, "bandwidth", min_value=1.0)
        (False, 'bandwidth must be >= 1.0, got 0.5')
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be numeric, got {type(value).__name__}"

    if value < min_value:
        return False, f"{name} must be >= {min_value}, got {value}"

    if max_value is not None and value > max_value:
        return False, f"{name} must be <= {max_value}, got {value}"

    return True, ""


def validate_range(value: Any, name: str, min_val: float, max_val: float) -> Tuple[bool, str]:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate
        name: Name of the parameter
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_range(5, "param", 0, 10)
        (True, '')
        >>> validate_range(15, "param", 0, 10)
        (False, 'param must be between 0 and 10, got 15')
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be numeric, got {type(value).__name__}"

    if value < min_val or value > max_val:
        return False, f"{name} must be between {min_val} and {max_val}, got {value}"

    return True, ""


def validate_parameters(params: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Validate a dictionary of parameters.

    Performs comprehensive validation of parameter dictionary, checking types,
    ranges, and required fields based on an optional schema.

    Args:
        params: Dictionary of parameters to validate
        schema: Optional parameter schema defining validation rules

    Returns:
        Tuple of (all_valid, list_of_error_messages)

    Examples:
        >>> schema = {
        ...     'snr_db': {'type': 'float', 'min': -10, 'max': 40, 'required': True},
        ...     'mod_order': {'type': 'int', 'min': 2, 'max': 256, 'required': True}
        ... }
        >>> validate_parameters({'snr_db': 10.0, 'mod_order': 16}, schema)
        (True, [])
    """
    errors = []

    if not isinstance(params, dict):
        return False, ["Parameters must be a dictionary"]

    if schema is None:
        # Basic validation without schema
        for key, value in params.items():
            if value is None:
                errors.append(f"Parameter '{key}' cannot be None")
        return len(errors) == 0, errors

    # Validate against schema
    for param_name, param_spec in schema.items():
        required = param_spec.get('required', False)

        # Check if required parameter is present
        if required and param_name not in params:
            errors.append(f"Missing required parameter: {param_name}")
            continue

        # Skip optional parameters that aren't provided
        if param_name not in params:
            continue

        value = params[param_name]
        param_type = param_spec.get('type', 'any')

        # Type validation
        if param_type == 'int' and not isinstance(value, int):
            errors.append(f"Parameter '{param_name}' must be int, got {type(value).__name__}")
            continue
        elif param_type == 'float' and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{param_name}' must be float, got {type(value).__name__}")
            continue
        elif param_type == 'str' and not isinstance(value, str):
            errors.append(f"Parameter '{param_name}' must be str, got {type(value).__name__}")
            continue

        # Range validation
        if param_type in ('int', 'float'):
            if 'min' in param_spec and value < param_spec['min']:
                errors.append(f"Parameter '{param_name}' must be >= {param_spec['min']}, got {value}")
            if 'max' in param_spec and value > param_spec['max']:
                errors.append(f"Parameter '{param_name}' must be <= {param_spec['max']}, got {value}")

        # Choices validation
        if 'choices' in param_spec and value not in param_spec['choices']:
            errors.append(f"Parameter '{param_name}' must be one of {param_spec['choices']}, got {value}")

    return len(errors) == 0, errors


def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...], name: str = "array") -> Tuple[bool, str]:
    """Validate numpy array shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape tuple (use None for variable dimensions)
        name: Name of the array (for error messages)

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> arr = np.zeros((10, 3))
        >>> validate_array_shape(arr, (10, 3), "data")
        (True, '')
        >>> validate_array_shape(arr, (5, 3), "data")
        (False, 'data shape (10, 3) does not match expected (5, 3)')
    """
    if not isinstance(array, np.ndarray):
        return False, f"{name} must be a numpy array, got {type(array).__name__}"

    if len(array.shape) != len(expected_shape):
        return False, f"{name} has {len(array.shape)} dimensions, expected {len(expected_shape)}"

    for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
        if expected is not None and actual != expected:
            return False, f"{name} shape {array.shape} does not match expected {expected_shape}"

    return True, ""


if __name__ == "__main__":
    print("=== Orbiter-2 Validation Utilities Demo ===\n")

    # Test modulation order validation
    print("1. Modulation Order Validation:")
    for order in [4, 16, 15, 256, 1024]:
        valid, msg = validate_modulation_order(order)
        status = "✓" if valid else "✗"
        print(f"   {status} Order {order}: {msg if msg else 'Valid'}")

    # Test SNR validation
    print("\n2. SNR Validation:")
    for snr in [-10.0, 0.0, 10.0, 40.0, 100.0]:
        valid, msg = validate_snr(snr)
        status = "✓" if valid else "✗"
        print(f"   {status} SNR {snr} dB: {msg if msg else 'Valid'}")

    # Test parameter dictionary validation
    print("\n3. Parameter Dictionary Validation:")
    schema = {
        'snr_db': {'type': 'float', 'min': -10.0, 'max': 40.0, 'required': True},
        'modulation_order': {'type': 'int', 'min': 2, 'max': 256, 'required': True},
        'num_symbols': {'type': 'int', 'min': 1, 'max': 100000, 'required': False}
    }

    # Valid parameters
    params1 = {'snr_db': 10.0, 'modulation_order': 16, 'num_symbols': 1000}
    valid, errors = validate_parameters(params1, schema)
    print(f"   {'✓' if valid else '✗'} Valid params: {valid}")

    # Missing required parameter
    params2 = {'snr_db': 10.0}
    valid, errors = validate_parameters(params2, schema)
    print(f"   {'✓' if valid else '✗'} Missing required: {errors if not valid else 'Valid'}")

    # Out of range
    params3 = {'snr_db': 100.0, 'modulation_order': 16}
    valid, errors = validate_parameters(params3, schema)
    print(f"   {'✓' if valid else '✗'} Out of range: {errors if not valid else 'Valid'}")

    print("\nValidation utilities ready for use!")
