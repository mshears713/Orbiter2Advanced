"""
Custom Exceptions for Orbiter-2 Simulation Modules

Defines exception classes for handling errors in wireless communication simulations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""


class SimulationError(Exception):
    """Base exception for all simulation-related errors.

    Use this exception or its subclasses to indicate errors during simulation
    execution, parameter validation, or result processing.

    Example:
        >>> raise SimulationError("Invalid modulation order")
    """
    pass


class ParameterValidationError(SimulationError):
    """Exception raised when simulation parameters fail validation.

    Example:
        >>> raise ParameterValidationError("SNR must be positive")
    """
    pass


class SimulationExecutionError(SimulationError):
    """Exception raised during simulation execution.

    Example:
        >>> raise SimulationExecutionError("IFFT failed due to invalid input")
    """
    pass


class VisualizationError(Exception):
    """Exception raised during visualization rendering or updates.

    Example:
        >>> raise VisualizationError("Cannot render with empty data")
    """
    pass
