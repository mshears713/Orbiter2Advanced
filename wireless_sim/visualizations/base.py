"""
Base Visualization Module Interface for Orbiter-2

Defines the abstract base class for all visualization components.
Ensures consistent rendering and update interfaces across visualizations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models.datamodels import SimulationResult
from simulations.exceptions import VisualizationError


class Visualization(ABC):
    """Abstract base class for all visualization modules.

    All visualizations in Orbiter-2 must inherit from this class and implement
    its abstract methods. This ensures consistent rendering and update patterns
    across different visualization types.

    Subclasses must implement:
        - render(simulation_result): Initialize and render the visualization
        - update(new_data): Update visualization with new data
        - get_name(): Return visualization name

    Example:
        >>> class MyViz(Visualization):
        ...     def get_name(self) -> str:
        ...         return "My Visualization"
        ...
        ...     def render(self, simulation_result: SimulationResult):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self):
        """Initialize visualization base."""
        self._initialized = False
        self._current_data = None

    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of this visualization.

        Returns:
            String name of the visualization

        Example:
            >>> viz = ConstellationPlot()
            >>> viz.get_name()
            'Constellation Diagram'
        """
        pass

    @abstractmethod
    def render(self, simulation_result: SimulationResult) -> Any:
        """Render the visualization from simulation results.

        This method initializes the visualization and displays the full simulation
        results. It should set up axes, titles, labels, and plot initial data.

        Args:
            simulation_result: SimulationResult object containing data to visualize

        Returns:
            Matplotlib figure, Plotly figure, or Streamlit component (depending on impl)

        Raises:
            VisualizationError: If rendering fails due to invalid or missing data

        Example:
            >>> result = simulation.run_simulation(params)
            >>> fig = viz.render(result)
        """
        pass

    @abstractmethod
    def update(self, new_data: Dict[str, Any]) -> None:
        """Update the visualization with new data.

        Used for animated or dynamic visualizations where data changes over time.
        Should efficiently update only the changed elements rather than re-rendering
        the entire visualization.

        Args:
            new_data: Dictionary containing new data to display. Structure depends
                     on specific visualization type.

        Raises:
            VisualizationError: If update fails due to invalid data format

        Example:
            >>> viz.update({'constellation_points': new_points})
        """
        pass

    def get_description(self) -> str:
        """Get a detailed description of this visualization.

        Returns:
            Multi-line string describing what this visualization shows

        Note:
            Subclasses can override this to provide custom descriptions
        """
        return f"{self.get_name()} visualization"

    def validate_data(self, data: Any) -> bool:
        """Validate data before rendering or updating.

        Override this method in subclasses to implement specific validation logic.

        Args:
            data: Data to validate

        Returns:
            True if data is valid

        Raises:
            VisualizationError: If data is invalid
        """
        if data is None:
            raise VisualizationError("Data cannot be None")
        return True

    def clear(self) -> None:
        """Clear the visualization.

        Override this method in subclasses to implement clearing logic.
        """
        self._initialized = False
        self._current_data = None

    @property
    def is_initialized(self) -> bool:
        """Check if visualization has been initialized.

        Returns:
            True if render() has been called successfully
        """
        return self._initialized


if __name__ == "__main__":
    # This module defines interfaces only
    print("=== Orbiter-2 Visualization Module Base Class ===\n")
    print("This module defines the abstract base class for all visualizations.")
    print("Subclasses must implement:")
    print("  - get_name()")
    print("  - render(simulation_result)")
    print("  - update(new_data)")
    print("\nSee individual visualization modules for concrete implementations.")
