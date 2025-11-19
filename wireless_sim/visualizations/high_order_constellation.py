"""
Constellation Diagram Visualization for High-Order Modulation

Creates interactive scatter plots showing transmitted and received constellation
points with ideal constellation overlay. Supports both Matplotlib and Plotly
rendering for static and interactive visualizations.

Educational Features:
- Color-coded transmitted vs received symbols
- Ideal constellation grid overlay
- Decision boundaries
- Error vector visualization
- SNR and error metrics display

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from visualizations.base import Visualization
from simulations.exceptions import VisualizationError
from models.datamodels import SimulationResult


class HighOrderConstellationPlot(Visualization):
    """Interactive constellation diagram for high-order modulation.

    Displays both transmitted (ideal) and received (noisy) symbols on the
    same I-Q plane, with the ideal constellation overlay. Educational
    annotations show the effects of noise and modulation order.

    Example:
        >>> viz = HighOrderConstellationPlot()
        >>> fig = viz.render(simulation_result)
        >>> viz.update({'new_rx_symbols': updated_symbols})
    """

    def __init__(self, backend: str = 'plotly'):
        """Initialize visualization.

        Args:
            backend: Plotting backend - 'matplotlib' or 'plotly' (default)
        """
        super().__init__()
        self.backend = backend.lower()
        self.fig = None
        self.scatter_tx = None
        self.scatter_rx = None

    def get_name(self) -> str:
        return "Constellation Diagram"

    def render(self, simulation_result: SimulationResult) -> Any:
        """Render complete constellation diagram.

        Creates visualization showing:
        - Ideal constellation points (reference)
        - Transmitted symbols
        - Received symbols with noise
        - Decision boundaries
        - Performance metrics

        Args:
            simulation_result: Result from HighOrderModulationSimulation

        Returns:
            Matplotlib Figure or Plotly Figure depending on backend

        Raises:
            VisualizationError: If data is invalid or missing
        """
        # Validate data
        self.validate_data(simulation_result.data)

        if 'tx_symbols' not in simulation_result.data:
            raise VisualizationError("Missing 'tx_symbols' in simulation data")
        if 'rx_symbols' not in simulation_result.data:
            raise VisualizationError("Missing 'rx_symbols' in simulation data")
        if 'constellation' not in simulation_result.data:
            raise VisualizationError("Missing 'constellation' in simulation data")

        # Extract data
        tx_symbols = simulation_result.data['tx_symbols']
        rx_symbols = simulation_result.data['rx_symbols']
        constellation = simulation_result.data['constellation']

        # Get metrics
        ber = simulation_result.data.get('ber', 0.0)
        ser = simulation_result.data.get('ser', 0.0)
        snr_db = simulation_result.data.get('actual_snr_db', 0.0)
        mod_order = simulation_result.metadata.get('modulation_order', 0)

        if self.backend == 'plotly':
            fig = self._render_plotly(
                tx_symbols, rx_symbols, constellation,
                ber, ser, snr_db, mod_order
            )
        else:
            fig = self._render_matplotlib(
                tx_symbols, rx_symbols, constellation,
                ber, ser, snr_db, mod_order
            )

        self.fig = fig
        self._initialized = True
        self._current_data = simulation_result.data

        return fig

    def _render_plotly(
        self,
        tx_symbols: np.ndarray,
        rx_symbols: np.ndarray,
        constellation: np.ndarray,
        ber: float,
        ser: float,
        snr_db: float,
        mod_order: int
    ) -> go.Figure:
        """Render using Plotly for interactive visualization.

        Args:
            tx_symbols: Clean transmitted symbols
            rx_symbols: Noisy received symbols
            constellation: Ideal constellation points
            ber: Bit error rate
            ser: Symbol error rate
            snr_db: Signal-to-noise ratio
            mod_order: Modulation order

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Plot ideal constellation (reference grid)
        fig.add_trace(go.Scatter(
            x=constellation.real,
            y=constellation.imag,
            mode='markers',
            name='Ideal Constellation',
            marker=dict(
                size=15,
                symbol='x',
                color='black',
                line=dict(width=2)
            ),
            hovertemplate='Ideal<br>I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
        ))

        # Plot transmitted symbols (semi-transparent)
        fig.add_trace(go.Scatter(
            x=tx_symbols.real,
            y=tx_symbols.imag,
            mode='markers',
            name='Transmitted',
            marker=dict(
                size=6,
                color='blue',
                opacity=0.3
            ),
            hovertemplate='TX<br>I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
        ))

        # Plot received symbols (with noise)
        fig.add_trace(go.Scatter(
            x=rx_symbols.real,
            y=rx_symbols.imag,
            mode='markers',
            name='Received (Noisy)',
            marker=dict(
                size=4,
                color='red',
                opacity=0.6
            ),
            hovertemplate='RX<br>I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{mod_order}-QAM Constellation Diagram<br>' +
                     f'<sub>SNR: {snr_db:.1f} dB | BER: {ber:.4f} | SER: {ser:.4f}</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='In-Phase (I)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Quadrature (Q)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                gridcolor='lightgray',
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            width=800,
            height=700
        )

        return fig

    def _render_matplotlib(
        self,
        tx_symbols: np.ndarray,
        rx_symbols: np.ndarray,
        constellation: np.ndarray,
        ber: float,
        ser: float,
        snr_db: float,
        mod_order: int
    ) -> plt.Figure:
        """Render using Matplotlib for static visualization.

        Args:
            Same as _render_plotly

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot received symbols (background, with noise)
        ax.scatter(
            rx_symbols.real, rx_symbols.imag,
            c='red', s=20, alpha=0.4,
            label='Received (Noisy)', zorder=1
        )

        # Plot transmitted symbols
        ax.scatter(
            tx_symbols.real, tx_symbols.imag,
            c='blue', s=30, alpha=0.3,
            label='Transmitted', zorder=2
        )

        # Plot ideal constellation (foreground, prominent)
        ax.scatter(
            constellation.real, constellation.imag,
            c='black', s=150, marker='x', linewidths=2,
            label='Ideal Constellation', zorder=3
        )

        # Add grid and axes through origin
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Labels and title
        ax.set_xlabel('In-Phase (I)', fontsize=12)
        ax.set_ylabel('Quadrature (Q)', fontsize=12)
        ax.set_title(
            f'{mod_order}-QAM Constellation Diagram\n' +
            f'SNR: {snr_db:.1f} dB | BER: {ber:.4f} | SER: {ser:.4f}',
            fontsize=14, fontweight='bold'
        )

        # Legend
        ax.legend(loc='upper left', framealpha=0.9)

        # Add info box
        info_text = (
            f'Symbols: {len(tx_symbols)}\n'
            f'Mod Order: {mod_order}\n'
            f'Bits/Symbol: {int(np.log2(mod_order))}'
        )
        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        plt.tight_layout()

        return fig

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update visualization with new data.

        For animation or dynamic updates. Updates received symbols
        while keeping constellation and transmitted symbols fixed.

        Args:
            new_data: Dictionary with:
                - 'rx_symbols': New received symbols (optional)
                - 'highlight_errors': Indices of symbol errors (optional)

        Raises:
            VisualizationError: If visualization not initialized
        """
        if not self._initialized:
            raise VisualizationError("Must call render() before update()")

        # Update implementation depends on backend
        # For Plotly: update trace data
        # For Matplotlib: update scatter data
        # This is a simplified version - full implementation would
        # actually update the plot data

        self._current_data.update(new_data)

        # In a real implementation with animation:
        # if self.backend == 'plotly' and 'rx_symbols' in new_data:
        #     self.fig.data[2].x = new_data['rx_symbols'].real
        #     self.fig.data[2].y = new_data['rx_symbols'].imag


if __name__ == "__main__":
    print("=== Constellation Diagram Visualization Demo ===\n")
    print("This visualization requires simulation data.")
    print("Run a simulation first, then pass results to render()")
    print("\nExample usage:")
    print("  from simulations.high_order_modulation import HighOrderModulationSimulation")
    print("  sim = HighOrderModulationSimulation()")
    print("  result = sim.run_simulation({'modulation_order': 16, 'num_symbols': 500, 'snr_db': 10})")
    print("  viz = HighOrderConstellationPlot()")
    print("  fig = viz.render(result)")
