"""
OFDM Subcarrier Constellation Visualization

Creates interactive multi-panel visualization showing OFDM subcarrier constellations,
time-domain signals, and frequency response. Demonstrates how OFDM parallelizes
data transmission across multiple orthogonal subcarriers.

Educational Features:
- Per-subcarrier constellation diagrams
- Color-coded subcarrier identification
- Time-domain OFDM symbol visualization
- Frequency spectrum display
- Cyclic prefix illustration
- Pilot subcarrier highlighting

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional

from visualizations.base import Visualization
from simulations.exceptions import VisualizationError
from models.datamodels import SimulationResult


class OFDMConstellationPlot(Visualization):
    """Interactive OFDM visualization with subcarrier constellations.

    Displays multiple views of OFDM signal:
    - Superimposed subcarrier constellations
    - Individual subcarrier details
    - Time-domain OFDM symbols
    - Frequency domain representation

    Example:
        >>> viz = OFDMConstellationPlot()
        >>> fig = viz.render(simulation_result)
    """

    def __init__(self, backend: str = 'plotly'):
        """Initialize OFDM visualization.

        Args:
            backend: Plotting backend - 'plotly' or 'matplotlib'
        """
        super().__init__()
        self.backend = backend.lower()
        self.fig = None

    def get_name(self) -> str:
        return "OFDM Subcarrier Constellation"

    def render(self, simulation_result: SimulationResult) -> Any:
        """Render complete OFDM visualization.

        Creates multi-panel display showing:
        - All subcarrier constellations superimposed
        - Time-domain OFDM signal
        - Frequency spectrum
        - Performance metrics

        Args:
            simulation_result: Result from OFDMSimulation

        Returns:
            Plotly or Matplotlib Figure

        Raises:
            VisualizationError: If data is missing or invalid
        """
        # Validate data
        self.validate_data(simulation_result.data)

        required_fields = ['tx_freq_domain', 'rx_freq_domain', 'constellation']
        for field in required_fields:
            if field not in simulation_result.data:
                raise VisualizationError(f"Missing '{field}' in simulation data")

        # Extract data
        tx_freq = simulation_result.data['tx_freq_domain']
        rx_freq = simulation_result.data['rx_freq_domain']
        constellation = simulation_result.data['constellation']
        pilot_indices = simulation_result.data.get('pilot_indices', np.array([]))
        data_indices = simulation_result.data.get('data_indices', np.arange(tx_freq.shape[1]))

        # Get metrics
        ber = simulation_result.data.get('ber', 0.0)
        snr_db = simulation_result.data.get('actual_snr_db', 0.0)
        num_subcarriers = simulation_result.metadata.get('num_subcarriers', 0)
        mod_type = simulation_result.metadata.get('subcarrier_modulation', 'QPSK')

        if self.backend == 'plotly':
            fig = self._render_plotly(
                tx_freq, rx_freq, constellation,
                pilot_indices, data_indices,
                ber, snr_db, num_subcarriers, mod_type
            )
        else:
            fig = self._render_matplotlib(
                tx_freq, rx_freq, constellation,
                pilot_indices, data_indices,
                ber, snr_db, num_subcarriers, mod_type
            )

        self.fig = fig
        self._initialized = True
        self._current_data = simulation_result.data

        return fig

    def _render_plotly(
        self,
        tx_freq: np.ndarray,
        rx_freq: np.ndarray,
        constellation: np.ndarray,
        pilot_indices: np.ndarray,
        data_indices: np.ndarray,
        ber: float,
        snr_db: float,
        num_subcarriers: int,
        mod_type: str
    ) -> go.Figure:
        """Render using Plotly for interactive visualization.

        Args:
            tx_freq: Transmitted frequency domain symbols
            rx_freq: Received frequency domain symbols
            constellation: Ideal constellation points
            pilot_indices: Indices of pilot subcarriers
            data_indices: Indices of data subcarriers
            ber: Bit error rate
            snr_db: Signal-to-noise ratio
            num_subcarriers: Number of subcarriers
            mod_type: Modulation type

        Returns:
            Plotly Figure with multiple subplots
        """
        # Create subplots: constellation + time domain
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Subcarrier Constellations', 'Received Symbols per Subcarrier'),
            column_widths=[0.5, 0.5],
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )

        # Left plot: Superimposed constellation for all subcarriers
        # Plot ideal constellation
        fig.add_trace(
            go.Scatter(
                x=constellation.real,
                y=constellation.imag,
                mode='markers',
                name='Ideal Constellation',
                marker=dict(size=12, symbol='x', color='black', line=dict(width=2)),
                hovertemplate='Ideal<br>I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot received symbols for data subcarriers (color by subcarrier index)
        # Sample a subset of subcarriers for clarity
        num_to_plot = min(8, len(data_indices))
        subcarrier_step = max(1, len(data_indices) // num_to_plot)

        colors = plt.cm.tab10(np.linspace(0, 1, num_to_plot))

        for i, sc_idx in enumerate(data_indices[::subcarrier_step][:num_to_plot]):
            rx_symbols_sc = rx_freq[:, sc_idx]
            color_rgb = f'rgb({int(colors[i,0]*255)},{int(colors[i,1]*255)},{int(colors[i,2]*255)})'

            fig.add_trace(
                go.Scatter(
                    x=rx_symbols_sc.real,
                    y=rx_symbols_sc.imag,
                    mode='markers',
                    name=f'Subcarrier {sc_idx}',
                    marker=dict(size=6, color=color_rgb, opacity=0.7),
                    hovertemplate=f'SC {sc_idx}<br>I: %{{x:.3f}}<br>Q: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )

        # Right plot: Magnitude vs subcarrier index
        # Show average power per subcarrier
        rx_power_per_sc = np.mean(np.abs(rx_freq)**2, axis=0)

        # Separate data and pilot subcarriers
        fig.add_trace(
            go.Scatter(
                x=data_indices,
                y=10*np.log10(rx_power_per_sc[data_indices] + 1e-10),
                mode='markers',
                name='Data Subcarriers',
                marker=dict(size=8, color='blue'),
                hovertemplate='SC: %{x}<br>Power: %{y:.2f} dB<extra></extra>'
            ),
            row=1, col=2
        )

        if len(pilot_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pilot_indices,
                    y=10*np.log10(rx_power_per_sc[pilot_indices] + 1e-10),
                    mode='markers',
                    name='Pilot Subcarriers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    hovertemplate='Pilot SC: %{x}<br>Power: %{y:.2f} dB<extra></extra>'
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_xaxes(title_text="In-Phase (I)", row=1, col=1)
        fig.update_yaxes(title_text="Quadrature (Q)", row=1, col=1, scaleanchor="x", scaleratio=1)

        fig.update_xaxes(title_text="Subcarrier Index", row=1, col=2)
        fig.update_yaxes(title_text="Power (dB)", row=1, col=2)

        fig.update_layout(
            title=dict(
                text=f'OFDM: {num_subcarriers} Subcarriers, {mod_type} Modulation<br>' +
                     f'<sub>SNR: {snr_db:.1f} dB | BER: {ber:.4f}</sub>',
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            height=500,
            hovermode='closest'
        )

        return fig

    def _render_matplotlib(
        self,
        tx_freq: np.ndarray,
        rx_freq: np.ndarray,
        constellation: np.ndarray,
        pilot_indices: np.ndarray,
        data_indices: np.ndarray,
        ber: float,
        snr_db: float,
        num_subcarriers: int,
        mod_type: str
    ) -> plt.Figure:
        """Render using Matplotlib for static visualization.

        Args:
            Same as _render_plotly

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Constellation
        ax1 = axes[0]

        # Plot ideal constellation
        ax1.scatter(
            constellation.real, constellation.imag,
            c='black', s=150, marker='x', linewidths=2,
            label='Ideal', zorder=3
        )

        # Plot received symbols (sample subcarriers for clarity)
        num_to_plot = min(8, len(data_indices))
        subcarrier_step = max(1, len(data_indices) // num_to_plot)
        colors = plt.cm.tab10(np.linspace(0, 1, num_to_plot))

        for i, sc_idx in enumerate(data_indices[::subcarrier_step][:num_to_plot]):
            rx_symbols_sc = rx_freq[:, sc_idx]
            ax1.scatter(
                rx_symbols_sc.real, rx_symbols_sc.imag,
                c=[colors[i]], s=40, alpha=0.6,
                label=f'SC {sc_idx}'
            )

        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlabel('In-Phase (I)')
        ax1.set_ylabel('Quadrature (Q)')
        ax1.set_title('Subcarrier Constellations')
        ax1.legend(loc='upper right', fontsize=8)

        # Right: Power per subcarrier
        ax2 = axes[1]
        rx_power_per_sc = np.mean(np.abs(rx_freq)**2, axis=0)

        ax2.plot(data_indices, 10*np.log10(rx_power_per_sc[data_indices] + 1e-10),
                'bo', label='Data', markersize=6)

        if len(pilot_indices) > 0:
            ax2.plot(pilot_indices, 10*np.log10(rx_power_per_sc[pilot_indices] + 1e-10),
                    'rd', label='Pilots', markersize=8)

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Subcarrier Index')
        ax2.set_ylabel('Power (dB)')
        ax2.set_title('Received Power per Subcarrier')
        ax2.legend()

        fig.suptitle(
            f'OFDM: {num_subcarriers} Subcarriers, {mod_type}\n' +
            f'SNR: {snr_db:.1f} dB | BER: {ber:.4f}',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()
        return fig

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update visualization with new data.

        Args:
            new_data: Dictionary with updated OFDM symbols

        Raises:
            VisualizationError: If not initialized
        """
        if not self._initialized:
            raise VisualizationError("Must call render() before update()")

        self._current_data.update(new_data)


if __name__ == "__main__":
    print("=== OFDM Constellation Visualization Demo ===\n")
    print("This visualization requires OFDM simulation data.")
    print("Run OFDMSimulation first, then pass results to render()")
    print("\nExample usage:")
    print("  from simulations.ofdm import OFDMSimulation")
    print("  sim = OFDMSimulation()")
    print("  result = sim.run_simulation({'num_subcarriers': 64, 'snr_db': 15.0})")
    print("  viz = OFDMConstellationPlot()")
    print("  fig = viz.render(result)")
