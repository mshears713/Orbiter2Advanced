"""
LDPC Iteration Heatmap Visualization

Creates interactive heatmap visualizations showing the evolution of log-likelihood
ratios (LLRs) and belief propagation messages across LDPC decoding iterations.
Demonstrates convergence behavior and message passing dynamics.

Educational Features:
- LLR evolution heatmap across iterations
- Syndrome weight convergence plot
- Bit flip tracking
- Convergence metrics
- Color-coded confidence levels

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

from visualizations.base import Visualization
from simulations.exceptions import VisualizationError
from models.datamodels import SimulationResult


class LDPCIterationHeatmap(Visualization):
    """Interactive heatmap for LDPC belief propagation iterations.

    Displays:
    - LLR values for each bit across iterations
    - Syndrome weight convergence
    - Bit decision evolution
    - Convergence indicators

    Example:
        >>> viz = LDPCIterationHeatmap()
        >>> fig = viz.render(simulation_result)
    """

    def __init__(self, backend: str = 'plotly'):
        """Initialize LDPC visualization.

        Args:
            backend: Plotting backend - 'plotly' or 'matplotlib'
        """
        super().__init__()
        self.backend = backend.lower()
        self.fig = None

    def get_name(self) -> str:
        return "LDPC Iteration Heatmap"

    def render(self, simulation_result: SimulationResult) -> Any:
        """Render complete LDPC iteration visualization.

        Creates multi-panel display showing:
        - LLR evolution heatmap
        - Syndrome weight convergence
        - Final decoded bits vs transmitted
        - Convergence metrics

        Args:
            simulation_result: Result from LDPCDecodingSimulation

        Returns:
            Plotly or Matplotlib Figure

        Raises:
            VisualizationError: If data is missing or invalid
        """
        # Validate data
        self.validate_data(simulation_result.data)

        required_fields = ['iteration_history', 'codeword', 'decoded_bits']
        for field in required_fields:
            if field not in simulation_result.data:
                raise VisualizationError(f"Missing '{field}' in simulation data")

        # Extract data
        iteration_history = simulation_result.data['iteration_history']
        codeword = simulation_result.data['codeword']
        decoded_bits = simulation_result.data['decoded_bits']

        # Get metrics
        ber = simulation_result.data.get('ber', 0.0)
        converged = simulation_result.data.get('converged', False)
        num_iterations = len(iteration_history)
        block_length = simulation_result.metadata.get('block_length', 0)

        if self.backend == 'plotly':
            fig = self._render_plotly(
                iteration_history, codeword, decoded_bits,
                ber, converged, num_iterations, block_length
            )
        else:
            fig = self._render_matplotlib(
                iteration_history, codeword, decoded_bits,
                ber, converged, num_iterations, block_length
            )

        self.fig = fig
        self._initialized = True
        self._current_data = simulation_result.data

        return fig

    def _render_plotly(
        self,
        iteration_history: List[Dict],
        codeword: np.ndarray,
        decoded_bits: np.ndarray,
        ber: float,
        converged: bool,
        num_iterations: int,
        block_length: int
    ) -> go.Figure:
        """Render using Plotly for interactive visualization.

        Args:
            iteration_history: List of iteration data
            codeword: Original transmitted codeword
            decoded_bits: Final decoded bits
            ber: Bit error rate
            converged: Whether decoding converged
            num_iterations: Number of iterations performed
            block_length: LDPC block length

        Returns:
            Plotly Figure with multiple subplots
        """
        # Create subplots: heatmap + syndrome convergence
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'LLR Evolution Across Iterations',
                'Syndrome Weight Convergence',
                'Decoded Bits vs Transmitted',
                'Bit Confidence (Final LLRs)'
            ),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # === Top Left: LLR Heatmap ===
        # Build matrix of LLR values: rows=bits, cols=iterations
        n_bits = block_length
        llr_matrix = np.zeros((n_bits, num_iterations))

        for iter_idx, iter_data in enumerate(iteration_history):
            llr_matrix[:, iter_idx] = iter_data['llr_posterior']

        fig.add_trace(
            go.Heatmap(
                z=llr_matrix,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="LLR", x=0.46, y=0.75, len=0.35),
                hovertemplate='Bit: %{y}<br>Iteration: %{x}<br>LLR: %{z:.2f}<extra></extra>',
                name='LLRs'
            ),
            row=1, col=1
        )

        # === Top Right: Syndrome Weight Convergence ===
        iterations = [d['iteration'] for d in iteration_history]
        syndrome_weights = [d['syndrome_weight'] for d in iteration_history]

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=syndrome_weights,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                name='Syndrome Weight',
                hovertemplate='Iteration: %{x}<br>Syndrome Weight: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Add horizontal line at y=0 (convergence)
        fig.add_hline(y=0, line_dash="dash", line_color="green",
                     annotation_text="Converged", row=1, col=2)

        # === Bottom Left: Bit Comparison ===
        bit_indices = np.arange(min(50, len(codeword)))  # Show first 50 bits for clarity
        tx_subset = codeword[bit_indices]
        rx_subset = decoded_bits[bit_indices]

        fig.add_trace(
            go.Scatter(
                x=bit_indices,
                y=tx_subset + 0.1,  # Offset for visibility
                mode='markers',
                marker=dict(size=8, color='blue', symbol='circle'),
                name='Transmitted',
                hovertemplate='Bit %{x}: %{y:.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=bit_indices,
                y=rx_subset - 0.1,  # Offset for visibility
                mode='markers',
                marker=dict(size=8, color='red', symbol='x'),
                name='Decoded',
                hovertemplate='Bit %{x}: %{y:.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        # === Bottom Right: Final LLR Confidence ===
        final_llrs = iteration_history[-1]['llr_posterior'] if iteration_history else np.zeros(n_bits)
        llr_magnitudes = np.abs(final_llrs)

        # Show first 50 bits
        bit_subset = np.arange(min(50, len(llr_magnitudes)))
        colors = ['green' if codeword[i] == decoded_bits[i] else 'red'
                 for i in bit_subset]

        fig.add_trace(
            go.Bar(
                x=bit_subset,
                y=llr_magnitudes[bit_subset],
                marker=dict(color=colors),
                name='LLR Magnitude',
                hovertemplate='Bit %{x}<br>Confidence: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="Bit Index", row=1, col=1)

        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Syndrome Weight", row=1, col=2)

        fig.update_xaxes(title_text="Bit Index", row=2, col=1)
        fig.update_yaxes(title_text="Bit Value", row=2, col=1)

        fig.update_xaxes(title_text="Bit Index (first 50)", row=2, col=2)
        fig.update_yaxes(title_text="|LLR|", row=2, col=2)

        # Overall layout
        convergence_status = "✓ Converged" if converged else "✗ Not Converged"

        fig.update_layout(
            title=dict(
                text=f'LDPC Belief Propagation Iterations<br>' +
                     f'<sub>{convergence_status} | Iterations: {num_iterations} | BER: {ber:.6f}</sub>',
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            height=800,
            hovermode='closest'
        )

        return fig

    def _render_matplotlib(
        self,
        iteration_history: List[Dict],
        codeword: np.ndarray,
        decoded_bits: np.ndarray,
        ber: float,
        converged: bool,
        num_iterations: int,
        block_length: int
    ) -> plt.Figure:
        """Render using Matplotlib for static visualization.

        Args:
            Same as _render_plotly

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # === Top Left: LLR Heatmap ===
        ax1 = axes[0, 0]
        n_bits = block_length
        llr_matrix = np.zeros((n_bits, num_iterations))

        for iter_idx, iter_data in enumerate(iteration_history):
            llr_matrix[:, iter_idx] = iter_data['llr_posterior']

        im = ax1.imshow(llr_matrix, aspect='auto', cmap='RdBu', origin='lower',
                       vmin=-np.max(np.abs(llr_matrix)), vmax=np.max(np.abs(llr_matrix)))
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Bit Index')
        ax1.set_title('LLR Evolution Across Iterations')
        plt.colorbar(im, ax=ax1, label='LLR')

        # === Top Right: Syndrome Weight ===
        ax2 = axes[0, 1]
        iterations = [d['iteration'] for d in iteration_history]
        syndrome_weights = [d['syndrome_weight'] for d in iteration_history]

        ax2.plot(iterations, syndrome_weights, 'r-o', linewidth=2, markersize=5)
        ax2.axhline(y=0, color='green', linestyle='--', label='Converged')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Syndrome Weight')
        ax2.set_title('Syndrome Weight Convergence')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # === Bottom Left: Bit Comparison ===
        ax3 = axes[1, 0]
        bit_indices = np.arange(min(50, len(codeword)))
        tx_subset = codeword[bit_indices]
        rx_subset = decoded_bits[bit_indices]

        ax3.plot(bit_indices, tx_subset + 0.1, 'bo', label='Transmitted', markersize=6)
        ax3.plot(bit_indices, rx_subset - 0.1, 'rx', label='Decoded', markersize=6)
        ax3.set_xlabel('Bit Index (first 50)')
        ax3.set_ylabel('Bit Value')
        ax3.set_title('Decoded Bits vs Transmitted')
        ax3.set_ylim([-0.5, 1.5])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # === Bottom Right: Final LLR Confidence ===
        ax4 = axes[1, 1]
        final_llrs = iteration_history[-1]['llr_posterior'] if iteration_history else np.zeros(n_bits)
        llr_magnitudes = np.abs(final_llrs)

        bit_subset = np.arange(min(50, len(llr_magnitudes)))
        colors = ['green' if codeword[i] == decoded_bits[i] else 'red'
                 for i in bit_subset]

        ax4.bar(bit_subset, llr_magnitudes[bit_subset], color=colors)
        ax4.set_xlabel('Bit Index (first 50)')
        ax4.set_ylabel('|LLR| (Confidence)')
        ax4.set_title('Bit Confidence (Final LLRs)')
        ax4.grid(True, alpha=0.3, axis='y')

        # Overall title
        convergence_status = "✓ Converged" if converged else "✗ Not Converged"
        fig.suptitle(
            f'LDPC Belief Propagation Iterations\n' +
            f'{convergence_status} | Iterations: {num_iterations} | BER: {ber:.6f}',
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()
        return fig

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update visualization with new data.

        Args:
            new_data: Dictionary with updated iteration data

        Raises:
            VisualizationError: If not initialized
        """
        if not self._initialized:
            raise VisualizationError("Must call render() before update()")

        self._current_data.update(new_data)


if __name__ == "__main__":
    print("=== LDPC Iteration Heatmap Visualization Demo ===\n")
    print("This visualization requires LDPC simulation data.")
    print("Run LDPCDecodingSimulation first, then pass results to render()")
    print("\nExample usage:")
    print("  from simulations.ldpc_decoding import LDPCDecodingSimulation")
    print("  sim = LDPCDecodingSimulation()")
    print("  result = sim.run_simulation({'block_length': 100, 'code_rate': 0.5, 'snr_db': 2.0})")
    print("  viz = LDPCIterationHeatmap()")
    print("  fig = viz.render(result)")
    print("  fig.show()  # For Plotly")
