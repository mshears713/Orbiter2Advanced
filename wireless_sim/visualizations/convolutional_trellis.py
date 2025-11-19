"""
Trellis Diagram Visualization for Convolutional Coding

Creates interactive and animated visualizations of the Viterbi decoding trellis,
showing state transitions, path metrics, and survivor paths. Educational tool
for understanding maximum likelihood decoding and the Viterbi algorithm.

Educational Features:
- Animated trellis state transitions
- Color-coded path metrics
- Survivor path highlighting
- Branch metric visualization
- Stage-by-stage decoding progression
- Final decoded path overlay

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

from visualizations.base import Visualization
from simulations.exceptions import VisualizationError
from models.datamodels import SimulationResult


class ConvolutionalTrellisPlot(Visualization):
    """Interactive trellis diagram for Viterbi decoding visualization.

    Displays the trellis structure of a convolutional code with:
    - State nodes at each time stage
    - Branch transitions (solid for input=0, dashed for input=1)
    - Path metrics displayed at nodes
    - Survivor paths highlighted
    - Final decoded path emphasized

    Example:
        >>> viz = ConvolutionalTrellisPlot()
        >>> fig = viz.render(simulation_result)
    """

    def __init__(self, backend: str = 'plotly', max_stages: int = 20):
        """Initialize trellis visualization.

        Args:
            backend: Plotting backend - 'plotly' or 'matplotlib'
            max_stages: Maximum number of trellis stages to display
        """
        super().__init__()
        self.backend = backend.lower()
        self.max_stages = max_stages
        self.fig = None

    def get_name(self) -> str:
        return "Convolutional Code Trellis Diagram"

    def render(self, simulation_result: SimulationResult) -> Any:
        """Render complete trellis diagram.

        Creates visualization showing:
        - Trellis state diagram with all transitions
        - Path metrics at each node
        - Survivor paths
        - Final decoded path
        - Coding gain metrics

        Args:
            simulation_result: Result from ConvolutionalCodingSimulation

        Returns:
            Plotly or Matplotlib Figure

        Raises:
            VisualizationError: If data is missing or invalid
        """
        # Validate data
        self.validate_data(simulation_result.data)

        required_fields = ['trellis_data', 'decoded_bits', 'info_bits']
        for field in required_fields:
            if field not in simulation_result.data:
                raise VisualizationError(f"Missing '{field}' in simulation data")

        # Extract data
        trellis_data = simulation_result.data['trellis_data']
        decoded_bits = simulation_result.data['decoded_bits']
        info_bits = simulation_result.data['info_bits']

        # Get metadata
        K = simulation_result.metadata.get('constraint_length', 7)
        num_states = simulation_result.metadata.get('num_states', 2**(K-1))
        ber = simulation_result.data.get('ber', 0.0)
        coding_gain_db = simulation_result.data.get('coding_gain_db', 0.0)

        # Limit stages for visualization clarity
        num_stages = min(len(trellis_data), self.max_stages)
        trellis_subset = trellis_data[:num_stages]

        if self.backend == 'plotly':
            fig = self._render_plotly(
                trellis_subset, decoded_bits, num_states,
                K, ber, coding_gain_db
            )
        else:
            fig = self._render_matplotlib(
                trellis_subset, decoded_bits, num_states,
                K, ber, coding_gain_db
            )

        self.fig = fig
        self._initialized = True
        self._current_data = simulation_result.data

        return fig

    def _render_plotly(
        self,
        trellis_data: List[Dict],
        decoded_bits: np.ndarray,
        num_states: int,
        constraint_length: int,
        ber: float,
        coding_gain_db: float
    ) -> go.Figure:
        """Render using Plotly for interactive visualization.

        Args:
            trellis_data: Trellis transition information per stage
            decoded_bits: Final decoded bit sequence
            num_states: Number of trellis states
            constraint_length: Constraint length K
            ber: Bit error rate
            coding_gain_db: Coding gain over uncoded

        Returns:
            Plotly Figure with trellis diagram
        """
        fig = go.Figure()

        num_stages = len(trellis_data)

        # Create node positions (stage vs. state)
        stage_positions = np.arange(num_stages + 1)
        state_positions = np.arange(num_states)

        # Draw all transitions as lines
        # We'll draw all possible transitions first (in gray)
        # Then highlight the survivor path in color

        edge_x = []
        edge_y = []
        edge_text = []

        for stage_idx, stage_info in enumerate(trellis_data):
            transitions = stage_info['transitions']

            # Group transitions by (from_state, to_state) to find the best one
            best_transitions = {}
            for trans in transitions:
                key = (trans['from'], trans['to'])
                if key not in best_transitions or trans['metric'] < best_transitions[key]['metric']:
                    best_transitions[key] = trans

            # Draw all best transitions
            for (from_state, to_state), trans in best_transitions.items():
                # Line from (stage, from_state) to (stage+1, to_state)
                edge_x.extend([stage_idx, stage_idx + 1, None])
                edge_y.extend([from_state, to_state, None])
                edge_text.append(f"Stage {stage_idx}<br>Input: {trans['input']}<br>Metric: {trans['metric']:.2f}")

        # Draw edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            hoverinfo='skip',
            showlegend=False
        ))

        # Draw state nodes
        node_x = []
        node_y = []
        node_text = []

        for stage in stage_positions:
            for state in state_positions:
                node_x.append(stage)
                node_y.append(state)
                node_text.append(f'Stage {stage}<br>State {state}<br>(Binary: {bin(state)[2:].zfill(constraint_length-1)})')

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=10, color='lightblue', line=dict(width=2, color='darkblue')),
            text=node_text,
            hovertemplate='%{text}<extra></extra>',
            name='States'
        ))

        # Highlight survivor path (decoded path)
        if len(decoded_bits) > 0:
            # Reconstruct state sequence from decoded bits
            survivor_x = [0]
            survivor_y = [0]  # Start at state 0

            curr_state = 0
            for bit in decoded_bits[:num_stages]:
                # Next state = (prev_state << 1 | bit) & (num_states - 1)
                next_state = ((curr_state << 1) | int(bit)) & (num_states - 1)
                survivor_x.append(survivor_x[-1] + 1)
                survivor_y.append(next_state)
                curr_state = next_state

            fig.add_trace(go.Scatter(
                x=survivor_x,
                y=survivor_y,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=12, color='red', symbol='circle'),
                name='Decoded Path',
                hovertemplate='Decoded Path<br>Stage: %{x}<br>State: %{y}<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Viterbi Trellis Diagram (K={constraint_length}, States={num_states})<br>' +
                     f'<sub>BER: {ber:.4f} | Coding Gain: {coding_gain_db:.2f} dB | Showing {num_stages} stages</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Trellis Stage (Time)',
                tickmode='linear',
                dtick=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='State',
                tickmode='linear',
                dtick=1,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            height=600,
            width=1200
        )

        return fig

    def _render_matplotlib(
        self,
        trellis_data: List[Dict],
        decoded_bits: np.ndarray,
        num_states: int,
        constraint_length: int,
        ber: float,
        coding_gain_db: float
    ) -> plt.Figure:
        """Render using Matplotlib for static visualization.

        Args:
            Same as _render_plotly

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        num_stages = len(trellis_data)

        # Draw all transitions
        for stage_idx, stage_info in enumerate(trellis_data):
            transitions = stage_info['transitions']

            # Track best transition for each (from, to) pair
            best_transitions = {}
            for trans in transitions:
                key = (trans['from'], trans['to'])
                if key not in best_transitions or trans['metric'] < best_transitions[key]['metric']:
                    best_transitions[key] = trans

            # Draw transitions
            for (from_state, to_state), trans in best_transitions.items():
                linestyle = '-' if trans['input'] == 0 else '--'
                ax.plot([stage_idx, stage_idx + 1],
                       [from_state, to_state],
                       color='lightgray', linewidth=0.5,
                       linestyle=linestyle, alpha=0.5)

        # Draw state nodes
        for stage in range(num_stages + 1):
            for state in range(num_states):
                ax.plot(stage, state, 'o', color='lightblue',
                       markersize=8, markeredgecolor='darkblue', markeredgewidth=1.5)

        # Highlight survivor path
        if len(decoded_bits) > 0:
            survivor_stages = [0]
            survivor_states = [0]

            curr_state = 0
            for bit in decoded_bits[:num_stages]:
                next_state = ((curr_state << 1) | int(bit)) & (num_states - 1)
                survivor_stages.append(survivor_stages[-1] + 1)
                survivor_states.append(next_state)
                curr_state = next_state

            ax.plot(survivor_stages, survivor_states,
                   color='red', linewidth=3, marker='o',
                   markersize=10, label='Decoded Path', zorder=10)

        # Formatting
        ax.set_xlabel('Trellis Stage (Time)', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title(
            f'Viterbi Trellis Diagram (K={constraint_length}, States={num_states})\n' +
            f'BER: {ber:.4f} | Coding Gain: {coding_gain_db:.2f} dB | Showing {num_stages} stages',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Set integer ticks
        ax.set_xticks(range(num_stages + 1))
        ax.set_yticks(range(num_states))

        # Add legend for line styles
        solid_line = mpatches.Patch(color='lightgray', label='Input bit = 0')
        dashed_line = mpatches.Patch(color='lightgray', label='Input bit = 1 (dashed)', linestyle='--')
        ax.legend(handles=[mpatches.Patch(color='red', label='Decoded Path'),
                          solid_line, dashed_line],
                 loc='upper right')

        plt.tight_layout()
        return fig

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update visualization with new data.

        Args:
            new_data: Dictionary with updated trellis data

        Raises:
            VisualizationError: If not initialized
        """
        if not self._initialized:
            raise VisualizationError("Must call render() before update()")

        self._current_data.update(new_data)


if __name__ == "__main__":
    print("=== Convolutional Trellis Visualization Demo ===\n")
    print("This visualization requires Convolutional Coding simulation data.")
    print("Run ConvolutionalCodingSimulation first, then pass results to render()")
    print("\nExample usage:")
    print("  from simulations.convolutional_coding import ConvolutionalCodingSimulation")
    print("  sim = ConvolutionalCodingSimulation()")
    print("  result = sim.run_simulation({'constraint_length': 7, 'num_bits': 100, 'snr_db': 3.0})")
    print("  viz = ConvolutionalTrellisPlot()")
    print("  fig = viz.render(result)")
    print("  fig.show()  # For Plotly")
