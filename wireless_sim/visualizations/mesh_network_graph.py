"""
Mesh Network Graph Visualization

Creates interactive network topology visualizations showing mesh nodes, wireless
links, and packet routing paths. Demonstrates network connectivity and multi-hop
communication paths.

Educational Features:
- Interactive node-link diagram
- Color-coded routing paths
- Node degree highlighting
- Link visualization
- Packet trajectory animation
- Network metrics overlay

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from visualizations.base import Visualization
from simulations.exceptions import VisualizationError
from models.datamodels import SimulationResult


class MeshNetworkGraph(Visualization):
    """Interactive network graph for mesh routing visualization.

    Displays:
    - Network topology with nodes and links
    - Routing paths for packets
    - Node connectivity
    - Network statistics

    Example:
        >>> viz = MeshNetworkGraph()
        >>> fig = viz.render(simulation_result)
    """

    def __init__(self, backend: str = 'plotly'):
        """Initialize mesh network visualization.

        Args:
            backend: Plotting backend - 'plotly' or 'matplotlib'
        """
        super().__init__()
        self.backend = backend.lower()
        self.fig = None

    def get_name(self) -> str:
        return "Mesh Network Graph"

    def render(self, simulation_result: SimulationResult) -> Any:
        """Render complete mesh network visualization.

        Creates visualization showing:
        - Network topology
        - Node positions
        - Wireless links
        - Routing paths
        - Network metrics

        Args:
            simulation_result: Result from MeshRoutingSimulation

        Returns:
            Plotly or Matplotlib Figure

        Raises:
            VisualizationError: If data is missing or invalid
        """
        # Validate data
        self.validate_data(simulation_result.data)

        required_fields = ['node_positions', 'adjacency_matrix', 'packet_records']
        for field in required_fields:
            if field not in simulation_result.data:
                raise VisualizationError(f"Missing '{field}' in simulation data")

        # Extract data
        node_positions = simulation_result.data['node_positions']
        adj_matrix = simulation_result.data['adjacency_matrix']
        packet_records = simulation_result.data['packet_records']

        # Get metrics
        pdr = simulation_result.data.get('packet_delivery_ratio', 0.0)
        is_connected = simulation_result.data.get('is_connected', False)
        num_nodes = simulation_result.metadata.get('num_nodes', 0)
        tx_range = simulation_result.metadata.get('transmission_range', 0)

        if self.backend == 'plotly':
            fig = self._render_plotly(
                node_positions, adj_matrix, packet_records,
                pdr, is_connected, num_nodes, tx_range
            )
        else:
            fig = self._render_matplotlib(
                node_positions, adj_matrix, packet_records,
                pdr, is_connected, num_nodes, tx_range
            )

        self.fig = fig
        self._initialized = True
        self._current_data = simulation_result.data

        return fig

    def _render_plotly(
        self,
        node_positions: np.ndarray,
        adj_matrix: np.ndarray,
        packet_records: List[Dict],
        pdr: float,
        is_connected: bool,
        num_nodes: int,
        tx_range: float
    ) -> go.Figure:
        """Render using Plotly for interactive visualization.

        Args:
            node_positions: Node (x,y) coordinates
            adj_matrix: Adjacency matrix
            packet_records: Packet routing records
            pdr: Packet delivery ratio
            is_connected: Whether network is fully connected
            num_nodes: Number of nodes
            tx_range: Transmission range

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # === Draw edges (wireless links) ===
        edge_x = []
        edge_y = []
        num_nodes_actual = len(node_positions)

        for i in range(num_nodes_actual):
            for j in range(i+1, num_nodes_actual):
                if adj_matrix[i, j] == 1:
                    edge_x.extend([node_positions[i, 0], node_positions[j, 0], None])
                    edge_y.extend([node_positions[i, 1], node_positions[j, 1], None])

        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            hoverinfo='skip',
            showlegend=False,
            name='Links'
        ))

        # === Draw routing paths (sample first 3 successful packets) ===
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        path_count = 0

        for pkt_record in packet_records[:5]:  # Show first 5 packets
            if not pkt_record['success']:
                continue

            path = pkt_record['path']
            if len(path) < 2:
                continue

            # Draw path
            path_x = []
            path_y = []

            for node_idx in path:
                path_x.append(node_positions[node_idx, 0])
                path_y.append(node_positions[node_idx, 1])

            fig.add_trace(go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines+markers',
                line=dict(color=colors[path_count % len(colors)], width=3),
                marker=dict(size=8),
                name=f"Packet {pkt_record['packet_id']}: {pkt_record['source']}→{pkt_record['destination']}",
                hovertemplate='Node: %{text}<extra></extra>',
                text=[f"Node {n}" for n in path]
            ))

            path_count += 1
            if path_count >= 3:  # Limit to 3 paths for clarity
                break

        # === Draw nodes ===
        # Color nodes by degree (number of connections)
        node_degrees = np.sum(adj_matrix, axis=1)

        fig.add_trace(go.Scatter(
            x=node_positions[:, 0],
            y=node_positions[:, 1],
            mode='markers+text',
            marker=dict(
                size=15,
                color=node_degrees,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node<br>Degree", x=1.15),
                line=dict(color='white', width=2)
            ),
            text=[str(i) for i in range(num_nodes_actual)],
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            hovertemplate='Node %{text}<br>Degree: %{marker.color}<br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>',
            name='Nodes',
            showlegend=False
        ))

        # Update layout
        connection_status = "✓ Connected" if is_connected else "✗ Disconnected"

        fig.update_layout(
            title=dict(
                text=f'Mesh Network Topology ({num_nodes} nodes)<br>' +
                     f'<sub>{connection_status} | PDR: {pdr:.1%} | Range: {tx_range}m</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='X Position (m)',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='Y Position (m)',
                gridcolor='lightgray',
                showgrid=True,
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            height=700,
            width=900
        )

        return fig

    def _render_matplotlib(
        self,
        node_positions: np.ndarray,
        adj_matrix: np.ndarray,
        packet_records: List[Dict],
        pdr: float,
        is_connected: bool,
        num_nodes: int,
        tx_range: float
    ) -> plt.Figure:
        """Render using Matplotlib for static visualization.

        Args:
            Same as _render_plotly

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        num_nodes_actual = len(node_positions)

        # Draw edges
        for i in range(num_nodes_actual):
            for j in range(i+1, num_nodes_actual):
                if adj_matrix[i, j] == 1:
                    ax.plot(
                        [node_positions[i, 0], node_positions[j, 0]],
                        [node_positions[i, 1], node_positions[j, 1]],
                        'gray', linewidth=0.5, alpha=0.5, zorder=1
                    )

        # Draw routing paths (first 3 successful)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        path_count = 0

        for pkt_record in packet_records[:5]:
            if not pkt_record['success'] or len(pkt_record['path']) < 2:
                continue

            path = pkt_record['path']
            path_x = [node_positions[n, 0] for n in path]
            path_y = [node_positions[n, 1] for n in path]

            ax.plot(
                path_x, path_y,
                color=colors[path_count % len(colors)],
                linewidth=2, marker='o', markersize=6,
                label=f"Pkt {pkt_record['packet_id']}: {pkt_record['source']}→{pkt_record['destination']}",
                zorder=3
            )

            path_count += 1
            if path_count >= 3:
                break

        # Draw nodes
        node_degrees = np.sum(adj_matrix, axis=1)
        scatter = ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=node_degrees,
            cmap='viridis',
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=4
        )

        # Add node labels
        for i, (x, y) in enumerate(node_positions):
            ax.text(x, y, str(i), color='white', fontsize=8,
                   ha='center', va='center', zorder=5)

        # Colorbar for node degree
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Node Degree', rotation=270, labelpad=20)

        # Formatting
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        connection_status = "✓ Connected" if is_connected else "✗ Disconnected"
        ax.set_title(
            f'Mesh Network Topology ({num_nodes} nodes)\n' +
            f'{connection_status} | PDR: {pdr:.1%} | Range: {tx_range}m',
            fontsize=14, fontweight='bold'
        )

        if path_count > 0:
            ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update visualization with new data.

        Args:
            new_data: Dictionary with updated network data

        Raises:
            VisualizationError: If not initialized
        """
        if not self._initialized:
            raise VisualizationError("Must call render() before update()")

        self._current_data.update(new_data)


if __name__ == "__main__":
    print("=== Mesh Network Graph Visualization Demo ===\n")
    print("This visualization requires mesh routing simulation data.")
    print("Run MeshRoutingSimulation first, then pass results to render()")
    print("\nExample usage:")
    print("  from simulations.mesh_routing import MeshRoutingSimulation")
    print("  sim = MeshRoutingSimulation()")
    print("  result = sim.run_simulation({'num_nodes': 15, 'network_area': 100, 'transmission_range': 30})")
    print("  viz = MeshNetworkGraph()")
    print("  fig = viz.render(result)")
    print("  fig.show()  # For Plotly")
