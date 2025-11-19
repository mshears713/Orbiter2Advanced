"""
Mesh Routing Simulation - Alien Civilization 5

Simulates wireless mesh network routing with multi-hop packet forwarding.
Demonstrates ad-hoc routing protocols, path discovery, and network topology
dynamics in decentralized wireless networks.

Mesh networks are used in IoT, smart cities, disaster recovery, and military
communications where infrastructure is unavailable or unreliable.

Educational Goals:
- Understand wireless mesh network topologies
- Learn routing protocols (AODV-like behavior)
- Visualize multi-hop packet forwarding
- Explore route discovery and maintenance
- Analyze network metrics (hop count, latency, packet delivery ratio)

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import time

from simulations.base import SimulationModule
from simulations.exceptions import SimulationError
from models.datamodels import SimulationResult, NetworkNode, Packet


class MeshRoutingSimulation(SimulationModule):
    """Wireless mesh network routing simulation.

    Implements multi-hop routing in ad-hoc wireless networks.
    Uses a simplified AODV (Ad-hoc On-Demand Distance Vector) approach
    for route discovery and packet forwarding.

    Example:
        >>> sim = MeshRoutingSimulation()
        >>> params = {'num_nodes': 15, 'network_area': 100, 'transmission_range': 30}
        >>> result = sim.run_simulation(params)
    """

    def get_name(self) -> str:
        return "Mesh Routing (Civilization 5)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'num_nodes': 15,            # Number of mesh nodes
            'network_area': 100,        # Network area side length (m)
            'transmission_range': 30,   # Radio transmission range (m)
            'num_packets': 10,          # Number of packets to route
            'routing_protocol': 'AODV', # Routing protocol
            'node_failure_prob': 0.0    # Probability of node failure
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'num_nodes': {
                'type': 'int',
                'description': 'Number of mesh nodes in network',
                'min': 5,
                'max': 50,
                'default': 15,
                'required': True
            },
            'network_area': {
                'type': 'float',
                'description': 'Network area side length in meters',
                'min': 50.0,
                'max': 200.0,
                'default': 100.0,
                'required': True
            },
            'transmission_range': {
                'type': 'float',
                'description': 'Node transmission range in meters',
                'min': 10.0,
                'max': 100.0,
                'default': 30.0,
                'required': True
            },
            'num_packets': {
                'type': 'int',
                'description': 'Number of data packets to simulate',
                'min': 1,
                'max': 50,
                'default': 10,
                'required': True
            }
        }

    def _generate_node_positions(
        self,
        num_nodes: int,
        area_size: float
    ) -> np.ndarray:
        """Generate random node positions in 2D area.

        Args:
            num_nodes: Number of nodes
            area_size: Square area side length

        Returns:
            Array of shape (num_nodes, 2) with (x, y) positions
        """
        positions = np.random.uniform(0, area_size, size=(num_nodes, 2))
        return positions

    def _compute_adjacency_matrix(
        self,
        positions: np.ndarray,
        tx_range: float
    ) -> np.ndarray:
        """Compute network adjacency matrix based on transmission range.

        Args:
            positions: Node positions (num_nodes x 2)
            tx_range: Transmission range

        Returns:
            Adjacency matrix (num_nodes x num_nodes)
        """
        num_nodes = len(positions)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                # Calculate Euclidean distance
                dist = np.linalg.norm(positions[i] - positions[j])

                # Create edge if within range
                if dist <= tx_range:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return adj_matrix

    def _dijkstra_shortest_path(
        self,
        adj_matrix: np.ndarray,
        source: int,
        dest: int
    ) -> Tuple[List[int], float]:
        """Find shortest path using Dijkstra's algorithm.

        Args:
            adj_matrix: Network adjacency matrix
            source: Source node index
            dest: Destination node index

        Returns:
            Tuple of (path, distance) where path is list of node indices
        """
        num_nodes = len(adj_matrix)

        # Initialize distances and visited set
        distances = np.full(num_nodes, np.inf)
        distances[source] = 0
        visited = set()
        predecessors = {i: None for i in range(num_nodes)}

        for _ in range(num_nodes):
            # Find unvisited node with minimum distance
            unvisited_dist = [(i, distances[i]) for i in range(num_nodes) if i not in visited]
            if not unvisited_dist:
                break

            current, _ = min(unvisited_dist, key=lambda x: x[1])
            visited.add(current)

            # If reached destination, can stop
            if current == dest:
                break

            # Update distances to neighbors
            for neighbor in range(num_nodes):
                if adj_matrix[current, neighbor] == 1 and neighbor not in visited:
                    new_dist = distances[current] + 1  # Hop count metric
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current

        # Reconstruct path
        if distances[dest] == np.inf:
            return [], np.inf  # No path exists

        path = []
        current = dest
        while current is not None:
            path.insert(0, current)
            current = predecessors[current]

        return path, distances[dest]

    def _simulate_packet_routing(
        self,
        adj_matrix: np.ndarray,
        num_packets: int,
        node_failure_prob: float = 0.0
    ) -> List[Dict]:
        """Simulate routing multiple packets through the mesh network.

        Args:
            adj_matrix: Network adjacency matrix
            num_packets: Number of packets to route
            node_failure_prob: Probability of node failure

        Returns:
            List of packet routing records
        """
        num_nodes = len(adj_matrix)
        packet_records = []

        # Simulate node failures
        active_nodes = np.random.random(num_nodes) > node_failure_prob
        active_nodes[0] = True  # Always keep first node active
        active_nodes[-1] = True  # Always keep last node active

        # Generate random source-destination pairs
        for pkt_id in range(num_packets):
            # Random source and destination
            source = np.random.randint(0, num_nodes)
            dest = np.random.randint(0, num_nodes)
            while dest == source:
                dest = np.random.randint(0, num_nodes)

            # Find route
            path, hop_count = self._dijkstra_shortest_path(adj_matrix, source, dest)

            # Check if path uses only active nodes
            if path and all(active_nodes[node] for node in path):
                success = True
            else:
                success = False
                path = []
                hop_count = np.inf

            # Calculate estimated latency (assume 1ms per hop + processing)
            latency_ms = hop_count * 1.5 if hop_count < np.inf else np.inf

            packet_records.append({
                'packet_id': pkt_id,
                'source': int(source),
                'destination': int(dest),
                'path': [int(n) for n in path],
                'hop_count': int(hop_count) if hop_count < np.inf else -1,
                'success': success,
                'latency_ms': float(latency_ms) if latency_ms < np.inf else -1.0
            })

        return packet_records

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute mesh routing simulation.

        Args:
            parameters: Simulation parameters

        Returns:
            SimulationResult with mesh network routing data
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate
            self.validate_parameters(parameters)

            # Extract parameters
            num_nodes = parameters['num_nodes']
            area_size = parameters['network_area']
            tx_range = parameters['transmission_range']
            num_packets = parameters['num_packets']
            node_fail_prob = parameters.get('node_failure_prob', 0.0)

            # Generate network topology
            node_positions = self._generate_node_positions(num_nodes, area_size)
            adj_matrix = self._compute_adjacency_matrix(node_positions, tx_range)

            # Calculate network statistics
            num_edges = np.sum(adj_matrix) // 2  # Undirected graph
            avg_degree = np.mean(np.sum(adj_matrix, axis=1))

            # Check if network is connected
            # Simple check: can we reach all nodes from node 0?
            visited = set()
            to_visit = [0]
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    continue
                visited.add(current)
                neighbors = np.where(adj_matrix[current] == 1)[0]
                to_visit.extend(neighbors)

            is_connected = len(visited) == num_nodes

            # Simulate packet routing
            packet_records = self._simulate_packet_routing(
                adj_matrix, num_packets, node_fail_prob
            )

            # Calculate metrics
            successful_packets = sum(1 for p in packet_records if p['success'])
            packet_delivery_ratio = successful_packets / num_packets if num_packets > 0 else 0.0

            successful_hops = [p['hop_count'] for p in packet_records if p['success']]
            avg_hop_count = np.mean(successful_hops) if successful_hops else 0.0

            successful_latencies = [p['latency_ms'] for p in packet_records if p['success']]
            avg_latency = np.mean(successful_latencies) if successful_latencies else 0.0

            # Execution time
            execution_time = (time.perf_counter() - start_perf) * 1000

            result_data = {
                'node_positions': node_positions,
                'adjacency_matrix': adj_matrix,
                'packet_records': packet_records,
                'num_edges': int(num_edges),
                'avg_node_degree': float(avg_degree),
                'is_connected': is_connected,
                'packet_delivery_ratio': packet_delivery_ratio,
                'avg_hop_count': avg_hop_count,
                'avg_latency_ms': avg_latency,
                'successful_packets': int(successful_packets),
                'failed_packets': int(num_packets - successful_packets)
            }

            metadata = {
                'module': self.get_name(),
                'num_nodes': num_nodes,
                'network_area': area_size,
                'transmission_range': tx_range,
                'num_packets': num_packets,
                'routing_protocol': parameters.get('routing_protocol', 'AODV')
            }

            return SimulationResult(
                timestamp=start_time,
                parameters=parameters,
                data=result_data,
                metadata=metadata,
                success=True,
                error_message="",
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_perf) * 1000
            return SimulationResult(
                timestamp=start_time,
                parameters=parameters,
                data={},
                metadata={'module': self.get_name()},
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )


if __name__ == "__main__":
    print("=== Mesh Routing Simulation Demo ===\n")

    sim = MeshRoutingSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    # Test with different network sizes
    for num_nodes in [10, 15, 20]:
        params = {
            'num_nodes': num_nodes,
            'network_area': 100.0,
            'transmission_range': 30.0,
            'num_packets': 10,
            'node_failure_prob': 0.0
        }

        result = sim.run_simulation(params)

        if result.success:
            print(f"Network with {num_nodes} nodes:")
            print(f"  Edges: {result.data['num_edges']}")
            print(f"  Avg Node Degree: {result.data['avg_node_degree']:.2f}")
            print(f"  Connected: {result.data['is_connected']}")
            print(f"  Packet Delivery Ratio: {result.data['packet_delivery_ratio']:.2%}")
            print(f"  Avg Hop Count: {result.data['avg_hop_count']:.2f}")
            print(f"  Avg Latency: {result.data['avg_latency_ms']:.2f} ms")
            print(f"  Successful Packets: {result.data['successful_packets']}/{params['num_packets']}")
            print()
        else:
            print(f"  Failed: {result.error_message}\n")

    print("Mesh routing simulation ready!")
