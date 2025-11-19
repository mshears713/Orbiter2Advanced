"""
Orbiter-2 Streamlit Dashboard Application

Main entry point for the interactive web-based simulation dashboard.
Provides UI for selecting simulations, configuring parameters, running
simulations, and viewing animated visualizations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from db.db_access import DBManager
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Orbiter-2: Wireless Communications Simulation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'selected_simulation' not in st.session_state:
        st.session_state.selected_simulation = None
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'db' not in st.session_state:
        st.session_state.db = DBManager()


def render_header():
    """Render application header."""
    st.title("🛰️ Orbiter-2: Deep-Space Wireless Communications Simulation")
    st.markdown("""
    **Educational wireless communications platform exploring advanced techniques
    through interactive simulations and visualizations.**

    Each module represents an alien civilization's unique wireless technology encountered
    during a CubeSat's deep-space journey.
    """)
    st.divider()


def render_sidebar():
    """Render sidebar with navigation and controls."""
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # Available simulation modules (placeholder - will be populated dynamically later)
        available_sims = [
            "Welcome",
            "High-Order Modulation (Civilization 1)",
            "OFDM Signal Processing (Civilization 2)",
            "Convolutional Coding (Civilization 3)",
            "LDPC Decoding (Civilization 4)",
            "Mesh Routing (Civilization 5)",
            "Interference Shaping (Civilization 6)",
            "Delay-Tolerant Networking (Civilization 7)",
            "Adaptive Link Strategy (Civilization 8)"
        ]

        st.subheader("Select Simulation")
        selected = st.selectbox(
            "Choose a simulation module:",
            available_sims,
            index=0
        )

        st.session_state.selected_simulation = selected

        st.divider()

        # Database statistics
        st.subheader("📊 Statistics")
        try:
            stats = st.session_state.db.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Runs", stats.get('total_runs', 0))
            with col2:
                st.metric("Successful", stats.get('successful_runs', 0))

            if stats.get('avg_execution_time_ms', 0) > 0:
                st.metric(
                    "Avg Time",
                    f"{stats['avg_execution_time_ms']:.1f} ms"
                )
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")

        st.divider()

        # Configuration section
        with st.expander("⚙️ Configuration"):
            st.info("Configuration settings loaded from config.json")
            config = st.session_state.config
            st.write(f"**Sampling Rate:** {config.simulation.sampling_rate / 1e6:.1f} MHz")
            st.write(f"**Default SNR:** {config.simulation.snr_db} dB")
            st.write(f"**Database:** {config.database.db_path}")

        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Orbiter-2** is an educational platform for learning advanced
            wireless communications concepts through interactive simulations.

            **Features:**
            - Real-time animated visualizations
            - Comprehensive parameter controls
            - Performance metrics tracking
            - Educational commentary

            **Version:** 1.0.0-dev
            """)


def render_welcome_page():
    """Render welcome/landing page."""
    st.header("Welcome to Orbiter-2")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🚀 Mission Overview

        You are piloting a CubeSat on a deep-space exploration mission. During your journey,
        you encounter eight alien civilizations, each using unique wireless communication
        technologies. Your mission: learn and master each civilization's techniques.

        ### 📡 Available Simulations

        Each simulation module teaches a specific wireless communications concept:

        1. **High-Order Modulation** - Learn QAM and PSK modulation schemes with
           constellation diagrams showing symbol mapping and noise effects.

        2. **OFDM Signal Processing** - Explore orthogonal frequency division multiplexing
           with subcarrier visualization and cyclic prefix demonstration.

        3. **Convolutional Coding** - Master forward error correction with animated
           trellis diagrams showing Viterbi decoding in action.

        4. **LDPC Decoding** - Understand iterative belief propagation with heatmaps
           showing convergence behavior.

        5. **Mesh Routing** - Simulate multi-hop wireless networks with dynamic
           routing and topology visualization.

        6. **Interference Shaping** - Control spatial interference patterns with
           beamforming and nulling techniques.

        7. **Delay-Tolerant Networking** - Explore store-and-forward routing for
           intermittent connectivity scenarios.

        8. **Adaptive Link Strategy** - Dynamic modulation and power control based
           on channel conditions.

        ### 🎯 Getting Started

        1. Select a simulation from the sidebar
        2. Configure parameters using the control panel
        3. Click "Run Simulation" to execute
        4. Explore interactive visualizations
        5. Analyze performance metrics

        ### 💡 Learning Approach

        Each simulation provides:
        - **Interactive controls** for exploring parameter space
        - **Real-time visualizations** showing algorithm behavior
        - **Educational commentary** explaining key concepts
        - **Performance metrics** demonstrating trade-offs
        """)

    with col2:
        st.info("""
        **Quick Tips:**

        🎛️ Use the sidebar to navigate between simulations

        📊 Check statistics to track your progress

        ⚙️ Adjust configuration for different scenarios

        💾 All simulation runs are logged to the database
        """)

        st.success("""
        **Ready to begin?**

        Select a simulation from the sidebar to start exploring!
        """)

        # Recent activity
        st.subheader("Recent Activity")
        try:
            recent = st.session_state.db.fetch_recent_runs(limit=5)
            if recent:
                for run in recent:
                    success_icon = "✅" if run['success'] else "❌"
                    st.text(f"{success_icon} {run['simulation_type']}")
            else:
                st.text("No simulations run yet")
        except:
            st.text("Database not available")


def render_placeholder_simulation(sim_name: str):
    """Render placeholder for simulation modules (to be implemented)."""
    st.header(f"📡 {sim_name}")

    st.info("""
    This simulation module is under development and will be available soon.

    **Planned features:**
    - Interactive parameter controls
    - Real-time animated visualizations
    - Performance metrics analysis
    - Educational commentary

    Check back after more modules are implemented!
    """)

    # Show where controls and visualization would go
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        st.text("Parameter controls will appear here")

        st.button("Run Simulation", disabled=True, help="Coming soon!")

    with col2:
        st.subheader("Visualization")
        st.text("Interactive visualization will appear here")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render UI components
    render_header()
    render_sidebar()

    # Main content area - route to appropriate page
    selected = st.session_state.selected_simulation

    if selected == "Welcome" or selected is None:
        render_welcome_page()
    else:
        # Placeholder for simulation modules (will be implemented in Phase 2)
        render_placeholder_simulation(selected)

    # Footer
    st.divider()
    st.caption("Orbiter-2 v1.0.0-dev | Educational Wireless Communications Platform | Claude AI Implementation")


if __name__ == "__main__":
    main()
