"""
Database Access Layer for Orbiter-2 Wireless Communications Simulation

Provides a comprehensive interface for CRUD operations on the simulation database.
All database interactions should go through the DBManager class for consistency
and proper error handling.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager


class DBManager:
    """Database manager providing CRUD operations for simulation data.

    This class encapsulates all database operations with proper connection management,
    error handling, and transaction support.

    Attributes:
        db_path: Path to the SQLite database file

    Example:
        >>> db = DBManager('wireless_sim.db')
        >>> sim_id = db.insert_simulation_run(
        ...     simulation_type="HighOrderModulation",
        ...     start_time=datetime.now(),
        ...     parameters={"snr_db": 10.0}
        ... )
        >>> db.update_simulation_run(sim_id, datetime.now(), success=True)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.

        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            db_dir = Path(__file__).parent
            db_path = db_dir / "wireless_sim.db"
        self.db_path = str(db_path)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections.

        Yields:
            sqlite3.Connection object

        Example:
            >>> with db.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM simulation_runs")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    # ========================================================================
    # Simulation Runs CRUD Operations
    # ========================================================================

    def insert_simulation_run(
        self,
        simulation_type: str,
        start_time: datetime,
        parameters: Dict[str, Any],
        simulation_name: Optional[str] = None,
        user_notes: Optional[str] = None
    ) -> int:
        """Insert a new simulation run record.

        Args:
            simulation_type: Type of simulation (e.g., "HighOrderModulation")
            start_time: When the simulation started
            parameters: Dictionary of simulation parameters
            simulation_name: Optional human-readable name
            user_notes: Optional notes from the user

        Returns:
            ID of the newly inserted simulation run

        Raises:
            sqlite3.Error: If database operation fails
        """
        params_json = json.dumps(parameters)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulation_runs
                (simulation_type, simulation_name, start_time, parameters,
                 success, user_notes)
                VALUES (?, ?, ?, ?, 0, ?)
            """, (simulation_type, simulation_name, start_time, params_json, user_notes))

            return cursor.lastrowid

    def update_simulation_run(
        self,
        sim_id: int,
        end_time: datetime,
        success: bool,
        execution_time_ms: Optional[float] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update simulation run with completion information.

        Args:
            sim_id: ID of the simulation run to update
            end_time: When the simulation completed
            success: Whether the simulation completed successfully
            execution_time_ms: Execution duration in milliseconds
            result_summary: Optional summary of results
            error_message: Error description if simulation failed

        Returns:
            True if update successful, False otherwise
        """
        result_json = json.dumps(result_summary) if result_summary else None

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE simulation_runs
                    SET end_time = ?,
                        success = ?,
                        execution_time_ms = ?,
                        result_summary = ?,
                        error_message = ?
                    WHERE id = ?
                """, (end_time, 1 if success else 0, execution_time_ms,
                      result_json, error_message, sim_id))

                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating simulation run: {e}")
            return False

    def fetch_simulation_run(self, sim_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a simulation run by ID.

        Args:
            sim_id: ID of the simulation run

        Returns:
            Dictionary containing simulation run data, or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM simulation_runs WHERE id = ?
                """, (sim_id,))

                row = cursor.fetchone()
                if row is None:
                    return None

                # Convert to dictionary and parse JSON fields
                result = dict(row)
                result['parameters'] = json.loads(result['parameters'])
                if result['result_summary']:
                    result['result_summary'] = json.loads(result['result_summary'])

                return result
        except sqlite3.Error as e:
            print(f"Error fetching simulation run: {e}")
            return None

    def fetch_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch most recent simulation runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of dictionaries containing simulation run data
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM simulation_runs
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (limit,))

                results = []
                for row in cursor.fetchall():
                    data = dict(row)
                    data['parameters'] = json.loads(data['parameters'])
                    if data['result_summary']:
                        data['result_summary'] = json.loads(data['result_summary'])
                    results.append(data)

                return results
        except sqlite3.Error as e:
            print(f"Error fetching recent runs: {e}")
            return []

    # ========================================================================
    # Logging Operations
    # ========================================================================

    def insert_log(
        self,
        sim_run_id: Optional[int],
        log_level: str,
        message: str,
        module_name: Optional[str] = None,
        function_name: Optional[str] = None,
        line_number: Optional[int] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> int:
        """Insert a log entry.

        Args:
            sim_run_id: ID of associated simulation run (None for global logs)
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message text
            module_name: Name of the module that generated the log
            function_name: Name of the function that generated the log
            line_number: Line number in source code
            extra_data: Additional context data

        Returns:
            ID of the newly inserted log entry
        """
        extra_json = json.dumps(extra_data) if extra_data else None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO logs
                (sim_run_id, log_level, message, module_name, function_name,
                 line_number, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sim_run_id, log_level, message, module_name, function_name,
                  line_number, extra_json))

            return cursor.lastrowid

    def fetch_logs(
        self,
        sim_run_id: Optional[int] = None,
        log_level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch log entries with optional filtering.

        Args:
            sim_run_id: Filter by simulation run ID (None for all)
            log_level: Filter by log level (None for all)
            limit: Maximum number of logs to return

        Returns:
            List of dictionaries containing log data
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Build query with filters
                query = "SELECT * FROM logs WHERE 1=1"
                params = []

                if sim_run_id is not None:
                    query += " AND sim_run_id = ?"
                    params.append(sim_run_id)

                if log_level is not None:
                    query += " AND log_level = ?"
                    params.append(log_level)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)

                results = []
                for row in cursor.fetchall():
                    data = dict(row)
                    if data['extra_data']:
                        data['extra_data'] = json.loads(data['extra_data'])
                    results.append(data)

                return results
        except sqlite3.Error as e:
            print(f"Error fetching logs: {e}")
            return []

    # ========================================================================
    # Metrics Operations
    # ========================================================================

    def insert_metric(
        self,
        sim_run_id: int,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None
    ) -> int:
        """Insert a simulation metric.

        Args:
            sim_run_id: ID of associated simulation run
            metric_name: Name of the metric (e.g., "BER", "SNR")
            metric_value: Numeric value of the metric
            metric_unit: Unit of measurement (e.g., "dB", "bps")

        Returns:
            ID of the newly inserted metric
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulation_metrics
                (sim_run_id, metric_name, metric_value, metric_unit)
                VALUES (?, ?, ?, ?)
            """, (sim_run_id, metric_name, metric_value, metric_unit))

            return cursor.lastrowid

    def fetch_metrics(self, sim_run_id: int) -> List[Dict[str, Any]]:
        """Fetch all metrics for a simulation run.

        Args:
            sim_run_id: ID of the simulation run

        Returns:
            List of dictionaries containing metric data
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM simulation_metrics
                    WHERE sim_run_id = ?
                    ORDER BY recorded_at DESC
                """, (sim_run_id,))

                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching metrics: {e}")
            return []

    # ========================================================================
    # Statistics and Analytics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics.

        Returns:
            Dictionary containing various statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Total runs
                cursor.execute("SELECT COUNT(*) FROM simulation_runs")
                stats['total_runs'] = cursor.fetchone()[0]

                # Successful runs
                cursor.execute("SELECT COUNT(*) FROM simulation_runs WHERE success = 1")
                stats['successful_runs'] = cursor.fetchone()[0]

                # Failed runs
                stats['failed_runs'] = stats['total_runs'] - stats['successful_runs']

                # Runs by type
                cursor.execute("""
                    SELECT simulation_type, COUNT(*) as count
                    FROM simulation_runs
                    GROUP BY simulation_type
                """)
                stats['runs_by_type'] = {row['simulation_type']: row['count']
                                        for row in cursor.fetchall()}

                # Average execution time
                cursor.execute("""
                    SELECT AVG(execution_time_ms) as avg_time
                    FROM simulation_runs
                    WHERE success = 1 AND execution_time_ms IS NOT NULL
                """)
                result = cursor.fetchone()
                stats['avg_execution_time_ms'] = result['avg_time'] if result['avg_time'] else 0

                return stats
        except sqlite3.Error as e:
            print(f"Error getting statistics: {e}")
            return {}


if __name__ == "__main__":
    print("=== Orbiter-2 Database Access Layer Demo ===\n")

    # Create database manager
    db = DBManager()

    # Insert a simulation run
    print("1. Inserting simulation run...")
    sim_id = db.insert_simulation_run(
        simulation_type="HighOrderModulation",
        start_time=datetime.now(),
        parameters={
            "modulation_order": 16,
            "snr_db": 10.0,
            "num_symbols": 1000
        },
        simulation_name="Test Run 1"
    )
    print(f"   Created simulation run ID: {sim_id}\n")

    # Insert a log entry
    print("2. Inserting log entry...")
    log_id = db.insert_log(
        sim_run_id=sim_id,
        log_level="INFO",
        message="Simulation started successfully",
        module_name="high_order_modulation"
    )
    print(f"   Created log entry ID: {log_id}\n")

    # Update simulation run
    print("3. Updating simulation run...")
    db.update_simulation_run(
        sim_id=sim_id,
        end_time=datetime.now(),
        success=True,
        execution_time_ms=125.5,
        result_summary={"ber": 0.01, "constellation_points": 1000}
    )
    print("   Simulation run updated\n")

    # Insert metrics
    print("4. Inserting metrics...")
    db.insert_metric(sim_id, "BER", 0.01, "ratio")
    db.insert_metric(sim_id, "SNR", 10.0, "dB")
    print("   Metrics inserted\n")

    # Fetch simulation run
    print("5. Fetching simulation run...")
    run_data = db.fetch_simulation_run(sim_id)
    if run_data:
        print(f"   Type: {run_data['simulation_type']}")
        print(f"   Success: {run_data['success']}")
        print(f"   Execution time: {run_data['execution_time_ms']} ms\n")

    # Get statistics
    print("6. Database statistics:")
    stats = db.get_statistics()
    print(f"   Total runs: {stats['total_runs']}")
    print(f"   Successful: {stats['successful_runs']}")
    print(f"   Failed: {stats['failed_runs']}")
    print(f"   Average execution time: {stats['avg_execution_time_ms']:.2f} ms\n")

    print("Database access layer demo complete!")
