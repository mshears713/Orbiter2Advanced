"""
Database Initialization Module for Orbiter-2

Provides functions to create and initialize the SQLite database with the
required schema for simulation metadata, logging, and metrics storage.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional


def get_schema_path() -> Path:
    """Get the path to the schema.sql file.

    Returns:
        Path object pointing to schema.sql in the db directory.
    """
    return Path(__file__).parent / "schema.sql"


def init_db(db_path: Optional[str] = None, force_recreate: bool = False) -> bool:
    """Initialize the SQLite database with the schema.

    Creates the database file if it doesn't exist and executes the schema.sql
    file to set up all tables, indices, views, and triggers. Handles errors
    gracefully and provides detailed feedback.

    Args:
        db_path: Path to the database file. If None, uses 'wireless_sim.db'
                in the db directory.
        force_recreate: If True, drops existing database and recreates from scratch.
                       USE WITH CAUTION - will delete all data!

    Returns:
        True if initialization successful, False otherwise.

    Example:
        >>> init_db('my_database.db')
        Database initialized successfully at my_database.db
        True

    Raises:
        No exceptions raised - errors are logged and False is returned.
    """
    # Determine database path
    if db_path is None:
        db_dir = Path(__file__).parent
        db_path = db_dir / "wireless_sim.db"
    else:
        db_path = Path(db_path)

    try:
        # Handle force recreate
        if force_recreate and db_path.exists():
            print(f"WARNING: Deleting existing database at {db_path}")
            os.remove(db_path)

        # Check if database already exists
        db_exists = db_path.exists()

        # Create parent directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Read schema file
        schema_path = get_schema_path()
        if not schema_path.exists():
            print(f"ERROR: Schema file not found at {schema_path}")
            return False

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Connect to database and execute schema
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()

        # Provide feedback
        if db_exists and not force_recreate:
            print(f"Database schema updated/verified at {db_path}")
        else:
            print(f"Database initialized successfully at {db_path}")

        return True

    except sqlite3.Error as e:
        print(f"SQLite error during initialization: {e}")
        return False
    except IOError as e:
        print(f"File I/O error during initialization: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        return False


def verify_db(db_path: Optional[str] = None) -> bool:
    """Verify that the database exists and has the expected schema.

    Checks for the presence of required tables and returns True if the
    database appears to be properly initialized.

    Args:
        db_path: Path to the database file. If None, uses default location.

    Returns:
        True if database exists and has required tables, False otherwise.

    Example:
        >>> verify_db('wireless_sim.db')
        True
    """
    # Determine database path
    if db_path is None:
        db_dir = Path(__file__).parent
        db_path = db_dir / "wireless_sim.db"
    else:
        db_path = Path(db_path)

    # Check if file exists
    if not db_path.exists():
        print(f"Database file not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check for required tables
        required_tables = [
            'simulation_runs',
            'logs',
            'simulation_metrics',
            'configuration_history',
            'visualization_cache'
        ]

        for table in required_tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            if cursor.fetchone() is None:
                print(f"Required table '{table}' not found in database")
                conn.close()
                return False

        conn.close()
        print(f"Database verification successful: {db_path}")
        return True

    except sqlite3.Error as e:
        print(f"Error during database verification: {e}")
        return False


def get_db_info(db_path: Optional[str] = None) -> dict:
    """Get information about the database.

    Retrieves metadata including table names, row counts, and database size.

    Args:
        db_path: Path to the database file. If None, uses default location.

    Returns:
        Dictionary containing database information, or empty dict on error.

    Example:
        >>> info = get_db_info()
        >>> print(info['num_tables'])
        5
    """
    # Determine database path
    if db_path is None:
        db_dir = Path(__file__).parent
        db_path = db_dir / "wireless_sim.db"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get table information
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Get row counts for each table
        table_counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_counts[table] = cursor.fetchone()[0]

        # Get database size
        db_size = db_path.stat().st_size

        conn.close()

        return {
            'db_path': str(db_path),
            'db_size_bytes': db_size,
            'db_size_kb': db_size / 1024,
            'num_tables': len(tables),
            'tables': tables,
            'table_counts': table_counts
        }

    except sqlite3.Error as e:
        print(f"Error getting database info: {e}")
        return {}


if __name__ == "__main__":
    print("=== Orbiter-2 Database Initialization ===\n")

    # Initialize database
    print("Initializing database...")
    if init_db():
        print("✓ Database initialization successful\n")
    else:
        print("✗ Database initialization failed\n")
        exit(1)

    # Verify database
    print("Verifying database schema...")
    if verify_db():
        print("✓ Database verification successful\n")
    else:
        print("✗ Database verification failed\n")
        exit(1)

    # Display database information
    print("Database Information:")
    info = get_db_info()
    if info:
        print(f"  Path: {info['db_path']}")
        print(f"  Size: {info['db_size_kb']:.2f} KB")
        print(f"  Number of tables: {info['num_tables']}")
        print(f"  Tables:")
        for table in info['tables']:
            count = info['table_counts'].get(table, 0)
            print(f"    - {table}: {count} rows")
    else:
        print("  Could not retrieve database information")

    print("\nDatabase ready for use!")
