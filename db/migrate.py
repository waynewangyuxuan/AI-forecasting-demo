"""
Database migration and initialization script.

This module handles database schema creation and migration.
It is idempotent and can be run multiple times safely.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_schema_path() -> Path:
    """
    Get the path to the schema.sql file.

    Returns:
        Path: Path to schema.sql
    """
    return Path(__file__).parent / "schema.sql"


def get_db_path(db_path: Optional[str] = None) -> Path:
    """
    Get the path to the database file.

    Args:
        db_path: Optional custom database path

    Returns:
        Path: Path to database file
    """
    if db_path:
        return Path(db_path)

    # Default path from project root
    project_root = Path(__file__).parent.parent
    return project_root / "data" / "forecast.db"


def ensure_data_directory(db_path: Path) -> None:
    """
    Ensure the data directory exists.

    Args:
        db_path: Path to the database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)


def read_schema() -> str:
    """
    Read the schema.sql file.

    Returns:
        str: SQL schema content

    Raises:
        FileNotFoundError: If schema.sql is not found
    """
    schema_path = get_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    return schema_path.read_text()


def execute_schema(db_path: Path, schema_sql: str) -> None:
    """
    Execute the schema SQL against the database.

    This function is idempotent - it uses CREATE TABLE IF NOT EXISTS
    and CREATE INDEX IF NOT EXISTS, so it can be run multiple times safely.

    Args:
        db_path: Path to the database file
        schema_sql: SQL schema to execute

    Raises:
        sqlite3.Error: If there's an error executing the schema
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute the schema (contains multiple statements)
        cursor.executescript(schema_sql)

        conn.commit()
        print(f"Successfully executed schema on database: {db_path}")

    except sqlite3.Error as e:
        print(f"Error executing schema: {e}")
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()


def verify_schema(db_path: Path) -> bool:
    """
    Verify that all expected tables exist in the database.

    Args:
        db_path: Path to the database file

    Returns:
        bool: True if all tables exist, False otherwise
    """
    expected_tables = [
        'Questions',
        'Runs',
        'SearchQueries',
        'SearchResults',
        'Documents',
        'Events',
        'Embeddings',
        'EventClusters',
        'Timeline',
        'Forecasts',
        'RunMetrics',
        'Errors'
    ]

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        # Check if all expected tables exist
        missing_tables = set(expected_tables) - existing_tables

        if missing_tables:
            print(f"Missing tables: {missing_tables}")
            return False

        print("All expected tables exist")
        return True

    except sqlite3.Error as e:
        print(f"Error verifying schema: {e}")
        return False

    finally:
        if conn:
            conn.close()


def verify_indices(db_path: Path) -> bool:
    """
    Verify that expected indices exist in the database.

    Args:
        db_path: Path to the database file

    Returns:
        bool: True if indices exist, False otherwise
    """
    expected_indices = [
        'idx_documents_content_hash',
        'idx_events_event_time',
        'idx_timeline_run_id',
        'idx_runs_status',
        'idx_runs_question_id',
        'idx_search_queries_run_id',
        'idx_documents_run_id',
        'idx_events_document_id',
        'idx_forecasts_run_id',
        'idx_errors_run_id'
    ]

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing_indices = {row[0] for row in cursor.fetchall() if row[0] is not None}

        # Check if all expected indices exist
        missing_indices = set(expected_indices) - existing_indices

        if missing_indices:
            print(f"Missing indices: {missing_indices}")
            return False

        print("All expected indices exist")
        return True

    except sqlite3.Error as e:
        print(f"Error verifying indices: {e}")
        return False

    finally:
        if conn:
            conn.close()


def get_database_info(db_path: Path) -> dict:
    """
    Get information about the database.

    Args:
        db_path: Path to the database file

    Returns:
        dict: Database information including tables and their row counts
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        # Get row count for each table
        table_counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_counts[table] = count

        return {
            'path': str(db_path),
            'tables': tables,
            'table_counts': table_counts,
            'total_tables': len(tables)
        }

    except sqlite3.Error as e:
        print(f"Error getting database info: {e}")
        return {}

    finally:
        if conn:
            conn.close()


def migrate(db_path: Optional[str] = None, verbose: bool = True) -> bool:
    """
    Run database migration.

    This is the main entry point for database initialization and migration.
    It is idempotent and can be run multiple times safely.

    Args:
        db_path: Optional custom database path (defaults to data/forecast.db)
        verbose: Whether to print verbose output

    Returns:
        bool: True if migration successful, False otherwise
    """
    try:
        # Get paths
        db_file = get_db_path(db_path)

        if verbose:
            print(f"Starting database migration...")
            print(f"Database path: {db_file}")

        # Ensure data directory exists
        ensure_data_directory(db_file)

        # Read schema
        if verbose:
            print("Reading schema...")
        schema_sql = read_schema()

        # Execute schema
        if verbose:
            print("Executing schema...")
        execute_schema(db_file, schema_sql)

        # Verify schema
        if verbose:
            print("Verifying schema...")

        tables_ok = verify_schema(db_file)
        indices_ok = verify_indices(db_file)

        if not tables_ok or not indices_ok:
            print("Schema verification failed!")
            return False

        # Print database info
        if verbose:
            print("\nDatabase migration completed successfully!")
            print("\nDatabase Information:")
            info = get_database_info(db_file)
            print(f"  Path: {info.get('path', 'unknown')}")
            print(f"  Total tables: {info.get('total_tables', 0)}")

            if info.get('table_counts'):
                print("\nTable row counts:")
                for table, count in sorted(info['table_counts'].items()):
                    print(f"  {table}: {count}")

        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def reset_database(db_path: Optional[str] = None, confirm: bool = False) -> bool:
    """
    Reset the database by deleting it and recreating it.

    WARNING: This will delete all data in the database!

    Args:
        db_path: Optional custom database path
        confirm: Must be True to actually delete the database

    Returns:
        bool: True if reset successful, False otherwise
    """
    if not confirm:
        print("ERROR: confirm=True required to reset database")
        return False

    try:
        db_file = get_db_path(db_path)

        # Delete the database file if it exists
        if db_file.exists():
            print(f"Deleting database: {db_file}")
            db_file.unlink()

        # Run migration to recreate
        print("Recreating database...")
        return migrate(db_path=str(db_file) if db_path else None)

    except Exception as e:
        print(f"Reset failed: {e}")
        return False


if __name__ == "__main__":
    """
    Run migration when executed as a script.

    Usage:
        python db/migrate.py                    # Run migration with default path
        python db/migrate.py <custom_path>      # Run migration with custom path
        python db/migrate.py --reset            # Reset database (requires confirmation)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Database migration script")
    parser.add_argument(
        'db_path',
        nargs='?',
        help='Path to database file (default: data/forecast.db)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (WARNING: deletes all data)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    if args.reset:
        print("WARNING: This will delete all data in the database!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            success = reset_database(args.db_path, confirm=True)
        else:
            print("Reset cancelled")
            success = False
    else:
        success = migrate(args.db_path, verbose=not args.quiet)

    sys.exit(0 if success else 1)
