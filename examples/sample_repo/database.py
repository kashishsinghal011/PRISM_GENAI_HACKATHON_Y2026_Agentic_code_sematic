import sqlite3


def init_connection(path):
    """Initialize the database connection and create the tables if missing."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, hash TEXT)")
    return conn


def query_user(conn, name):
    return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
