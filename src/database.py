"""Database module for storing experiment results and execution traces."""

import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import sys
from pathlib import Path

# Fix import for standalone execution
try:
    from .models import ExecutionTrace, LayerResult
except ImportError:
    from models.execution_trace import ExecutionTrace
    from models.layer_result import LayerResult

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for experiment tracking."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection and create tables."""
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Execution traces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                experiment_id TEXT,
                user_input TEXT NOT NULL,
                attack_label TEXT,
                attack_successful INTEGER,
                violation_detected INTEGER NOT NULL,
                blocked_at_layer TEXT,
                final_output TEXT,
                total_latency_ms REAL,
                configuration TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Layer results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                layer_name TEXT NOT NULL,
                passed INTEGER NOT NULL,
                confidence REAL NOT NULL,
                flags TEXT,
                annotations TEXT,
                risk_score REAL,
                latency_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES execution_traces(request_id)
            )
        """)
        
        # Attack prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attack_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT UNIQUE NOT NULL,
                attack_type TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Benign prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benign_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                expected_behavior TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Experiments metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                configuration TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_execution_trace(self, trace: ExecutionTrace, user_input: Optional[str] = None) -> bool:
        """Save an execution trace to the database."""
        try:
            cursor = self.conn.cursor()
            
            # Use user_input from trace annotations if not provided
            if user_input is None:
                user_input = trace.configuration.get("user_input", "")
            
            # Insert execution trace
            cursor.execute("""
                INSERT INTO execution_traces (
                    request_id, session_id, experiment_id, user_input,
                    attack_label, attack_successful, violation_detected,
                    blocked_at_layer, final_output, total_latency_ms,
                    configuration, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.request_id,
                trace.session_id,
                trace.experiment_id,
                user_input,
                trace.attack_label,
                1 if trace.attack_successful else 0 if trace.attack_successful is not None else None,
                1 if trace.violation_detected else 0,
                trace.blocked_at_layer,
                trace.final_output,
                trace.total_latency_ms,
                json.dumps(trace.configuration),
                trace.timestamp.isoformat()
            ))
            
            # Insert layer results
            # Handle both list and dict formats
            layer_results_items = (
                trace.layer_results.items() if isinstance(trace.layer_results, dict)
                else [(r.layer_name, r) for r in trace.layer_results]
            )
            
            for layer_name, result in layer_results_items:
                cursor.execute("""
                    INSERT INTO layer_results (
                        request_id, layer_name, passed, confidence,
                        flags, annotations, risk_score, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.request_id,
                    result.layer_name,
                    1 if result.passed else 0,
                    result.confidence,
                    json.dumps(result.flags),
                    json.dumps(result.annotations),
                    result.risk_score,
                    result.latency_ms
                ))
            
            self.conn.commit()
            logger.debug(f"Saved execution trace {trace.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving execution trace: {e}")
            self.conn.rollback()
            return False
    
    def save_attack_prompt(self, prompt_id: str, attack_type: str, text: str, source: str = None) -> bool:
        """Save an attack prompt to the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO attack_prompts (prompt_id, attack_type, text, source)
                VALUES (?, ?, ?, ?)
            """, (prompt_id, attack_type, text, source))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving attack prompt: {e}")
            return False
    
    def save_benign_prompt(self, prompt_id: str, text: str, expected_behavior: str = None) -> bool:
        """Save a benign prompt to the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO benign_prompts (prompt_id, text, expected_behavior)
                VALUES (?, ?, ?)
            """, (prompt_id, text, expected_behavior))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving benign prompt: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all results for a specific experiment."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM execution_traces WHERE experiment_id = ?
            ORDER BY timestamp
        """, (experiment_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_attack_success_rate(self, experiment_id: str, attack_type: str = None) -> float:
        """Calculate attack success rate for an experiment."""
        cursor = self.conn.cursor()
        
        if attack_type:
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN attack_successful = 1 THEN 1 ELSE 0 END) as successful
                FROM execution_traces
                WHERE experiment_id = ? AND attack_label = ?
            """, (experiment_id, attack_type))
        else:
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN attack_successful = 1 THEN 1 ELSE 0 END) as successful
                FROM execution_traces
                WHERE experiment_id = ? AND attack_label IS NOT NULL
            """, (experiment_id,))
        
        row = cursor.fetchone()
        if row['total'] == 0:
            return 0.0
        return row['successful'] / row['total']
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
