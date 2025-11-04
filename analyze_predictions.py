#!/usr/bin/env python3
"""
Script to query and analyze prediction logs from the SQLite database
"""

import sqlite3
import json
import argparse
from datetime import datetime
from collections import Counter


def connect_db(db_path):
    """Connect to the SQLite database"""
    return sqlite3.connect(db_path)


def get_all_predictions(conn, limit=None):
    """Get all predictions from the database"""
    cursor = conn.cursor()
    query = "SELECT * FROM predictions ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    cursor.execute(query)

    columns = [description[0] for description in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))

    return results


def get_predictions_by_intent(conn, intent):
    """Get all predictions for a specific intent"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE predicted_intent = ?
        ORDER BY timestamp DESC
    """, (intent,))

    columns = [description[0] for description in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))

    return results


def get_error_predictions(conn):
    """Get all predictions that resulted in errors"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE error IS NOT NULL
        ORDER BY timestamp DESC
    """)

    columns = [description[0] for description in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))

    return results


def get_statistics(conn):
    """Get statistics about predictions"""
    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    # Error count
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE error IS NOT NULL")
    errors = cursor.fetchone()[0]

    # Intent distribution
    cursor.execute("""
        SELECT predicted_intent, COUNT(*) as count
        FROM predictions
        WHERE predicted_intent IS NOT NULL
        GROUP BY predicted_intent
        ORDER BY count DESC
    """)
    intent_counts = dict(cursor.fetchall())

    # Average processing time
    cursor.execute("SELECT AVG(processing_time_ms) FROM predictions WHERE processing_time_ms IS NOT NULL")
    avg_time = cursor.fetchone()[0]

    # Min/Max processing time
    cursor.execute("SELECT MIN(processing_time_ms), MAX(processing_time_ms) FROM predictions WHERE processing_time_ms IS NOT NULL")
    min_time, max_time = cursor.fetchone()

    return {
        "total_predictions": total,
        "errors": errors,
        "success_rate": ((total - errors) / total * 100) if total > 0 else 0,
        "intent_distribution": intent_counts,
        "avg_processing_time_ms": avg_time,
        "min_processing_time_ms": min_time,
        "max_processing_time_ms": max_time
    }


def search_by_text(conn, search_term):
    """Search predictions by request text"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE request_text LIKE ?
        ORDER BY timestamp DESC
    """, (f"%{search_term}%",))

    columns = [description[0] for description in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))

    return results


def export_to_json(conn, output_file):
    """Export all predictions to a JSON file"""
    predictions = get_all_predictions(conn)

    # Parse JSON strings in response_data and predicted_slots
    for pred in predictions:
        if pred.get('response_data'):
            try:
                pred['response_data'] = json.loads(pred['response_data'])
            except:
                pass
        if pred.get('predicted_slots'):
            try:
                pred['predicted_slots'] = json.loads(pred['predicted_slots'])
            except:
                pass

    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Exported {len(predictions)} predictions to {output_file}")


def print_prediction(pred, verbose=False):
    """Print a single prediction in a readable format"""
    print(f"\n{'='*80}")
    print(f"ID: {pred['id']}")
    print(f"Timestamp: {pred['timestamp']}")
    print(f"Request: {pred['request_text']}")
    print(f"Intent: {pred['predicted_intent']}")

    if pred.get('predicted_slots'):
        try:
            slots = json.loads(pred['predicted_slots']) if isinstance(pred['predicted_slots'], str) else pred['predicted_slots']
            print(f"Slots: {json.dumps(slots, indent=2)}")
        except:
            print(f"Slots: {pred['predicted_slots']}")

    if pred.get('processing_time_ms'):
        print(f"Processing Time: {pred['processing_time_ms']:.2f}ms")

    if pred.get('error'):
        print(f"ERROR: {pred['error']}")

    if verbose and pred.get('response_data'):
        try:
            response = json.loads(pred['response_data']) if isinstance(pred['response_data'], str) else pred['response_data']
            print(f"Full Response: {json.dumps(response, indent=2)}")
        except:
            print(f"Full Response: {pred['response_data']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction logs from SQLite database")
    parser.add_argument("--db", default="predictions.db", help="Path to SQLite database")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--all", type=int, metavar="N", help="Show last N predictions (or all if not specified)")
    parser.add_argument("--intent", type=str, help="Show predictions for a specific intent")
    parser.add_argument("--errors", action="store_true", help="Show only error predictions")
    parser.add_argument("--search", type=str, help="Search predictions by text")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export all predictions to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full response data")

    args = parser.parse_args()

    # Connect to database
    conn = connect_db(args.db)

    # Show statistics
    if args.stats:
        stats = get_statistics(conn)
        print("\n" + "="*80)
        print("PREDICTION STATISTICS")
        print("="*80)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Errors: {stats['errors']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"\nIntent Distribution:")
        for intent, count in stats['intent_distribution'].items():
            print(f"  {intent}: {count}")
        if stats['avg_processing_time_ms']:
            print(f"\nProcessing Time:")
            print(f"  Average: {stats['avg_processing_time_ms']:.2f}ms")
            print(f"  Min: {stats['min_processing_time_ms']:.2f}ms")
            print(f"  Max: {stats['max_processing_time_ms']:.2f}ms")

    # Show all predictions
    elif args.all is not None:
        predictions = get_all_predictions(conn, limit=args.all if args.all > 0 else None)
        print(f"\nShowing {len(predictions)} predictions:")
        for pred in predictions:
            print_prediction(pred, args.verbose)

    # Show predictions by intent
    elif args.intent:
        predictions = get_predictions_by_intent(conn, args.intent)
        print(f"\nShowing {len(predictions)} predictions for intent '{args.intent}':")
        for pred in predictions:
            print_prediction(pred, args.verbose)

    # Show errors
    elif args.errors:
        predictions = get_error_predictions(conn)
        print(f"\nShowing {len(predictions)} error predictions:")
        for pred in predictions:
            print_prediction(pred, args.verbose)

    # Search by text
    elif args.search:
        predictions = search_by_text(conn, args.search)
        print(f"\nShowing {len(predictions)} predictions matching '{args.search}':")
        for pred in predictions:
            print_prediction(pred, args.verbose)

    # Export to JSON
    elif args.export:
        export_to_json(conn, args.export)

    # Default: show statistics
    else:
        stats = get_statistics(conn)
        print("\n" + "="*80)
        print("PREDICTION STATISTICS")
        print("="*80)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Errors: {stats['errors']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"\nIntent Distribution:")
        for intent, count in stats['intent_distribution'].items():
            print(f"  {intent}: {count}")
        if stats['avg_processing_time_ms']:
            print(f"\nProcessing Time:")
            print(f"  Average: {stats['avg_processing_time_ms']:.2f}ms")
            print(f"  Min: {stats['min_processing_time_ms']:.2f}ms")
            print(f"  Max: {stats['max_processing_time_ms']:.2f}ms")
        print(f"\nUse --help to see available options")

    conn.close()


if __name__ == "__main__":
    main()
