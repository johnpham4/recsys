"""
Event Generator for Streaming Recommendation System

Generates synthetic user events (click, purchase, view, add_to_cart)
for testing streaming pipeline and real-time recommendation system.
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse


@dataclass
class EventConfig:
    """Configuration for event generation."""
    num_users: int = 1000
    num_items: int = 5000
    num_events: int = 10000
    event_types: List[str] = None
    time_window_hours: int = 24
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = ["click", "purchase", "view", "add_to_cart"]


class EventGenerator:
    """Generate synthetic user events for streaming pipeline."""
    
    def __init__(self, config: EventConfig):
        self.config = config
        self.event_weights = {
            "click": 0.5,
            "view": 0.3,
            "add_to_cart": 0.15,
            "purchase": 0.05
        }
    
    def generate_user_id(self) -> str:
        """Generate random user ID."""
        return f"user_{random.randint(1, self.config.num_users)}"
    
    def generate_item_id(self) -> str:
        """Generate random item ID."""
        return f"item_{random.randint(1, self.config.num_items)}"
    
    def generate_event_type(self) -> str:
        """Generate event type based on weights."""
        event_types = list(self.event_weights.keys())
        weights = list(self.event_weights.values())
        return random.choices(event_types, weights=weights, k=1)[0]
    
    def generate_timestamp(self, event_index: int) -> str:
        """Generate timestamp within time window."""
        start_time = datetime.now() - timedelta(hours=self.config.time_window_hours)
        time_offset = (event_index / self.config.num_events) * timedelta(hours=self.config.time_window_hours)
        event_time = start_time + time_offset
        return event_time.isoformat()
    
    def generate_event(self, event_index: int) -> Dict[str, Any]:
        """Generate single event."""
        event = {
            "event_id": f"event_{event_index}",
            "user_id": self.generate_user_id(),
            "item_id": self.generate_item_id(),
            "event_type": self.generate_event_type(),
            "timestamp": self.generate_timestamp(event_index),
            "session_id": f"session_{random.randint(1, 100)}",
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
            "page_url": f"/product/{random.randint(1, self.config.num_items)}",
        }
        
        # Add event-specific fields
        if event["event_type"] == "purchase":
            event["quantity"] = random.randint(1, 5)
            event["price"] = round(random.uniform(10.0, 500.0), 2)
        elif event["event_type"] == "add_to_cart":
            event["quantity"] = random.randint(1, 3)
        
        return event
    
    def generate_events(self) -> List[Dict[str, Any]]:
        """Generate all events."""
        events = []
        for i in range(self.config.num_events):
            event = self.generate_event(i)
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        return events
    
    def save_events(self, events: List[Dict[str, Any]], output_file: str):
        """Save events to JSON file."""
        with open(output_file, 'w') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        print(f"Generated {len(events)} events and saved to {output_file}")
    
    def print_statistics(self, events: List[Dict[str, Any]]):
        """Print event statistics."""
        event_type_counts = {}
        for event in events:
            event_type = event["event_type"]
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        print("\n=== Event Statistics ===")
        print(f"Total events: {len(events)}")
        print(f"Unique users: {len(set(e['user_id'] for e in events))}")
        print(f"Unique items: {len(set(e['item_id'] for e in events))}")
        print("\nEvent type distribution:")
        for event_type, count in event_type_counts.items():
            percentage = (count / len(events)) * 100
            print(f"  {event_type}: {count} ({percentage:.2f}%)")


def main():
    """Main function to generate events."""
    parser = argparse.ArgumentParser(description="Generate synthetic user events")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users")
    parser.add_argument("--num-items", type=int, default=5000, help="Number of items")
    parser.add_argument("--num-events", type=int, default=10000, help="Number of events")
    parser.add_argument("--time-window-hours", type=int, default=24, help="Time window in hours")
    parser.add_argument("--output", type=str, default="events.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    config = EventConfig(
        num_users=args.num_users,
        num_items=args.num_items,
        num_events=args.num_events,
        time_window_hours=args.time_window_hours
    )
    
    generator = EventGenerator(config)
    events = generator.generate_events()
    generator.print_statistics(events)
    generator.save_events(events, args.output)


if __name__ == "__main__":
    main()
