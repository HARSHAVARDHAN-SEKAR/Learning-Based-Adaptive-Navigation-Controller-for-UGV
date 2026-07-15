"""Topic-based message bus — the ROS2 stand-in.

Each node publishes dict messages to named topics and subscribes with
callbacks, exactly mirroring rclpy publishers/subscriptions. Migrating to
ROS2 later = swap this class for rclpy, keep every node's logic unchanged.

Synchronous and deterministic on purpose: a fixed-step scheduler calls
nodes in a defined order, so runs are perfectly reproducible (unlike real
ROS2 — that determinism is a feature for a laboratory).
"""
from collections import defaultdict


class Bus:
    def __init__(self):
        self._subs = defaultdict(list)
        self.latest = {}                 # topic -> last message (for viz/log)

    def subscribe(self, topic, callback):
        self._subs[topic].append(callback)

    def publish(self, topic, msg):
        self.latest[topic] = msg
        for cb in self._subs[topic]:
            cb(msg)
