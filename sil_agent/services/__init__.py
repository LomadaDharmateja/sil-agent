"""Cross-cutting services: model routing, retries, replay, locking, cost.

Everything in here sits between the agent and the outside world. The agent asks
for a completion; it never learns which provider answered, how many times the
request was retried, or whether the answer came from a cache.
"""
