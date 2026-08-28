# Morning schedule guard

The morning workflow is scheduled for 02:45 UTC (08:15 IST). GitHub Actions schedules can be delayed by platform load, so `morning.py` also enforces an 08:05-08:25 IST execution window for scheduled events. A delayed scheduled event exits without generating or resending a prediction. Manual `workflow_dispatch` runs remain available for testing.
