from enum import Enum


class AppState(Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    VIOLATION_TRIGGERED = "VIOLATION_TRIGGERED"
    ERROR = "ERROR"
