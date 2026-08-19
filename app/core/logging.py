import json
import logging
import sys

from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(
                record,
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for field in (
            "request_id",
            "method",
            "endpoint",
            "status",
            "duration_ms",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    root.setLevel(
        getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        )
    )

    root.handlers.clear()
    root.addHandler(handler)
