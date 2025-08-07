"""Logging configuration for milvus-ingest using loguru.

This module provides a centralized logging configuration following best practices:
- Structured logging with context
- Different log levels for development and production
- File rotation and retention
- Performance-aware logging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Default log directory
DEFAULT_LOG_DIR = Path.home() / ".milvus-ingest" / "logs"

# Log format for different environments
VERBOSE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

NORMAL_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

PRODUCTION_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logging(
    verbose: bool = False,
    log_file: Path | str | None = None,
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    rich_console: Any = None,
) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging with detailed format
        log_file: Custom log file path. If None, uses default location
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
        rich_console: Rich Console instance to use for logging (for progress bar compatibility)
    """
    # Remove default logger first
    logger.remove()

    # Console logging setup
    console_format = VERBOSE_FORMAT  # Always use verbose format for better debugging
    console_level = "DEBUG" if verbose else log_level

    # If rich_console is provided, use it for logging to avoid conflicts with progress bars
    if rich_console is not None:
        logger.add(
            rich_console.print,
            format=console_format,
            level=console_level,
            colorize=True,
            backtrace=verbose,
            diagnose=verbose,
        )
    else:
        logger.add(
            sys.stderr,
            format=console_format,
            level=console_level,
            colorize=True,
            backtrace=verbose,
            diagnose=verbose,
        )

    # File logging setup
    if enable_file_logging:
        if log_file is None:
            DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = DEFAULT_LOG_DIR / "milvus-ingest.log"

        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=PRODUCTION_FORMAT,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Async logging for better performance
        )

    # Add context enricher
    logger.configure(
        extra={
            "component": "milvus-ingest",
            "version": "0.1.0",
        }
    )


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


def log_performance(func_name: str, duration: float, **kwargs: Any) -> None:
    """Log performance metrics.

    Args:
        func_name: Name of the function being measured
        duration: Duration in seconds
        **kwargs: Additional context data
    """
    logger.info(
        "Performance: {func} took {duration:.3f}s",
        func=func_name,
        duration=duration,
        extra=kwargs,
    )


def log_error_with_context(
    error: Exception, context: dict[str, Any] | None = None
) -> None:
    """Log error with additional context.

    Args:
        error: The exception that occurred
        context: Additional context information
    """
    context = context or {}
    logger.error(
        "Error occurred: {error}",
        error=str(error),
        error_type=type(error).__name__,
        extra=context,
    )


def log_data_operation(
    operation: str,
    rows: int,
    batch_size: int | None = None,
    schema_name: str | None = None,
    **kwargs: Any,
) -> None:
    """Log data generation/processing operations.

    Args:
        operation: Type of operation (e.g., 'generate', 'write', 'validate')
        rows: Number of rows involved
        batch_size: Batch size if applicable
        schema_name: Schema name if applicable
        **kwargs: Additional operation-specific data
    """
    logger.info(
        "Data operation: {operation} - {rows} rows",
        operation=operation,
        rows=rows,
        batch_size=batch_size,
        schema_name=schema_name,
        extra=kwargs,
    )


def log_schema_validation(
    schema_name: str,
    fields_count: int,
    validation_result: str = "success",
    errors: list[str] | None = None,
) -> None:
    """Log schema validation results.

    Args:
        schema_name: Name of the schema being validated
        fields_count: Number of fields in the schema
        validation_result: Result of validation ('success' or 'failed')
        errors: List of validation errors if any
    """
    if validation_result == "success":
        logger.info(
            "Schema validation successful: {schema} with {fields} fields",
            schema=schema_name,
            fields=fields_count,
        )
    else:
        logger.error(
            "Schema validation failed: {schema} - {errors}",
            schema=schema_name,
            errors=errors or [],
            fields=fields_count,
        )


def disable_logging() -> None:
    """Disable all logging (useful for tests)."""
    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")


# Create a default logger instance
default_logger = get_logger(__name__)
