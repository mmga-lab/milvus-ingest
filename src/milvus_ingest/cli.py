"""Command-line interface for milvus-ingest.

Usage::

    # Data generation
    milvus-ingest generate --schema schema.json --rows 1000
    milvus-ingest generate --builtin quickstart --rows 100 --preview

    # Schema management
    milvus-ingest schema list
    milvus-ingest schema show quickstart
    milvus-ingest schema add my_schema schema.json

    # Utilities
    milvus-ingest clean --yes

The script is installed as ``milvus-ingest`` when the package is
installed via PDM/pip.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

from .logging_config import (
    get_logger,
    log_error_with_context,
    log_performance,
    setup_logging,
)
from .milvus_inserter import MilvusInserter
from .models import get_schema_help, validate_schema_data
from .report.models import ReportConfig

# CSV report generator removed - using LLMGenerator directly
from .rich_display import (
    display_error,
    display_schema_details,
    display_schema_list,
    display_schema_preview,
    display_schema_validation,
    display_success,
)
from .schema_manager import get_schema_manager
from .uploader import S3Uploader, parse_s3_url

_OUTPUT_FORMATS = {"parquet", "json"}

# Default directory for generated data files: ~/.milvus-ingest/data
DEFAULT_DATA_DIR = Path.home() / ".milvus-ingest" / "data"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging with detailed debug information.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool = False) -> None:
    """Generate mock data for Milvus with schema management."""
    # Setup logging first
    setup_logging(verbose=verbose, log_level="DEBUG" if verbose else "INFO")
    logger = get_logger(__name__)

    # Store verbose in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    logger.info(
        "Starting milvus-ingest CLI",
        extra={"verbose": verbose},
    )


@main.command()
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to schema JSON/YAML file.",
)
@click.option(
    "--builtin",
    "builtin_schema",
    help="Use a built-in schema (e.g., 'ecommerce', 'documents').",
)
@click.option(
    "--total-rows",
    "-r",
    default=1000,
    show_default=True,
    type=int,
    help="Total number of rows to generate.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    default="parquet",
    show_default=True,
    type=click.Choice(sorted(_OUTPUT_FORMATS)),
    help="Output file format.",
)
@click.option(
    "-p",
    "--preview",
    is_flag=True,
    help="Print first 5 rows to terminal after generation.",
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory path (will create directory with data files + meta.json). Default: <collection_name>/",
)
@click.option("--seed", type=int, help="Random seed for reproducibility.")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate schema without generating data.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar display for large datasets.",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=50000,
    show_default=True,
    type=int,
    help="Number of rows to generate and process in each batch (larger batches = better performance).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force overwrite output directory if it exists.",
)
@click.option(
    "--file-size",
    "file_size",
    type=str,
    help="File size limit. Can be exact size ('10GB', '200MB') or max size in MB (e.g., '256'). Default: 256MB.",
)
@click.option(
    "--file-count",
    "file_count",
    type=int,
    help="Target number of files. If used with --file-size, total rows will be calculated (ignores --total-rows).",
)
@click.option(
    "--partitions",
    "num_partitions",
    type=int,
    help="Number of partitions to simulate. Requires partition key field in schema.",
)
@click.option(
    "--shards",
    "num_shards",
    type=int,
    help="Number of shards (VChannels) to simulate. Data distributed based on primary key hash.",
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    help="Number of parallel worker processes for file generation. Default: CPU count.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging output.",
)
@click.pass_context
def generate(
    ctx: click.Context,
    schema_path: Path | None = None,
    builtin_schema: str | None = None,
    total_rows: int = 1000,
    output_format: str = "parquet",
    output_path: Path | None = None,
    seed: int | None = None,
    preview: bool = False,
    validate_only: bool = False,
    no_progress: bool = False,
    batch_size: int = 50000,
    force: bool = False,
    file_size: str | None = None,
    file_count: int | None = None,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
) -> None:
    """Generate high-performance mock data from schema using optimized vectorized operations.

    This tool is optimized for large-scale data generation with NumPy vectorized operations
    (automatically utilizing multiple CPU cores through optimized BLAS), efficient memory
    management, and high-speed file I/O. Uses intelligent file partitioning for maximum
    performance on large datasets.

    Output is always a directory containing data files and collection schema.json file.
    """
    verbose = ctx.obj["verbose"]
    logger = get_logger(__name__)

    logger.info(
        "Starting data generation",
        extra={
            "total_rows": total_rows,
            "format": output_format,
            "verbose": verbose,
            "batch_size": batch_size,
            "seed": seed,
        },
    )

    # Validate argument combinations
    provided_args = [
        ("--schema", schema_path is not None),
        ("--builtin", builtin_schema is not None),
    ]
    provided_count = sum(provided for _, provided in provided_args)
    if provided_count == 0:
        click.echo("One of --schema or --builtin is required", err=True)
        raise SystemExit(1)
    if provided_count > 1:
        provided_names = [name for name, provided in provided_args if provided]
        click.echo(f"Cannot use {', '.join(provided_names)} together", err=True)
        raise SystemExit(1)

    # Handle built-in or custom schema
    if builtin_schema:
        logger.debug("Loading builtin schema", schema_name=builtin_schema)
        manager = get_schema_manager()
        try:
            # Try to load from schema manager (supports both built-in and custom)
            schema_data = manager.load_schema(builtin_schema)
            schema_type = (
                "built-in" if manager.is_builtin_schema(builtin_schema) else "custom"
            )
            logger.info(
                "Schema loaded successfully",
                schema_name=builtin_schema,
                schema_type=schema_type,
            )
            click.echo(f"✓ Loaded {schema_type} schema: {builtin_schema}")

            # Create temporary file for the schema
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
                json.dump(schema_data, tmp)
                schema_path = Path(tmp.name)
                logger.debug(
                    "Created temporary schema file", temp_file=str(schema_path)
                )
        except ValueError as e:
            log_error_with_context(
                e, {"schema_name": builtin_schema, "operation": "load_schema"}
            )
            click.echo(f"✗ Error with schema: {e}", err=True)
            click.echo("Available schemas:", err=True)
            all_schemas = manager.list_all_schemas()
            for schema_id in sorted(all_schemas.keys()):
                schema_type = (
                    "built-in" if manager.is_builtin_schema(schema_id) else "custom"
                )
                click.echo(f"  - {schema_id} ({schema_type})", err=True)
            raise SystemExit(1) from e

    # builtin_schema case is already handled above, schema_path is set

    # Validate schema if --validate-only flag is used
    if validate_only:
        try:
            import yaml
            from pydantic import ValidationError

            assert schema_path is not None
            content = schema_path.read_text("utf-8")
            if schema_path.suffix.lower() in {".yaml", ".yml"}:
                schema_data = yaml.safe_load(content)
            else:
                schema_data = json.loads(content)

            validated_schema = validate_schema_data(schema_data)

            # Prepare validation info for rich display
            validation_info: dict[str, Any] = {}
            if isinstance(validated_schema, list):
                validation_info["fields_count"] = len(validated_schema)
                validation_info["fields"] = [
                    {
                        "name": field.name,
                        "type": field.type,
                        "is_primary": field.is_primary,
                    }
                    for field in validated_schema
                ]
            else:
                validation_info["collection_name"] = validated_schema.collection_name
                validation_info["fields_count"] = len(validated_schema.fields)
                validation_info["fields"] = [
                    {
                        "name": field.name,
                        "type": field.type,
                        "is_primary": field.is_primary,
                    }
                    for field in validated_schema.fields
                ]

            # Get schema ID for display
            if schema_path:
                schema_id = schema_path.stem
            elif builtin_schema:
                schema_id = builtin_schema
            else:
                schema_id = "schema"

            display_schema_validation(schema_id, validation_info)
            return
        except ValidationError as e:
            click.echo("✗ Schema validation failed:", err=True)
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                click.echo(f"  • {loc}: {error['msg']}", err=True)
            click.echo(
                f"\nFor help with schema format, run: {sys.argv[0]} --schema-help",
                err=True,
            )
            raise SystemExit(1) from e
        except Exception as e:
            click.echo(f"✗ Error reading schema file: {e}", err=True)
            raise SystemExit(1) from e

    assert schema_path is not None

    # Load and validate schema for preview
    logger.debug("Loading schema for preview", schema_file=str(schema_path))
    try:
        import yaml

        content = schema_path.read_text("utf-8")
        if schema_path.suffix.lower() in {".yaml", ".yml"}:
            schema_data = yaml.safe_load(content)
        else:
            schema_data = json.loads(content)

        # Validate schema
        validated_schema = validate_schema_data(schema_data)

        # Extract fields and collection info
        if isinstance(validated_schema, list):
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema
            ]
            collection_name = None
            schema_display_name = schema_path.stem
        else:
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema.fields
            ]
            collection_name = validated_schema.collection_name
            schema_display_name = collection_name or schema_path.stem

        # Calculate effective total rows for display
        # Check if both file_count and file_size are specified (rows will be calculated)
        effective_display_rows: int | str = (
            0 if file_count and file_size else total_rows
        )

        # If both file_count and file_size are specified, estimate total rows for display
        if file_count and file_size and effective_display_rows == 0:
            try:
                # Import here to avoid circular imports
                from .optimized_writer import (
                    _enhanced_estimate_row_size_from_sample,
                    _parse_file_size,
                )

                # Quick estimation for display purposes
                sample_size = min(1000, 5000)  # Smaller sample for speed
                estimated_bytes_per_row, _ = _enhanced_estimate_row_size_from_sample(
                    fields,
                    sample_size,
                    0,
                    output_format,
                    {"fields": fields},
                    seed,
                    num_iterations=1,
                )

                # Parse file size and calculate rows
                file_size_bytes = _parse_file_size(file_size)
                rows_per_file = int(file_size_bytes / estimated_bytes_per_row)
                estimated_total_rows = rows_per_file * file_count

                effective_display_rows = f"~{estimated_total_rows:,} (calculated)"
            except Exception:
                effective_display_rows = f"calculated ({file_count} × {file_size})"

        # Display schema preview
        generation_config = {
            "total_rows": effective_display_rows,
            "batch_size": batch_size,
            "seed": seed,
            "format": output_format,
        }

        display_schema_preview(
            schema_name=schema_display_name,
            collection_name=collection_name,
            fields=fields,
            generation_config=generation_config,
        )

        # Handle preview mode - generate a few rows and display them
        if preview:
            import json as json_mod
            import tempfile as tempfile_mod

            import pandas as pd

            from .optimized_writer import generate_data_optimized

            with tempfile_mod.TemporaryDirectory() as temp_dir:
                # Generate 5 rows for preview
                preview_rows = 5
                temp_output = Path(temp_dir) / "preview"
                temp_output.mkdir(exist_ok=True)

                # Create a temporary schema file for the generator
                temp_schema_file = Path(temp_dir) / "temp_schema.json"
                temp_schema = {
                    "collection_name": collection_name or "preview",
                    "fields": fields,
                }
                with open(temp_schema_file, "w") as f:
                    json_mod.dump(temp_schema, f)

                try:
                    files_created, actual_preview_rows = generate_data_optimized(
                        schema_path=temp_schema_file,
                        total_rows=preview_rows,
                        output_dir=temp_output,
                        batch_size=preview_rows,
                        format="parquet",  # Always use parquet for preview
                        num_partitions=num_partitions,
                        num_shards=num_shards,
                        seed=seed,
                        file_size=None,  # Don't use size controls for preview
                        file_count=None,
                        skip_validation=True,  # Skip validation for preview
                    )

                    if files_created:
                        # Read and display the generated data
                        parquet_file = files_created[0]  # First file
                        df = pd.read_parquet(parquet_file)

                        from rich.console import Console
                        from rich.table import Table

                        console = Console()

                        console.print(
                            "\n[bold green]Preview (top 5 rows):[/bold green]"
                        )

                        # Create table for preview
                        table = Table(show_header=True, header_style="bold magenta")

                        # Add columns
                        for col in df.columns:
                            table.add_column(col, style="cyan", no_wrap=True)

                        # Add rows (limit to 5)
                        for i in range(min(len(df), 5)):
                            row_values = []
                            for col in df.columns:
                                val = df.iloc[i][col]
                                # Handle different data types for display
                                if isinstance(val, list):
                                    # For arrays/lists, show first few items
                                    if len(val) > 3:
                                        display_val = (
                                            f"[{', '.join(map(str, val[:3]))}...]"
                                        )
                                    else:
                                        display_val = str(val)
                                elif isinstance(val, dict):
                                    display_val = "{...}" if val else "{}"
                                else:
                                    display_val = str(val)
                                    # Truncate long strings
                                    if len(display_val) > 30:
                                        display_val = display_val[:27] + "..."
                                row_values.append(display_val)
                            table.add_row(*row_values)

                        console.print(table)
                        console.print()

                except Exception as e:
                    logger.error("Failed to generate preview data", error=str(e))
                    from rich.console import Console

                    console = Console()
                    console.print(f"[red]Error generating preview: {e}[/red]")

            return

    except Exception as e:
        logger.error("Failed to load or validate schema for preview", error=str(e))
        click.echo(f"✗ Error loading schema: {e}", err=True)
        raise SystemExit(1) from e

    logger.info(
        "Starting data generation",
        schema_file=str(schema_path),
        rows=total_rows,
        batch_size=batch_size,
    )
    if output_path is None:
        # derive default file name using default data directory (~/.milvus-ingest/data)
        try:
            content = schema_path.read_text("utf-8")
            data = json.loads(content)
            schema_collection_name: str | None = (
                data.get("collection_name") if isinstance(data, dict) else None
            )
        except Exception:
            schema_collection_name = None
        base_name = schema_collection_name or schema_path.stem
        # Ensure target directory exists
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_DATA_DIR / base_name
        logger.debug(
            "Output path determined", output_path=str(output_path), base_name=base_name
        )

    # Handle force cleanup of output directory
    if force and output_path.exists():
        import shutil

        logger.info(
            "Force cleanup enabled, removing existing output directory",
            output_path=str(output_path),
        )
        shutil.rmtree(output_path)
        logger.debug("Output directory removed successfully")

    # Handle file-count + file-size parameter conflict
    # When both are specified, total_rows should be calculated by the generator
    effective_total_rows = total_rows
    if file_count and file_size:
        logger.info(
            "Both --file-count and --file-size specified. Total rows will be calculated based on file sizes."
        )
        # Use a special value to indicate rows should be calculated
        effective_total_rows = 0  # This tells the generator to calculate rows

    # Use high-performance data generation (default and only mode)
    logger.info(
        "Starting high-performance data generation",
        output_dir=str(output_path),
        format=output_format,
        rows=effective_total_rows if effective_total_rows > 0 else "calculated",
    )
    try:
        # High-performance parallel generator (default and only mode)
        logger.info("Using vectorized high-performance generator")

        # If progress bar will be shown, suppress INFO logs before starting generation
        if not no_progress:
            logger.info("Starting high-performance vectorized generator")
            setup_logging(verbose=verbose, log_level="WARNING")

        actual_total_rows = _save_with_high_performance_generator(
            schema_path,
            effective_total_rows,
            output_path,
            output_format,
            batch_size=batch_size,
            seed=seed,
            show_progress=not no_progress,
            file_size=file_size,
            num_partitions=num_partitions,
            num_shards=num_shards,
            file_count=file_count,
            num_workers=num_workers,
            verbose=verbose,
        )
        # Calculate directory size for logging (output is always a directory now)
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        file_size_mb = total_size / (1024 * 1024)

        logger.info(
            "Data generation completed successfully",
            rows=actual_total_rows,
            output_file=str(output_path),
            file_size_mb=file_size_mb,
        )

        # Output is always a directory now
        click.echo(
            f"Saved {actual_total_rows} rows to directory {output_path.resolve()}"
        )
    except Exception as e:
        log_error_with_context(
            e,
            {
                "operation": "data_generation",
                "rows": total_rows,
                "output_path": str(output_path),
                "batch_size": batch_size,
            },
        )
        raise


@main.group()
def schema() -> None:
    """Manage schemas (built-in and custom)."""
    pass


@schema.command("list")
def list_schemas() -> None:
    """List all available schemas."""
    manager = get_schema_manager()
    all_schemas = manager.list_all_schemas()

    # Separate built-in and custom schemas
    builtin_schemas = {
        k: v for k, v in all_schemas.items() if manager.is_builtin_schema(k)
    }
    custom_schemas = {
        k: v for k, v in all_schemas.items() if not manager.is_builtin_schema(k)
    }

    if builtin_schemas:
        display_schema_list(builtin_schemas, "Built-in Schemas")

    if custom_schemas:
        display_schema_list(custom_schemas, "Custom Schemas")

    if not builtin_schemas and not custom_schemas:
        click.echo("No schemas found.")

    click.echo(
        "\nFor detailed schema information: milvus-ingest schema show <schema_id>"
    )


@schema.command()
@click.argument("schema_id")
def show(schema_id: str) -> None:
    """Show details of a specific schema."""
    manager = get_schema_manager()
    try:
        info = manager.get_schema_info(schema_id)
        if not info:
            display_error(
                f"Schema '{schema_id}' not found.",
                "Use 'milvus-ingest schema list' to see available schemas.",
            )
            raise SystemExit(1)

        schema_data = manager.load_schema(schema_id)
        is_builtin = manager.is_builtin_schema(schema_id)

        display_schema_details(schema_id, info, schema_data, is_builtin)

    except Exception as e:
        display_error(f"Error showing schema: {e}")
        raise SystemExit(1) from e


@schema.command()
@click.argument("schema_id")
@click.argument(
    "schema_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def add(schema_id: str, schema_file: Path) -> None:
    """Add a custom schema."""
    manager = get_schema_manager()
    try:
        # Load and validate schema
        try:
            import yaml

            content = schema_file.read_text("utf-8")
            if schema_file.suffix.lower() in {".yaml", ".yml"}:
                schema_data = yaml.safe_load(content)
            else:
                schema_data = json.loads(content)
        except Exception as e:
            display_error(f"Error reading schema file: {e}")
            raise SystemExit(1) from e

        # Get additional info from user
        description = click.prompt(
            "Schema description (optional)", default="", show_default=False
        )
        use_cases_input = click.prompt(
            "Use cases (comma-separated, optional)", default="", show_default=False
        )
        use_cases = (
            [uc.strip() for uc in use_cases_input.split(",") if uc.strip()]
            if use_cases_input
            else []
        )

        manager.add_schema(schema_id, schema_data, description, use_cases)

        details = f"Description: {description or 'N/A'}\n"
        details += f"Use cases: {', '.join(use_cases) if use_cases else 'N/A'}\n"
        details += f"Usage: milvus-ingest schema show {schema_id}"

        display_success(f"Added custom schema: {schema_id}", details)

    except ValueError as e:
        display_error(f"Error adding schema: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        display_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@schema.command()
@click.argument("schema_id")
def remove(schema_id: str) -> None:
    """Remove a custom schema."""
    manager = get_schema_manager()
    try:
        if not manager.schema_exists(schema_id):
            display_error(f"Schema '{schema_id}' does not exist.")
            raise SystemExit(1)

        if manager.is_builtin_schema(schema_id):
            display_error(f"Cannot remove built-in schema '{schema_id}'.")
            raise SystemExit(1)

        if click.confirm(f"Are you sure you want to remove schema '{schema_id}'?"):
            manager.remove_schema(schema_id)
            display_success(f"Removed custom schema: {schema_id}")
        else:
            click.echo("Cancelled.")

    except ValueError as e:
        display_error(f"Error removing schema: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        display_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@schema.command()
def help() -> None:
    """Show schema format help and examples."""
    click.echo(get_schema_help())


@main.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-confirm all prompts and proceed without interactive confirmation.",
)
def clean(yes: bool = False) -> None:
    """Clean up generated output files."""
    logger = get_logger(__name__)
    _handle_clean_command(yes, logger)


@main.command()
@click.option(
    "--local-path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    help="Local path to data directory to upload",
)
@click.option(
    "--s3-path",
    required=True,
    type=str,
    help="S3 destination path (e.g., s3://bucket/prefix/)",
)
@click.option(
    "--endpoint-url",
    help="S3-compatible endpoint URL (e.g., http://localhost:9000 for MinIO)",
)
@click.option(
    "--access-key-id",
    envvar="AWS_ACCESS_KEY_ID",
    help="AWS access key ID (can also use AWS_ACCESS_KEY_ID env var)",
)
@click.option(
    "--secret-access-key",
    envvar="AWS_SECRET_ACCESS_KEY",
    help="AWS secret access key (can also use AWS_SECRET_ACCESS_KEY env var)",
)
@click.option(
    "--region",
    default="us-east-1",
    help="AWS region name (default: us-east-1)",
)
@click.option(
    "--no-verify-ssl",
    is_flag=True,
    help="Disable SSL certificate verification (useful for self-signed certs)",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar during upload",
)
@click.option(
    "--use-boto3",
    is_flag=True,
    help="Use boto3 instead of AWS CLI for uploads (AWS CLI is default and more reliable)",
)
@click.option(
    "--use-mc",
    is_flag=True,
    help="Use MinIO Client (mc) CLI for uploads (prioritized over AWS CLI and boto3)",
)
def upload(
    local_path: Path,
    s3_path: str,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str = "us-east-1",
    no_verify_ssl: bool = False,
    no_progress: bool = False,
    use_boto3: bool = False,
    use_mc: bool = False,
) -> None:
    """Upload generated data files to S3/MinIO.

    \b
    Examples:
        # Upload to AWS S3 (using AWS CLI by default)
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/

        # Upload to MinIO using mc CLI
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ --endpoint-url http://localhost:9000 --use-mc

        # Upload using boto3 (legacy mode)
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ --use-boto3

        # With explicit credentials
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ \\
            --access-key-id mykey --secret-access-key mysecret --use-mc
    """
    try:
        # Parse S3 URL
        bucket, prefix = parse_s3_url(s3_path)

        # Create uploader
        uploader = S3Uploader(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name=region,
            verify_ssl=not no_verify_ssl,
            use_aws_cli=not use_boto3 and not use_mc,
            use_mc_cli=use_mc,
        )

        # Test connection
        click.echo("Testing S3 connection...")
        if not uploader.test_connection():
            return

        # Upload files
        click.echo(f"Uploading {local_path} to s3://{bucket}/{prefix}")
        result = uploader.upload_directory(
            local_path=local_path,
            bucket=bucket,
            prefix=prefix,
            show_progress=not no_progress,
        )

        # Display results
        if result["uploaded_files"] > 0:
            display_success(
                f"Successfully uploaded {result['uploaded_files']} files to s3://{bucket}/{prefix}"
            )

        if result["failed_files"]:
            display_error(
                f"Failed to upload {len(result['failed_files'])} files:\n"
                + "\n".join(
                    f"  - {f['file']}: {f['error']}" for f in result["failed_files"]
                )
            )

        # Display validation results
        validation = result.get("validation")
        if validation:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            if validation["valid"]:
                console.print("\n✅ [bold green]Upload validation passed[/bold green]")
            else:
                console.print("\n❌ [bold red]Upload validation failed[/bold red]")

            # Summary table
            table = Table(title="Upload Validation Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            # Handle different validation result formats
            summary = validation.get("summary", {})
            if "total_files" in validation:
                # Old format (backwards compatibility)
                table.add_row("Total Files", f"{validation['total_files']}")
                table.add_row(
                    "Validated Files", f"{validation.get('validated_files', 0)}"
                )
                table.add_row(
                    "Failed Validations",
                    f"{len(validation.get('failed_validations', []))}",
                )
            else:
                # New S3MinimalValidator format
                table.add_row("Total Files", f"{summary.get('total_files', 0)}")
                table.add_row("Total Rows", f"{summary.get('total_rows', 0):,}")
                table.add_row(
                    "Total Size",
                    f"{summary.get('total_size', 0) / (1024 * 1024):.2f} MB",
                )
                table.add_row("Format", f"{summary.get('format', 'unknown')}")
                table.add_row("Errors", f"{len(validation.get('errors', []))}")

            console.print(table)

            # Show failed validations if any
            failed_validations = validation.get("failed_validations", [])
            errors = validation.get("errors", [])

            if failed_validations:
                console.print("\n[bold red]Failed File Validations:[/bold red]")
                for failure in failed_validations:
                    console.print(
                        f"  • [red]{failure['file']}[/red]: {failure['s3_key']}"
                    )
                    for error in failure.get("errors", []):
                        console.print(f"    - {error}")
            elif errors:
                console.print("\n[bold red]Validation Errors:[/bold red]")
                for error in errors:
                    console.print(f"  • [red]{error}[/red]")

            # Show file details in verbose mode (optional)
            file_details = validation.get("file_details", [])
            if file_details and len(file_details) <= 5:
                console.print("\n[dim]File Details:[/dim]")
                for detail in file_details:
                    status = "✅" if detail["valid"] else "❌"
                    size_mb = detail["file_size_bytes"] / (1024 * 1024)
                    console.print(
                        f"  {status} [cyan]{detail['filename']}[/cyan]: "
                        f"{detail.get('row_count', 0):,} rows, {size_mb:.2f} MB"
                    )

    except ValueError as e:
        display_error(f"Invalid input: {e}")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
    except Exception as e:
        log_error_with_context(e, {"local_path": str(local_path), "s3_path": s3_path})
        display_error(f"Upload failed: {e}")


@main.group(name="to-milvus")
def to_milvus() -> None:
    """Send data to Milvus using different methods."""
    pass


@to_milvus.command("insert")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI (default: http://localhost:19530)",
)
@click.option(
    "--token",
    default="",
    help="Token for authentication",
)
@click.option(
    "--db-name",
    default="default",
    help="Database name (default: default)",
)
@click.option(
    "--collection-name",
    help="Override collection name from metadata",
)
@click.option(
    "--drop-if-exists",
    is_flag=True,
    help="Drop collection if it already exists",
)
@click.option(
    "--no-index",
    is_flag=True,
    help="Skip creating indexes on vector fields",
)
@click.option(
    "--batch-size",
    default=10000,
    type=int,
    help="Batch size for inserting data (default: 10000)",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar during import",
)
@click.option(
    "--use-autoindex",
    is_flag=True,
    help="Use AUTOINDEX for dense vector fields (faster but ~90-95% recall, overrides default FLAT index)",
)
def insert_to_milvus(
    data_path: Path,
    uri: str = "http://localhost:19530",
    token: str = "",
    db_name: str = "default",
    collection_name: str | None = None,
    drop_if_exists: bool = False,
    no_index: bool = False,
    batch_size: int = 10000,
    no_progress: bool = False,
    use_autoindex: bool = False,
) -> None:
    """Insert generated data directly to Milvus.

    \b
    Examples:
        # Insert to local Milvus
        milvus-ingest to-milvus insert ./output

        # Insert to remote Milvus with token
        milvus-ingest to-milvus insert ./output --uri http://192.168.1.100:19530 --token your_token

        # Drop existing collection and recreate
        milvus-ingest to-milvus insert ./output --drop-if-exists

        # Insert with custom collection name
        milvus-ingest to-milvus insert ./output --collection-name my_collection
    """
    try:
        # Create inserter
        inserter = MilvusInserter(
            uri=uri,
            token=token,
            db_name=db_name,
        )

        # Test connection
        click.echo("Testing Milvus connection...")
        if not inserter.test_connection():
            return

        # Insert data
        click.echo(f"Inserting data from {data_path} to Milvus...")
        result = inserter.insert_data(
            data_path=data_path,
            collection_name=collection_name,
            drop_if_exists=drop_if_exists,
            create_index=not no_index,
            batch_size=batch_size,
            show_progress=not no_progress,
            use_flat_index=not use_autoindex,  # Default to FLAT unless --use-autoindex specified
        )

        # Display results
        if result["total_inserted"] > 0:
            display_success(
                f"Successfully inserted {result['total_inserted']:,} rows to collection '{result['collection_name']}'",
                details=f"Indexes created: {len(result['indexes_created'])}",
            )

        if result["failed_batches"]:
            display_error(
                f"Failed to insert {len(result['failed_batches'])} batches:\n"
                + "\n".join(
                    f"  - {b['file']} batch {b['batch']}: {b['error']}"
                    for b in result["failed_batches"]
                )
            )

        # Close connection
        inserter.close()

    except ValueError as e:
        display_error(f"Invalid input: {e}")
    except Exception as e:
        log_error_with_context(e, {"data_path": str(data_path), "uri": uri})
        display_error(f"Insert failed: {e}")


@to_milvus.command("verify")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--collection-name",
    help="Collection name to verify (overrides collection name from meta.json)",
)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI (default: http://localhost:19530)",
)
@click.option(
    "--token",
    default="",
    help="Token for authentication",
)
@click.option(
    "--db-name",
    default="default",
    help="Database name (default: default)",
)
@click.option(
    "--level",
    type=click.Choice(["count", "scalar", "full"], case_sensitive=False),
    default="count",
    help="Verification level: count (row count + query tests), scalar (scalar fields + query tests), full (all fields + query tests)",
)
def verify_milvus_data(
    data_path: Path,
    collection_name: str | None = None,
    uri: str = "http://localhost:19530",
    token: str = "",
    db_name: str = "default",
    level: str = "count",
) -> None:
    """Verify that data in Milvus matches the original generated data.

    Supports three verification levels (all include query/search correctness tests):
    - count: Row count + query tests (default, fastest)
    - scalar: Row count + scalar field values + query tests (excludes vectors)
    - full: Row count + all field values + query tests (includes vectors)

    \b
    Examples:
        # Row count + query tests (default)
        milvus-ingest to-milvus verify ./output

        # Scalar fields + query tests (excludes vectors)
        milvus-ingest to-milvus verify ./output --level scalar

        # Full verification with all fields + query tests
        milvus-ingest to-milvus verify ./output --level full

        # Verify specific collection on remote Milvus
        milvus-ingest to-milvus verify ./output --level full --collection-name my_collection --uri http://remote:19530
    """
    from pymilvus import MilvusClient

    from .verifier import MilvusVerifier

    try:
        # Load metadata to get collection name
        meta_file = data_path / "meta.json"
        if not meta_file.exists():
            display_error(f"meta.json not found in {data_path}")
            raise SystemExit(1)

        with open(meta_file) as f:
            metadata = json.load(f)

        # Get collection name
        final_collection_name = collection_name or metadata.get("schema", {}).get(
            "collection_name"
        )
        if not final_collection_name:
            display_error(
                "Collection name not found in meta.json and not provided via --collection-name"
            )
            raise SystemExit(1)

        # Connect to Milvus using MilvusClient
        click.echo("Connecting to Milvus...")
        client = MilvusClient(
            uri=uri,
            token=token,
            db_name=db_name,
        )

        # Check if collection exists
        if not client.has_collection(final_collection_name):
            display_error(
                f"Collection '{final_collection_name}' does not exist in Milvus"
            )
            raise SystemExit(1)

        # Create verifier and run verification based on level
        verifier = MilvusVerifier(client, final_collection_name, data_path)

        if level == "count":
            # Level 1: Row count + query tests
            click.echo("Running row count verification with query tests...")
            results = verifier.verify_count_with_queries()
            if not all(results.values()):
                failed_checks = [k for k, v in results.items() if not v]
                display_error(f"Verification failed: {', '.join(failed_checks)}")
                raise SystemExit(1)

        elif level == "scalar":
            # Level 2: Row count + scalar field comparison + query tests
            click.echo("Running scalar field verification with query tests...")
            results = verifier.verify_scalar_fields_with_queries()
            if not all(results.values()):
                failed_checks = [k for k, v in results.items() if not v]
                display_error(f"Verification failed: {', '.join(failed_checks)}")
                raise SystemExit(1)

        elif level == "full":
            # Level 3: Full verification including all fields + query tests
            click.echo("Running full field verification with query tests...")
            results = verifier.verify_full_fields_with_queries()
            if not all(results.values()):
                failed_checks = [k for k, v in results.items() if not v]
                display_error(f"Verification failed: {', '.join(failed_checks)}")
                raise SystemExit(1)

    except Exception as e:
        log_error_with_context(
            e,
            {
                "data_path": str(data_path),
                "collection": final_collection_name
                if "final_collection_name" in locals()
                else "unknown",
            },
        )
        display_error(f"Verification failed: {e}")
        raise SystemExit(1) from e


@main.group()
def report() -> None:
    """Generate LLM context documents from Loki and Prometheus data."""
    pass


@report.command("generate")
@click.option(
    "--job-ids", multiple=True, help="Job IDs to analyze (can specify multiple)"
)
@click.option("--collection-name", help="Collection name")
@click.option("--start-time", help="Start time (ISO format: 2024-01-01T10:00:00)")
@click.option("--end-time", help="End time (ISO format: 2024-01-01T11:00:00)")
@click.option(
    "--output", "-o", default="/tmp/import_analysis.md", help="Output file path"
)
@click.option(
    "--format",
    default="analysis",
    type=click.Choice(["analysis", "raw"]),
    help="Output format: analysis (GLM-powered) or raw (data only)",
)
@click.option("--loki-url", default="http://10.100.36.154:80", help="Loki server URL")
@click.option(
    "--prometheus-url",
    default="http://10.100.36.157:9090",
    help="Prometheus server URL",
)
@click.option("--timeout", type=int, default=30, help="Query timeout in seconds")
@click.option(
    "--test-scenario", help="Test scenario name (e.g., 'Large Parquet Import')"
)
@click.option("--notes", help="Additional notes for the test")
@click.option("--release-name", help="Milvus release name (derives pod pattern if not specified)")
@click.option("--milvus-namespace", help="Milvus namespace (for Prometheus queries)")
@click.option(
    "--import-info-file", help="Path to import_info.json file for additional metadata"
)
@click.option(
    "--glm-api-key",
    envvar="GLM_API_KEY",
    help="GLM API key for analysis (or set GLM_API_KEY env var)",
)
@click.option(
    "--glm-model", default="glm-4-flash", help="GLM model to use (default: glm-4-flash)"
)
def generate_report(
    job_ids: tuple,
    collection_name: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    output: str,
    format: str,
    loki_url: str,
    prometheus_url: str,
    timeout: int,
    test_scenario: Optional[str],
    notes: Optional[str],
    release_name: Optional[str],
    milvus_namespace: Optional[str],
    import_info_file: Optional[str],
    glm_api_key: Optional[str],
    glm_model: str,
) -> None:
    """Generate import performance report with GLM analysis or raw data export."""
    from datetime import datetime, timedelta

    # Parse time parameters
    end_dt = datetime.fromisoformat(end_time) if end_time else datetime.now()
    start_dt = (
        datetime.fromisoformat(start_time)
        if start_time
        else end_dt - timedelta(hours=1)
    )

    # Convert job_ids tuple to list
    job_id_list = list(job_ids) if job_ids else None

    # Check GLM API key for analysis format
    if format == "analysis" and not glm_api_key:
        click.echo(
            "❌ GLM API key required for analysis format. Set GLM_API_KEY environment variable or use --glm-api-key option."
        )
        click.echo("💡 Use --format raw to export raw data without analysis.")
        raise click.Abort()

    # Create config
    config = ReportConfig(
        loki_url=loki_url,
        prometheus_url=prometheus_url,
        release_name=release_name,
        start_time=start_dt,
        end_time=end_dt,
        timeout_seconds=timeout,
        namespace=milvus_namespace or "chaos-testing",
    )

    # Display generation message
    if format == "analysis":
        click.echo("🤖 Generating GLM-powered analysis report...")
        click.echo(f"   Model: {glm_model}")
    else:
        click.echo("📊 Generating raw data export...")

    if job_id_list:
        click.echo(f"   Job IDs: {', '.join(job_id_list)}")

    # Use ReportGenerator
    from .report.report_generator import ReportGenerator

    generator = ReportGenerator(config)

    try:
        result = generator.generate_report(
            job_ids=job_id_list,
            collection_name=collection_name,
            start_time=start_dt,
            end_time=end_dt,
            output_file=output,
            output_format=format,
            test_scenario=test_scenario,
            notes=notes,
            release_name=release_name,
            milvus_namespace=milvus_namespace,
            import_info_file=import_info_file,
            glm_api_key=glm_api_key,
            glm_model=glm_model,
        )

        # Display success message
        if format == "analysis":
            click.echo(f"✅ Analysis report saved to: {output}")
            if "glm_model" in result:
                click.echo(f"   GLM model used: {result['glm_model']}")
        else:
            click.echo(f"✅ Raw data exported to: {output}")

        click.echo(f"   Jobs analyzed: {result['jobs_analyzed']}")
        click.echo(f"   Total log entries: {result['total_logs']}")

        if result["jobs_analyzed"] > 0 and job_id_list:
            click.echo("\n📋 Analyzed job IDs:")
            for job_id in job_id_list:
                click.echo(f"   - {job_id}")

    except Exception as e:
        click.echo(f"❌ Report generation failed: {e}")
        if format == "analysis":
            click.echo("💡 Try --format raw to export raw data without GLM analysis.")
        raise click.Abort()


@to_milvus.command("import")
@click.option(
    "--collection-name",
    type=str,
    help="Target collection name for import (overrides collection name from meta.json)",
)
@click.option(
    "--local-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Local path to data directory (output from 'generate' command)",
)
@click.option(
    "--s3-path",
    required=True,
    type=str,
    help="S3 path to upload to (relative to bucket, e.g., 'data/' or 'prefix/data/')",
)
@click.option(
    "--bucket",
    required=True,
    type=str,
    help="S3/MinIO bucket name",
)
@click.option(
    "--endpoint-url",
    type=str,
    help="S3/MinIO endpoint URL (e.g., http://localhost:9000 for MinIO)",
)
@click.option(
    "--access-key-id",
    type=str,
    help="S3/MinIO access key ID",
)
@click.option(
    "--secret-access-key",
    type=str,
    help="S3/MinIO secret access key",
)
@click.option(
    "--no-verify-ssl",
    is_flag=True,
    help="Disable SSL certificate verification",
)
@click.option(
    "--uri",
    default="http://127.0.0.1:19530",
    help="Milvus instance URI (default: http://127.0.0.1:19530)",
)
@click.option(
    "--token",
    default="",
    help="Token for authentication",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for import to complete",
)
@click.option(
    "--timeout",
    type=int,
    help="Timeout in seconds when waiting (no timeout by default)",
)
@click.option(
    "--drop-if-exists",
    is_flag=True,
    help="Drop collection if it already exists before creating",
)
@click.option(
    "--use-autoindex",
    is_flag=True,
    help="Use AUTOINDEX for dense vector fields (faster but ~90-95% recall, overrides default FLAT index)",
)
@click.option(
    "--use-boto3",
    is_flag=True,
    help="Use boto3 instead of AWS CLI for uploads (AWS CLI is default and more reliable)",
)
@click.option(
    "--use-mc",
    is_flag=True,
    help="Use MinIO Client (mc) CLI for uploads (prioritized over AWS CLI and boto3)",
)
@click.option(
    "--output-import-info",
    type=click.Path(path_type=Path),
    help="Output import information to JSON file (job IDs, collection name, timing, etc.)",
)
def import_to_milvus(
    collection_name: str | None,
    local_path: Path,
    s3_path: str,
    bucket: str,
    endpoint_url: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    no_verify_ssl: bool,
    uri: str = "http://127.0.0.1:19530",
    token: str = "",
    wait: bool = False,
    timeout: int | None = None,
    drop_if_exists: bool = False,
    use_autoindex: bool = False,
    use_boto3: bool = False,
    use_mc: bool = False,
    output_import_info: Path | None = None,
) -> None:
    """Upload data to S3/MinIO and bulk import to Milvus in one step.

    Automatically creates the collection if it doesn't exist (using meta.json from data directory).
    This combines the upload and import steps for convenience.

    \b
    Examples:
        # Upload and import using collection name from meta.json (AWS CLI by default)
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000

        # Upload and import using mc CLI
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-mc

        # Upload and import with custom collection name
        milvus-ingest to-milvus import --collection-name my_collection --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-mc

        # Upload and import with credentials using mc CLI
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret --use-mc

        # Upload and import then wait for completion
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait --use-mc
    """
    import json
    from datetime import datetime

    from .milvus_importer import MilvusBulkImporter
    from .uploader import S3Uploader

    # Record import start time
    import_start_time = datetime.now()

    try:
        # Load metadata from local path
        meta_file = local_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.json not found in {local_path}")

        with open(meta_file) as f:
            metadata = json.load(f)

        # Get collection name
        final_collection_name = collection_name or metadata.get("schema", {}).get(
            "collection_name"
        )
        if not final_collection_name:
            raise ValueError(
                "Collection name not found in meta.json and not provided via --collection-name"
            )

        # Step 1: Upload data to S3/MinIO
        print("Step 1: Uploading data to S3/MinIO...")
        uploader = S3Uploader(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            verify_ssl=not no_verify_ssl,
            use_aws_cli=not use_boto3 and not use_mc,
            use_mc_cli=use_mc,
        )

        # Ensure s3_path ends with /
        if not s3_path.endswith("/"):
            s3_path = s3_path + "/"

        destination = f"s3://{bucket}/{s3_path}"
        uploader.upload_directory(local_path, bucket, s3_path, show_progress=True)
        print(f"✓ Data uploaded to {destination}")

        # Step 2: Import to Milvus
        print("Step 2: Importing data to Milvus...")
        importer = MilvusBulkImporter(uri=uri, token=token)

        # Get list of data files to import (parquet and JSON files)
        data_files = []

        # Find parquet files
        for parquet_file in local_path.glob("*.parquet"):
            data_files.append(s3_path + parquet_file.name)

        # Find JSON files (exclude meta.json)
        for json_file in local_path.glob("*.json"):
            if json_file.name != "meta.json":
                data_files.append(s3_path + json_file.name)

        if not data_files:
            raise ValueError(f"No parquet or json data files found in {local_path}")

        # Start import (returns list of job IDs if batching is needed)
        job_ids = importer.bulk_import_files(
            collection_name=final_collection_name,
            files=[str(local_path)],  # For metadata loading
            import_files=data_files,  # S3 file paths
            show_progress=True,
            create_collection=True,  # Always try to create with metadata
            drop_if_exists=drop_if_exists,
            use_flat_index=not use_autoindex,  # Default to FLAT unless --use-autoindex specified
            max_files_per_batch=250,  # Max 250 files per batch to avoid large meta requests
        )

        if len(job_ids) == 1:
            print(f"✓ Import job started: {job_ids[0]}")
        else:
            print(
                f"✓ {len(job_ids)} import jobs started (batched due to {len(data_files)} files)"
            )
            print(f"✓ Job IDs: {', '.join(job_ids)}")
        print(f"✓ Collection: {final_collection_name}")

        # Record import completion time
        import_end_time = datetime.now()

        # Wait for completion if requested
        import_success = True
        if wait:
            # Use the new wait_for_multiple_jobs method
            success = importer.wait_for_multiple_jobs(job_ids, timeout=timeout or 600)
            if success:
                print("✓ All imports completed successfully!")
                import_success = True
            else:
                print("✗ Some imports failed or timed out")
                import_success = False
                raise SystemExit(1)
        else:
            if len(job_ids) == 1:
                print(
                    f"Import job is running asynchronously. Use job ID {job_ids[0]} to check status."
                )
            else:
                print(
                    f"{len(job_ids)} import jobs are running asynchronously. Monitor with job IDs: {', '.join(job_ids)}"
                )

        # Prepare import information structure
        import_info = {
            "status": "completed" if import_success else "failed",
            "collection_name": final_collection_name,
            "job_ids": job_ids,
            "import_start_time": import_start_time.isoformat(),
            "import_end_time": import_end_time.isoformat(),
            "total_duration_seconds": (
                import_end_time - import_start_time
            ).total_seconds(),
            "milvus_uri": uri,
            "milvus_token_used": bool(token.strip()),
            "s3_bucket": bucket,
            "s3_path": s3_path,
            "s3_endpoint": endpoint_url,
            "upload_method": "mc_cli"
            if use_mc
            else ("boto3" if use_boto3 else "aws_cli"),
            "drop_if_exists": drop_if_exists,
            "use_autoindex": use_autoindex,
            "wait_for_completion": wait,
            "timeout": timeout,
            "total_files": len(data_files),
            "file_types": list({f.split(".")[-1] for f in data_files}),
            "schema_metadata": {
                "schema_type": metadata.get("schema_type", "unknown"),
                "fields_count": len(metadata.get("schema", {}).get("fields", [])),
                "vector_fields": [
                    f["name"]
                    for f in metadata.get("schema", {}).get("fields", [])
                    if f.get("type")
                    in ["FLOAT_VECTOR", "BINARY_VECTOR", "SPARSE_FLOAT_VECTOR"]
                ],
                "has_dynamic_fields": any(
                    f.get("is_dynamic", False)
                    for f in metadata.get("schema", {}).get("fields", [])
                ),
                "has_partition_key": any(
                    f.get("is_partition_key", False)
                    for f in metadata.get("schema", {}).get("fields", [])
                ),
            },
            "generation_metadata": metadata.get("generation", {}),
        }

        # Add data generation info directly from meta.json for report generation
        generation_info = metadata.get("generation_info", {})
        if generation_info:
            # Add total_rows if available
            if "total_rows" in generation_info:
                import_info["total_rows"] = generation_info["total_rows"]

            # Add file information if available
            if "data_files" in generation_info:
                data_files_info = generation_info["data_files"]
                total_size = sum(f.get("file_size_bytes", 0) for f in data_files_info)

                import_info["file_info"] = {
                    "file_count": len(data_files_info),
                    "total_size_bytes": total_size,
                    "file_sizes": [
                        f.get("file_size_bytes", 0) for f in data_files_info
                    ],
                }

            # Add schema type if available
            if "schema_type" in generation_info:
                import_info["schema_type"] = generation_info["schema_type"]

        # Output import info to file if requested
        if output_import_info:
            try:
                # Ensure parent directory exists
                output_import_info.parent.mkdir(parents=True, exist_ok=True)

                with open(output_import_info, "w", encoding="utf-8") as f:
                    json.dump(import_info, f, indent=2, ensure_ascii=False)

                print(f"✓ Import information saved to: {output_import_info}")
            except Exception as e:
                print(f"⚠ Failed to save import information: {e}")

        # Always print a summary line for easy parsing (can be parsed by Jenkins)
        summary_data = {
            "job_ids": ",".join(job_ids),
            "collection_name": final_collection_name,
            "status": "completed" if import_success else "failed",
            "start_time": import_start_time.isoformat(),
            "end_time": import_end_time.isoformat(),
        }
        print(f"IMPORT_SUMMARY: {json.dumps(summary_data)}")

    except Exception as e:
        from .rich_display import display_error

        display_error(f"Import failed: {e}")
        raise SystemExit(1) from e


def _save_with_high_performance_generator(
    schema_path: Path,
    total_rows: int,
    output_path: Path,
    fmt: str,
    batch_size: int = 10000,
    seed: int | None = None,
    show_progress: bool = True,
    file_size: str | None = None,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    file_count: int | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
) -> int:
    """Save using high-performance vectorized generator optimized for large-scale data.

    Returns:
        int: The actual number of rows generated (may differ from total_rows when file_count and file_size are specified)
    """
    import time

    from .optimized_writer import generate_data_optimized

    logger = get_logger(__name__)
    start_time = time.time()

    # Use larger batch size for high-performance mode
    optimized_batch_size = max(batch_size, 50000)

    try:
        if show_progress:
            from rich.console import Console

            # Immediately set log level to WARNING before starting progress to suppress all INFO messages
            setup_logging(
                verbose=verbose,
                log_level="WARNING",  # Suppress INFO messages during progress
            )

            # Generate data (simplified without progress bar)
            print("🚀 Generating data with high-performance mode...")

            # Simple progress callback for print-based progress
            def update_progress(completed_rows: int) -> None:
                if completed_rows % 50000 == 0:  # Print every 50k rows
                    progress_pct = 100.0 * completed_rows / total_rows
                    print(
                        f"📊 Progress: {completed_rows:,}/{total_rows:,} rows ({progress_pct:.1f}%)"
                    )

            # Run optimized generator with progress callback
            files_created, actual_total_rows = generate_data_optimized(
                schema_path=schema_path,
                total_rows=total_rows,
                output_dir=output_path,
                format=fmt,
                batch_size=optimized_batch_size,
                seed=seed,
                file_size=file_size,
                num_partitions=num_partitions,
                num_shards=num_shards,
                file_count=file_count,
                num_workers=num_workers,
                progress_callback=update_progress,
                skip_validation=False,  # Always validate final output
            )

            print(f"✅ Generation completed: {actual_total_rows:,} rows")

            # Show completion summary after progress bar

            console = Console()
            console.print(
                f"\n✅ [bold green]Generation completed![/bold green] {len(files_created)} files created with {actual_total_rows:,} rows total"
            )
        else:
            # Run without progress bar
            files_created, actual_total_rows = generate_data_optimized(
                schema_path=schema_path,
                total_rows=total_rows,
                output_dir=output_path,
                format=fmt,
                batch_size=optimized_batch_size,
                seed=seed,
                file_size=file_size,
                num_partitions=num_partitions,
                num_shards=num_shards,
                file_count=file_count,
                num_workers=num_workers,
                skip_validation=False,  # Always validate final output
            )

        # Log performance metrics
        total_time = time.time() - start_time
        total_size = sum(
            Path(f).stat().st_size for f in files_created if Path(f).exists()
        )
        file_size_mb = total_size / (1024 * 1024)

        log_performance(
            "high_performance_generator",
            total_time,
            total_rows=actual_total_rows,
            batch_size=optimized_batch_size,
            file_size_mb=file_size_mb,
        )
        logger.info(
            "High-performance generator completed",
            total_rows=actual_total_rows,
            output_dir=str(output_path),
            file_size_mb=file_size_mb,
            duration_seconds=total_time,
            rows_per_second=actual_total_rows / total_time if total_time > 0 else 0,
        )
        return actual_total_rows
    except Exception as e:
        logger.error(f"High-performance generator failed: {e}")
        raise


def _handle_clean_command(yes: bool, logger: Any) -> None:
    """Handle the clean command to remove generated output files."""
    # Default data directory
    data_dir = DEFAULT_DATA_DIR

    # Collect paths to clean
    paths_to_clean: list[tuple[str, Path | list[Path]]] = []

    # Check data directory
    if data_dir.exists() and any(data_dir.iterdir()):
        paths_to_clean.append(("Generated data directory", data_dir))

    # Check for any generated files in current directory
    current_dir = Path.cwd()
    generated_files: list[Path] = []
    for pattern in ["*.parquet", "*.csv", "*.npy"]:
        generated_files.extend(current_dir.glob(pattern))

    # For JSON files, be more selective to avoid schema files
    json_files = current_dir.glob("*.json")
    excluded_json_files = {
        "package.json",
        "pyproject.toml",
        "schema.json",
        "example_schema.json",
        "demo.json",
        "meta.json",
    }
    # Include JSON files that look like generated data files
    for json_file in json_files:
        if (
            json_file.name not in excluded_json_files
            and "schema" not in json_file.name.lower()
        ):
            generated_files.append(json_file)

    if generated_files:
        paths_to_clean.append(("Generated files in current directory", generated_files))

    if not paths_to_clean:
        display_success("No files or directories to clean.")
        logger.info("Clean command completed - nothing to clean")
        return

    # Display what will be cleaned
    click.echo("The following will be cleaned:")
    for description, path in paths_to_clean:
        if isinstance(path, list):
            click.echo(f"  • {description}:")
            for file_path in path:
                click.echo(f"    - {file_path}")
        else:
            click.echo(f"  • {description}: {path}")

    # Confirm deletion unless --yes flag is used
    if not yes:
        click.echo()
        if not click.confirm(
            "Are you sure you want to delete these files and directories?"
        ):
            click.echo("Clean operation cancelled.")
            logger.info("Clean command cancelled by user")
            return

    # Perform cleanup
    cleaned_items: list[str] = []
    errors: list[str] = []

    for _, path in paths_to_clean:
        try:
            if isinstance(path, list):
                # Handle list of files
                for file_path in path:
                    if file_path.exists():
                        file_path.unlink()
                        cleaned_items.append(str(file_path))
                        logger.debug(f"Removed file: {file_path}")
            else:
                # Handle directory
                if path.exists():
                    shutil.rmtree(path)
                    cleaned_items.append(str(path))
                    logger.debug(f"Removed directory: {path}")
        except Exception as e:
            error_msg = f"Failed to remove {path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

    # Report results
    if cleaned_items:
        display_success(f"Successfully cleaned {len(cleaned_items)} items.")
        logger.info(
            "Clean command completed successfully",
            cleaned_items=len(cleaned_items),
            errors=len(errors),
        )

    if errors:
        click.echo("\nErrors occurred during cleanup:")
        for error in errors:
            display_error(error)


if __name__ == "__main__":  # pragma: no cover
    # Allow ``python -m milvus_fake_data``
    sys.exit(main())
