"""Command-line interface for milvus-ingest.

Usage::

    # Data generation
    milvus-ingest generate --schema schema.json --rows 1000
    milvus-ingest generate --builtin simple --rows 100 --preview

    # Schema management
    milvus-ingest schema list
    milvus-ingest schema show simple
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
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .logging_config import (
    get_logger,
    log_error_with_context,
    log_performance,
    setup_logging,
)
from .milvus_inserter import MilvusInserter
from .models import get_schema_help, validate_schema_data
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
    "--rows-per-file",
    "rows_per_file",
    default=1000000,
    show_default=True,
    type=int,
    help="Maximum rows per file (for automatic file partitioning).",
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
    rows_per_file: int = 1000000,
    file_count: int | None = None,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    num_workers: int | None = None,
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

        # Display schema preview
        generation_config = {
            "total_rows": total_rows,
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
                    files_created = generate_data_optimized(
                        schema_path=temp_schema_file,
                        total_rows=preview_rows,
                        output_dir=temp_output,
                        batch_size=preview_rows,
                        format="parquet",  # Always use parquet for preview
                        num_partitions=num_partitions,
                        num_shards=num_shards,
                        seed=seed,
                        rows_per_file=preview_rows,
                        file_size=None,  # Don't use size controls for preview
                        file_count=None,
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

    # Use high-performance data generation (default and only mode)
    logger.info(
        "Starting high-performance data generation",
        output_dir=str(output_path),
        format=output_format,
        rows=total_rows,
    )
    try:
        # High-performance parallel generator (default and only mode)
        logger.info("Using vectorized high-performance generator")

        # If progress bar will be shown, suppress INFO logs before starting generation
        if not no_progress:
            logger.info("Starting high-performance vectorized generator")
            setup_logging(verbose=verbose, log_level="WARNING")

        _save_with_high_performance_generator(
            schema_path,
            total_rows,
            output_path,
            output_format,
            batch_size=batch_size,
            seed=seed,
            show_progress=not no_progress,
            file_size=file_size,
            rows_per_file=rows_per_file,
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
            rows=total_rows,
            output_file=str(output_path),
            file_size_mb=file_size_mb,
        )
        # Output is always a directory now
        click.echo(f"Saved {total_rows} rows to directory {output_path.resolve()}")
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
def upload(
    local_path: Path,
    s3_path: str,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str = "us-east-1",
    no_verify_ssl: bool = False,
    no_progress: bool = False,
) -> None:
    """Upload generated data files to S3/MinIO.

    \b
    Examples:
        # Upload to AWS S3
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/

        # Upload to MinIO
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ --endpoint-url http://localhost:9000

        # With explicit credentials
        milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ \\
            --access-key-id mykey --secret-access-key mysecret
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
def verify_milvus_data(
    data_path: Path,
    collection_name: str | None = None,
    uri: str = "http://localhost:19530",
    token: str = "",
    db_name: str = "default",
) -> None:
    """Verify that data in Milvus matches the original generated data count.

    \b
    Examples:
        # Verify data using collection name from meta.json
        milvus-ingest to-milvus verify ./output

        # Verify specific collection
        milvus-ingest to-milvus verify ./output --collection-name my_collection

        # Verify on remote Milvus
        milvus-ingest to-milvus verify ./output --uri http://192.168.1.100:19530 --token your_token
    """
    from pymilvus import MilvusClient

    try:
        # Load metadata
        meta_file = data_path / "meta.json"
        if not meta_file.exists():
            display_error(f"meta.json not found in {data_path}")
            raise SystemExit(1)

        with open(meta_file) as f:
            metadata = json.load(f)

        # Get expected row count from metadata
        expected_rows = metadata.get("generation_info", {}).get("total_rows", 0)
        if expected_rows == 0:
            display_error("Could not find total_rows in meta.json")
            raise SystemExit(1)

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

        # Count rows in collection
        click.echo(f"Counting rows in collection '{final_collection_name}'...")

        # Query to count all rows - we need to use a count query
        # First, load the collection
        client.load_collection(final_collection_name)

        # Query count using empty filter to get all rows
        result = client.query(
            collection_name=final_collection_name,
            filter="",  # Empty filter to match all
            output_fields=["count(*)"],
        )

        # Extract row count from result
        actual_rows = result[0].get("count(*)", 0) if result else 0

        # Compare counts
        click.echo("\n" + "=" * 50)
        click.echo(f"Collection: {final_collection_name}")
        click.echo(f"Expected rows (from meta.json): {expected_rows:,}")
        click.echo(f"Actual rows (in Milvus): {actual_rows:,}")
        click.echo("=" * 50)

        if actual_rows == expected_rows:
            display_success(
                "✓ Verification passed!", f"Row count matches: {actual_rows:,} rows"
            )
        else:
            difference = actual_rows - expected_rows
            percentage = (actual_rows / expected_rows * 100) if expected_rows > 0 else 0

            display_error(
                "✗ Verification failed!",
                f"Row count mismatch:\n"
                f"  Expected: {expected_rows:,} rows\n"
                f"  Actual: {actual_rows:,} rows\n"
                f"  Difference: {difference:+,} rows ({percentage:.1f}% of expected)",
            )

            # Provide additional debugging info
            if actual_rows < expected_rows:
                click.echo("\nPossible causes:")
                click.echo("  - Some batches may have failed during insert/import")
                click.echo("  - The import job may still be in progress")
                click.echo("  - Data files may have been corrupted or incomplete")
            else:
                click.echo("\nPossible causes:")
                click.echo("  - Duplicate data may have been inserted")
                click.echo("  - Multiple import operations may have been performed")

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
) -> None:
    """Upload data to S3/MinIO and bulk import to Milvus in one step.

    Automatically creates the collection if it doesn't exist (using meta.json from data directory).
    This combines the upload and import steps for convenience.

    \b
    Examples:
        # Upload and import using collection name from meta.json
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000

        # Upload and import with custom collection name
        milvus-ingest to-milvus import --collection-name my_collection --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000

        # Upload and import with credentials
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret

        # Upload and import then wait for completion
        milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait
    """
    from .milvus_importer import MilvusBulkImporter
    from .uploader import S3Uploader

    try:
        # Load metadata from local path
        meta_file = local_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.json not found in {local_path}")

        import json

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

        # Start import
        job_id = importer.bulk_import_files(
            collection_name=final_collection_name,
            files=[str(local_path)],  # For metadata loading
            import_files=data_files,  # S3 file paths
            show_progress=True,
            create_collection=True,  # Always try to create with metadata
            drop_if_exists=drop_if_exists,
        )

        print(f"✓ Import job started: {job_id}")
        print(f"✓ Collection: {final_collection_name}")

        # Wait for completion if requested
        if wait:
            success = importer.wait_for_completion(job_id, timeout=timeout or 300)
            if success:
                print("✓ Import completed successfully!")
            else:
                raise SystemExit(1)
        else:
            print(
                f"Import job is running asynchronously. Use job ID {job_id} to check status."
            )

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
    rows_per_file: int = 1000000,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    file_count: int | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
) -> None:
    """Save using high-performance vectorized generator optimized for large-scale data."""
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

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=Console(stderr=True),  # Use stderr for progress output
            ) as progress:
                # Reconfigure logging to use Rich's console for proper coordination
                setup_logging(
                    verbose=verbose,
                    log_level="WARNING",  # Keep WARNING level during progress
                    rich_console=progress.console,
                )

                task = progress.add_task(
                    "Generating data with high-performance mode...", total=total_rows
                )

                # Progress callback to update the progress bar
                def update_progress(completed_rows: int) -> None:
                    progress.update(task, completed=completed_rows)

                # Run optimized generator with progress callback
                files_created = generate_data_optimized(
                    schema_path=schema_path,
                    total_rows=total_rows,
                    output_dir=output_path,
                    format=fmt,
                    batch_size=optimized_batch_size,
                    seed=seed,
                    file_size=file_size,
                    rows_per_file=rows_per_file,
                    num_partitions=num_partitions,
                    num_shards=num_shards,
                    file_count=file_count,
                    num_workers=num_workers,
                    progress_callback=update_progress,
                )

                # Ensure progress shows 100% at the end
                progress.update(task, completed=total_rows)

            # Restore original logging configuration after progress bar is closed
            setup_logging(verbose=verbose, log_level="DEBUG" if verbose else "INFO")

            # Show completion summary after progress bar
            from rich.console import Console

            console = Console()
            console.print(
                f"\n✅ [bold green]Generation completed![/bold green] {len(files_created)} files created with {total_rows:,} rows total"
            )
        else:
            # Run without progress bar
            files_created = generate_data_optimized(
                schema_path=schema_path,
                total_rows=total_rows,
                output_dir=output_path,
                format=fmt,
                batch_size=optimized_batch_size,
                seed=seed,
                file_size=file_size,
                rows_per_file=rows_per_file,
                num_partitions=num_partitions,
                num_shards=num_shards,
                file_count=file_count,
                num_workers=num_workers,
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
            total_rows=total_rows,
            batch_size=optimized_batch_size,
            file_size_mb=file_size_mb,
        )
        logger.info(
            "High-performance generator completed",
            total_rows=total_rows,
            output_dir=str(output_path),
            file_size_mb=file_size_mb,
            duration_seconds=total_time,
            rows_per_second=total_rows / total_time if total_time > 0 else 0,
        )
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
