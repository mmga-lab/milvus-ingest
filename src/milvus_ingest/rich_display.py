"""Rich terminal display utilities for schema information."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def display_schema_list(schemas: dict[str, dict[str, Any]], title: str) -> None:
    """Display a list of schemas with rich formatting.

    Args:
        schemas: Dictionary of schema_id -> schema_info
        title: Title for the display
    """
    console = Console()

    if not schemas:
        console.print(f"[yellow]No {title.lower()} found.[/yellow]")
        return

    console.print(f"\n[bold blue]{title}[/bold blue]\n")

    for schema_id, info in schemas.items():
        # Create a panel for each schema
        content = []
        content.append(f"[dim]{info['description']}[/dim]")

        # Add metadata
        metadata_parts = []
        metadata_parts.append(f"Fields: [cyan]{info['fields_count']}[/cyan]")

        if info["vector_dims"]:
            dims_str = ", ".join(map(str, info["vector_dims"]))
            metadata_parts.append(f"Vector dims: [green]{dims_str}[/green]")

        content.append(" • ".join(metadata_parts))

        # Add use cases if available
        if info.get("use_cases"):
            use_cases_str = ", ".join(info["use_cases"])
            content.append(f"[italic]Use cases: {use_cases_str}[/italic]")

        # Create panel
        panel_content = "\n".join(content)
        panel = Panel(
            panel_content,
            title=f"[bold]{schema_id}[/bold] - {info['name']}",
            border_style="bright_blue" if info.get("type") == "custom" else "green",
            width=80,
        )
        console.print(panel)


def display_schema_details(
    schema_id: str, info: dict[str, Any], schema_data: dict[str, Any], is_builtin: bool
) -> None:
    """Display detailed schema information with rich formatting.

    Args:
        schema_id: Schema identifier
        info: Schema metadata information
        schema_data: Full schema data
        is_builtin: Whether the schema is built-in
    """
    console = Console()

    # Header
    schema_type = "Built-in" if is_builtin else "Custom"
    type_color = "green" if is_builtin else "bright_blue"

    console.print(
        f"\n[bold {type_color}]Schema: {schema_id}[/bold {type_color}] [dim]({schema_type})[/dim]\n"
    )

    # Basic information table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Name", info["name"])
    info_table.add_row(
        "Description", info["description"] or "[dim]No description[/dim]"
    )
    info_table.add_row("Collection", schema_data.get("collection_name", "N/A"))
    info_table.add_row("Fields", str(info["fields_count"]))

    if info["vector_dims"]:
        dims_str = ", ".join(map(str, info["vector_dims"]))
        info_table.add_row("Vector dimensions", f"[green]{dims_str}[/green]")

    if info.get("use_cases"):
        use_cases_str = ", ".join(info["use_cases"])
        info_table.add_row("Use cases", f"[italic]{use_cases_str}[/italic]")

    console.print(
        Panel(
            info_table, title="[bold]Schema Information[/bold]", border_style=type_color
        )
    )

    # Fields table
    fields_table = Table(title="Fields", show_header=True, header_style="bold magenta")
    fields_table.add_column("Name", style="cyan", no_wrap=True)
    fields_table.add_column("Type", style="yellow")
    fields_table.add_column("Properties", style="dim")
    fields_table.add_column("Description", style="dim", max_width=40)

    for field in schema_data.get("fields", []):
        field_name = field["name"]
        field_type = field["type"]

        # Build properties string
        properties = []
        if field.get("is_primary"):
            properties.append("[bold red]PRIMARY[/bold red]")
        if field.get("auto_id"):
            properties.append("[blue]AUTO_ID[/blue]")
        if field.get("nullable"):
            properties.append("[yellow]NULLABLE[/yellow]")
        if "dim" in field:
            properties.append(f"[green]dim={field['dim']}[/green]")
        if "max_length" in field:
            properties.append(f"[cyan]max_length={field['max_length']}[/cyan]")
        if "min" in field or "max" in field:
            range_info = []
            if "min" in field:
                range_info.append(f"min={field['min']}")
            if "max" in field:
                range_info.append(f"max={field['max']}")
            properties.append(f"[dim]{', '.join(range_info)}[/dim]")

        properties_str = " ".join(properties) if properties else ""
        description = field.get("description", "")

        fields_table.add_row(field_name, field_type, properties_str, description)

    console.print(fields_table)

    # Usage example
    usage_text = Text()
    usage_text.append("Usage: ", style="bold")
    if is_builtin:
        usage_text.append(
            f"milvus-ingest --builtin {schema_id} --rows 1000", style="code"
        )
    else:
        usage_text.append(
            f"milvus-ingest --builtin {schema_id} --rows 1000", style="code"
        )

    console.print(
        Panel(
            usage_text, title="[bold]Usage Example[/bold]", border_style="bright_green"
        )
    )


def display_error(message: str, details: str = "") -> None:
    """Display error message with rich formatting.

    Args:
        message: Main error message
        details: Additional details
    """
    console = Console()

    error_text = f"[bold red]✗[/bold red] {message}"
    if details:
        error_text += f"\n[dim]{details}[/dim]"

    console.print(
        Panel(error_text, border_style="red", title="[bold red]Error[/bold red]")
    )


def display_success(message: str, details: str = "") -> None:
    """Display success message with rich formatting.

    Args:
        message: Main success message
        details: Additional details
    """
    console = Console()

    success_text = f"[bold green]✓[/bold green] {message}"
    if details:
        success_text += f"\n[dim]{details}[/dim]"

    console.print(
        Panel(
            success_text, border_style="green", title="[bold green]Success[/bold green]"
        )
    )


def display_info(message: str, details: str = "") -> None:
    """Display info message with rich formatting.

    Args:
        message: Main info message
        details: Additional details
    """
    console = Console()

    info_text = f"[bold blue]ℹ[/bold blue] {message}"
    if details:
        info_text += f"\n[dim]{details}[/dim]"

    console.print(
        Panel(info_text, border_style="blue", title="[bold blue]Info[/bold blue]")
    )


def display_schema_validation(schema_id: str, validation_info: dict[str, Any]) -> None:
    """Display schema validation results with rich formatting.

    Args:
        schema_id: Schema identifier
        validation_info: Validation result information
    """
    console = Console()

    console.print(f"\n[bold green]✓ Schema '{schema_id}' is valid![/bold green]\n")

    # Create validation summary
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Property", style="bold")
    summary_table.add_column("Value", style="cyan")

    if "collection_name" in validation_info:
        summary_table.add_row(
            "Collection", validation_info["collection_name"] or "unnamed"
        )

    if "fields_count" in validation_info:
        summary_table.add_row("Fields found", str(validation_info["fields_count"]))

    console.print(
        Panel(
            summary_table, title="[bold]Validation Summary[/bold]", border_style="green"
        )
    )

    # Display fields if available
    if "fields" in validation_info:
        fields_tree = Tree("[bold]Fields:[/bold]")
        for field in validation_info["fields"]:
            field_name = field.get("name", "unknown")
            field_type = field.get("type", "unknown")
            is_primary = field.get("is_primary", False)

            field_label = f"[cyan]{field_name}[/cyan]: [yellow]{field_type}[/yellow]"
            if is_primary:
                field_label += " [bold red](PRIMARY)[/bold red]"

            fields_tree.add(field_label)

        console.print(fields_tree)


def display_schema_preview(
    schema_name: str,
    collection_name: str | None,
    fields: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> None:
    """Display detailed schema preview before data generation.

    Args:
        schema_name: Name of the schema being used
        collection_name: Collection name if available
        fields: List of field definitions
        generation_config: Configuration for data generation (rows, batch_size, etc.)
    """
    console = Console()

    # Main header
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Schema Preview: {schema_name}[/bold blue]", border_style="blue"
        )
    )

    # Collection info
    if collection_name:
        console.print(f"[dim]Collection:[/dim] [cyan]{collection_name}[/cyan]")

    # Generation summary
    summary_items = []
    # Get rows value and format safely
    rows_value = generation_config.get(
        "total_rows", generation_config.get("rows", "N/A")
    )
    rows_display = f"{rows_value:,}" if isinstance(rows_value, int) else str(rows_value)
    summary_items.append(
        f"[bold]Rows to generate:[/bold] [green]{rows_display}[/green]"
    )

    # Get batch size and format safely
    batch_size_value = generation_config.get("batch_size", "N/A")
    batch_size_display = (
        f"{batch_size_value:,}"
        if isinstance(batch_size_value, int)
        else str(batch_size_value)
    )
    summary_items.append(
        f"[bold]Batch size:[/bold] [yellow]{batch_size_display}[/yellow]"
    )

    if generation_config.get("seed") is not None:
        summary_items.append(
            f"[bold]Random seed:[/bold] [magenta]{generation_config['seed']}[/magenta]"
        )

    console.print()
    for item in summary_items:
        console.print(f"  • {item}")

    # Fields table
    console.print()
    console.print("[bold]Field Definitions:[/bold]")

    fields_table = Table(show_header=True, header_style="bold magenta")
    fields_table.add_column("Field Name", style="cyan", min_width=15)
    fields_table.add_column("Type", style="yellow", min_width=12)
    fields_table.add_column("Properties", style="dim", min_width=20)
    fields_table.add_column("Constraints", style="green", min_width=15)

    for field in fields:
        name = field.get("name", "unknown")
        field_type = field.get("type", "unknown")

        # Properties column
        properties = []
        if field.get("is_primary"):
            properties.append("[bold red]PRIMARY[/bold red]")
        if field.get("auto_id"):
            properties.append("[bold blue]AUTO_ID[/bold blue]")
        if field.get("nullable"):
            properties.append("[dim]nullable[/dim]")

        properties_str = " • ".join(properties) if properties else "[dim]—[/dim]"

        # Constraints column
        constraints = []
        if "min" in field or "max" in field:
            if "min" in field and "max" in field:
                constraints.append(f"range: {field['min']}–{field['max']}")
            elif "min" in field:
                constraints.append(f"min: {field['min']}")
            elif "max" in field:
                constraints.append(f"max: {field['max']}")

        if "max_length" in field:
            constraints.append(f"max_len: {field['max_length']}")

        if "dim" in field:
            constraints.append(f"dim: {field['dim']}")

        if "max_capacity" in field:
            constraints.append(f"capacity: {field['max_capacity']}")

        if "element_type" in field:
            constraints.append(f"element: {field['element_type']}")

        constraints_str = " • ".join(constraints) if constraints else "[dim]—[/dim]"

        fields_table.add_row(name, field_type, properties_str, constraints_str)

    console.print(fields_table)

    # Statistics summary
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Stat", style="bold")
    stats_table.add_column("Count", style="cyan")

    total_fields = len(fields)
    vector_fields = len([f for f in fields if "vector" in f.get("type", "").lower()])
    primary_fields = len([f for f in fields if f.get("is_primary")])
    nullable_fields = len([f for f in fields if f.get("nullable")])

    stats_table.add_row("Total fields:", str(total_fields))
    stats_table.add_row("Vector fields:", str(vector_fields))
    stats_table.add_row("Primary fields:", str(primary_fields))
    stats_table.add_row("Nullable fields:", str(nullable_fields))

    console.print()
    console.print(
        Panel(stats_table, title="[bold]Schema Statistics[/bold]", border_style="green")
    )

    # Estimated output size
    if generation_config.get("rows"):
        estimated_size = _estimate_output_size(fields, generation_config["rows"])
        console.print()
        console.print(f"[dim]Estimated output size: ~{estimated_size}[/dim]")


def _estimate_output_size(fields: list[dict[str, Any]], rows: int) -> str:
    """Estimate the output file size based on field types and row count."""
    size_bytes = 0

    for field in fields:
        field_type = field.get("type", "").lower()

        # Estimate bytes per field per row
        if field_type in ["bool"] or field_type in ["int8"]:
            size_bytes += 1
        elif field_type in ["int16"]:
            size_bytes += 2
        elif field_type in ["int32", "float"]:
            size_bytes += 4
        elif field_type in ["int64", "double"]:
            size_bytes += 8
        elif field_type in ["varchar", "string"]:
            # Estimate average string length
            max_len = field.get("max_length", 100)
            avg_len = min(max_len * 0.6, 50)  # Assume 60% of max or 50 chars
            size_bytes += int(avg_len)
        elif "vector" in field_type:
            dim = field.get("dim", 128)
            if "binary" in field_type:
                size_bytes += dim // 8  # Binary vectors are packed
            else:
                size_bytes += dim * 4  # Float vectors are 4 bytes per dimension
        elif field_type == "json":
            size_bytes += 100  # Estimate for JSON fields
        elif field_type == "array":
            capacity = field.get("max_capacity", 5)
            element_type = field.get("element_type", "varchar")
            if element_type.lower() == "varchar":
                element_size = field.get("max_length", 20)
            else:
                element_size = 4  # Assume 4 bytes for numeric elements
            size_bytes += capacity * element_size
        else:
            size_bytes += 10  # Default estimate

    total_size = size_bytes * rows

    # Format size in human readable format
    if total_size < 1024:
        return f"{total_size} bytes"
    elif total_size < 1024 * 1024:
        return f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        return f"{total_size / (1024 * 1024):.1f} MB"
    else:
        return f"{total_size / (1024 * 1024 * 1024):.1f} GB"


def display_confirmation_prompt(rows: int, estimated_size: str) -> None:
    """Display a formatted confirmation prompt before data generation.

    Args:
        rows: Number of rows to generate
        estimated_size: Estimated file size string
    """
    console = Console()

    # Create a summary box for confirmation
    if rows > 100000:
        icon = "⚠️"
        style = "yellow"
        warning_text = "\n[dim]⏱️  Large dataset - this may take some time and use significant resources[/dim]"
    else:
        icon = "📝"
        style = "green"
        warning_text = ""

    summary_text = f"""[bold]{icon} Ready to Generate Mock Data[/bold]

[cyan]Rows:[/cyan] {rows:,}
[cyan]Estimated size:[/cyan] ~{estimated_size}{warning_text}"""

    console.print()
    console.print(
        Panel(
            summary_text,
            title="[bold]Confirmation[/bold]",
            border_style=style,
            padding=(1, 2),
        )
    )
