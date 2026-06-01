from __future__ import annotations


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def per_request_us(row: dict[str, str | int | float], field: str) -> float:
    iterations = int(row["iterations"])
    if iterations <= 0:
        return 0.0
    return float(row[field]) / iterations


def print_markdown_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    table = [header, *rows]
    widths = [max(len(row[idx]) for row in table) for idx in range(len(header))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            row[idx].ljust(widths[idx]) for idx in range(len(row))
        ) + " |"

    separator = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(widths))) + "-|"
    print(f"\n=== {title} ===")
    print(render(table[0]))
    print(separator)
    for row in table[1:]:
        print(render(row))
