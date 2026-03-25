"""Small SVG reporting helpers without third-party dependencies."""

from __future__ import annotations

from pathlib import Path


PALETTE = [
    "#0b5fff",
    "#ff6b00",
    "#00a37a",
    "#d13b40",
    "#7c4dff",
    "#b68a00",
]


def _safe_range(values: list[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = 1.0 if minimum == 0.0 else abs(minimum) * 0.1
        return minimum - padding, maximum + padding
    padding = 0.05 * (maximum - minimum)
    return minimum - padding, maximum + padding


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_line_chart_svg(
    path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    width: int = 960,
    height: int = 540,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_points = [point for points in series.values() for point in points]
    x_min, x_max = _safe_range([point[0] for point in all_points])
    y_min, y_max = _safe_range([point[1] for point in all_points])

    left = 80
    right = 24
    top = 48
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + ((x - x_min) / (x_max - x_min)) * plot_width
        py = top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_height
        return px, py

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" font-size="20" text-anchor="middle" font-family="sans-serif">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="{width / 2}" y="{height - 16}" font-size="14" text-anchor="middle" font-family="sans-serif">{_escape(x_label)}</text>',
        f'<text x="22" y="{height / 2}" font-size="14" text-anchor="middle" transform="rotate(-90, 22, {height / 2})" font-family="sans-serif">{_escape(y_label)}</text>',
    ]

    for tick in range(5):
        ratio = tick / 4
        y_value = y_min + (1.0 - ratio) * (y_max - y_min)
        y_pos = top + ratio * plot_height
        svg_lines.append(
            f'<line x1="{left}" y1="{y_pos}" x2="{left + plot_width}" y2="{y_pos}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left - 8}" y="{y_pos + 4}" font-size="12" text-anchor="end" font-family="sans-serif">{y_value:.3g}</text>'
        )

    if y_min < 0.0 < y_max:
        zero_ratio = (0.0 - y_min) / (y_max - y_min)
        zero_y = top + plot_height - zero_ratio * plot_height
        svg_lines.append(
            f'<line x1="{left}" y1="{zero_y}" x2="{left + plot_width}" y2="{zero_y}" stroke="#555" stroke-width="1.2"/>'
        )

    for index, (name, points) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        polyline = " ".join(
            f"{project(point)[0]:.2f},{project(point)[1]:.2f}" for point in points
        )
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>'
        )
        legend_y = top + 16 + index * 20
        legend_x = left + plot_width - 160
        svg_lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="12" font-family="sans-serif">{_escape(name)}</text>'
        )

    svg_lines.append("</svg>")
    output.write_text("\n".join(svg_lines), encoding="utf-8")


def write_bar_chart_svg(
    path: str | Path,
    title: str,
    y_label: str,
    categories: list[str],
    values: list[float],
    width: int = 960,
    height: int = 540,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    y_min, y_max = _safe_range([0.0, *values])
    left = 80
    right = 24
    top = 48
    bottom = 128
    plot_width = width - left - right
    plot_height = height - top - bottom
    slot_width = plot_width / max(len(categories), 1)
    bar_width = slot_width * 0.7

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" font-size="20" text-anchor="middle" font-family="sans-serif">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="22" y="{height / 2}" font-size="14" text-anchor="middle" transform="rotate(-90, 22, {height / 2})" font-family="sans-serif">{_escape(y_label)}</text>',
    ]

    for tick in range(5):
        ratio = tick / 4
        y_value = y_min + (1.0 - ratio) * (y_max - y_min)
        y_pos = top + ratio * plot_height
        svg_lines.append(
            f'<line x1="{left}" y1="{y_pos}" x2="{left + plot_width}" y2="{y_pos}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left - 8}" y="{y_pos + 4}" font-size="12" text-anchor="end" font-family="sans-serif">{y_value:.3g}</text>'
        )

    for index, (category, value) in enumerate(zip(categories, values)):
        color = PALETTE[index % len(PALETTE)]
        x = left + index * slot_width + (slot_width - bar_width) / 2
        zero_ratio = (0.0 - y_min) / (y_max - y_min)
        zero_y = top + plot_height - zero_ratio * plot_height
        value_ratio = (value - y_min) / (y_max - y_min)
        value_y = top + plot_height - value_ratio * plot_height
        y = min(zero_y, value_y)
        bar_height = abs(zero_y - value_y)
        svg_lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" opacity="0.85"/>'
        )
        svg_lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{(y - 6 if value >= 0 else y + bar_height + 14):.2f}" font-size="12" text-anchor="middle" font-family="sans-serif">{value:.3g}</text>'
        )
        label_x = x + bar_width / 2
        label_y = top + plot_height + 16
        svg_lines.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" font-size="11" text-anchor="end" transform="rotate(-35, {label_x:.2f}, {label_y:.2f})" font-family="sans-serif">{_escape(category)}</text>'
        )

    svg_lines.append("</svg>")
    output.write_text("\n".join(svg_lines), encoding="utf-8")
