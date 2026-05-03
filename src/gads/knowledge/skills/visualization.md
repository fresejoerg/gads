---
id: visualization_best_practices
triggers: ["plot", "chart", "visualize", "visualization"]
---
# Visualization Best Practices

- **Y-Axis Anchoring**: For all bar charts and line charts, ensure the Y-axis is NOT truncated. It MUST start at 0.
- **Interactivity**: Use `fig.update_layout(hovermode='x unified')` for better readability.
- **Monochrome Consistency**: Stick to the `plotly_white` template and avoid overly vibrant or distracting colors.
