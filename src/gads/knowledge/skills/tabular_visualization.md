---
id: tabular_visualization
triggers: ["list", "table", "tabular", "top", "bottom", "show records", "display dataframe"]
---
# Tabular Visualization with Plotly

When a user or task explicitly requests a **list**, **table**, or a specific number of records (e.g., "Top 5"), you MUST create an interactive Plotly Table (`plotly.graph_objects.Table`) to ensure the data is clearly visible in the final dashboard.

## Implementation Pattern
1. Create a `go.Table` using the data from your DataFrame.
2. Update the layout for professional appearance.
3. Save as an HTML file and call `.show()`.

```python
import plotly.graph_objects as go

# Create the table
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Top 5 Records", height=400)
fig.write_html("top_records_table.html")
fig.show()
```

## When to Use
- **Top/Bottom N**: Always use a table for small sets of records (e.g., Top 5, Top 10).
- **Categorical Summaries**: Use a table if there are many columns or text-heavy data that doesn't fit well in a bar chart.
- **Contract Fulfillment**: If the user asks for a "list", a narrative description is NOT enough; a visual table artifact is required.
