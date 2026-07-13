---
id: visualization_best_practices
description: "Chart rules: Y-axis from 0, plotly_white, unified hover; px.imshow for heatmaps; artifact naming and save conventions (Plotly write_json, matplotlib Agg+png)"
triggers: ["plot", "chart", "visualize", "visualization", "figure", "heatmap", "confusion matrix"]
---
# Visualization Best Practices

## Chart design
- **Y-axis anchoring**: bar and line charts must NOT truncate the Y-axis — start at 0.
- **Interactivity**: `fig.update_layout(hovermode='x unified')`.
- **Template**: stick to `template='plotly_white'`; avoid loud color schemes.
- Horizontal bar rankings read best sorted: `fig.update_layout(yaxis={'categoryorder': 'total ascending'})`.

## API gotchas
- Heatmaps / confusion matrices: `px.imshow(cm, text_auto=True)` — **`px.heatmap()` does not exist**.
- Feature-importance DataFrames from AutoGluon keep names in the **index**: plot with
  `x=fi_top['importance'], y=fi_top.index`.

## Saving artifacts (the dashboard collects files by extension)
- Plotly (preferred — interactive in the dashboard): `fig.write_json('figure_1_<short_name>.json')`.
- matplotlib (only when Plotly can't express it): set the headless backend first —
  `import matplotlib; matplotlib.use('Agg')` — then `plt.savefig('figure_1_<short_name>.png', dpi=120, bbox_inches='tight')` and `plt.close()`.
- Number figures sequentially (`figure_1_*`, `figure_2_*`) with a short descriptive name.
- Never call `plt.show()`.
