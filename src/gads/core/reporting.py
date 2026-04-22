import os
from typing import List, Dict, Any
import uuid

def create_master_reports(
    project_id: uuid.UUID,
    workspace_dir: str,
    narrative: str,
    takeaways: List[str],
    artifacts: List[Any]
):
    """
    Assembles the final integrated HTML dashboard and Markdown research report.
    """
    
    # 1. Generate Markdown Report
    takeaways_md = "\n".join([f"- {t}" for t in takeaways])
    md_content = f"""# Research Report: Project {project_id}

## Executive Summary
{narrative}

## Key Takeaways
{takeaways_md}

## Methodology & Findings
The following artifacts were generated during this analysis:
"""
    for a in artifacts:
        if a.type == "interactive_plot":
            fname = a.content_json.get("filename")
            md_content += f"- **{a.description}**: [View Interactive Plot]({fname})\n"
        elif a.type == "plot":
            md_content += f"- **{a.description}** (Static Image)\n"

    with open(os.path.join(workspace_dir, "research_report.md"), "w") as f:
        f.write(md_content)

    # 2. Generate Integrated HTML Dashboard
    # Filter for interactive plots
    plot_divs = []
    for a in artifacts:
        if a.type == "interactive_plot":
            fname = a.content_json.get("filename")
            fpath = os.path.join(workspace_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as pf:
                    content = pf.read()
                    # Extract just the div part if it's a full plotly html
                    if "<div>" in content:
                        div = content.split("<body>")[1].split("</body>")[0] if "<body>" in content else content
                        plot_divs.append({
                            "title": a.description,
                            "div": div
                        })

    takeaways_html = "".join([f"<li>{t}</li>" for t in takeaways])
    cards_html = ""
    for p in plot_divs:
        cards_html += f"""
        <div class='card'>
            <h2>{p['title']}</h2>
            <div class='plot-container'>{p['div']}</div>
        </div>
        """

    html_content = f"""
    <html>
    <head>
        <title>GADS Research Synthesis</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 1000px; background: #f0f2f5; color: #1a1a1a; line-height: 1.6; }}
            .report-header {{ background: #1e293b; color: white; padding: 40px; border-radius: 12px 12px 0 0; margin-bottom: 0; }}
            .report-body {{ background: white; padding: 40px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; margin: 30px 0; background: #ffffff; }}
            .plot-container {{ height: 420px; overflow: hidden; }}
            h1, h2, h3 {{ color: #0f172a; }}
            .takeaways {{ background: #f8fafc; border-left: 4px solid #3b82f6; padding: 20px; margin: 20px 0; }}
            .narrative {{ font-size: 1.1rem; color: #334155; margin-bottom: 40px; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class='report-header'>
            <h1>GADS Research Synthesis</h1>
            <p>Project ID: {project_id}</p>
        </div>
        <div class='report-body'>
            <h2>Executive Summary</h2>
            <div class='narrative'>{narrative}</div>
            
            <div class='takeaways'>
                <h3>Key Takeaways</h3>
                <ul>{takeaways_html}</ul>
            </div>

            <hr style='margin: 40px 0; border: 0; border-top: 1px solid #e2e8f0;' />
            
            <h2>Supporting Visualizations</h2>
            {cards_html}
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(workspace_dir, "final_dashboard.html"), "w") as f:
        f.write(html_content)
