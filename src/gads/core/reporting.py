import os
import re
import json
from typing import List, Dict, Any, Optional
import uuid
import base64
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Task
from gads.core import report_sections

def create_master_reports(
    project_id: uuid.UUID,
    workspace_dir: str,
    narrative: str,
    takeaways: List[str],
    artifacts: List[Any],
    artifact_insights: Optional[List[Any]] = None,
    followups: Optional[List[Dict[str, Any]]] = None,
    section_notes: Optional[List[Any]] = None,
):
    """
    Assembles the final integrated HTML dashboard and Markdown research report.

    The body of the report is composed from the **recipe DAG**, not from the artifact list:
    `report_sections.build_sections` produces one section per pipeline step, in order, and
    every card, metric and insight is filed under the step that produced it (see that
    module for why). Steps that drew no chart — the reasoning and audit nodes — therefore
    appear in the report, and steps that never ran appear as gaps instead of vanishing.
    Artifacts that cannot be attributed to any step are still rendered, in a trailing
    section, so restructuring the report can never lose evidence.

    `followups` appends analyst-directed sections AFTER the autonomous result — each with its
    own instruction text, timestamp, optional narrative/stdout and its own cards. The main
    section is never mutated or reordered by a follow-up (approach_docs/020). Each entry:
    {instruction_text, created_at, model_used, narrative, stdout, artifacts}.
    """
    insights_map = {}
    if artifact_insights:
        # artifact_insights can be list of dicts or pydantic models
        for ins in artifact_insights:
            if hasattr(ins, 'model_dump'):
                data = ins.model_dump()
            else:
                data = ins
            insights_map[data.get('artifact_id')] = data

    # 1. Generate Integrated HTML Dashboard using Jinja2
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("dashboard.html.j2")

    # Build a reverse lookup from artifact filenames back to the 'Figure N' labels used in tasks
    # This ensures that if the Synthesizer says 'Figure 1', we can find which task produced it
    # and match it to the actual file on disk.
    artifact_to_figure_label = {}
    with Session(engine) as session:
        tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
        for t in tasks:
            # Look for all "Figure X" mentions in the description
            fig_matches = re.findall(r"Figure\s*(\d+)", t.description, re.IGNORECASE)
            
            # CRITICAL: Only use tasks that mention EXACTLY ONE figure.
            # These are the 'Producer' tasks. 
            # Reference tasks (like synthesis) that mention multiple figures are ignored.
            if len(fig_matches) == 1:
                fig_label = f"Figure {fig_matches[0]}"
                t_res = t.result_json or {}
                
                # GROUND TRUTH: Only trust filenames found in the orchestrator summary.
                # This summary records exactly which files were created/modified by THIS specific task.
                summary = t_res.get("orchestrator_summary", "")
                created_filenames = re.findall(r"[\w\d\-_]+\.(?:json|html|png|csv|parquet)", summary, re.IGNORECASE)
                
                for fn in created_filenames:
                    fn_l = fn.lower()
                    # An artifact belongs to the task that created it.
                    if fn_l not in artifact_to_figure_label:
                        artifact_to_figure_label[fn_l] = fig_label

        # The report's spine: one section per recipe DAG node, built while the tasks are
        # still attached to their session.
        sections = report_sections.build_sections(list(tasks))

    # We also need the inverse: Figure N -> Filename
    figure_label_to_artifact = {v.lower(): k for k, v in artifact_to_figure_label.items()}

    cards = _build_cards(artifacts, workspace_dir, insights_map, artifact_to_figure_label)

    orphan_cards = report_sections.attach_cards(sections, cards)
    report_sections.apply_section_notes(sections, section_notes)
    report_sections.finalize(sections)
    if orphan_cards:
        print(f"  [Reporting] {len(orphan_cards)} artifact(s) could not be attributed to a "
              f"pipeline step; rendering them in a trailing section.")

    # Load persisted metrics (written by the metrics guarantee probe)
    key_metrics: Dict[str, Any] = {}
    metrics_path = os.path.join(workspace_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                raw_metrics = json.load(f)
            for k, v in raw_metrics.items():
                key_metrics[k] = report_sections.format_metric(v)
        except Exception as e:
            print(f"  [Reporting] Warning: could not load metrics.json: {e}")

    # 2. Markdown research report — the same section order as the dashboard, so the two
    # artefacts describe the run identically.
    md_content = _build_markdown(project_id, narrative, takeaways, key_metrics,
                                 sections, orphan_cards)
    with open(os.path.join(workspace_dir, "research_report.md"), "w") as f:
        f.write(md_content)

    # Follow-up sections: each carries its own cards, built with the same logic so they
    # render identically to the main run (approach_docs/020).
    followup_views = []
    for fu in (followups or []):
        followup_views.append({
            "instruction_text": fu.get("instruction_text", ""),
            "created_at": fu.get("created_at", ""),
            "model_used": fu.get("model_used", ""),
            "narrative": fu.get("narrative", ""),
            "stdout": (fu.get("stdout") or "").strip()[-4000:],
            "cards": _build_cards(fu.get("artifacts", []), workspace_dir, {}, {}),
        })
    if followup_views:
        md_content += "\n## Follow-up Analyses\n"
        md_content += ("Analyst-directed work performed after the autonomous run; the "
                       "findings above are unchanged.\n\n")
        for fu in followup_views:
            md_content += f"### {fu['instruction_text']}\n"
            md_content += f"*{fu['created_at']}*\n\n"
            if fu["narrative"]:
                md_content += fu["narrative"] + "\n\n"
            for c in fu["cards"]:
                md_content += f"- **{c['description']}**\n"
        with open(os.path.join(workspace_dir, "research_report.md"), "w") as f:
            f.write(md_content)

    html_content = template.render(
        project_id=str(project_id),
        narrative=narrative,
        takeaways=takeaways,
        cards=cards,
        sections=sections,
        orphan_cards=orphan_cards,
        key_metrics=key_metrics,
        followups=followup_views,
    )

    with open(os.path.join(workspace_dir, "final_dashboard.html"), "w") as f:
        f.write(html_content)

    return html_content


def _build_markdown(project_id, narrative, takeaways, key_metrics, sections, orphan_cards):
    """Markdown twin of the dashboard: same sections, same order, same gaps."""
    md = [f"# Research Report: Project {project_id}", "", "## Executive Summary", narrative, ""]
    if key_metrics:
        md += ["## Key Metrics", ""]
        md += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in key_metrics.items()]
        md += [""]
    md += ["## Key Takeaways", ""]
    md += [f"- {t}" for t in takeaways]
    md += ["", "## Methodology, Step by Step",
           "", "Each step below is a node of the applied recipe, in execution order.", ""]

    for sec in sections:
        status = "not executed" if sec["status"] == "not_executed" else sec["status"]
        md.append(f"### {sec['index']}. {sec['title']}  *({status})*")
        if sec["summary"]:
            md.append(f"*{sec['summary']}*")
        md.append("")
        if sec["note"]:
            md += [sec["note"], ""]
        if sec["metrics"]:
            md += ["| Metric | Value |", "| --- | --- |"]
            md += [f"| {k} | {v} |" for k, v in sec["metrics"].items()]
            md.append("")
        for ins in sec["insights"]:
            md.append(f"- **{ins.get('artifact', 'insight')}**: {ins.get('insight', '')}")
        if sec["insights"]:
            md.append("")
        if sec.get("state"):
            shapes = ", ".join(f"`{st.get('artifact')}` "
                               f"{str(st.get('insight', '')).split('(')[-1].rstrip(')')}"
                               for st in sec["state"])
            md += [f"New in the kernel after this step: {shapes}.", ""]
        for c in sec["cards"]:
            if c.get("type") == "handover_bundle":
                md.append(f"- **{c['description']}**: [Download Offline Script]({c.get('filename')})")
            elif c.get("source_filename"):
                md.append(f"- **{c['description']}**: [View Plot]({c['source_filename']})")
            else:
                md.append(f"- **{c['description']}** (static image)")
        if sec["cards"]:
            md.append("")
        if sec["error"]:
            md += [f"> **Failed:** {sec['error']}", ""]
        if sec["status"] == "not_executed":
            md += ["> This step of the recipe did not run, so the analysis it was to "
                   "contribute is missing from this report.", ""]
        if sec["fallback"]:
            md += [f"> Completed by the {sec['fallback']} fallback rather than by the "
                   f"assigned model.", ""]

    if orphan_cards:
        md += ["### Additional Artifacts", "",
               "Produced during the run but not attributable to a single recipe step.", ""]
        md += [f"- **{c['description']}**" for c in orphan_cards]
        md.append("")
    return "\n".join(md)


def _build_cards(artifacts, workspace_dir, insights_map, artifact_to_figure_label):
    """Turn Artifact rows into renderable dashboard cards (shared by the main run and each
    follow-up section, so both look the same)."""
    cards = []
    for a in artifacts:
        # ... logic ...
        card = {
            "description": a.description,
            "type": a.type,
            "caption": "Generated by GADS workflow.",
            "ctx_text": "",
            "json_data": None,
            "html_content": None,
            "img_b64": None,
            "filename": None,
            # Provenance for section attribution (see report_sections.attach_cards).
            # Absent on artifacts created before the stamp existed — those fall back to
            # the filename match, and failing that render in the trailing section.
            "task_id": (a.content_json or {}).get("task_id"),
            "node_id": (a.content_json or {}).get("node_id"),
            "source_filename": (a.content_json or {}).get("filename"),
        }

        actual_filename = ""
        if a.type == "json_plot":
            actual_filename = a.content_json.get("filename")
            fpath = os.path.join(workspace_dir, actual_filename)
            if os.path.exists(fpath):
                with open(fpath, "r") as pf: card["json_data"] = pf.read()
        
        elif a.type == "interactive_plot":
            actual_filename = a.content_json.get("filename")
            fpath = os.path.join(workspace_dir, actual_filename)
            if os.path.exists(fpath):
                with open(fpath, "r") as pf: card["html_content"] = pf.read()
        
        elif a.type == "plot":
            img_b64 = a.content_json.get("image_base64")
            if img_b64:
                card["img_b64"] = img_b64
                card["type"] = "static_plot"
                actual_filename = a.description # Use description as ID for static plots
        
        elif a.type == "handover_bundle":
            actual_filename = a.content_json.get("filename")
            card["filename"] = actual_filename

        if not (card["json_data"] or card["html_content"] or card["img_b64"] or card["filename"]):
            continue

        # FIND THE RIGHT INSIGHT
        # Look for an insight where the artifact_id matches the figure label assigned to THIS artifact
        assigned_fig_label = artifact_to_figure_label.get(actual_filename.lower()) if actual_filename else None
        
        insight = None
        if assigned_fig_label:
            # Look for the specific figure label in the insights map
            insight = insights_map.get(assigned_fig_label)
            if not insight:
                # Try normalization (e.g. "Figure 1" vs "figure 1")
                for k, v in insights_map.items():
                    if k.lower() == assigned_fig_label.lower():
                        insight = v; break
        
        # If no explicit figure match, try filename match or fuzzy match
        if not insight:
            insight = insights_map.get(actual_filename)
        
        if not insight:
            norm_desc = re.sub(r'[^a-z0-9]', '', a.description.lower())
            for k, v in insights_map.items():
                norm_k = re.sub(r'[^a-z0-9]', '', k.lower())
                if norm_k in norm_desc:
                    insight = v; break

        if insight:
            card["ctx_text"] = insight.get('contextual_text', "")
            card["caption"] = insight.get('caption', "Generated by GADS workflow.")

        # FINAL SAFETY CHECK: If it's a JSON plot but doesn't look like Plotly data, skip it
        if a.type == "json_plot" and card["json_data"]:
            try:
                js_test = json.loads(card["json_data"])
                if "data" not in js_test or not isinstance(js_test["data"], list):
                    print(f"  [Reporting] Skipping non-plotly JSON artifact: {a.description}")
                    continue
            except: continue

        cards.append(card)
    return cards


def rebuild_dashboard(project_id: uuid.UUID, workspace_dir: str) -> Optional[str]:
    """Regenerate dashboard + report for a project from persisted state, including every
    follow-up section.

    Deliberately rebuilt from the DB rather than patched into the existing HTML: that makes
    regeneration **idempotent and additive** — running it after the 3rd follow-up reproduces
    follow-ups 1 and 2 unchanged, and never double-appends. Artifacts are partitioned by the
    `followup`/`instruction_id` stamps the follow-up lane writes into `content_json`.

    Best-effort: returns None on failure. A reporting problem must not fail a follow-up whose
    analysis already succeeded.
    """
    from gads.core.models import Artifact, Instruction, Project
    try:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            artifacts = session.exec(
                select(Artifact).where(Artifact.project_id == project_id)
                .order_by(Artifact.created_at)
            ).all()
            instructions = {
                str(i.id): i for i in session.exec(
                    select(Instruction).where(Instruction.project_id == project_id)).all()
            }
            tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
            # Recover the Synthesizer's prose so a rebuilt dashboard keeps its section
            # commentary and captions instead of degrading to bare evidence.
            synth_tasks = sorted([t for t in tasks if t.assigned_to == "Synthesizer"
                                  and t.status == "completed"],
                                 key=lambda t: t.created_at or datetime.min)
            last_synth = (synth_tasks[-1].result_json or {}) if synth_tasks else {}

        main_artifacts, by_instruction = [], {}
        for a in artifacts:
            cj = a.content_json or {}
            iid = cj.get("instruction_id") if cj.get("followup") else None
            (by_instruction.setdefault(iid, []) if iid else main_artifacts).append(a)

        # One section per follow-up instruction, oldest first, carrying that task's stdout.
        followups = []
        for iid, arts in by_instruction.items():
            instr = instructions.get(iid)
            ftask = next((t for t in tasks
                          if str(t.instruction_id) == iid
                          and (t.result_json or {}).get("mode") == "followup"), None)
            rj = (ftask.result_json or {}) if ftask else {}
            created = (instr.created_at if instr else None) or (ftask.created_at if ftask else None)
            followups.append({
                "instruction_text": (instr.content if instr else "(follow-up)"),
                "created_at": created.strftime("%Y-%m-%d %H:%M") if created else "",
                "model_used": rj.get("model_used", ""),
                "narrative": "",
                "stdout": rj.get("stdout", ""),
                "artifacts": arts,
                "_sort": created,
            })
        # Follow-up tasks that produced no artifact still belong in the record.
        for t in tasks:
            rj = t.result_json or {}
            iid = str(t.instruction_id) if t.instruction_id else None
            if rj.get("mode") == "followup" and iid and iid not in by_instruction:
                instr = instructions.get(iid)
                followups.append({
                    "instruction_text": (instr.content if instr else "(follow-up)"),
                    "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
                    "model_used": rj.get("model_used", ""),
                    "narrative": "", "stdout": rj.get("stdout", ""),
                    "artifacts": [], "_sort": t.created_at,
                })
        followups.sort(key=lambda f: f["_sort"] or 0)
        for f in followups:
            f.pop("_sort", None)

        # SAFETY: never replace a richer dashboard with a poorer one. If the DB holds no
        # main-run artifacts and no narrative (e.g. the project's rows were lost and only the
        # workspace survived — see kernel_state.replay_code_from_workspace), a rebuild would
        # overwrite a complete existing dashboard with an almost-empty one. Skip instead; the
        # follow-up's own outputs are still in the workspace and the DB.
        dash = os.path.join(workspace_dir, "final_dashboard.html")
        has_main = bool(main_artifacts) or bool(project and project.narrative)
        if not has_main and os.path.exists(dash) and os.path.getsize(dash) > 2048:
            print("  [Reporting] Existing dashboard preserved: the database has no main-run "
                  "artifacts to rebuild from (follow-up outputs remain in the workspace).",
                  flush=True)
            return None

        return create_master_reports(
            project_id=project_id,
            workspace_dir=workspace_dir,
            narrative=(project.narrative if project and project.narrative else ""),
            takeaways=(project.takeaways if project and project.takeaways else []),
            artifacts=main_artifacts,
            artifact_insights=last_synth.get("artifact_insights") or [],
            section_notes=last_synth.get("section_notes") or [],
            followups=followups,
        )
    except Exception as e:
        print(f"  [Reporting] Warning: dashboard rebuild failed: {e}", flush=True)
        return None
