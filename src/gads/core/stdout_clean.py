"""Display hygiene for captured sandbox stdout.

Two transforms, both needed anywhere task stdout is shown to a user:

1. **Carriage-return collapsing.** tqdm / AutoGluon / sklearn `verbose=` output redraws
   a progress bar in place with `\\r`. Rendered verbatim, a 100-step bar becomes 100
   near-identical lines — the single biggest source of unreadable output on the long
   training runs this view exists for.
2. **Sentinel stripping.** GADS prints internal telemetry to stdout behind reserved
   prefixes (see CLAUDE.md "Conventions & Gotchas"). Several of those probes run as
   *separate* `sandbox.execute` calls against the SAME session while the log poller is
   still alive, so they land in the live stream even though they were never part of the
   task's own output. They are machine plumbing and must never reach a user-facing view.

Both the live stream and the persisted copy go through here, so the view a user watches
while a node runs matches the view they get once it completes. Keeping the prefix list
in one place is the point: it was already duplicated in the notebook exporter, and that
copy had silently drifted — it was missing `GADS_METRICS_JSON:`.
"""

# Every reserved stdout prefix GADS parses back into structured data. Add new sentinels
# HERE and nowhere else; the parsers match on these same strings.
SENTINEL_PREFIXES = (
    "GADS_INSIGHTS_JSON:",
    "GADS_FLOOR_JSON:",
    "GADS_STATE_SNAPSHOT:",
    "GADS_HYPOTHESIS_JSON:",
    "GADS_METRICS_JSON:",
)


def collapse_carriage_returns(text: str) -> str:
    """Keep only the final segment of any `\\r`-redrawn line (the last frame a terminal
    would have shown), so an in-place progress bar renders as one line, not hundreds."""
    if not text:
        return text
    out = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        out.append(line)
    return "\n".join(out)


def strip_sentinels(text: str) -> str:
    """Drop lines carrying an internal telemetry prefix."""
    if not text:
        return text
    return "\n".join(
        line for line in text.split("\n")
        if not line.startswith(SENTINEL_PREFIXES)
    )


def clean_stdout(text: str) -> str:
    """Full display cleanup: collapse progress-bar redraws, then drop sentinel lines.

    Order matters — a sentinel emitted right after a `\\r` redraw only starts the line
    once the carriage returns have been collapsed.
    """
    if not text:
        return text
    return strip_sentinels(collapse_carriage_returns(text))
