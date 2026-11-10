# -*- coding: utf-8 -*-
"""
Per-stage recorder of numeric outputs.

Tables and JSON objects are written to results/<stage>/<site>/ together with a
_manifest.json and a human-readable _SUMMARY.md (settings, checks, warnings,
outputs and table previews).

Usage
-----
    from pipeline.common.report import StageCheck

    chk = StageCheck(stage="10", site="taal", title="Post-eruption response")
    chk.param(window_days=180)
    chk.table(df, "per_eruption_delta", desc="per-onset change")   # -> CSV
    chk.json_(meta, "site_meta")                                    # -> JSON
    chk.gate("SITES_SELECTED", passed=True, detail="...", blocking=False)
    chk.finish()
"""

from __future__ import annotations

import json
import platform
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pipeline.config import settings as S


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


class StageCheck:
    """Recorder for the intermediate outputs of one stage run."""

    def __init__(self, stage: int, title: str, site: Optional[str] = None,
                 desc: str = ""):
        self.stage = stage
        self.title = title
        self.site = site
        self.desc = desc
        self.dir = S.records_dir(stage, site)          # numeric records
        self._vis_key = (stage, site)                  # key of the visual-output folder (settings.check_dir)
        self.started = _now()
        self.artifacts: List[Dict[str, Any]] = []
        self.gates: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.params: Dict[str, Any] = {}
        self._warnings: List[str] = []
        print(f"[stage {stage}] {title}  →  {self.dir}")


    # -- recording --------------------------------------------------------
    def param(self, **kwargs) -> "StageCheck":
        """Record the settings used in this run (for reproducibility)."""
        self.params.update(kwargs)
        return self

    def note(self, text: str) -> "StageCheck":
        """Add a note to the summary's checklist section."""
        self.notes.append(textwrap.dedent(text).strip())
        return self


    def table(self, df, name: str, desc: str = "", preview_rows: int = 12):
        """Save a DataFrame as CSV and record a summary and preview."""
        path = self.dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding=S.CSV_ENCODING)
        info = dict(
            kind="table", name=name, file=path.name, desc=desc,
            rows=int(len(df)), cols=list(map(str, df.columns)),
        )
        # Numeric-column summary: flags columns that are entirely NaN or constant.
        try:
            num = df.select_dtypes("number")
            if len(num.columns):
                info["numeric_summary"] = json.loads(
                    num.describe().T[["count", "mean", "std", "min", "max"]]
                    .round(4).to_json(orient="index")
                )
                allnan = [c for c in num.columns if num[c].notna().sum() == 0]
                if allnan:
                    self.warn(f"{name}: all-NaN columns {allnan}")
        except Exception:
            pass
        info["preview"] = df.head(preview_rows).to_markdown(index=False)
        self.artifacts.append(info)
        print(f"    · {path.name}  ({len(df)} rows)")
        return path


    def json_(self, obj: Any, name: str, desc: str = ""):
        """Save an object as JSON under results/."""
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
        self.artifacts.append(dict(kind="json", name=name, file=path.name, desc=desc))
        print(f"    · {path.name}")
        return path


    def gate(self, name: str, passed: bool, detail: str = "",
             blocking: bool = True) -> bool:
        """Record a pass/fail check. A failed blocking gate makes finish() raise."""
        self.gates.append(dict(name=name, passed=bool(passed), detail=detail,
                               blocking=bool(blocking)))
        mark = "PASS" if passed else ("FAIL (blocking)" if blocking else "FAIL (warning)")
        print(f"    [{name}] {mark} — {detail}")
        return bool(passed)

    # -- finalisation -----------------------------------------------------
    def finish(self, raise_on_gate: bool = True) -> Dict[str, Any]:
        """Write _manifest.json and _SUMMARY.md; raise GateFailure on a failed blocking gate."""
        manifest = dict(
            stage=self.stage, title=self.title, site=self.site, desc=self.desc,
            started=self.started, finished=_now(),
            python=sys.version.split()[0], platform=platform.platform(),
            params=self.params, gates=self.gates, warnings=self._warnings,
            artifacts=[{k: v for k, v in a.items() if k != "preview"}
                       for a in self.artifacts],
        )
        (self.dir / "_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        (self.dir / "_SUMMARY.md").write_text(self._summary_md(), encoding="utf-8")

        failed = [g for g in self.gates if not g["passed"] and g["blocking"]]
        if failed and raise_on_gate:
            names = ", ".join(g["name"] for g in failed)
            raise GateFailure(
                f"Stage {self.stage} failed checks: {names}\n"
                f"  → see {self.dir / '_SUMMARY.md'}, resolve the cause and rerun."
            )
        return manifest

    def _summary_md(self) -> str:
        L: List[str] = []
        L.append(f"# Stage {self.stage} — {self.title}")
        if self.site:
            L.append(f"**Site**: `{self.site}`")
        L.append(f"**Run**: {self.started} → {_now()}")
        if self.desc:
            L.append("")
            L.append(self.desc)

        L.append("\n## Checks\n")
        if self.gates:
            L.append("| Check | Result | Detail |")
            L.append("|---|---|---|")
            for g in self.gates:
                m = "✅ pass" if g["passed"] else ("❌ **fail**" if g["blocking"] else "⚠️ fail (warning)")
                L.append(f"| {g['name']} | {m} | {g['detail']} |")
        else:
            L.append("_No checks (informational stage)_")

        if self._warnings:
            L.append("\n## ⚠️ Warnings\n")
            for w in self._warnings:
                L.append(f"- {w}")

        L.append("\n## Outputs\n")
        L.append("| File | Kind | Description |")
        L.append("|---|---|---|")
        for a in self.artifacts:
            extra = f" ({a['rows']} rows)" if a.get("rows") is not None else ""
            L.append(f"| `{a['file']}` | {a['kind']} | {a.get('desc','')}{extra} |")

        tables = [a for a in self.artifacts if a["kind"] == "table"]
        if tables:
            L.append("\n## Table previews\n")
            for a in tables:
                L.append(f"### `{a['file']}` — {a.get('desc','')}\n")
                L.append(a.get("preview", "_(none)_"))
                ns = a.get("numeric_summary")
                if ns:
                    L.append("\n<details><summary>Numeric summary</summary>\n")
                    L.append("| Column | count | mean | std | min | max |")
                    L.append("|---|---|---|---|---|---|")
                    for col, v in ns.items():
                        L.append(f"| {col} | {v.get('count')} | {v.get('mean')} | "
                                 f"{v.get('std')} | {v.get('min')} | {v.get('max')} |")
                    L.append("\n</details>")
                L.append("")

        if self.params:
            L.append("\n## Run settings\n```json")
            L.append(json.dumps(self.params, ensure_ascii=False, indent=2, default=str))
            L.append("```")

        if self.notes:
            L.append("\n## Points to verify\n")
            for n in self.notes:
                L.append(f"- {n}")

        return "\n".join(L) + "\n"


class GateFailure(RuntimeError):
    """A blocking check failed; the next stage must not run."""


