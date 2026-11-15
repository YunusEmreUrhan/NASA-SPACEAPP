import { datasets, type Profile, type Site, type Target } from "../data/catalog";
import { compare } from "../analysis/similarity";
export function makeReport(site: Site, target: Target, profile: Profile) {
  return {
    generatedAt: new Date().toISOString(),
    site,
    target,
    profile,
    result: compare(site, target, profile),
    datasets,
    disclaimer:
      "Heuristic screening only. Coverage is not confidence. One crater-scale variable cannot establish overall analogy or base suitability. Unknown measurement uncertainty; no Earth site reproduces extraterrestrial gravity, vacuum, or radiation.",
  };
}
export type Report = ReturnType<typeof makeReport>;
function download(text: string, type: string, extension: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `planetary-analog-comparison.${extension}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const csv = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
const escape = (v: unknown) =>
  String(v ?? "—").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!,
  );
export function exportReport(report: Report, format: "json" | "csv" | "html") {
  if (format === "json")
    return download(JSON.stringify(report, null, 2), "application/json", "json");
  if (format === "csv") {
    const meta = [
      ["Method", report.result.method],
      ["Analysis date", report.generatedAt],
      ["Earth", report.site.name],
      ["Target", report.target.name],
      ["Objective", report.profile.label],
      ["Score", report.result.score],
      ["Coverage %", report.result.coverage],
      ["Caveat", report.disclaimer],
    ];
    const rows = report.result.rows.map((r) => [
      r.label,
      r.earth,
      r.target,
      r.unit,
      r.weight,
      r.similarity,
      r.status,
      r.exclusion,
      r.sources
        .map((id) => datasets.find((d) => d.id === id)?.officialDocumentationUrl)
        .join(" | "),
    ]);
    return download(
      [
        ...meta,
        [],
        [
          "Criterion",
          "Earth",
          "Target",
          "Unit",
          "Weight",
          "Similarity 0–1",
          "Status",
          "Excluded because",
          "Sources",
        ],
        ...rows,
      ]
        .map((r) => r.map(csv).join(","))
        .join("\r\n"),
      "text/csv",
      "csv",
    );
  }
  const text = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Planetary Analog Lab — Report</title><style>body{font:15px/1.7 system-ui;max-width:1000px;margin:60px auto;color:#17232a;padding:20px}h1{letter-spacing:-1px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #ccc;padding:12px}small{color:#555}@media print{body{margin:0}}</style><h1>PLANETARY ANALOG LAB</h1><p>${escape(report.site.name)} ↔ ${escape(report.target.name)}</p><p>${escape(report.profile.label)} · ${escape(report.generatedAt)} · ${escape(report.result.method)}</p><h2>Screening score: ${report.result.score?.toFixed(1) ?? "Not supported"} / 100 · Coverage: ${report.result.coverage.toFixed(1)}%</h2><p>${escape(report.disclaimer)}</p><p>Method: s = min(a,b)/max(a,b); S = 100 Σ(w s)/Σ(w available); C = 100 Σ(w available)/Σ(w required). Published whole-crater diameters only; no imputation. Spatial delineations and epochs differ; formal error bars are unknown.</p><table><tr><th>Criterion</th><th>Earth</th><th>Target</th><th>Unit</th><th>Weight</th><th>Similarity</th><th>Status / exclusion</th></tr>${report.result.rows.map((r) => `<tr><td>${escape(r.label)}</td><td>${escape(r.earth)}</td><td>${escape(r.target)}</td><td>${escape(r.unit)}</td><td>${r.weight}</td><td>${r.similarity?.toFixed(4) ?? "—"}</td><td>${escape(r.exclusion ?? r.status)}</td></tr>`).join("")}</table><h2>Differences</h2><p>${escape(report.site.limitation)}</p><p>${escape(report.target.context)}</p><h2>Provenance</h2>${report.datasets.map((d) => `<p><b>${escape(d.title)}</b> — ${escape(d.provider)}<br><a href="${escape(d.officialDocumentationUrl)}">${escape(d.officialDocumentationUrl)}</a><br>${escape(d.temporalCoverage)} · ${escape(d.spatialResolution)} · ${escape(d.license)}<br>${escape(d.limitations)}</p>`).join("")}<small>Independent Space Apps project. Not affiliated with or endorsed by NASA. Use your browser’s Print command to save as PDF.</small></html>`;
  download(text, "text/html", "html");
}
