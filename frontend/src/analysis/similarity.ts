import { criteria, type Profile, type Site, type Target } from "../data/catalog";
export const METHOD_VERSION = "PAL-screening-1.0.0";
export function ratioSimilarity(a: number, b: number): number | null {
  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0) return null;
  return Math.min(a, b) / Math.max(a, b);
}
export function weightedScore(values: { weight: number; similarity: number | null }[]) {
  const validWeight = (n: number) => (Number.isFinite(n) && n > 0 ? n : 0);
  const required = values.reduce((n, v) => n + validWeight(v.weight), 0);
  const valid = values.filter(
    (v) =>
      validWeight(v.weight) &&
      v.similarity !== null &&
      Number.isFinite(v.similarity) &&
      v.similarity >= 0 &&
      v.similarity <= 1,
  );
  const available = valid.reduce((n, v) => n + v.weight, 0);
  return {
    score: available
      ? (100 * valid.reduce((n, v) => n + v.weight * v.similarity!, 0)) / available
      : null,
    coverage: required ? (100 * available) / required : 0,
  };
}
export function compare(site: Site, target: Target, profile: Profile) {
  const rows = criteria.map((c) => {
    const earth = c.id === "diameter" ? site.diameter : undefined;
    const other = c.id === "diameter" ? target.diameter : undefined;
    const similarity =
      earth && other && earth.unit === other.unit
        ? ratioSimilarity(earth.value, other.value)
        : null;
    return {
      ...c,
      weight: profile.weights[c.id],
      earth: earth?.value ?? null,
      target: other?.value ?? null,
      similarity,
      sources: [earth?.datasetId, other?.datasetId].filter(Boolean),
      status: similarity === null ? "UNAVAILABLE" : "PROCESSED",
      exclusion:
        similarity === null ? "Missing compatible paired observations; not imputed." : null,
    };
  });
  return { method: METHOD_VERSION, ...weightedScore(rows), rows };
}
