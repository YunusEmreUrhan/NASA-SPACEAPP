import { describe, expect, it } from "vitest";
import { compare, ratioSimilarity, weightedScore } from "./similarity";
import { sites, targets, profiles } from "../data/catalog";
describe("ratio normalization", () => {
  it("identical and deliberately dissimilar", () => {
    expect(ratioSimilarity(20, 20)).toBe(1);
    expect(ratioSimilarity(1, 100)).toBe(0.01);
  });
  it("is symmetric", () => expect(ratioSimilarity(20, 21)).toBe(ratioSimilarity(21, 20)));
  it("rejects invalid and zero observations", () => {
    for (const n of [0, -1, NaN, Infinity]) expect(ratioSimilarity(n, 20)).toBeNull();
  });
});
describe("weighted screening", () => {
  it("normalizes weights", () =>
    expect(
      weightedScore([
        { weight: 2, similarity: 1 },
        { weight: 6, similarity: 0 },
      ]).score,
    ).toBe(25));
  it("omits missing values but reduces coverage", () =>
    expect(
      weightedScore([
        { weight: 1, similarity: 1 },
        { weight: 3, similarity: null },
      ]),
    ).toEqual({ score: 100, coverage: 25 }));
  it("handles zero weights", () =>
    expect(weightedScore([{ weight: 0, similarity: 1 }])).toEqual({ score: null, coverage: 0 }));
  it("excludes malformed inputs", () =>
    expect(
      weightedScore([
        { weight: NaN, similarity: 1 },
        { weight: 1, similarity: 2 },
      ]),
    ).toEqual({ score: null, coverage: 0 }));
  it("stays bounded", () => {
    for (let a = 1; a < 100; a++) {
      const s = weightedScore([{ weight: a, similarity: a / 100 }]);
      expect(s.score).toBeGreaterThanOrEqual(0);
      expect(s.score).toBeLessThanOrEqual(100);
    }
  });
  it("is repeatable with a sourced incomplete pair", () => {
    const r = compare(sites[0], targets[0], profiles[0]);
    expect(r).toEqual(compare(sites[0], targets[0], profiles[0]));
    expect(r.score).toBeCloseTo(95.238);
    expect(r.coverage).toBe(25);
  });
  it("does not rank an unsupported site", () =>
    expect(compare(sites[1], targets[0], profiles[0]).score).toBeNull());
  it("shows the larger Mars crater as less similar in scale", () =>
    expect(compare(sites[0], targets[1], profiles[0]).score).toBeCloseTo(44.4444));
  it("does not score a target with no compatible observations", () =>
    expect(
      compare(sites[0], { ...targets[1], diameter: undefined }, profiles[0]).score,
    ).toBeNull());
});
