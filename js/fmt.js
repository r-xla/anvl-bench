// Formatting. Everything here has to survive values JSON could not carry:
// NaN, both infinities and signed zero all occur in this data and all mean
// something specific, so none of them may be flattened into "0" or "-".

export const MINUS = "−";

const sign = (s) => s.replace(/-/g, MINUS);

/** A measurement. `null` is absent data; NaN/Inf/-0 are findings, not noise. */
export function num(v, sig = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "∞";
  if (v === -Infinity) return MINUS + "∞";
  if (Object.is(v, -0)) return MINUS + "0";
  if (v === 0) return "0";
  const a = Math.abs(v);
  return sign(a >= 1e6 || a < 1e-4 ? v.toExponential(sig - 1) : v.toPrecision(sig));
}

/** Counts, with thin-space grouping so long digit strings stay readable. */
export function int(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Math.round(v).toLocaleString("en-US").replace(/,/g, " ");
}

/** A count short enough for a tile: 24.6k, 7.8M. Exact figures go in a tooltip. */
export function compact(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "\u2014";
  const a = Math.abs(v);
  if (a < 1e3) return String(Math.round(v));
  if (a < 1e6) return (v / 1e3).toFixed(a < 1e4 ? 1 : 0) + "k";
  if (a < 1e9) return (v / 1e6).toFixed(a < 1e7 ? 1 : 0) + "M";
  return (v / 1e9).toFixed(a < 1e10 ? 1 : 0) + "G";
}

export function pct(v, dp = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (v > 0 && v < 10 ** -dp / 100) return "<" + (10 ** -dp / 100 * 100).toFixed(dp) + "%";
  return (100 * v).toFixed(dp) + "%";
}

/**
 * An error decade: `hist` counts samples whose relative error lies in
 * [10^d, 10^(d+1)). The same axis appears in `bands` as m = -d - 1, which the
 * site does not use -- `bands` also carries the continuous worst_rel_err, and
 * charting that rather than its decade keeps the structure within a decade.
 */
export const decadeTick = (d) => `1e${sign(String(d))}`;

/** cell_id is spec/backend/dtype/kind/param_set/flags. */
export function cellParts(cellId) {
  const [spec, backend, dtype, kind, param_set, ...rest] = String(cellId).split("/");
  return { spec, backend, dtype, kind, param_set, flags: rest.join("/") };
}

/** Flags arrive as `lower_tail=FALSE,log_p=TRUE`; render them as chips. */
export const flagList = (flags) => (flags ? String(flags).split(",") : []);
