// Copy-to-clipboard reproductions: one input on a result page, as a script
// that runs in a fresh session and prints what this site recorded beside what
// the reader's own machine computes.
//
// Every snippet mirrors the harness's own call for its cell (the sweeps under
// anvl-sweeps/benchmarks/api-distributions/sweeps) -- the same function, the
// same parameters, a length-1 array like the sweep's length-n one, and for a
// gradient the same jit(gradient(...)) over all three arguments -- so a
// disagreement with the recorded figure means the platform, not the snippet.
//
// Three things here are easy to get wrong and each was measured:
//
//   - R's decimal parser is not correctly rounded where long double is 64
//     bits (Apple Silicon): ~80% of random 17-digit literals parse to a
//     neighbouring double. Hex literals are exact for normal doubles, and
//     R's parser flushes a hex subnormal to 0, so a subnormal is written as
//     an integer times 2^-1074, which is exact. The decimal goes in a comment.
//   - The backend and the reference see different parameters in an f32 cell:
//     the backend gets the full-precision doubles (and rounds them itself),
//     the reference gets them rounded to f32 (summary.ref_params).
//   - The error figures follow the harness's score_pair() and ulp_size():
//     against the unrounded double reference, ulp at the result's precision.

import { cellParts, flagList } from "./fmt.js";

const PREC = 256;

// --- exact literals -----------------------------------------------------

const f64Bits = (v) => {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, v);
  return dv.getBigUint64(0);
};

/** An R expression that evaluates to exactly `v`. */
export function rNum(v) {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Inf";
  if (v === -Infinity) return "-Inf";
  if (v === 0) return Object.is(v, -0) ? "-0" : "0";
  // integers accumulate exactly in R's parser below 2^53
  if (Number.isInteger(v) && Math.abs(v) < 2 ** 53) return String(v);
  const b = f64Bits(v);
  const sign = b >> 63n ? "-" : "";
  const e = Number((b >> 52n) & 0x7ffn);
  const m = b & ((1n << 52n) - 1n);
  if (e === 0) return `${sign}${m} * 2^-1074`;
  const frac = m.toString(16).padStart(13, "0").replace(/0+$/, "");
  const p = e - 1023;
  return `${sign}0x1${frac ? "." + frac : ""}p${p < 0 ? "" : "+"}${p}`;
}

/** A Python float literal for exactly `v` (Python's parser is correctly rounded). */
export function pyNum(v) {
  if (Number.isNaN(v)) return "math.nan";
  if (v === Infinity) return "math.inf";
  if (v === -Infinity) return "-math.inf";
  if (Object.is(v, -0)) return "-0.0";
  const s = String(v);
  return /[.e]/.test(s) ? s : s + ".0";
}

/** The shortest decimal that round-trips, for comments. */
export const dec = (v) => {
  if (v === null || v === undefined) return "—";
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Inf";
  if (v === -Infinity) return "-Inf";
  if (Object.is(v, -0)) return "-0";
  return String(v);
};

/** "mean=-0x1.921fb6p+1;sd=0x1p+0" -> { mean: -3.14159..., sd: 1 } */
function parseHexParams(s) {
  const out = {};
  for (const kv of String(s ?? "").split(";").filter(Boolean)) {
    const [k, hex] = kv.split("=");
    out[k] = hexToNum(hex);
  }
  return out;
}

function hexToNum(s) {
  const m = /^([+-]?)0x([0-9a-f]+)(?:\.([0-9a-f]*))?p([+-]?\d+)$/i.exec(String(s).trim());
  if (!m) return Number(s);
  const frac = m[3] ?? "";
  const mant = BigInt("0x" + m[2] + frac);
  const e = Number(m[4]) - 4 * frac.length;
  // mant has at most 53 significant bits, so Number(mant) is exact; the scale
  // is split so neither factor under- or overflows on the way.
  const v = Number(mant) * 2 ** Math.max(e, -1000) * 2 ** Math.min(0, e + 1000);
  return m[1] === "-" ? -v : v;
}

// --- what each function is ----------------------------------------------

// The parameter sets as the harness writes them (NORM_PARAMS, UNIF_INTERVALS):
// the backend is handed these doubles, whatever the cell's precision.
const PARAM_SETS = {
  standard: { mean: 0, sd: 1 },
  shifted: { mean: -Math.PI, sd: 2 * Math.PI },
  unit: { min: 0, max: 1 },
  wide: { min: -Math.PI, max: 2 * Math.PI },
};

const NAMED = [
  [Math.PI, "pi", "math.pi"], [-Math.PI, "-pi", "-math.pi"],
  [2 * Math.PI, "2 * pi", "2 * math.pi"], [-2 * Math.PI, "-2 * pi", "-2 * math.pi"],
];
const rParam = (v) => NAMED.find(([n]) => n === v)?.[1] ?? rNum(v);
const pyParam = (v) => NAMED.find(([n]) => n === v)?.[2] ?? pyNum(v);

const NORM = ["mean", "sd"];
const UNIF = ["min", "max"];
const SPECS = {
  nv_dnorm: { arg: "x", params: NORM, base: "dnorm", flags: ["log"], jax: "norm" },
  nv_pnorm: { arg: "q", params: NORM, base: "pnorm", flags: ["lower_tail", "log_p"], jax: "norm" },
  nv_qnorm: { arg: "p", params: NORM, base: "qnorm", flags: ["lower_tail", "log_p"], jax: "norm" },
  nv_dunif: { arg: "x", params: UNIF, base: "dunif", flags: ["log"], jax: "uniform" },
  nv_punif: { arg: "q", params: UNIF, base: "punif", flags: ["lower_tail", "log_p"], jax: "uniform" },
  nv_qunif: { arg: "p", params: UNIF, base: "qunif", flags: ["lower_tail", "log_p"], jax: "uniform" },
};

const BASE_FLAG = { log: "log", lower_tail: "lower.tail", log_p: "log.p" };

function parseFlags(flags) {
  const out = {};
  for (const f of flagList(flags)) {
    const [k, v] = f.split("=");
    out[k] = v === "TRUE";
  }
  return out;
}

/** The JAX function for a value cell, in the harness's parameterisation. */
function jaxFn(spec, f) {
  const lower = f.lower_tail !== false;
  switch (spec) {
    case "nv_dnorm": return f.log ? "norm.logpdf" : "norm.pdf";
    case "nv_pnorm": return lower ? (f.log_p ? "norm.logcdf" : "norm.cdf") : (f.log_p ? "norm.logsf" : "norm.sf");
    case "nv_qnorm": return lower ? "norm.ppf" : "norm.isf";
    case "nv_dunif": return f.log ? "uniform.logpdf" : "uniform.pdf";
    case "nv_punif": return "uniform.cdf";
    case "nv_qunif": return "uniform.ppf";
  }
  return null;
}

// --- the 256-bit truth ----------------------------------------------------
//
// Ported from the harness's own MPFR truths (ref_grad_mpfr, ref_stable_mpfr
// and the helpers in _normal.R), for one finite input. The value truths for
// which the harness has no MPFR function are the same mathematics. Parameters
// are the reference's (X, and M/S or A/B, all exact mpfr of doubles).

const NORMAL_HELPERS = {
  phi: `phi <- function(z) exp(-z^2 / 2) / sqrt(2 * Rmpfr::Const("pi", ${PREC}))`,
  // phi(z)/Phi(z); the continued fraction below -30, where Phi would need exp()
  imills: `imills <- function(z) {  # phi(z) / Phi(z); a continued fraction below -30
  if (z >= -30) return(phi(z) / Rmpfr::pnorm(z))
  v <- -z
  for (n in 120:1) v <- -z + n / v
  v
}`,
  logPhi: `log_Phi <- function(z) {  # log Phi(z), without exp() in the far lower tail
  if (z < -30) return(-z^2 / 2 - log(sqrt(2 * Rmpfr::Const("pi", ${PREC}))) - log(imills(z)))
  if (z > 0) log1p(-Rmpfr::pnorm(-z)) else log(Rmpfr::pnorm(z))
}`,
};

/** { helpers: [names], lines: [R lines], expr: R expression giving the truth } */
function truthR(spec, kind, output, f) {
  const lower = f.lower_tail !== false;
  const s = lower ? 1 : -1;
  const h = new Set();
  const L = [];
  let expr;
  const S = s === 1 ? "" : "-";
  if (spec === "nv_dnorm") {
    L.push("z <- (X - M) / S");
    if (kind === "value") {
      if (f.log) expr = `-z^2 / 2 - log(S) - log(sqrt(2 * Rmpfr::Const("pi", ${PREC})))`;
      else { h.add("phi"); expr = "phi(z) / S"; }
    } else {
      const core = { x: "-z / S", mean: "z / S", sd: "(z^2 - 1) / S" }[output];
      if (f.log) expr = core;
      else { h.add("phi"); expr = `${core} * phi(z) / S`; }
    }
  } else if (spec === "nv_pnorm") {
    L.push("z <- (X - M) / S");
    const zs = lower ? "z" : "-z";
    if (kind === "value") {
      if (f.log_p) { h.add("phi"); h.add("imills"); h.add("logPhi"); expr = `log_Phi(${zs})`; }
      else expr = `Rmpfr::pnorm(${zs})`;
    } else if (f.log_p) {
      h.add("phi"); h.add("imills");
      if (output === "sd") {
        L.push(`zs <- ${zs}`);
        L.push(`r <- if (zs >= -30) phi(zs) * z / S / Rmpfr::pnorm(zs) else z * imills(zs) / S`);
        expr = `${s === 1 ? "-" : ""}r`;
      } else {
        const sign = (output === "q") === lower ? "" : "-";
        expr = `${sign}imills(${zs}) / S`;
      }
    } else {
      h.add("phi");
      if (output === "sd") expr = `${S === "" ? "-" : ""}phi(z) * z / S`;
      else {
        const sign = (output === "q") === lower ? "" : "-";
        expr = `${sign}phi(z) / S`;
      }
    }
  } else if (spec === "nv_qnorm") {
    h.add("phi");
    if (f.log_p) { h.add("imills"); h.add("logPhi"); }
    // Newton from base R's double, four steps: quadratic convergence from ~1e-16
    const step = f.log_p
      ? `z <- z - (log_Phi(${S}z) - X) / (${s === 1 ? "" : "-"}imills(${S}z))`
      : `z <- z - (Rmpfr::pnorm(${S}z) - X) / (${s === 1 ? "" : "-"}phi(${S}z))`;
    L.push(`z0 <- suppressWarnings(qnorm(x, lower.tail = ${lower ? "TRUE" : "FALSE"}, log.p = ${f.log_p ? "TRUE" : "FALSE"}))  # base R's start`);
    L.push(`z <- Rmpfr::mpfr(z0, ${PREC})`);
    L.push(`if (is.finite(z0)) for (i in 1:4) ${step}  # Newton, in MPFR`);
    const inRange = f.log_p ? "x <= 0" : "x >= 0 && x <= 1";
    if (kind === "value") expr = "M + S * z";
    else {
      const d = { p: f.log_p ? `${S}S / imills(${S}z)` : `${S}S / phi(z)`, mean: "Rmpfr::mpfr(1, " + PREC + ")", sd: "z" }[output];
      expr = `if (${inRange}) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_dunif") {
    L.push("W <- B - A");
    if (kind === "value") {
      expr = f.log ? "if (x >= a && x <= b) -log(W) else Rmpfr::mpfr(-Inf, " + PREC + ")"
        : "if (x >= a && x <= b) 1 / W else Rmpfr::mpfr(0, " + PREC + ")";
    } else {
      const d = { x: `Rmpfr::mpfr(0, ${PREC})`, min: f.log ? "1 / W" : "1 / W^2", max: f.log ? "-1 / W" : "-1 / W^2" }[output];
      expr = `if (x >= a && x <= b) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_punif") {
    L.push("W <- B - A");
    if (kind === "value") {
      const lo = lower ? (f.log_p ? "-Inf" : "0") : (f.log_p ? "0" : "1");
      const hi = lower ? (f.log_p ? "0" : "1") : (f.log_p ? "-Inf" : "0");
      L.push(`small <- ${lower ? "(X - A) / W" : "(B - X) / W"}  # the tail that is small, directly`);
      L.push(`other <- ${lower ? "(B - X) / W" : "(X - A) / W"}`);
      const mid = f.log_p ? "if (small <= 0.5) log(small) else log1p(-other)" : "small";
      expr = `if (x <= a) Rmpfr::mpfr(${lo}, ${PREC}) else if (x >= b) Rmpfr::mpfr(${hi}, ${PREC}) else ${mid}`;
    } else {
      let d;
      if (!f.log_p) {
        const sg = lower ? "-" : "";
        const ng = lower ? "" : "-";
        d = { q: `${ng}1 / W`, min: `${sg}(B - X) / W^2`, max: `${sg}(X - A) / W^2` }[output];
      } else if (lower) {
        d = { q: "1 / (X - A)", min: "-(B - X) / (W * (X - A))", max: "-1 / W" }[output];
      } else {
        d = { q: "-1 / (B - X)", min: "1 / W", max: "(X - A) / (W * (B - X))" }[output];
      }
      expr = `if (x > a && x < b) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_qunif") {
    L.push("W <- B - A");
    const inRange = f.log_p ? "x <= 0" : "x >= 0 && x <= 1";
    const u = lower ? (f.log_p ? "exp(X)" : "X") : (f.log_p ? "-expm1(X)" : "1 - X");
    const u1 = lower ? (f.log_p ? "-expm1(X)" : "1 - X") : (f.log_p ? "exp(X)" : "X");
    if (kind === "value") {
      expr = `if (${inRange}) A + W * (${u}) else Rmpfr::mpfr(NaN, ${PREC})`;
    } else {
      const dp = `${lower ? "" : "-"}W${f.log_p ? " * exp(X)" : ""}`;
      const d = { p: dp, min: u1, max: u }[output];
      expr = `if (${inRange}) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  }
  const order = ["phi", "imills", "logPhi"];
  return { helpers: order.filter((k) => h.has(k)).map((k) => NORMAL_HELPERS[k]), lines: L, expr };
}

// --- the scoring, as the harness does it ------------------------------------

const R_ERR = `# relative and ulp error exactly as the site scores them (score_pair, ulp_size):
# against the unrounded reference g, ulp at precision p with minimum exponent emin
err <- function(f, g, p, emin) {
  if (isTRUE(f == g) || (is.nan(f) && is.nan(g))) return(c(rel = 0, ulp = 0))
  d <- abs(f - g)
  if (is.infinite(d) && is.finite(f) && is.finite(g)) d <- 2 * abs(f / 2 - g / 2)
  e <- floor(log2(abs(g)))
  e <- e - (2^e > abs(g)) + (2^(e + 1) <= abs(g))
  out <- c(rel = d / abs(g), ulp = d / 2^(max(e, emin) - p))
  if (!is.finite(out[["rel"]])) out[] <- Inf
  out
}
show_err <- function(label, f, g, p, emin) {
  if (is.na(g) && !is.nan(g)) return(cat(sprintf("%-18s (no MPFR value at a non-finite input)\\n", label)))
  e <- err(f, g, p, emin)
  cat(sprintf("%-18s rel %.3g   %.3g ulp\\n", label, e[["rel"]], e[["ulp"]]))
}`;

const PY_ERR = `# relative and ulp error exactly as the site scores them (score_pair, ulp_size):
# against the unrounded reference g, ulp at precision p with minimum exponent emin
def err(f, g, p, emin):
    if f == g or (math.isnan(f) and math.isnan(g)):
        return 0.0, 0.0
    d = abs(f - g)
    if math.isinf(d) and math.isfinite(f) and math.isfinite(g):
        d = 2 * abs(f / 2 - g / 2)
    if math.isnan(d) or not math.isfinite(g) or g == 0 or math.isinf(d / abs(g)):
        return math.inf, math.inf
    e = math.frexp(abs(g))[1] - 1  # the exact binary exponent of g
    return d / abs(g), d / 2.0 ** (max(e, emin) - p)

def show_err(label, f, g, p, emin):
    if g is None:
        print(f"{label:<18} (no MPFR value at a non-finite input)")
        return
    e = err(f, g, p, emin)
    print(f"{label:<18} rel {e[0]:.3g}   {e[1]:.3g} ulp")`;

// The sweep evaluates batches, and XLA compiles a lone element differently: at
// n = 1 it keeps (q - min) / (max - min) a true division, at n >= 2 it
// multiplies by a hoisted reciprocal, 1 ulp apart (nv_punif, measured). Where
// in the batch the input sits does not matter.
// Printed before the results: base R, XLA and libm are built differently per
// platform (e.g. fused multiply-add), so a value can differ by an ulp or two.
const CAVEAT = 'Note: results may differ from the web interface unless run on exactly the same setup as the benchmark environment.';
const BATCH_R = "n <- 1024  # evaluated in a batch, as the sweep is: XLA compiles a lone element differently";
const BATCH_PY = "n = 1024  # evaluated in a batch, as the sweep is: XLA compiles a lone element differently";

// --- the snippet --------------------------------------------------------------

/**
 * @param ctx.result    the summary row of the result whose figures these are
 * @param ctx.run       its runs row, or undefined
 * @param ctx.x, ctx.bits  the input
 * @param ctx.recorded  [[label, value], ...] figures this site recorded for it
 * @param ctx.where     a few words on which table the input came from
 * @returns {{ lang: "R" | "Python", text: string }}
 */
export function snippet(ctx) {
  const r = ctx.result;
  const parts = cellParts(r.cell_id);
  const sp = SPECS[r.spec];
  if (!sp) return null;
  const f = parseFlags(parts.flags);
  const dtype = parts.dtype;
  const kind = parts.kind;
  const output = r.output;
  const x = ctx.x;

  // Parameters: the backend's (full precision) and the reference's (summary).
  const refP = parseHexParams(r.ref_params);
  let beP = PARAM_SETS[parts.param_set];
  const round = dtype === "f32" ? Math.fround : (v) => v;
  if (!beP || !sp.params.every((k) => k in beP && k in refP && round(beP[k]) === refP[k])) beP = refP;
  const differ = sp.params.some((k) => beP[k] !== refP[k]);
  const isDefault = sp.params.every((k) => beP[k] === (k === "sd" || k === "max" ? 1 : 0));

  const [p1, p2] = sp.params;
  // Python names: min and max would shadow the builtins the scoring uses.
  const [q1, q2] = sp.params === UNIF ? ["mn", "mx"] : sp.params;
  const flagsR = sp.flags.map((k) => `${k} = ${f[k] ? "TRUE" : "FALSE"}`).join(", ");
  const baseFlags = sp.flags.map((k) => `${BASE_FLAG[k]} = ${f[k] ? "TRUE" : "FALSE"}`).join(", ");
  const prec = dtype === "f32" ? "23L, -126L" : "52L, -1022L";
  const precPy = dtype === "f32" ? "23, -126" : "52, -1022";
  const who = r.backend === "jax" ? "JAX" : "anvl";
  const what = kind === "grad" ? `d/d${output}` : "value";
  const refIsBase = kind === "value";

  const header = (c) => {
    const run = ctx.run;
    const rec = (ctx.recorded ?? []).filter(([, v]) => v !== null && v !== undefined);
    const w = Math.max(0, ...rec.map(([k]) => k.length));
    return [
      `${c} anvl-bench: ${r.spec} ${dtype} ${what}, ${parts.param_set}${parts.flags ? ", " + flagList(parts.flags).join(", ") : ""} (${who})`,
      `${c} ${ctx.where ? ctx.where + ": " : ""}${sp.arg} = ${dec(x)}${ctx.bits ? `  [bits ${ctx.bits}]` : ""}`,
      run ? `${c} Recorded on ${run.platform_key} (${run.cpu}), R ${run.r_version}, anvl ${run.anvl_version} @ ${String(run.anvl_sha ?? "").slice(0, 7)}, ${String(run.started_at ?? "").slice(0, 10)}:` : `${c} Recorded:`,
      ...(rec.length ? rec.map(([k, v]) => `${c}   ${k.padEnd(w)}  ${typeof v === "number" ? dec(v) : v}`) : [`${c}   (no figures recorded for this input)`]),
      kind === "grad" ? `${c} The site scores gradients against the harness's double reference (within 16 ulp of f64 truth), so the MPFR comparison below can differ from the recorded figures in the last few f64 ulp.` : null,
    ].filter((l) => l !== null);
  };

  // The truth in R, shared by both languages' snippets.
  const t = truthR(r.spec, kind, output, f);
  // How the reference's parameters are named in R: literals where the snippet
  // leaves the function's defaults implicit (a bare `mean` would be base::mean).
  const refExpr = (k) => (differ ? `ref_${k}` : isDefault ? rParam(refP[k]) : k);
  const refLits = sp.params.map((k) => `${k === p1 ? (sp.params === NORM ? "M" : "A") : (sp.params === NORM ? "S" : "B")} <- Rmpfr::mpfr(${refExpr(k)}, ${PREC})`);
  const refDouble = sp.params === UNIF ? [`a <- ${refExpr("min")}; b <- ${refExpr("max")}`] : [];
  const truthBlock = [
    `# the true value, to ${PREC} bits (Rmpfr); evaluated only at a finite input`,
    ...t.helpers,
    `mpfr_value <- NA_real_`,
    `if (is.finite(x)) {`,
    `  X <- Rmpfr::mpfr(x, ${PREC})`,
    ...refLits.map((l) => "  " + l),
    ...refDouble.map((l) => "  " + l),
    ...t.lines.map((l) => "  " + l),
    `  mpfr_value <- Rmpfr::asNumeric(${t.expr})`,
    `}`,
  ];

  const paramLines = (lang) => {
    if (isDefault && !differ) return lang === "R" ? [] : [`${q1}, ${q2} = ${pyParam(beP[p1])}, ${pyParam(beP[p2])}`];
    if (lang === "R") {
      const out = [`${p1} <- ${rParam(beP[p1])}; ${p2} <- ${rParam(beP[p2])}${differ ? "  # as the harness passes them to the backend, which rounds them to f32" : ""}`];
      if (differ) out.push(`ref_${p1} <- ${rNum(refP[p1])}; ref_${p2} <- ${rNum(refP[p2])}  # the same rounded to f32: what the reference is given`);
      return out;
    }
    return [`${q1}, ${q2} = ${pyParam(beP[p1])}, ${pyParam(beP[p2])}${differ ? "  # as the harness passes them; rounded to f32 by the array's dtype" : ""}`];
  };
  const refArgs = (isDefault && !differ) ? "" : differ ? `, ref_${p1}, ref_${p2}` : `, ${p1}, ${p2}`;
  const baseCall = `${sp.base}(x${refArgs}, ${baseFlags})`;
  const xComment = rNum(x) === dec(x) ? "" : `  # ${dec(x)}, written exactly: R's decimal parser is not correctly rounded everywhere`;

  // anvl's own evaluation, which the JAX snippet runs too: anvl is the
  // subject, and where JAX is out is exactly where anvl is worth a look.
  const beArgs = isDefault && !differ ? "" : `, ${p1}, ${p2}`;
  const call = kind === "value"
      ? [`anvl_value <- as.double(${r.spec}(nv_array(rep(x, n), dtype = "${dtype}")${beArgs}, ${flagsR}))[1]`]
      : [
        `grad_fn <- jit(gradient(`,
        `  \\(${sp.arg}, ${p1}, ${p2}) sum(${r.spec}(${sp.arg}, ${p1}, ${p2}, ${flagsR})),`,
        `  wrt = c("${sp.arg}", "${p1}", "${p2}")`,
        `))`,
        `g <- grad_fn(`,
        `  nv_array(rep(x, n), dtype = "${dtype}"),`,
        `  nv_array(rep(${isDefault && !differ ? rParam(beP[p1]) : p1}, n), dtype = "${dtype}"),`,
        `  nv_array(rep(${isDefault && !differ ? rParam(beP[p2]) : p2}, n), dtype = "${dtype}")`,
        `)`,
        `anvl_value <- as.double(g$${output})[1]`,
      ];

  if (r.backend !== "jax") {
    const lines = [
      ...header("#"),
      "",
      "library(anvl)",
      "",
      `x <- ${rNum(x)}${xComment}`,
      ...paramLines("R"),
      BATCH_R,
      ...call,
      refIsBase ? `base_value <- ${baseCall}` : null,
      "",
      ...truthBlock,
      "",
      R_ERR,
      "",
      `cat(${JSON.stringify(CAVEAT)}, "\\n", sep = "")`,
      `cat(sprintf("%-8s %.17g\\n", c("anvl", ${refIsBase ? '"base R", ' : ""}"MPFR"), c(anvl_value, ${refIsBase ? "base_value, " : ""}mpfr_value)), sep = "")`,
      refIsBase ? `show_err("anvl vs base R", anvl_value, base_value, ${prec})` : null,
      `show_err("anvl vs MPFR", anvl_value, mpfr_value, ${prec})`,
      refIsBase ? `show_err("base R vs MPFR", base_value, mpfr_value, 52L, -1022L)` : null,
    ].filter((l) => l !== null);
    return { lang: "R", text: lines.join("\n") + "\n" };
  }

  // JAX: the value from Python, base R and the truth from R through Rscript.
  const fn = jaxFn(r.spec, f);
  const unif = sp.params === UNIF;
  const jdt = dtype === "f32" ? "jnp.float32" : "jnp.float64";
  const jaxCall = kind === "value"
    ? [`jax_value = float(${fn}(jnp.full(n, x, dtype=${jdt}), ${unif ? "mn, mx - mn" : "mean, sd"})[0])`]
    : [
      unif
        ? `f = lambda ${sp.arg}, mn, mx: ${fn}(${sp.arg}, mn, mx - mn)  # restated in (min, max) before differentiating`
        : `f = ${fn}`,
      `g = jax.jit(jax.vmap(jax.grad(f, argnums=${{ [sp.arg]: 0, [p1]: 1, [p2]: 2 }[output]}), in_axes=(0, None, None)))`,
      `jax_value = float(g(jnp.full(n, x, dtype=${jdt}), ${q1}, ${q2})[0])`,
    ];
  const rCode = [
    "library(anvl)",
    `x <- ${rNum(x)}`,
    ...paramLines("R"),
    BATCH_R,
    ...call,
    refIsBase ? `base_value <- ${baseCall}` : null,
    ...truthBlock,
    `cat(sprintf("%.17g", c(anvl_value, ${refIsBase ? "base_value, " : ""}mpfr_value)))`,
  ].filter((l) => l !== null);
  const lines = [
    ...header("#"),
    "",
    "import math",
    "import subprocess",
    "",
    "import jax",
    'jax.config.update("jax_enable_x64", True)  # before any array exists, or f64 silently becomes f32',
    "import jax.numpy as jnp",
    `from jax.scipy.stats import ${sp.jax}`,
    "",
    `x = ${pyNum(x)}`,
    ...paramLines("Python"),
    BATCH_PY,
    ...jaxCall,
    "",
    `# anvl, ${refIsBase ? "base R " : ""}and the ${PREC}-bit truth at the same input, from R (needs anvl and Rmpfr)`,
    'r_code = r"""',
    ...rCode,
    '"""',
    'out = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True, check=True).stdout',
    `anvl_value, ${refIsBase ? "base_value, " : ""}mpfr_value = [None if v == "NA" else float(v) for v in out.split()]`,
    "",
    PY_ERR,
    "",
    `print(${JSON.stringify(CAVEAT)})`,
    `for label, v in [("JAX", jax_value), ("anvl", anvl_value), ${refIsBase ? '("base R", base_value), ' : ""}("MPFR", mpfr_value)]:`,
    `    print(f"{label:<8} {v!r}")  # repr: the shortest decimal that round-trips`,
    refIsBase ? `show_err("anvl vs base R", anvl_value, base_value, ${precPy})` : null,
    `show_err("anvl vs MPFR", anvl_value, mpfr_value, ${precPy})`,
    refIsBase ? `show_err("JAX vs base R", jax_value, base_value, ${precPy})` : null,
    `show_err("JAX vs MPFR", jax_value, mpfr_value, ${precPy})`,
    refIsBase ? `show_err("base R vs MPFR", base_value, mpfr_value, 52, -1022)` : null,
  ].filter((l) => l !== null);
  return { lang: "Python", text: lines.join("\n") + "\n" };
}
