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

// The parameter sets as the harness writes them (NORM_PARAMS, UNIF_INTERVALS,
// EXP_PARAMS, BINOM_D_PARAMS, BINOM_PQ_PARAMS), per family since set names
// repeat across families (and within the binomial, where the density's
// `large` is not the distribution function's): the backend
// is handed these doubles, whatever the cell's precision. Each is the same
// correctly rounded product of doubles in JS as in R.
const NORM_SETS = { standard: { mean: 0, sd: 1 }, shifted: { mean: -Math.PI, sd: 2 * Math.PI } };
const UNIF_SETS = { unit: { min: 0, max: 1 }, wide: { min: -Math.PI, max: 2 * Math.PI } };
const EXP_SETS = { standard: { rate: 1 }, small: { rate: Math.PI * 1e-10 }, large: { rate: Math.PI * 1e10 } };
const BINOM_D_SETS = {
  tabulated: { x: 3, size: 10 }, next_to_zero: { x: 1, size: 10 }, next_to_size: { x: 9, size: 10 },
  at_zero: { x: 0, size: 1000 }, at_size: { x: 1000, size: 1000 },
  large: { x: 3e5, size: 1e6 }, huge: { x: 3e12, size: 1e13 },
};
const BINOM_PQ_SETS = {
  small: { size: 20, prob: 0.3 }, large: { size: 1e6, prob: 0.3 },
  skewed: { size: 1e6, prob: 1e-3 }, rare: { size: 1e10, prob: 1e-10 },
};

// Each parameter's default in the anvl and base R signatures, and its name in
// the MPFR truth. Python avoids min and max, which would shadow the builtins
// the scoring uses. The binomial's parameters have no defaults; its count x is
// K in the truth, since X is always the swept input.
const DEFAULT = { mean: 0, sd: 1, min: 0, max: 1, rate: 1 };
const MP_NAME = { mean: "M", sd: "S", min: "A", max: "B", rate: "R", x: "K", size: "N", prob: "P" };
const PY_NAME = { min: "mn", max: "mx" };

// The binomial probabilities as the harness writes them: each short decimal was
// checked to parse to exactly the same double as its hex form in R on Apple
// Silicon, where long 17-digit literals do not.
const NAMED = [
  [Math.PI, "pi", "math.pi"], [-Math.PI, "-pi", "-math.pi"],
  [2 * Math.PI, "2 * pi", "2 * math.pi"], [-2 * Math.PI, "-2 * pi", "-2 * math.pi"],
  [0.3, "0.3", "0.3"], [1e-3, "1e-3", "1e-3"], [1e-10, "1e-10", "1e-10"],
];
const rParam = (v) => NAMED.find(([n]) => n === v)?.[1] ?? rNum(v);
const pyParam = (v) => NAMED.find(([n]) => n === v)?.[2] ?? pyNum(v);

const NORM = ["mean", "sd"];
const UNIF = ["min", "max"];
const EXPO = ["rate"];
// jaxArgs: the JAX call's trailing arguments, in the harness's parameterisation
// (expon takes scale = 1/rate, formed inside the traced function as the
// harness does, so d/drate is comparable with anvl's).
const NORM_JAX = { jax: "norm", jaxArgs: "mean, sd", sets: NORM_SETS };
const UNIF_JAX = { jax: "uniform", jaxArgs: "mn, mx - mn", sets: UNIF_SETS };
const EXPO_JAX = { jax: "expon", jaxArgs: "scale=1 / rate", sets: EXP_SETS };
// The binomial departs from the others in ways the harness's calls dictate
// (sweeps/_binomial.R, dbinom.R, pbinom.R, qbinom.R):
//   - the density is swept in prob, not x, so the swept variable is `prob`
//     (`variable`) and sits last in the signature (`sig`);
//   - the density's count is an array of the cell's dtype, nv_scalar(x), since
//     it is the operand that sets the precision (`scalar`), and its gradient is
//     taken in prob alone (`wrt`), with x and size scalars too (`gradScalar`);
//   - every JAX twin takes size and prob at the cell's precision (`jaxTyped`),
//     and the distribution function's twin is the harness's own betainc
//     formula, there being no binom.cdf in jax.scipy.stats. There is no
//     quantile twin.
const BINOM = ["size", "prob"];
const BINOM_PQ = { params: BINOM, sets: BINOM_PQ_SETS, jaxTyped: true };
const SPECS = {
  nv_dnorm: { arg: "x", params: NORM, base: "dnorm", flags: ["log"], ...NORM_JAX },
  nv_pnorm: { arg: "q", params: NORM, base: "pnorm", flags: ["lower_tail", "log_p"], ...NORM_JAX },
  nv_qnorm: { arg: "p", params: NORM, base: "qnorm", flags: ["lower_tail", "log_p"], ...NORM_JAX },
  nv_dunif: { arg: "x", params: UNIF, base: "dunif", flags: ["log"], ...UNIF_JAX },
  nv_punif: { arg: "q", params: UNIF, base: "punif", flags: ["lower_tail", "log_p"], ...UNIF_JAX },
  nv_qunif: { arg: "p", params: UNIF, base: "qunif", flags: ["lower_tail", "log_p"], ...UNIF_JAX },
  nv_dexp: { arg: "x", params: EXPO, base: "dexp", flags: ["log"], ...EXPO_JAX },
  nv_pexp: { arg: "q", params: EXPO, base: "pexp", flags: ["lower_tail", "log_p"], ...EXPO_JAX },
  nv_qexp: { arg: "p", params: EXPO, base: "qexp", flags: ["lower_tail", "log_p"], ...EXPO_JAX },
  nv_dbinom: {
    arg: "prob", variable: "prob", params: ["x", "size"], sig: ["x", "size", "prob"], base: "dbinom", flags: ["log"],
    sets: BINOM_D_SETS, scalar: ["x"], gradScalar: ["x", "size"], wrt: ["prob"],
    jax: "binom", jaxImport: "from jax.scipy.stats import binom", jaxTyped: true,
  },
  nv_pbinom: {
    arg: "q", base: "pbinom", flags: ["lower_tail", "log_p"], ...BINOM_PQ,
    jax: "betainc", jaxImport: "from jax.scipy.special import betainc",
  },
  nv_qbinom: { arg: "p", base: "qbinom", flags: ["lower_tail", "log_p"], ...BINOM_PQ },
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
    case "nv_dexp": return f.log ? "expon.logpdf" : "expon.pdf";
    case "nv_pexp": return lower ? (f.log_p ? "expon.logcdf" : "expon.cdf") : (f.log_p ? "expon.logsf" : "expon.sf");
    case "nv_qexp": return "expon.ppf";  // the only quantile JAX has; the harness pairs no other variant
  }
  return null;
}

// --- the 256-bit truth ----------------------------------------------------
//
// Ported from the harness's own MPFR truths (ref_grad_mpfr, ref_stable_mpfr
// and the helpers in _normal.R and _exponential.R), for one finite input. The
// value truths for which the harness has no MPFR function are the same
// mathematics. Parameters are the reference's (X, and M/S, A/B or R, all
// exact mpfr of doubles).

const HELPERS = {
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
  // mp_log1mexp: even at 256 bits 1 - exp(v) rounds to 1 below v ~ -177
  log1mexp: `log1mexp <- function(v) {  # log(1 - exp(v)), v <= 0, split at -log 2
  if (v > -log(2)) log(-expm1(v)) else log1p(-exp(v))
}`,
  // mp_binom_log_density: by lgamma, which at 256 bits keeps ~180 bits below
  // the point even at lgamma(1e13 + 1)
  binomLd: `binom_ld <- function(J, N, P) {  # log dbinom(J; N, P): J whole in [0, N], 0 < P < 1
  lgamma(N + 1) - lgamma(J + 1) - lgamma(N - J + 1) + J * log(P) + (N - J) * log1p(-P)
}`,
  // The harness's own MPFR distribution function (mp_log_pbinom_lower) is a
  // continued fraction for the lower log tail only. This is the same quantity
  // by its definition, for both tails: the mass summed outward from k, away
  // from the mode, where it falls monotonically, so the sum stops once a term
  // is below 2^-(precision + 8) of it. Each term is taken relative to the
  // first, so nothing underflows however far out k is.
  binomTail: `log_binom_tail <- function(k, N, P, lower) {  # log P(X <= k) or log P(X > k), whole 0 <= k < N
  n <- Rmpfr::asNumeric(N)  # exact: N is the mpfr of a double
  down <- k < floor((n + 1) * Rmpfr::asNumeric(P))  # below the mode, sum k, k - 1, ...; else k + 1, k + 2, ...
  j0 <- if (down) k else k + 1
  end <- if (down) 0 else n
  first <- binom_ld(Rmpfr::mpfr(j0, ${PREC}), N, P)
  total <- Rmpfr::mpfr(0, ${PREC})
  repeat {
    j <- seq(j0, by = if (down) -1 else 1, length.out = min(4096, abs(end - j0) + 1))
    t <- exp(binom_ld(Rmpfr::mpfr(j, ${PREC}), N, P) - first)
    total <- total + sum(t)
    j0 <- j[length(j)] + if (down) -1 else 1
    if (t[length(t)] < 2^-(${PREC} + 8) * total || (if (down) j0 < 0 else j0 > n)) break
  }
  near <- first + log(total)  # the lower tail if down, else the upper
  if (down == lower) near else log1mexp(near)
}`,
  // The quantile is the least whole k whose tail passes the target: from base
  // R's own answer, stepped down while the count below also passes, and up
  // until k does. Base R widens each comparison by 64 ulp (its fuzz), so at a
  // step of the distribution function it can answer one count off.
  binomQuantile: `binom_quantile <- function(q0, t, N, P, lower) {  # q0 base R's answer, t the target log probability
  passes <- function(k) {
    if (k >= Rmpfr::asNumeric(N)) return(TRUE)
    v <- log_binom_tail(k, N, P, lower)
    if (lower) v >= t else v <= t
  }
  k <- q0
  while (k > 0 && passes(k - 1)) k <- k - 1
  while (!passes(k)) k <- k + 1
  k
}`,
};

/** { helpers: [names], lines: [R lines], expr: R expression giving the truth } */
function truthR(spec, kind, output, f, refName = (k) => k) {
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
  } else if (spec === "nv_dexp") {
    // the support is closed, x == 0 included, as in nv_dexp()
    L.push("t <- R * X");
    if (kind === "value") {
      expr = f.log ? `if (x >= 0) log(R) - t else Rmpfr::mpfr(-Inf, ${PREC})`
        : `if (x >= 0) R * exp(-t) else Rmpfr::mpfr(0, ${PREC})`;
    } else {
      const d = f.log
        ? { x: "-R", rate: "(1 - t) / R" }[output]
        : { x: "-(R * R) * exp(-t)", rate: "(1 - t) * exp(-t)" }[output];
      expr = `if (x >= 0) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_pexp") {
    L.push("t <- R * X");
    if (kind === "value") {
      if (f.log_p && lower) h.add("log1mexp");
      const at0 = lower ? (f.log_p ? "-Inf" : "0") : (f.log_p ? "0" : "1");
      const v = lower ? (f.log_p ? "log1mexp(-t)" : "-expm1(-t)") : (f.log_p ? "-t" : "exp(-t)");
      expr = `if (x <= 0) Rmpfr::mpfr(${at0}, ${PREC}) else ${v}`;
    } else {
      // every derivative is 0 at and below q = 0, as in the harness's reference
      let d;
      if (!f.log_p) d = { q: `${S}R * exp(-t)`, rate: `${S}X * exp(-t)` }[output];
      else if (!lower) d = { q: "-R", rate: "-X" }[output];
      else d = { q: "R / expm1(t)", rate: "X / expm1(t)" }[output];
      expr = `if (x > 0) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_qexp") {
    // x = -L / rate, L the log of the upper-tail probability
    const inRange = f.log_p ? "x <= 0" : "x >= 0 && x <= 1";
    if (f.log_p && lower) h.add("log1mexp");
    const Lx = lower ? (f.log_p ? "log1mexp(X)" : "log1p(-X)") : (f.log_p ? "X" : "log(X)");
    if (kind === "value") {
      expr = `if (${inRange}) -(${Lx}) / R else Rmpfr::mpfr(NaN, ${PREC})`;
    } else {
      // one-sided at the ends of the range: the limits from inside it
      const dp = f.log_p
        ? (lower ? "1 / (R * expm1(abs(X)))" : "-1 / R")
        : (lower ? "1 / (R * (1 - X))" : "-1 / (R * abs(X))");
      const d = { p: dp, rate: `${Lx} / (R * R)` }[output];
      expr = `if (${inRange}) ${d} else Rmpfr::mpfr(0, ${PREC})`;
    }
  } else if (spec === "nv_dbinom") {
    // swept in prob, so X is the probability and K the count; prob the double
    const at0 = kind === "value"
      ? (f.log ? "if (K == 0) 0 else -Inf" : "if (K == 0) 1 else 0")
      : (f.log ? "if (K == 0) -N else Inf" : "N * ((K == 1) - (K == 0))");
    const at1 = kind === "value"
      ? (f.log ? "if (K == N) 0 else -Inf" : "if (K == N) 1 else 0")
      : (f.log ? "if (K == N) N else -Inf" : "N * ((K == N) - (K == N - 1))");
    h.add("binomLd");
    L.push(`out <- if (prob < 0 || prob > 1) ${kind === "value" ? "NaN" : "0"}`);
    L.push(`  else if (prob == 0) ${at0}  # the polynomial's limits at the ends of [0, 1]`);
    L.push(`  else if (prob == 1) ${at1}`);
    if (kind === "value") {
      L.push(`  else ${f.log ? "binom_ld(K, N, X)" : "exp(binom_ld(K, N, X))"}`);
    } else {
      // d/dp f = f d / (p (1 - p)), d = x - n p, exact at this precision
      L.push(`  else (K - N * X) / (X * (1 - X))${f.log ? "" : " * exp(binom_ld(K, N, X))"}`);
    }
    expr = `Rmpfr::mpfr(out, ${PREC})`;
  } else if (spec === "nv_pbinom") {
    h.add("binomLd"); h.add("log1mexp"); h.add("binomTail");
    // below and above the support, as base R resolves them
    const [below, above] = lower ? [f.log_p ? "-Inf" : "0", f.log_p ? "0" : "1"] : [f.log_p ? "0" : "1", f.log_p ? "-Inf" : "0"];
    L.push("k <- floor(x + 1e-7)  # the count, as base R takes it, in double");
    const tail = `log_binom_tail(k, N, P, lower = ${lower ? "TRUE" : "FALSE"})`;
    expr = `if (x < 0) Rmpfr::mpfr(${below}, ${PREC}) else if (k >= N) Rmpfr::mpfr(${above}, ${PREC}) else ${f.log_p ? tail : `exp(${tail})`}`;
  } else if (spec === "nv_qbinom") {
    h.add("binomLd"); h.add("log1mexp"); h.add("binomTail"); h.add("binomQuantile");
    const inRange = f.log_p ? "x <= 0" : "x >= 0 && x <= 1";
    const lt = lower ? "TRUE" : "FALSE";
    L.push(`q0 <- suppressWarnings(qbinom(x, ${refName("size")}, ${refName("prob")}, lower.tail = ${lt}, log.p = ${f.log_p ? "TRUE" : "FALSE"}))  # base R's start`);
    expr = `if (${inRange}) Rmpfr::mpfr(binom_quantile(q0, ${f.log_p ? "X" : "log(X)"}, N, P, lower = ${lt}), ${PREC}) else Rmpfr::mpfr(NaN, ${PREC})`;
  }
  const order = ["phi", "imills", "logPhi", "log1mexp", "binomLd", "binomTail", "binomQuantile"];
  return { helpers: order.filter((k) => h.has(k)).map((k) => HELPERS[k]), lines: L, expr };
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

// --- the binomial's JAX twins ---------------------------------------------------
//
// As jax_binom() in sweeps/_binomial.R: the density's twin is logpmf/pmf with
// prob the traced argument, its gradient vmapped over prob alone, and the
// distribution function's the betainc formula SciPy's binom.cdf and binom.sf
// use (Cephes bdtr and bdtrc), specialised here to this cell's flags. size,
// prob and the count are arrays at the cell's precision, as the harness
// passes them.
function binomJax(spec, kind, f, jdt) {
  const typed = (v) => `jnp.asarray(${v}, dtype=${jdt})`;
  if (spec === "nv_dbinom") {
    const fn = f.log ? "binom.logpmf" : "binom.pmf";
    const args = `jnp.full(n, prob, dtype=${jdt}), ${typed("x")}, ${typed("size")}`;
    return kind === "value"
      ? [
        `f = jax.jit(lambda p, k, size: ${fn}(k, size, p))  # swept in prob, as the harness does`,
        `jax_value = float(f(${args})[0])`,
      ]
      : [
        `g = jax.jit(jax.vmap(jax.grad(lambda p, k, size: ${fn}(k, size, p)), in_axes=(0, None, None)))`,
        `jax_value = float(g(${args})[0])`,
      ];
  }
  const lower = f.lower_tail !== false;
  const [below, above] = lower ? ["0.0", "1.0"] : ["1.0", "0.0"];
  return [
    "def cdf(q, size, prob):  # the harness's twin: no binom.cdf in jax.scipy.stats",
    "    k = jnp.floor(q + 1e-7)  # base R's count",
    "    inside = (k >= 0) & (k < size)",
    "    kk = jnp.where(inside, k, 0)",
    lower
      ? "    v = betainc(size - kk, kk + 1, 1 - prob)  # P(X <= k) = I_(1 - p)(n - k, k + 1)"
      : "    v = betainc(kk + 1, size - kk, prob)  # P(X > k) = I_p(k + 1, n - k)",
    `    v = jnp.where(k < 0, ${below}, jnp.where(k >= size, ${above}, v))`,
    "    v = jnp.where(jnp.isnan(q), jnp.nan, v)",
    `    return ${f.log_p ? "jnp.log(v)" : "v"}`,
    `jax_value = float(jax.jit(cdf)(jnp.full(n, x, dtype=${jdt}), ${typed("size")}, ${typed("prob")})[0])`,
  ];
}

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
  let beP = sp.sets[parts.param_set];
  const round = dtype === "f32" ? Math.fround : (v) => v;
  if (!beP || !sp.params.every((k) => k in beP && k in refP && round(beP[k]) === refP[k])) beP = refP;
  const differ = sp.params.some((k) => beP[k] !== refP[k]);
  const isDefault = sp.params.every((k) => beP[k] === DEFAULT[k]);
  const plain = isDefault && !differ;  // the call can leave every parameter at its default

  const P = sp.params;
  // The swept input's variable (x everywhere but the binomial density, which
  // is swept in prob), and every argument in the function's own order.
  const V = sp.variable ?? "x";
  const sig = sp.sig ?? [sp.arg, ...P];
  const pyName = (k) => PY_NAME[k] ?? k;
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

  // How the reference's parameters are named in R: literals where the snippet
  // leaves the function's defaults implicit (a bare `mean` would be base::mean).
  const refExpr = (k) => (differ ? `ref_${k}` : isDefault ? rParam(refP[k]) : k);
  // The truth in R, shared by both languages' snippets.
  const t = truthR(r.spec, kind, output, f, refExpr);
  const refLits = P.map((k) => `${MP_NAME[k]} <- Rmpfr::mpfr(${refExpr(k)}, ${PREC})`);
  const refDouble = sp.params === UNIF ? [`a <- ${refExpr("min")}; b <- ${refExpr("max")}`] : [];
  const truthBlock = [
    `# the true value, to ${PREC} bits (Rmpfr); evaluated only at a finite input`,
    ...t.helpers,
    `mpfr_value <- NA_real_`,
    `if (is.finite(${V})) {`,
    `  X <- Rmpfr::mpfr(${V}, ${PREC})`,
    ...refLits.map((l) => "  " + l),
    ...refDouble.map((l) => "  " + l),
    ...t.lines.map((l) => "  " + l),
    `  mpfr_value <- Rmpfr::asNumeric(${t.expr})`,
    `}`,
  ];

  const pyAssign = () => `${P.map(pyName).join(", ")} = ${P.map((k) => pyParam(beP[k])).join(", ")}`;
  const paramLines = (lang) => {
    if (plain) return lang === "R" ? [] : [pyAssign()];
    if (lang === "R") {
      const out = [`${P.map((k) => `${k} <- ${rParam(beP[k])}`).join("; ")}${differ ? "  # as the harness passes them to the backend, which rounds them to f32" : ""}`];
      if (differ) out.push(`${P.map((k) => `ref_${k} <- ${rNum(refP[k])}`).join("; ")}  # the same rounded to f32: what the reference is given`);
      return out;
    }
    return [`${pyAssign()}${differ ? "  # as the harness passes them; rounded to f32 by the array's dtype" : ""}`];
  };
  const refArg = (k) => (k === sp.arg ? V : `${differ ? "ref_" : ""}${k}`);
  const baseCall = `${sp.base}(${(plain ? [sp.arg] : sig).map(refArg).join(", ")}, ${baseFlags})`;
  const xComment = rNum(x) === dec(x) ? "" : `  # ${dec(x)}, written exactly: R's decimal parser is not correctly rounded everywhere`;

  // anvl's own evaluation, which the JAX snippet runs too: anvl is the
  // subject, and where JAX is out is exactly where anvl is worth a look.
  // Each argument as the harness passes it: the swept one as a batch, a
  // parameter as a bare R value (a scalar array of the cell's dtype where the
  // harness makes one), and for a gradient every parameter as a batch.
  const swept = `nv_array(rep(${V}, n), dtype = "${dtype}")`;
  const valueArg = (k) => (k === sp.arg ? swept : sp.scalar?.includes(k) ? `nv_scalar(${k}, dtype = "${dtype}")` : k);
  const gradArg = (k) => (k === sp.arg ? swept
    : sp.gradScalar?.includes(k) ? `nv_scalar(${k}, dtype = "${dtype}")`
      : `nv_array(rep(${plain ? rParam(beP[k]) : k}, n), dtype = "${dtype}")`);
  const wrt = sp.wrt ?? sig;
  const call = kind === "value"
      ? [`anvl_value <- as.double(${r.spec}(${(plain ? [sp.arg] : sig).map(valueArg).join(", ")}, ${flagsR}))[1]`]
      : [
        `grad_fn <- jit(gradient(`,
        `  \\(${sig.join(", ")}) sum(${r.spec}(${sig.join(", ")}, ${flagsR})),`,
        `  wrt = ${wrt.length === 1 ? `"${wrt[0]}"` : `c(${wrt.map((k) => `"${k}"`).join(", ")})`}`,
        `))`,
        `g <- grad_fn(`,
        ...sig.map((k, i) => `  ${gradArg(k)}${i < sig.length - 1 ? "," : ""}`),
        `)`,
        `anvl_value <- as.double(g$${output})[1]`,
      ];

  if (r.backend !== "jax") {
    const lines = [
      ...header("#"),
      "",
      "library(anvl)",
      "",
      `${V} <- ${rNum(x)}${xComment}`,
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
  const jdt = dtype === "f32" ? "jnp.float32" : "jnp.float64";
  const pyParams = P.map(pyName).join(", ");
  const restated = {
    uniform: "  # restated in (min, max) before differentiating",
    expon: "  # in terms of the rate, as the harness differentiates it",
  }[sp.jax];
  const jaxCall = sp.jaxTyped ? binomJax(r.spec, kind, f, jdt) : kind === "value"
    ? [`jax_value = float(${fn}(jnp.full(n, x, dtype=${jdt}), ${sp.jaxArgs})[0])`]
    : [
      restated
        ? `f = lambda ${sp.arg}, ${pyParams}: ${fn}(${sp.arg}, ${sp.jaxArgs})${restated}`
        : `f = ${fn}`,
      `g = jax.jit(jax.vmap(jax.grad(f, argnums=${[sp.arg, ...P].indexOf(output)}), in_axes=(0, ${P.map(() => "None").join(", ")})))`,
      `jax_value = float(g(jnp.full(n, x, dtype=${jdt}), ${pyParams})[0])`,
    ];
  const rCode = [
    "library(anvl)",
    `${V} <- ${rNum(x)}`,
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
    sp.jaxImport ?? `from jax.scipy.stats import ${sp.jax}`,
    "",
    `${V} = ${pyNum(x)}`,
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
