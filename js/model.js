// What a result's figures add up to. A port of the harness's result_state()
// (R/store.R), so the site and the terminal can never classify a result
// differently. Every flag states a fact; nothing here judges.

/** Input classes, as the harness splits them (input_class() in R/engine.R). */
export const CLASSES = {
  normal: {
    label: "normal", long: "Normal inputs",
    expect: "Finite, non-subnormal inputs within the valid domain; small relative errors are expected",
  },
  zero: {
    label: "±0", long: "±0",
    expect: "checked like any input; also among the exact points",
  },
  subnormal: {
    label: "subnormal", long: "Subnormal inputs",
    expect: "When the backend converts these inputs to zero, compare with the expected result for zero",
  },
  outside_domain: {
    label: "outside the domain", long: "Outside the valid domain",
    expect: "NaN by specification",
  },
  inf_nan: {
    label: "±∞ & NaN", long: "±∞ and NaN inputs",
    expect: "Match the reference exactly",
  },
};

/** No-finite-error region categories (the four the harness keeps visible, and
 * the fifth that validation adds). */
export const CATEGORIES = {
  failure: {
    label: "unexplained disagreement", cls: "warn",
    hint: "Relative error is undefined or infinite, and no recognised limitation or undefined-domain convention explains the disagreement",
  },
  boundary: {
    label: "domain boundary", cls: "neutral",
    hint: "at an endpoint of the valid input domain, where the reference gives a limiting value or convention",
  },
  backend_limitation: {
    label: "backend limitation", cls: "neutral",
    hint: "The backend converted a subnormal input to zero and returned the expected result for zero",
  },
  undefined_domain: {
    label: "undefined-domain convention", cls: "muted",
    hint: "Both function values are NaN, so no derivative exists. Differences in gradient conventions are reported separately from unexplained failures",
  },
  reference_limitation: {
    label: "verified base R limitation", cls: "ok",
    hint: "Base R exceeds its error tolerance while anvl meets its tolerance against a validated stable reference. Adjusted error statistics exclude these samples",
  },
};

export const CAUSES = {
  nan_input: "the input is NaN",
  input_flushing: "Input converted to zero; result matches the expected value for zero",
  flush_inherits_zero_error: "Input converted to zero; result for zero differs from the reference",
  domain_boundary: "an endpoint of the valid domain",
  outside_domain: "outside the valid domain",
  zero_input: "the input is ±0",
  inf_input: "the input is ±∞",
  unidentified: "Cause not identified by the benchmark",
};

/** Validation status of a reference, failing closed (reference_status()). */
export const REF_STATUS = {
  validated: { cls: "ok", label: "validated" },
  failed: { cls: "warn", label: "failed validation" },
  "not validated": { cls: "neutral", label: "not validated" },
  "no identity": { cls: "warn", label: "Reference identity unavailable" },
};

const n0 = (v) => (typeof v === "number" && !Number.isNaN(v) ? v : 0);

export function state(r) {
  const n = (k) => n0(r[k]);
  const verified = r.ref_stable_status === "validated";
  const differ = n("n_samples") - n("n_exact");
  const refCand = verified ? n("n_ref_candidate") : 0;
  const refPts = verified ? n("n_points_ref_candidate") : 0;
  const worstAny = Math.max(n("worst_rel_err"), n("worst_point_rel_err"));
  return {
    verified,
    failing: n("n_runs_unclassified") > 0 || n("n_points_failure") > 0,
    boundary: n("n_regions_boundary") > 0 || n("n_points_boundary") > 0,
    backend: n("n_regions_backend") > 0 || n("n_points_backend") > 0,
    conventions: n("n_regions_domain") > 0 || n("n_points_domain") > 0,
    reference: verified && (n("n_ref_candidate") > 0 || n("n_points_ref_candidate") > 0),
    candidates: !verified && (n("n_ref_candidate") > 0 || n("n_points_ref_candidate") > 0),
    identical: r.n_samples != null && differ === 0 && n("n_zero_sign") === 0 &&
      n("n_points_identical") === n("n_points"),
    setAsideOnly: differ > 0 && n("n_zero_sign") === 0 &&
      differ === n("n_failing_domain") + refCand &&
      n("n_points_identical") + n("n_points_domain") + refPts === n("n_points"),
    worstAny,
    worstSetAside: verified
      ? Math.max(n("worst_rel_err_excl"), n("worst_point_rel_err_excl"))
      : worstAny,
  };
}

/**
 * The headline figure of a class: the worst relative error among samples
 * whose base R value is a normal float -- normal in, normal out, where a small
 * relative error is the right expectation. `setAside` is the same with
 * verified base R limitations left out, when there are any; it is never
 * reported in place of the other, only beside it.
 */
export function headline(c, verified) {
  if (!c) return null;
  const all = c.worst_out_normal ?? null;
  const excl = verified && typeof c.worst_out_normal_excl === "number" ? c.worst_out_normal_excl : null;
  return { all, setAside: excl !== null && excl !== all ? excl : null };
}
