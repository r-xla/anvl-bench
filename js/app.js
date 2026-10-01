// Router and views.
//
// The URL is the whole of the navigable state, so any finding on this site can
// be linked from an issue:
//
//   #/                                             overview
//   #/spec/nv_qnorm                                one function
//   #/cell/<cell_id>/<output>[?z=<from>-<to>]      one result, zoomed to a range
//
// Only the leaf route fetches anything beyond the index.

import { listDeployed, urlSource, fileSource } from "./source.js";
import { openStore } from "./store.js";
import { binadeChart, histChart, bandKey, BEHAVIOUR } from "./chart.js";
import { num, int, pct, cellParts, flagList } from "./fmt.js";
import { snippet } from "./snippet.js";
import { CLASSES, CATEGORIES, CAUSES, REF_STATUS, state, headline } from "./model.js";

// --- tiny DOM helper ---------------------------------------------------

function h(tag, attrs = {}, ...kids) {
  const [name, ...cls] = tag.split(".");
  const n = document.createElement(name || "div");
  if (cls.length) n.className = cls.join(" ");
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = `${n.className} ${v}`.trim();
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids.flat(9)) {
    if (k === null || k === undefined || k === false) continue;
    n.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return n;
}

/**
 * Replace a node's children, skipping empty entries. The DOM's own
 * replaceChildren() stringifies null, which once printed a literal "null" on
 * every page where an optional panel part (the JAX key, the zoom note) was
 * absent.
 */
const put = (node, ...kids) =>
  node.replaceChildren(...kids.flat(9).filter((k) => k !== null && k !== undefined && k !== false));

// Wrapped, so a wide table scrolls within itself rather than dragging the
// whole page sideways on a narrow screen. `groups`, when given, is a header
// row above the column headers: a list of { label, span } naming runs of
// columns, so related figures read as a set.
const table = (headers, rows, groups = null) =>
  h("div.table-wrap", { tabindex: "0", role: "region", "aria-label": "results" },
    h("table.grid", {},
      h("thead", {},
        groups ? h("tr.groups", {}, groups.map((g) =>
          h("th", { colspan: g.span ?? 1, class: g.cls ?? null, title: g.hint ?? null }, g.label ?? ""))) : null,
        h("tr", {}, headers.map((c) =>
          h("th", { class: [c.align === "r" ? "r" : null, c.cls].filter(Boolean).join(" ") || null,
            title: c.hint ?? null }, c.label ?? c)))),
      h("tbody", {}, rows)));

/** How a backend is named on screen. */
const backendLabel = (b) => ({ anvl: "anvl", jax: "JAX" })[b] ?? b;

const referenceLabel = (r) => r.kind === "grad" ? "analytic gradient reference" : "base R";

const pill = (cls, text, title = null) => h(`span.pill.${cls}`, { title }, text);

/** A link to a panel further down the page. Not an href: the hash is the
 * router's, and "#regions" would navigate to the overview. */
const jump = (id, text) => h("button.link", {
  type: "button",
  onclick: () => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }),
}, text);
// --- reproducing one input ----------------------------------------------

let toastTimer = null;
function toast(msg, cls = "") {
  let t = document.getElementById("toast");
  if (!t) {
    t = h("div.toast", { id: "toast", role: "status", "aria-live": "polite" });
    document.body.append(t);
  }
  t.className = `toast show ${cls}`.trim();
  t.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = "toast"; }, 3200);
}

// The async clipboard needs a secure context; a local copy opened from file://
// is not one, so fall back to a selected textarea.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = h("textarea", { style: "position:fixed;opacity:0", readonly: "" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

/** What the site recorded for one input of a result, labelled for a snippet. */
const recOf = (res, value, reference, rel = null, ulp = null, extra = []) => [
  [backendLabel(res.backend), value],
  [referenceLabel(res), reference],
  ["Relative error", rel],
  ["Error (ULP)", ulp],
  ...extra,
];

/**
 * An input the reader can click to copy a script that reproduces it: R for
 * anvl, Python for JAX, each with base R and a 256-bit MPFR truth beside it
 * and the figures recorded here in its header. Built on click, not per row.
 */
const repro = (content, res, x, bits, where, recorded = []) => {
  if (!res || typeof x !== "number") return content;
  const lang = res.backend === "jax" ? "Python" : "R";
  return h("button.repro", {
    type: "button",
    title: `${bits ? bits + " · " : ""}click to copy ${lang === "R" ? "an R" : "a Python"} script that reproduces this input on your machine`,
    onclick: async () => {
      const s = snippet({ result: res, run: app.store.runById.get(res.run_id), x, bits, recorded, where });
      if (!s) return toast("No reproduction is available for this function.", "error");
      const ok = await copyText(s.text);
      toast(ok ? `Copied ${s.lang} script for ${res.spec} at ${num(x, 6)}` : "Could not copy to the clipboard", ok ? "" : "error");
    },
  }, content);
};

const catPill = (cat, text = null) => {
  const c = CATEGORIES[cat] ?? { label: cat, cls: "neutral" };
  return pill(c.cls, text ?? c.label, c.hint ?? null);
};
const statusPill = (status, prefix = "") => {
  if (!status) return null;
  const s = REF_STATUS[status] ?? { cls: "neutral", label: status };
  return pill(s.cls, `${prefix}${s.label}`);
};

// --- routes ------------------------------------------------------------

const cellHref = (cellId, output, extra = "") =>
  `#/cell/${encodeURIComponent(cellId)}/${encodeURIComponent(output)}${extra}`;
const specHref = (spec) => `#/spec/${encodeURIComponent(spec)}`;

function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const q = new URLSearchParams(qs ?? "");
  if (parts[0] === "spec" && parts[1]) return { view: "spec", spec: parts[1] };
  if (parts[0] === "cell" && parts[1] && parts[2]) {
    // The zoom is a half-open range of columns along the chart's axis.
    let zoom = null;
    const z = /^(\d+)-(\d+)$/.exec(q.get("z") ?? "");
    if (z && Number(z[2]) > Number(z[1])) zoom = [Number(z[1]), Number(z[2])];
    return { view: "cell", cellId: parts[1], output: parts[2], zoom };
  }
  return { view: "overview" };
}

// --- state -------------------------------------------------------------

const app = {
  store: null,
  deployed: [],
  chartCell: null,
  tab: {}, // which backend each per-backend panel shows; reset with the result
  redrawCell: null, // redraws the open result page for a new zoom, in place
  gen: 0, // bumped per result render, so a slow one cannot claim the page
};

const main = () => document.getElementById("main");
const setStatus = (msg, cls = "") => {
  put(main(), h(`p.status.${cls}`.replace(/\.$/, ""), {}, msg));
};

// --- header ------------------------------------------------------------

function renderHeader() {
  const bar = document.getElementById("source-bar");
  const m = app.store?.manifest;
  const kids = [];

  if (app.deployed.length > 1) {
    const sel = h("select", {
      "aria-label": "Results dataset",
      onchange: async (e) => { await load(urlSource(e.target.value)); route(); },
    }, app.deployed.map((a) =>
      h("option", { value: a.dir, selected: app.store?.source?.origin === `${a.dir}/` },
        a.label ?? a.id)));
    kids.push(h("label.field", {}, "Results dataset ", sel));
  }

  const picker = h("input", {
    type: "file",
    multiple: true,
    id: "file-picker",
    onchange: async (e) => {
      if (!e.target.files.length) return;
      try {
        await load(fileSource(e.target.files));
        location.hash = "#/";
        route();
      } catch (err) {
        setStatus(String(err.message ?? err), "error");
      }
    },
  });
  kids.push(h("details.local-results", {}, h("summary", {}, "Open downloaded results"),
    h("p.note", {}, "Download and extract a results archive, then select manifest.json and all Parquet files together."),
    h("label.field", {}, "Results files ", picker)));

  if (m) {
    kids.push(h("div.source-meta", {},
      h("span.tag", {}, m.platforms?.join(", ") ?? "?"),
      h("span.tag", {}, `anvl ${m.anvl_version?.join(", ") ?? "?"}`),
      h("span.tag", {}, `Test coverage: ${m.depths?.join(", ") ?? "unknown"}`)));
  }
  put(bar, ...kids);
}

// --- figures -----------------------------------------------------------

const frac = (c) => (c && c.n ? c.n_identical / c.n : null);

/** A result's figures by input class. */
const classesOf = (r) => app.store.classes(r) ?? {};

/**
 * The headline cell of a table: worst relative error for normal inputs with
 * normal outputs. When verified base R limitations are set aside the figure
 * shown is the set-aside one, marked, and the figure against base R is in the
 * mark's tooltip and on the result page -- never silently replaced.
 */
function headlineCell(r) {
  const st = state(r);
  const hd = headline(classesOf(r).normal, st.verified);
  if (!hd || hd.all === null) return h("span.muted", {}, "—");
  if (hd.setAside === null) return num(hd.all, 3);
  return [num(hd.setAside, 3), " ", h("span.sa", {
    title: `Excluding verified base R limitations. Against base R, including them: ${num(hd.all, 3)}.`,
  }, "†")];
}

/** Aggregate over results, e.g. one tile row. */
function aggregate(rows) {
  let n = 0, same = 0, worst = null, setAside = false;
  const count = { failing: 0, boundary: 0, backend: 0, reference: 0, candidates: 0, conventions: 0 };
  for (const r of rows) {
    const st = state(r);
    for (const k of Object.keys(count)) if (st[k]) count[k]++;
    const c = classesOf(r).normal;
    if (!c) continue;
    n += c.n ?? 0;
    same += c.n_identical ?? 0;
    const hd = headline(c, st.verified);
    const w = hd?.setAside ?? hd?.all;
    if (typeof w === "number" && !Number.isNaN(w) && (worst === null || w > worst)) {
      worst = w;
      setAside = hd.setAside !== null;
    }
  }
  return {
    n: rows.length,
    samples: rows.reduce((a, r) => a + (r.n_samples ?? 0), 0),
    normalFrac: n ? same / n : null,
    worst,
    setAside,
    ...count,
  };
}

/** The findings of one result, as a row of pills: every category shown. */
function findingPills(r) {
  const out = [];
  const regions = r.n_runs_unclassified ?? 0;
  const points = r.n_points_failure ?? 0;
  if (regions || points) out.push(catPill("failure", "Unexplained: " + [regions ? `${regions} region${regions === 1 ? "" : "s"}` : null, points ? `${points} test point${points === 1 ? "" : "s"}` : null].filter(Boolean).join("; ")));
  const nb = (r.n_regions_boundary ?? 0) + (r.n_points_boundary ?? 0);
  if (nb) out.push(catPill("boundary", `${nb} boundary findings`));
  const nl = (r.n_regions_backend ?? 0) + (r.n_points_backend ?? 0);
  if (nl) out.push(catPill("backend_limitation", `${nl} findings caused by backend limitations`));
  const nc = (r.n_regions_domain ?? 0) + (r.n_points_domain ?? 0);
  if (nc) out.push(catPill("undefined_domain", `${nc} undefined-domain findings`));
  const st = state(r);
  if (st.reference) out.push(catPill("reference_limitation", "verified base R limitation"));
  if (st.candidates) out.push(pill("neutral", "Possible base R error — unverified", "candidate base R disputes whose stable reference is not validated; nothing is excluded"));
  if (!out.length) out.push(pill("ok", st.identical ? "bit-identical" : "No classified findings"));
  return h("span.pills", {}, out);
}

/** A result's references, by their validation status. */
function refPills(r, plain = false) {
  const out = [];
  const label = (status, prefix) => plain
    ? h("div.muted", {}, prefix, REF_STATUS[status]?.label ?? status)
    : statusPill(status, prefix);
  if (r.ref_stable_status) out.push(label(r.ref_stable_status, "Stable reference: "));
  if (r.ref_grad_status) out.push(label(r.ref_grad_status, "Gradient reference: "));
  if (plain) return out.length ? out : h("span.muted", {}, referenceLabel(r));
  return out.length ? h("span.pills", {}, out) : h("span.muted", {}, referenceLabel(r));
}

// --- overview ----------------------------------------------------------

function renderOverview() {
  const s = app.store;
  const dtypes = [...new Set(s.summary.map((r) => r.dtype))].sort();
  const all = aggregate(s.summary);

  const out = [h("h1", {}, "Accuracy of anvl's distribution functions")];

  out.push(h("p.lede", {}, "Explore numerical agreement for anvl’s distribution functions and gradients. Choose a function below to inspect its results and compare with JAX where available."));
  out.push(h("p.note", {}, "Function values are compared with base R; gradients are compared with analytic gradient references. Each test configuration specifies the precision, parameters, options and output."));
  out.push(h("details.fold", {}, h("summary", {}, "Test coverage and terminology"),
    h("p.note", {}, "At full coverage, the tests examine every 32-bit floating-point bit pattern (f32), and sample one pattern from each block of 2³² patterns at 64-bit precision (f64). Shorter runs use fewer samples; the current coverage is shown in the header. Special inputs, including zero, infinities and domain boundaries, are tested separately. This dataset contains ", int(all.samples), " samples across ", int(all.n), " configurations."),
    h("p.note", {}, "Subnormal numbers are very small floating-point values close to zero. ULP (unit in the last place) measures error using the spacing of representable numbers. A stable reference is an additional numerical implementation checked against high-precision MPFR calculations.")));
  if (s.comparators.length) {
    const cmp = backendLabel(s.comparators[0]);
    const nTwin = s.summary.filter((r) => s.twin(r)).length;
    out.push(h("p.lede", {},
      `${cmp} is swept alongside as a comparator, against the same reference and on the same inputs. `,
      `${nTwin} of ${all.n} anvl results have a ${cmp} equivalent; the rest are variants ${cmp} does not offer. `,
      `The overview shows anvl’s results. Open a function or result for the JAX comparison.`));
  }

  // The headline: failures -- no finite error on valid inputs, for no tested
  // cause -- across the sweep and the exact points.
  const bySpec = s.specs
    .map((spec) => ({ spec, n: s.bySpec(spec).filter((r) => state(r).failing).length }))
    .filter((d) => d.n > 0)
    .sort((a, b) => b.n - a.n);
  out.push(h(`div.callout.${all.failing ? "warn" : "ok"}`, {},
    h("strong", {}, all.failing
      ? `${all.failing} of ${all.n} test configurations contain unexplained disagreements`
      : "No unexplained failures were found among the tested inputs"),
    all.failing
      ? h("p", {}, "These comparisons have undefined or infinite relative error and are not explained by a recognised boundary, backend or reference limitation, or an undefined-domain convention. Affected functions: ", bySpec.map((d, i) =>
          [i ? ", " : "", h("a", { href: specHref(d.spec) }, `${d.spec} (${d.n})`)]), ". ",
          h("span.muted", {}, "Further analysis is needed to determine which implementation is more accurate."))
      : null,
    h("p", {}, "Other findings (a configuration may appear in more than one category): ",
      [
        [all.boundary, "boundary", "with behaviour at a domain endpoint"],
        [all.backend, "backend_limitation", "with backend limitations (subnormal inputs flushed to zero)"],
        [all.reference, "reference_limitation", "with verified base R limitations"],
        [all.conventions, "undefined_domain", "with differences in undefined-domain conventions"],
      ].filter(([k]) => k).map(([k, cat, what], i) => [i ? "; " : "", catPill(cat, String(k)), ` ${what}`]),
      all.boundary + all.backend + all.reference + all.conventions ? "." : "none.")));

  // The references themselves: a reference is only as good as its validation.
  const refs = { stable: {}, gradient: {} };
  for (const r of s.summary) {
    if (r.ref_stable_status) refs.stable[r.ref_stable_status] = (refs.stable[r.ref_stable_status] ?? 0) + 1;
    if (r.ref_grad_status) refs.gradient[r.ref_grad_status] = (refs.gradient[r.ref_grad_status] ?? 0) + 1;
  }
  const refLine = (k, label) => {
    const e = Object.entries(refs[k]);
    if (!e.length) return null;
    return h("p", {}, h("span.muted", {}, `${label}: `),
      e.map(([st, nn], i) => [i ? " " : "", statusPill(st, `${nn} `)]));
  };
  const refBad = [...Object.keys(refs.stable), ...Object.keys(refs.gradient)].some((k) => k !== "validated");
  out.push(h(`div.callout.${refBad ? "warn" : "ok"}`, {},
    h("strong", {}, refBad
      ? "Not every reference is validated"
      : "All recorded auxiliary references passed high-precision validation"),
    h("p", {}, "Analytic gradient references and additional stable references are checked against 256-bit MPFR calculations. Only a validated stable reference can support excluding a verified base R limitation from adjusted error statistics."),
    refLine("stable", "stable references (base R disputes)"),
    refLine("gradient", "gradient references")));

  // One tile per function, split by precision: normal inputs with normal
  // outputs, and whether anything fails.
  const grid = h("div.tiles");
  for (const spec of s.specs) {
    const rows = s.bySpec(spec);
    const tile = h("a.tile", { href: specHref(spec) }, h("h2", {}, spec));
    const body = h("div.tile-body", {},
      h("div.tile-row.tile-head", {},
        h("span", {}), h("span.r", { title: "Maximum finite relative error for normal inputs and outputs" }, "Max. error"),
        h("span.r", { title: "Normal-input samples matching the reference; signed zeros count as equal" }, "Match %"),
        h("span", { title: "Configurations with unexplained disagreements" }, "Flagged")));
    for (const dt of dtypes) {
      const a = aggregate(rows.filter((r) => r.dtype === dt));
      if (!a.n) continue;
      body.append(h("div.tile-row", {},
        h("span.dt", {}, dt),
        h("span.metric", {}, num(a.worst, 3), a.setAside ? [" ", h("span.sa", { title: "excluding verified base R limitations; the figures against base R are on the function page" }, "†")] : null),
        h("span.metric.muted", {}, pct(a.normalFrac)),
        a.failing
          ? h("span.pill.warn", { title: "Configurations with unexplained disagreements" }, `${a.failing}`)
          : h("span.pill.ok", { title: "No unexplained failures found" }, "✓")));
    }
    tile.append(body);
    grid.append(tile);
  }
  out.splice(3, 0, grid);
  const metricLegend = h("p.legend", {},
    h("strong", {}, "Max. error"), ` is the maximum finite relative error against the relevant reference over `, h("strong", {}, "normal inputs with normal outputs"),
    ` within the valid domain. Here “normal” describes floating-point representation, not the normal distribution. `,
    h("strong", {}, "Match %"), ` is the percentage of normal-input samples matching the reference. Signed-zero differences are reported separately on result pages. `,
    h("span.sa", {}, "†"), ` marks an error statistic that excludes verified base R limitations. The unadjusted statistic is available on the result page. Flagged counts configurations containing unexplained disagreements, rather than individual samples.`);
  out.splice(4, 0, metricLegend);

  out.push(renderRuns());
  put(main(), ...out);
}

function renderRuns() {
  const s = app.store;
  const fields = [
    ["host", "host"], ["os", "os"], ["os_version", "os version"], ["arch", "arch"],
    ["cpu", "cpu"], ["n_cores", "cores"], ["device", "device"], ["r_version", "R"],
    ["depth", "depth"], ["branch", "branch"], ["sweep_seed", "seed"],
    ["default_float", "default float"], ["default_int", "default int"],
    ["anvl_version", "anvl"], ["anvl_sha", "anvl sha"],
    ["stablehlo_version", "stablehlo"], ["stablehlo_sha", "stablehlo sha"],
    ["pjrt_version", "pjrt"], ["pjrt_sha", "pjrt sha"],
    ["tengen_version", "tengen"], ["tengen_sha", "tengen sha"],
    ["xlamisc_version", "xlamisc"], ["xlamisc_sha", "xlamisc sha"],
  ];
  const body = h("div.runs");
  for (const run of s.runs) {
    const dl = h("dl.fingerprint");
    for (const [k, label] of fields) {
      if (!(k in run) || run[k] === null || run[k] === undefined) continue;
      const v = String(run[k]);
      dl.append(h("div.stat", {}, h("dt", {}, label),
        h("dd", { class: /_sha$/.test(k) ? "mono" : null, title: v },
          /_sha$/.test(k) ? v.slice(0, 12) : v)));
    }
    body.append(h("div.run", {},
      h("h3", {}, run.started_at ?? run.run_id), dl));
  }
  return h("details.fold", {}, h("summary", {}, `Environment (${s.runs.length} run${s.runs.length === 1 ? "" : "s"})`), body);
}

// --- one function ------------------------------------------------------

const worstOf = (r) => {
  const hd = headline(classesOf(r).normal, state(r).verified);
  return hd?.setAside ?? hd?.all ?? -1;
};
const SORTS = {
  worst: (a, b) => worstOf(b) - worstOf(a),
  exact: (a, b) => (frac(classesOf(a).normal) ?? 2) - (frac(classesOf(b).normal) ?? 2),
  failing: (a, b) =>
    ((b.n_runs_unclassified ?? 0) + (b.n_points_failure ?? 0)) - ((a.n_runs_unclassified ?? 0) + (a.n_points_failure ?? 0)) ||
    worstOf(b) - worstOf(a),
  name: (a, b) => a.cell_id.localeCompare(b.cell_id) || a.output.localeCompare(b.output),
};
let specSort = "failing";
const specFilters = { precision: "all", kind: "all", findings: "all" };

function renderSpec(spec) {
  const s = app.store;
  const rows = s.bySpec(spec);
  if (!rows.length) return setStatus(`No results for ${spec} in this artifact.`, "error");
  const cmp = s.comparators[0];
  const a = aggregate(rows);

  // The comparator's headline beside anvl's. No verdict column: the two
  // numbers are shown and the reader compares them.
  const cmpCells = (r) => {
    if (!cmp) return null;
    const t = s.twin(r, cmp);
    if (!t) {
      return h("td.muted.cmp.none", { colspan: 3, title: `${backendLabel(cmp)} has no equivalent of this variant` },
        `no ${backendLabel(cmp)} equivalent`);
    }
    return [
      h("td.r.cmp", {}, headlineCell(t)),
      h("td.r", {}, pct(frac(classesOf(t).normal))),
      h("td", {}, findingPills(t)),
    ];
  };

  const zeroCell = (r) => {
    const z = classesOf(r).zero;
    // The f64 sweep essentially never samples ±0; the exact points always do.
    if (!z) return h("span.muted", { title: "not among this sweep's samples; ±0 is checked among the exact points" }, "Tested separately");
    if (z.n_identical < z.n) return pill("warn", `${z.n - z.n_identical} samples differ`, "Zero inputs that differ from the reference");
    return z.n_zero_sign
      ? pill("neutral", "sign", "Matches the reference value, but the sign of zero differs")
      : pill("ok", "✓", "Matches the reference exactly");
  };
  const pointsCell = (r) => {
    if (!r.n_points) return h("span.muted", {}, "—");
    const bad = r.n_points - r.n_points_identical;
    return h("span", { title: `${r.n_points_identical} of ${r.n_points} special test points match the reference, including the sign of zero` },
      bad ? `${r.n_points_identical} of ${r.n_points} match` : pill("ok", `✓ ${r.n_points}`));
  };

  const filtered = rows.filter((r) =>
    (specFilters.precision === "all" || r.dtype === specFilters.precision) &&
    (specFilters.kind === "all" || r.kind === specFilters.kind) &&
    (specFilters.findings === "all" || state(r).failing));
  const sorted = [...filtered].sort(SORTS[specSort] ?? SORTS.failing);
  const filter = (key, label, choices) => h("label.field", {}, label, h("select", {
    onchange: (e) => { specFilters[key] = e.target.value; renderSpec(spec); },
  }, choices.map(([value, text]) => h("option", { value, selected: specFilters[key] === value }, text))));
  const body = sorted.map((r) => {
    const st = state(r);
    const c = classesOf(r);
    return h("tr", { class: st.failing ? "row-warn" : null },
      h("td", {}, h("a", { href: cellHref(r.cell_id, r.output) },
        h("span.dt", {}, r.dtype), " ", r.kind, " ", h("span.muted", {}, r.param_set))),
      h("td.flags", {}, flagList(r.flags).map((fl) => h("span.chip.flag", {}, fl))),
      h("td", {}, h("code", {}, r.output)),
      h("td.r.grp", {}, headlineCell(r)),
      h("td.r", {}, num(c.normal?.worst_rel_err, 3)),
      h("td.r", {}, pct(frac(c.normal))),
      h("td.grp", {}, zeroCell(r)),
      h("td.r", {}, pct(frac(c.subnormal))),
      h("td.r", {}, pointsCell(r)),
      h("td.grp", {}, findingPills(r)),
      h("td", {}, refPills(r, true)),
      cmpCells(r));
  });

  const sortSel = h("select", { onchange: (e) => { specSort = e.target.value; renderSpec(spec); } },
    [["failing", "Unexplained disagreements first"], ["worst", "Maximum relative error (normal inputs and outputs)"],
     ["exact", "Lowest match rate (normal inputs)"], ["name", "name"]].map(([v, l]) =>
      h("option", { value: v, selected: v === specSort }, l)));

  const groups = [
    { label: "", span: 3 },
    { label: "Normal inputs", span: 3, cls: "grp", hint: CLASSES.normal.expect },
    { label: "±0, subnormal, exact points", span: 3, cls: "grp" },
    { label: "", span: 2, cls: "grp" },
    ...(cmp ? [{ label: backendLabel(cmp), span: 3, cls: "cmp" }] : []),
  ];

  put(main(),
    h("nav.crumbs", {}, h("a", { href: "#/" }, "overview"), " / ", h("span", {}, spec)),
    h("h1", {}, spec),
    h("p.lede", {}, `${a.n} results, ${int(a.samples)} samples in all. `,
      a.failing
        ? h("strong", {}, `${a.failing} contain unexplained disagreements.`)
        : "No unexplained failures were found among the tested inputs."),
    h("p.note", {}, "Values use base R as the reference; gradients use an analytic gradient reference. Error columns show the maximum finite relative error. Undefined or infinite errors are listed separately in Findings. “Normal” refers to floating-point representation. Signed-zero differences are reported separately."),
    h("div.filters", {},
      filter("precision", "Precision ", [["all", "All precisions"], ["f32", "32-bit (f32)"], ["f64", "64-bit (f64)"]]),
      filter("kind", "Output type ", [["all", "Values and gradients"], ["value", "Values"], ["grad", "Gradients"]]),
      filter("findings", "Findings ", [["all", "All configurations"], ["unexplained", "Unexplained disagreements"]]),
      h("label.field", {}, "Sort by ", sortSel)),
    h("p.note", { role: "status" }, `Showing ${filtered.length} of ${rows.length} configurations.`, filtered.length ? " Scroll horizontally to see all comparison columns." : " Choose different filters to see results."),
    table([
      { label: "Configuration", hint: "precision, value or gradient, parameter set" },
      "Options",
      { label: "output", hint: "value, or the argument differentiated" },
      { label: "Max. relative error: normal outputs", align: "r", cls: "grp", hint: "worst relative error where the reference value is a normal float too" },
      { label: "Max. relative error: all outputs", align: "r", hint: "worst relative error over every output, subnormal and zero outputs included" },
      { label: "Matches reference (%)", align: "r" },
      { label: "±0", cls: "grp", hint: "the sweep's ±0 samples: bit-identical, down to the sign?" },
      { label: "Subnormal inputs matching (%)", align: "r", hint: "Percentage of subnormal inputs matching the reference" },
      { label: "Special test points matching", align: "r", hint: "±0, ±∞, NaN, extremes, ±½, ±1, domain and support edges, branch points, and their neighbours" },
      { label: "findings", cls: "grp", hint: "no-finite-error regions and exact points, by category" },
      { label: "reference", hint: "what this result is scored against, and whether that reference is validated" },
      ...(cmp ? [
        { label: "Max. relative error: normal outputs", align: "r", cls: "cmp", hint: `${backendLabel(cmp)}, on the same inputs, against the same reference` },
        { label: "Matches reference (%)", align: "r" },
        { label: "findings" },
      ] : []),
    ], body, groups),
    h("p.legend", {}, h("span.sa", {}, "†"),
      " marks an error statistic excluding verified base R limitations. Open the result to see both adjusted and unadjusted values."));
  main().querySelector(".table-wrap")?.classList.add("configuration-table");
}

// --- one result --------------------------------------------------------

async function renderCell(cellId, output, zoom) {
  app.redrawCell = null;
  const gen = ++app.gen;
  const s = app.store;
  const r = s.result(cellId, output);
  if (!r) return setStatus(`No result for ${cellId} / ${output} in this artifact.`, "error");
  const parts = cellParts(cellId);
  const st = state(r);

  // The comparator's twin: the same cell and output, from another backend. Only
  // anvl's results are compared; a comparator is never a page's subject.
  const cmp = r.backend === s.primary ? s.comparators[0] : undefined;
  const t = cmp ? s.twin(r, cmp) : undefined;
  const tst = t ? state(t) : null;
  const cmpLabel = cmp ? backendLabel(cmp) : null;
  const me = backendLabel(r.backend);

  // The per-backend tabs are per result; a new one starts on anvl.
  const key = `${cellId}/${output}`;
  if (app.chartCell !== key) {
    app.chartCell = key;
    app.tab = {};
  }

  // --- by input class: what each should do, and what it did ---
  const mine = classesOf(r);
  const theirs = t ? classesOf(t) : null;
  const worstWithExcl = (c, verified, field = "worst_out_normal") => {
    if (!c || c[field] === undefined || c[field] === null) return "—";
    const excl = verified ? c[`${field}_excl`] : null;
    if (typeof excl !== "number" || excl === c[field]) return num(c[field], 3);
    return [num(c[field], 3), h("div.sa-line", { title: "excluding verified base R limitations" }, `${num(excl, 3)} excluding verified base R limitations`)];
  };
  const classRow = (k, c, tc) => h("tr", { class: k === "all" ? "all-row" : null },
    h("th.cls", { scope: "row" }, k === "all" ? "All inputs" : CLASSES[k].long,
      k === "all" ? null : h("div.expect", {}, CLASSES[k].expect)),
    h("td.r", {}, int(c?.n)),
    h("td.r", {}, pct(frac(c))),
    h("td.r", {}, c?.n_rounded ? int(c.n_rounded) : "—"),
    h("td.r", {}, worstWithExcl(c, st.verified, "worst_out_normal")),
    h("td.r", {}, worstWithExcl(c, st.verified, "worst_rel_err")),
    h("td.r", {}, c && c.worst_rel_err > 0
      ? repro(num(c.worst_x), r, c.worst_x, c.worst_bits, `worst input, ${k === "all" ? "all inputs" : CLASSES[k].long}`,
        recOf(r, c.worst_value, c.worst_reference, c.worst_rel_err))
      : "—"),
    t ? h("td.r.cmp", {}, pct(frac(tc))) : null,
    t ? h("td.r", {}, worstWithExcl(tc, tst.verified, "worst_out_normal")) : null);
  const asClass = (x) => x && ({
    n: x.n_samples, n_identical: x.n_exact, n_rounded: x.n_rounded, worst_rel_err: x.worst_rel_err,
    worst_rel_err_excl: x.worst_rel_err_excl, worst_out_normal: x.worst_out_normal,
    worst_out_normal_excl: x.worst_out_normal_excl, worst_x: x.worst_x, worst_bits: x.worst_bits,
  });
  const classTable = h("div.compare.classes", {}, table([
    { label: "input, and what should happen" }, { label: "samples", align: "r" },
    { label: "Matches reference (%)", align: "r" },
    { label: "rounded", align: "r", hint: "correctly rounded to the result's precision without being identical: the best the precision allows, though its relative error is not zero" },
    { label: "Max. relative error: normal outputs", align: "r", hint: "worst relative error where the reference value is a normal float" },
    { label: "Max. relative error: all outputs", align: "r" }, { label: "worst at x", align: "r" },
    ...(t ? [{ label: "Matches reference (%)", align: "r", cls: "cmp" }, { label: "Max. relative error: normal outputs", align: "r" }] : []),
  ], [
    ...Object.keys(CLASSES).filter((k) => mine[k]).map((k) => classRow(k, mine[k], theirs?.[k])),
    classRow("all", asClass(r), asClass(t)),
  ], t ? [{ label: "", span: 7 }, { label: cmpLabel, span: 2, cls: "cmp" }] : null));

  // Facts that sit across the classes, each kept visible.
  const facts = [
    [r.n_zero_sign, "samples return a zero of the opposite sign (+0 against −0): counted as identical above, and shown here"],
    [r.n_flushed, "subnormal inputs differ from the reference because the backend converted them to ±0 and returned the expected result for zero"],
    [r.n_flushed_zero_error, "subnormal inputs were converted to ±0, and the result for zero also differs from the reference"],
  ].filter(([n]) => n > 0);
  const factList = facts.length
    ? h("ul.facts", {}, facts.map(([n, what]) => h("li", {}, h("strong", {}, int(n)), " ", what, ".")))
    : null;

  const STATS = [
    ["worst relative error", (x) => num(x.worst_rel_err)],
    ["worst ulp error", (x) => num(x.worst_ulp_err), "the worst ulp error over every sample, which need not be the worst relative error's sample"],
    ["Matches reference (%)", (x) => pct(x.n_exact / x.n_samples)],
    ["worst relative error at x", (x) => [
      repro(num(x.worst_x), x, x.worst_x, x.worst_bits, "worst relative error", recOf(x, x.worst_value, x.worst_reference, x.worst_rel_err)), " ",
      repro(h("code.bits", {}, x.worst_bits ?? ""), x, x.worst_x, x.worst_bits, "worst relative error", recOf(x, x.worst_value, x.worst_reference, x.worst_rel_err))]],
    ["Computed value at this input", (x) => num(x.worst_value)],
    ["Reference value at this input", (x) => num(x.worst_reference)],
    ["exact points: worst finite error", (x) => (x.worst_point_rel_err > 0 ? [num(x.worst_point_rel_err), " at ", h("code", {}, x.worst_point_label ?? "")] : "0")],
    ["samples", (x) => int(x.n_samples)],
  ];
  const overall = t
    ? h("div.compare", {}, table(
      [{ label: "" }, { label: me, align: "r" }, { label: cmpLabel, align: "r" }],
      STATS.map(([k, f, hint]) => h("tr", {},
        h("th", { scope: "row", title: hint ?? null }, k), h("td.r", {}, f(r)), h("td.r.cmp", {}, f(t))))))
    : h("dl.stats", {}, STATS.map(([k, f, hint]) => h("div.stat", { title: hint ?? null }, h("dt", {}, k), h("dd", {}, f(r)))));

  // --- what the categories add up to, in words ---
  const callouts = [];
  const nFailR = r.n_runs_unclassified ?? 0;
  const nFailP = r.n_points_failure ?? 0;
  if (st.failing) {
    callouts.push(h("div.callout.warn", {},
      h("strong", {}, `${me} has unexplained disagreements: ${[nFailR ? `${nFailR} region${nFailR === 1 ? "" : "s"}` : null, nFailP ? `${nFailP} special test point${nFailP === 1 ? "" : "s"}` : null].filter(Boolean).join(" and ")} with undefined or infinite relative error`),
      h("p", {}, "Listed below, under ", jump("regions", "regions"), " and ", jump("points", "special test points"), ". ",
        h("span.muted", {}, "These findings are not explained by a recognised limitation or an undefined-domain convention. Further analysis is needed to determine which implementation is more accurate."))));
  }
  if (st.reference) {
    callouts.push(h("div.callout", {},
      h("strong", {}, "Verified base R limitations excluded from adjusted error statistics"),
      h("p", {}, `${int(r.n_ref_candidate)} samples${r.n_points_ref_candidate ? ` and ${r.n_points_ref_candidate} exact points` : ""} where base R exceeds its error tolerance and ${me} meets its tolerance against a validated stable reference. `,
        `Worst relative error against base R: ${num(r.worst_rel_err, 3)}; excluding these samples: ${num(r.worst_rel_err_excl, 3)}. `,
        jump("disputes", "The evidence"), " is below.")));
  } else if (st.candidates) {
    callouts.push(h("div.callout", {},
      h("strong", {}, "Possible base R errors remain included in the results"),
      h("p", {}, `${int(r.n_ref_candidate)} samples look like base R limitations against a stable reference, but that reference is ${REF_STATUS[r.ref_stable_status]?.label ?? r.ref_stable_status}, so they are counted against ${me} like any other disagreement. `,
        jump("disputes", "The candidates"), " are below.")));
  }
  if (r.ref_grad_status && r.ref_grad_status !== "validated") {
    callouts.push(h("div.callout.warn", {},
      h("strong", {}, `The gradient reference is ${REF_STATUS[r.ref_grad_status]?.label ?? r.ref_grad_status}`),
      h("p", {}, "Treat the gradient error statistics as provisional until the reference passes validation. ",
        jump("validation", "See its validation"), ".")));
  }
  if (t && tst.failing && !st.failing) {
    callouts.push(h("div.callout", {},
      h("strong", {}, `${cmpLabel} has unexplained disagreements; none were found for ${me}`),
      h("p", {}, `${cmpLabel} has ${(t.n_runs_unclassified ?? 0) + (t.n_points_failure ?? 0)} failure(s) on this cell, listed below beside ${me}'s findings.`)));
  }

  const noTwin = !t && s.comparators.length && r.backend === s.primary
    ? h("p.note", {}, `${backendLabel(s.comparators[0])} has no equivalent of this variant, so there is nothing to compare it with here.`)
    : null;

  const head = [
    h("nav.crumbs", {},
      h("a", { href: "#/" }, "overview"), " / ",
      h("a", { href: specHref(r.spec) }, r.spec), " / ",
      h("span", {}, `${parts.dtype} ${parts.kind} ${parts.param_set}`), " / ",
      h("code", {}, output)),
    h("h1", {}, r.spec, " ", h("span.dt", {}, parts.dtype), " ",
      h("span.muted", {}, parts.kind === "grad" ? `d/d${output}` : "value")),
    h("div.chips", {}, h("span.chip", {}, parts.param_set),
      flagList(parts.flags).map((f) => h("span.chip.flag", {}, f)),
      h("span.chip", {}, me), t ? h("span.chip", {}, `compared with ${cmpLabel}`) : null,
      Number.isFinite(r.domain_lo) || Number.isFinite(r.domain_hi)
        ? h("span.chip", { title: "the valid input domain: outside it the value is NaN by specification" }, `domain [${num(r.domain_lo)}, ${num(r.domain_hi)}]`)
        : null,
      Number.isFinite(r.support_lo) || Number.isFinite(r.support_hi)
        ? h("span.chip", { title: "the distribution's support; reporting only, it excuses nothing" }, `support [${num(r.support_lo)}, ${num(r.support_hi)}]`)
        : null,
      refPills(r)),
    h("div.findings", {}, findingPills(r)),
    ...callouts,
    h("p.note", {}, "Reference: ", h("strong", {}, referenceLabel(r)), ". Error statistics describe finite relative errors; undefined or infinite errors appear separately in the findings. Matching percentages include those other comparisons. Signed zeros count as equal in sweep percentages and are reported separately."),
    h("p.note", {}, "Select an underlined input to copy a reproduction script."),
    classTable,
    factList,
    h("details.fold", {}, h("summary", {}, "Worst case over all inputs: ulp error, the values there, and the exact points"), overall),
    noTwin,
  ];

  const panel = (id, title) => h("section.panel", { id }, h("h2", {}, title), h("p.status", {}, "Loading result details…"));
  const slots = {
    hist: panel("hist", "Distribution of relative error"),
    bands: panel("bands", "Maximum relative error across input ranges"),
    detail: panel("detail", "Worst inputs"),
    points: panel("points", "Special test points"),
    ranges: panel("regions", "Regions with undefined or infinite relative error"),
    disputes: r.ref_stable_id || r.ref_stable_status ? panel("disputes", "Disputes with base R") : null,
    validation: r.ref_stable_status || r.ref_grad_status ? panel("validation", "Reference validation") : null,
  };
  // The histogram is recorded per result, so no zoom can reach it. Placing it
  // above the chart leaves the zoom governing everything that follows.
  put(main(), ...head.filter(Boolean), ...Object.values(slots).filter(Boolean));

  const forOutput = (rows) => rows.filter((x) => x.output === output);
  const rowsOf = (tbl, id) => (id ? s.cellRows(tbl, id).then(forOutput) : Promise.resolve([]));
  const tId = t?.cell_id ?? null;
  const [bands, hist, detail, ranges, points, disputes, tBands, tHist, tDetail, tRanges, tPoints] = await Promise.all([
    rowsOf("bands", cellId), rowsOf("hist", cellId), rowsOf("detail", cellId), rowsOf("ranges", cellId),
    rowsOf("points", cellId), rowsOf("disputes", cellId),
    rowsOf("bands", tId), rowsOf("hist", tId), rowsOf("detail", tId), rowsOf("ranges", tId), rowsOf("points", tId),
  ]);
  // The reader may have moved on while this was loading.
  if (gen !== app.gen) return;

  const cmpKey = t ? h("span.key-item", {}, h("i.k-cmp"), `${cmpLabel} (line)`) : null;

  // Columns in the chart's order: along the real line, most negative first,
  // with each sign's ±0 as a column of its own. A zoom is a range of these,
  // and everything below follows it by matching band keys -- the same interval
  // for any backend at one precision.
  const columns = bands.filter((b) => !b.special)
    .sort((a, b) => (a.sign < 0 ? -1 : 1) - (b.sign < 0 ? -1 : 1) ||
      (a.sign < 0 ? rank(b) - rank(a) : rank(a) - rank(b)));
  const zoomHref = (v) => cellHref(cellId, output, v ? `?z=${v[0]}-${v[1]}` : "");

  // A bit pattern's column, read straight off its fields: exponent 0 with a
  // zero mantissa is ±0, exponent 0 otherwise a subnormal, the top exponent
  // the ±∞/NaN field (no column).
  const layout = parts.dtype === "f64"
    ? { sign: 63n, exp: 52n, mask: 0x7ffn, mant: (1n << 52n) - 1n }
    : { sign: 31n, exp: 23n, mask: 0xffn, mant: (1n << 23n) - 1n };
  const fieldsOf = (hex) => {
    const v = BigInt(hex);
    const e = Number((v >> layout.exp) & layout.mask);
    return { sign: (v >> layout.sign) & 1n ? -1 : 1, binade: e, zero: e === 0 && (v & layout.mant) === 0n };
  };
  const keyOfBits = (hex) => {
    const f = fieldsOf(hex);
    return f.binade === Number(layout.mask) ? null : f.zero ? `${f.sign}:z` : `${f.sign}:${f.binade}`;
  };
  // A region is a run of bit patterns; the columns it covers follow from its
  // two ends. Bit order within a sign runs 0, subnormals, normals, ±∞/NaN, so
  // a run that starts at a sign's zero pattern covers that sign's ±0 column.
  const rangeKeys = (x) => {
    const a = fieldsOf(x.bits_from);
    const b = fieldsOf(x.bits_to);
    const keys = [];
    const top = Number(layout.mask);
    const run = (sign, lo, hi, withZero) => {
      if (withZero) keys.push(`${sign}:z`);
      for (let i = lo; i <= hi; i++) keys.push(`${sign}:${i}`);
    };
    if (a.sign === b.sign) {
      const [lo, hi] = a.binade <= b.binade ? [a, b] : [b, a];
      run(a.sign, lo.binade, hi.binade, lo.zero);
    } else {
      run(a.sign, a.binade, top, a.zero);
      run(b.sign, 0, b.binade, true);
    }
    return keys;
  };

  const draw = (view) => {
    const v = view && view[0] < columns.length ? [view[0], Math.min(view[1], columns.length)] : null;
    const inView = v ? new Set(columns.slice(v[0], v[1]).map(bandKey)) : null;
    const within = (key) => !inView || (key !== null && inView.has(key));
    const detailKey = (d) => (d.x === 0 ? `${d.sign}:z` : `${d.sign}:${d.binade}`);
    const lo = v ? columns[v[0]] : null;
    const hi = v ? columns[v[1] - 1] : null;
    // A function, not a value: a DOM node lives in one place, so each use of the
    // range's bounds needs its own copy or the earlier ones are emptied.
    const span = () => (v ? [h("code", {}, num(lo.x_from)), " … ", h("code", {}, num(hi.x_to))] : null);
    const tabs = (panelId, draw1) => {
      if (!t) return null;
      const which = app.tab[panelId] ?? r.backend;
      return h("div.seg", { role: "tablist", "aria-label": `${panelId === "points" ? "Special test points" : "Worst inputs"}: implementation` },
        [r.backend, cmp].map((b) => h("button", {
          role: "tab",
          "aria-selected": String(which === b),
          class: which === b ? "on" : null,
          onclick: () => { app.tab[panelId] = b; draw1(); },
        }, backendLabel(b))));
    };
    const whose = (panelId) => (t && app.tab[panelId] === cmp ? cmp : r.backend);

    // --- the chart ---
    const chart = binadeChart({
      domain: Number.isFinite(r.domain_lo) || Number.isFinite(r.domain_hi) ? [r.domain_lo, r.domain_hi] : null,
      bands,
      view: v,
      compare: t ? { label: cmpLabel, bands: tBands } : null,
      onZoom: (z) => { location.hash = zoomHref(z); },
    });

    // What the zoomed range holds, summed exactly from the per-binade counts.
    const tally = (rows) => {
      let n = 0, same = 0, worst = null;
      for (const b of rows) {
        if (b.special || !within(bandKey(b))) continue;
        n += (b.n_identical ?? 0) + (b.n_differ ?? 0) + (b.n_nonfinite ?? 0);
        same += b.n_identical ?? 0;
        const e = b.worst_rel_err;
        if (typeof e === "number" && Number.isFinite(e) && (worst === null || e > worst)) worst = e;
      }
      return { n, same, worst };
    };
    const summary = (label, x) =>
      h("span", {}, h("strong", {}, label), ` ${pct(x.n ? x.same / x.n : null)} matching reference, maximum finite relative error ${num(x.worst, 3)}`);
    const rangeLine = v
      ? h("p.range-line", {},
        h("span", {}, `In this range, ${v[1] - v[0]} of ${columns.length} columns (`, span(), `), ${int(tally(bands).n)} samples: `),
        summary(me, tally(bands)),
        t ? [" · ", summary(cmpLabel, tally(tBands))] : null)
      : null;

    put(slots.bands,
      h("div.panel-head", {},
        h("h2", {}, "Maximum relative error across input ranges"),
        v ? h("button.reset", { onclick: () => { location.hash = zoomHref(null); } }, "Reset view") : null),
      h("p.note", {}, "Inputs run from negative to positive. Bars group floating-point values into intervals between successive powers of two (binades), with separate groups for subnormal values and signed zeros. When space is limited, a bar shows the maximum across several groups. ",
        h("strong", {}, "Drag across the chart to zoom"), " — the worst inputs, exact points, regions and disputes below follow the range.",
        t ? ` The line is ${cmpLabel}'s worst relative error over the same binades, against the same reference; it breaks where ${cmpLabel} was not swept and rests on the axis where it matches the reference exactly.` : ""),
      chart.wrap ?? chart,
      rangeLine,
      h("div.chart-foot", {},
        h("div.key", {}, Object.values(BEHAVIOUR).map((b) =>
          h("span.key-item", {}, h("i", { class: b.cls }), b.label)), cmpKey,
          h("span.key-sep"),
          h("span.key-item", { title: CLASSES.zero.expect }, h("i.k-bg-zero"), "±0"),
          h("span.key-item", { title: CLASSES.subnormal.expect }, h("i.k-bg-sub"), "subnormal inputs"),
          Number.isFinite(r.domain_lo) || Number.isFinite(r.domain_hi)
            ? h("span.key-item", { title: CLASSES.outside_domain.expect }, h("i.k-bg-oos"), "outside the domain")
            : null),
        h("span.muted", {}, v ? `${v[1] - v[0]} of ${columns.length} columns` : `${columns.length} columns`)),
    );

    // --- the histogram: whole result only, and says so when zoomed ---
    const candInHist = hist.reduce((a, d) => a + (d.count_ref_candidate ?? 0), 0);
    put(slots.hist,
      h("h2", {}, "Distribution of relative error"),
      v ? h("p.note", {}, h("strong", {}, "Whole result — the zoom below does not apply here."),
        " The sweep records this distribution per result rather than per binade; the line under the chart gives the zoomed range's own figures.") : null,
      histChart(hist, t ? { label: cmpLabel, rows: tHist } : null),
      candInHist
        ? h("p.note", {}, `${int(candInHist)} of these finite errors are `,
          st.verified ? "verified base R limitations" : "candidate base R disputes (not validated)",
          " — counted here like every other sample.")
        : null,
      t ? h("div.chart-foot", {}, h("div.key", {},
        h("span.key-item", {}, h("i.k-anvl"), `${me} (bars)`), cmpKey)) : null,
    );

    // --- worst inputs ---
    const isUf = (d) => d.value === 0 && Number.isFinite(d.reference) && d.reference !== 0 &&
      Math.abs(d.reference) < (parts.dtype === "f32" ? 2 ** -126 : 2 ** -1022);
    const drawDetail = () => {
      const which = whose("detail");
      const rows = (which === r.backend ? detail : tDetail).filter((d) => within(detailKey(d)));
      const res = which === r.backend ? r : t;
      const dRepro = (content, d) => repro(content, res, d.x, d.bits, "worst inputs", recOf(res, d.value, d.reference, d.rel_err, d.ulp_err));
      const shown = [...rows].sort((a, b) => (b.rel_err ?? -1) - (a.rel_err ?? -1)).slice(0, 25);
      put(slots.detail,
        h("h2", {}, "Worst inputs"),
        tabs("detail", drawDetail),
        h("p.note", {},
          v ? ["Up to 25 recorded inputs with the largest finite relative errors in the selected range, ", span(), "."]
            : "Up to 25 recorded inputs with the largest finite relative errors. Drag across the chart above to filter the recorded inputs by range.",
          t ? " Each backend has its own recorded inputs, so the lists may show different input values." : ""),
        shown.length
          ? table([
            { label: "bits" }, { label: "x", align: "r" },
            { label: backendLabel(which), align: "r" }, { label: referenceLabel(res), align: "r" },
            { label: "Relative error", align: "r" }, { label: "ulp", align: "r" }, { label: "" },
          ], shown.map((d) => h("tr", {},
            h("td", {}, dRepro(h("code.bits", {}, d.bits ?? ""), d)),
            h("td.r", {}, dRepro(num(d.x), d)),
            h("td.r", {}, num(d.value)),
            h("td.r", {}, num(d.reference)),
            h("td.r", {}, num(d.rel_err, 3)),
            h("td.r", {}, num(d.ulp_err, 3)),
            h("td", {}, [
              d.rounded ? pill("ok", "rounded", "correctly rounded to the result's precision: the best it allows") : null,
              d.flushed ? pill("neutral", "Converted to zero", "The backend converted a subnormal input to zero and returned the expected result for zero") : null,
              isUf(d) ? pill("neutral", "Underflow to zero", "The result is zero while the reference value is subnormal at this precision") : null,
            ]))))
          : h("p.muted", {}, v ? "No differing samples recorded in this range." : "No differing samples recorded here."),
      );
    };
    drawDetail();

    // --- exact points ---
    const pointWhat = (p) => {
      if (p.failure) {
        return [catPill(p.category), " ", h("span.muted", {}, CAUSES[p.cause] ?? p.cause),
          p.ref_candidate && p.category !== "reference_limitation"
            ? h("div.muted.small", {}, `base R disputed by the stable reference (candidate; stable value ${num(p.stable)})`) : null,
          p.evidence ? h("div.muted.small", {}, p.evidence) : null];
      }
      if (p.zero_sign) return pill("neutral", "signed zero differs");
      if (p.rounded) return pill("ok", "correctly rounded");
      if (p.ref_candidate) return catPill(st.verified ? "reference_limitation" : "boundary", st.verified ? "verified base R limitation" : "Possible base R error — unverified");
      return h("span", {}, `Relative error ${num(p.rel_err, 3)}`);
    };
    const drawPoints = () => {
      const which = whose("points");
      const all = which === r.backend ? points : tPoints;
      const res = which === r.backend ? r : t;
      const pRepro = (content, p) => repro(content, res, p.x, p.bits, `exact point ${nameOf(p)}`,
        recOf(res, p.value, p.reference, p.rel_err, p.ulp_err, p.stable !== null && p.stable !== undefined ? [["stable ref", p.stable]] : []));
      const inRange = all.filter((p) => within(keyOfBits(p.bits)));
      // Rows for everything that differs in value; a sign-of-zero difference
      // alone is one line, so two dozen of them cannot bury a failure.
      const bad = inRange.filter((p) => !p.identical);
      const signOnly = inRange.filter((p) => p.identical && p.zero_sign);
      const nameOf = (p) => String(p.label).split("+").filter(Boolean).join(" · ");
      const order = (p) => (p.failure ? (p.category === "failure" ? 0 : 1) : 2);
      put(slots.points,
        h("h2", {}, "Special test points"),
        tabs("points", drawPoints),
        h("p.note", {}, "Checked beside every sweep, at inputs a sweep can miss: ±0, ±∞, NaN, the subnormal and normal extremes, ±½, ±1, the edges of the valid domain and the support, and ",
          r.backend === "anvl" ? "anvl's branch points" : "(for anvl) its branch points",
          " — each at this precision, with its two neighbours. Never added to the sweep's counts. ",
          all.length ? `${all.filter((p) => p.identical && !p.zero_sign).length} of ${all.length} test points match the reference, including the sign of zero.` : "",
          v ? [" Shown: points in ", span(), "."] : ""),
        signOnly.length
          ? h("p.note", { title: signOnly.map(nameOf).join("\n") },
            h("strong", {}, `${signOnly.length} point${signOnly.length === 1 ? "" : "s"}`),
            ` return a zero of the opposite sign to the reference (${signOnly.slice(0, 3).map(nameOf).join("; ")}${signOnly.length > 3 ? "; …" : ""}). Equal by value, and listed apart from those that are not.`)
          : null,
        bad.length
          ? table([{ label: "point" }, { label: "x", align: "r" }, { label: "bits" },
            { label: backendLabel(which), align: "r" }, { label: referenceLabel(res), align: "r" }, { label: "Finding" }],
          [...bad].sort((a, b) => order(a) - order(b) || (b.rel_err ?? 0) - (a.rel_err ?? 0)).map((p) =>
            h("tr", { class: p.failure && p.category === "failure" ? "row-warn" : null },
              h("td.wrap", { title: p.role }, h("code", {}, nameOf(p))),
              h("td.r", {}, pRepro(num(p.x), p)),
              h("td", {}, pRepro(h("code.bits", {}, p.bits), p)),
              h("td.r", {}, num(p.value)),
              h("td.r", {}, num(p.reference)),
              h("td", {}, pointWhat(p)))))
          : h("p.muted", {}, !all.length ? "No exact points recorded."
            : signOnly.length ? "Every other point is bit-identical."
              : v ? "Every exact point in this range is bit-identical." : "Every exact point is bit-identical."),
      );
    };
    drawPoints();

    // --- regions ---
    const overlaps = (x) => !inView || rangeKeys(x).some((k) => inView.has(k));
    const tagged = [
      ...ranges.filter(overlaps).map((x) => ({ ...x, be: r.backend })),
      ...tRanges.filter(overlaps).map((x) => ({ ...x, be: cmp })),
    ];
    const catOrder = { failure: 0, boundary: 1, backend_limitation: 2, reference_limitation: 3, undefined_domain: 4 };
    const rsorted = tagged.sort((a, b) =>
      (a.be === r.backend ? 0 : 1) - (b.be === r.backend ? 0 : 1) ||
      (catOrder[a.category] ?? 9) - (catOrder[b.category] ?? 9) ||
      (b.n_patterns ?? 0) - (a.n_patterns ?? 0));
    const resOf = (x) => (x.be === r.backend ? r : t);
    const extent = (x) => {
      // f32 bounds are sampled inputs; f64 bounds are those of the 2^32-pattern
      // blocks the failing samples fell in, not inputs that were evaluated --
      // so only sampled inputs offer a reproduction.
      const res = resOf(x);
      if (x.bounds_are_samples) {
        return [h("td.r", {}, repro(num(x.x_from), res, x.x_from, x.bits_from, "region start")),
          h("td.r", {}, repro(num(x.x_to), res, x.x_to, x.bits_to, "region end"))];
      }
      return [
        h("td.r", { title: `block bounds; sampled failing inputs from ${num(x.sampled_from)} to ${num(x.sampled_to)}` },
          num(x.x_from), h("div.muted.small", {}, "sampled ", repro(num(x.sampled_from), res, x.sampled_from, x.sampled_bits_from, "first sampled failing input of a region"))),
        h("td.r", { title: `block bounds; sampled failing inputs from ${num(x.sampled_from)} to ${num(x.sampled_to)}` },
          num(x.x_to), h("div.muted.small", {}, "sampled ", repro(num(x.sampled_to), res, x.sampled_to, x.sampled_bits_to, "last sampled failing input of a region"))),
      ];
    };
    put(slots.ranges,
      h("h2", {}, "Regions with undefined or infinite relative error"),
      h("p.note", {}, "Input ranges where relative error is undefined or infinite. Each range shows the category and cause identified by the benchmark, or states that the cause is unknown. ",
        parts.dtype === "f64" ? "For f64 the bounds are those of the 2³² blocks the failing samples fell in; the sampled inputs are shown beneath them. " : "",
        v ? ["Shown: regions that reach into ", span(), ", at their full extent."] : ""),
      rsorted.length
        ? table([
          ...(t ? [{ label: "backend" }] : []),
          { label: "from", align: "r" }, { label: "to", align: "r" },
          { label: "failing", align: "r", hint: "failing samples actually evaluated" },
          { label: "category, cause" }, { label: "returned", hint: "Types of values returned by the implementation and reference, with counts" },
          { label: "for example", hint: "one representative failing input" }],
          rsorted.map((x) => h("tr", { class: x.category === "failure" ? "row-warn" : null },
            t ? h("td", { class: x.be === cmp ? "cmp" : null }, backendLabel(x.be)) : null,
            extent(x),
            h("td.r", {}, int(x.n_failing)),
            h("td", {}, catPill(x.category), " ", h("span.muted", {}, CAUSES[x.cause] ?? x.cause),
              x.ref_candidate && x.category !== "reference_limitation"
                ? h("div.muted.small", {}, "base R disputed by the stable reference (candidate)") : null,
              x.evidence ? h("div.muted.small", {}, x.evidence) : null),
            h("td.small", {}, x.pairs ?? ""),
            h("td.small", {}, x.rep_x !== null && x.rep_x !== undefined
              ? [repro(h("code", {}, num(x.rep_x)), resOf(x), x.rep_x, x.rep_bits, "a region's example input", recOf(resOf(x), x.rep_value, x.rep_reference)),
                `: ${num(x.rep_value)} vs ${num(x.rep_reference)}`]
              : ""))))
        : h("p.muted", {}, v ? "None in this range." : "None."),
    );

    // --- disputes with base R ---
    if (slots.disputes) {
      const cands = disputes.filter((d) => within(detailKey(d)));
      const ex = [...cands].sort((a, b) => (b.beyond_tolerance ?? 0) - (a.beyond_tolerance ?? 0)).slice(0, 12);
      put(slots.disputes,
        h("h2", {}, "Disputes with base R"),
        h("p.note", {}, st.verified
          ? h("strong", {}, "Verified base R limitations: the stable reference passed validation, so adjusted error statistics exclude these samples. ")
          : h("strong", {}, `Candidates only: the stable reference is ${REF_STATUS[r.ref_stable_status]?.label ?? r.ref_stable_status}, so these samples remain included in the reported results. `),
          r.ref_stable_note ? `Why: ${r.ref_stable_note}. ` : "",
          "Candidates are inputs where base R exceeds its tolerance against the stable reference, while anvl stays within its tolerance and is at least as close to that reference."),
        h("details.fold.inner", {}, h("summary", {}, "Candidate selection thresholds"),
          h("p.note", {}, "Let s be the stable reference and B its declared error bound. A candidate must satisfy all three conditions: |base R − s| > 4 ULP at f64 precision + B; |anvl − s| ≤ 2 ULP at the tested precision + B; and |anvl − s| ≤ |base R − s|.")),
        h("dl.stats", {},
          h("div.stat", {}, h("dt", {}, "candidates"), h("dd", {}, int(r.n_ref_candidate))),
          h("div.stat", {}, h("dt", {}, "Candidates with undefined or infinite relative error"), h("dd", {}, int(r.n_ref_candidate_nonfinite))),
          h("div.stat", { title: "anvl and base R return the same value, and both are beyond anvl's tolerance against the stable reference" },
            h("dt", {}, "Both agree but exceed tolerance"), h("dd", {}, int(r.n_ref_shared))),
          h("div.stat", {}, h("dt", {}, "Maximum relative error, all samples"), h("dd", {}, num(r.worst_rel_err, 3))),
          h("div.stat", {}, h("dt", {}, st.verified ? "Maximum error excluding verified limitations" : "Maximum error excluding unverified candidates"), h("dd", {}, num(r.worst_rel_err_excl, 3)))),
        ex.length
          ? table([{ label: "kind" }, { label: "x", align: "r" }, { label: me, align: "r" }, { label: "base R", align: "r" },
            { label: "stable", align: "r" }, { label: `|${me} − s|`, align: "r" }, { label: "tolerance", align: "r" },
            { label: "|base R − s|", align: "r" }, { label: "tolerance", align: "r" }],
          ex.map((d) => h("tr", {},
            h("td", {}, d.kind === "shared" ? pill("warn", "Both exceed tolerance", "anvl and base R agree, and both are beyond anvl's tolerance") : pill("neutral", "candidate")),
            h("td.r", {}, repro(num(d.x), r, d.x, d.bits, "dispute with base R",
              recOf(r, d.value, d.reference, null, null, [["stable ref", d.stable]]))),
            h("td.r", {}, num(d.value, 6)), h("td.r", {}, num(d.reference, 6)), h("td.r", {}, num(d.stable, 6)),
            h("td.r", {}, num(d.d_anvl, 3)), h("td.r", {}, num(d.t_anvl, 3)),
            h("td.r", {}, num(d.d_base, 3)), h("td.r", {}, num(d.t_base, 3)))))
          : h("p.muted", {}, v ? "No retained disputes in this range." : "No disputes recorded."),
      );
    }
  };

  draw(zoom);
  app.redrawCell = draw;
  if (slots.validation) drawValidation(slots.validation, r, t);
}

// Whose rank puts ±0 nearest the sign change, as the chart orders columns.
const rank = (b) => (b.zero ? -1 : b.binade);

/** The latest validation of each reference this result depends on. */
async function drawValidation(slot, r, t) {
  const s = app.store;
  const latest = (vs) => vs.slice().sort((a, b) => String(b.validated_at).localeCompare(String(a.validated_at)))[0];
  const refs = [
    r.ref_stable_status ? { kind: "stable", label: "Stable reference (evidence in base R disputes)", status: r.ref_stable_status,
      vs: s.validationsOf(r.ref_stable_id), output: "value" } : null,
    r.ref_grad_status ? { kind: "gradient", label: "Gradient reference (what this result is scored against)", status: r.ref_grad_status,
      vs: s.validationsOf(r.ref_grad_id, r.output), output: r.output } : null,
  ].filter(Boolean);
  const blocks = [];
  for (const ref of refs) {
    const v = latest(ref.vs);
    const samples = v ? await s.samplesOf(v.validation_id, v.ref_id, ref.output) : [];
    blocks.push(h("div.vblock", {},
      h("h3", {}, ref.label, " ", statusPill(ref.status)),
      !v
        ? h("p.muted", {}, ref.status === "no identity"
          ? "The benchmark could not identify the reference implementation, so it cannot associate this result with a validation record."
          : "No validation of this reference is recorded.")
        : [
          h("dl.stats", {},
            h("div.stat", {}, h("dt", {}, "worst error"), h("dd", {}, `${num(v.max_err_ulp64, 3)} ulp`)),
            h("div.stat", {}, h("dt", {}, "declared bound"), h("dd", {}, `${v.bound_ulp64} ulp`)),
            h("div.stat", {}, h("dt", {}, "samples"), h("dd", {}, int(v.n_samples))),
            h("div.stat", {}, h("dt", {}, "precision"), h("dd", {}, `${v.precision} bits`)),
            h("div.stat", {}, h("dt", {}, "validated"), h("dd", {}, v.validated_at))),
          v.reason ? h("p.note", {}, h("strong", {}, v.pass ? "" : "Failed: "), v.reason) : null,
          h("p.note", {}, "Validation must use the current method, the same reference implementation as this test, and the latest MPFR reference calculation. ",
            ref.vs.length > 1 ? `${ref.vs.length} validations of this reference are recorded; the latest is shown. ` : "",
            "These checks provide evidence from sampled inputs; they do not prove the error bound for every input."),
          h("details.fold.inner", {}, h("summary", {}, "Sampling method and software versions"),
            h("p.note", {}, v.selection),
            h("dl.fingerprint", {},
              [["seed", v.seed], ["R", v.r_version], ["Rmpfr", v.rmpfr_version], ["MPFR", v.mpfr_version],
               ["reference identity", v.ref_id], ["truth identity", v.truth_id], ["method identity", v.method_id]]
                .map(([k, x]) => h("div.stat", {}, h("dt", {}, k), h("dd", { class: "mono", title: String(x ?? "") }, String(x ?? "—").slice(0, 16)))))),
          samples.length
            ? h("details.fold.inner", {}, h("summary", {}, `Worst samples (${samples.length} recorded)`),
              table([{ label: "source" }, { label: "x", align: "r" }, { label: "reference", align: "r" },
                { label: "High-precision MPFR value", align: "r" }, { label: "error, ulp", align: "r" }],
              samples.slice().sort((a, b) => (b.err_ulp64 ?? 0) - (a.err_ulp64 ?? 0)).slice(0, 20).map((x) =>
                h("tr", { class: x.pass === false ? "row-warn" : null },
                  h("td.small", {}, x.source),
                  h("td.r", {}, repro(num(x.x), s.result(x.cell_id, x.output) ?? r, x.x, x.bits, `validation sample (${x.source})`,
                    [["harness ref", x.ref], ["High-precision MPFR value", x.truth], ["error, f64 ulp", x.err_ulp64]])),
                  h("td.r", {}, num(x.ref, 6)), h("td.r", {}, num(x.truth, 6)), h("td.r", {}, num(x.err_ulp64, 3))))))
            : null,
        ]));
  }
  put(slot, h("h2", {}, "Reference validation"),
    h("p.note", {}, "Each reference checked against 256-bit MPFR on the exact inputs and parameters the sweep used", t ? `; ${backendLabel(t.backend)}'s result is scored against the same reference, so one validation serves both` : "", "."),
    blocks);
}

// --- wiring ------------------------------------------------------------

async function load(sourcePromise) {
  setStatus("Loading results…");
  const source = await sourcePromise;
  app.store = await openStore(source);
  app.chartCell = null;
  renderHeader();
}

let lastPage = null;

function route() {
  if (!app.store) return;
  const r = parseRoute();
  // Zooming is a move within the page, not a move to another one.
  const page = `${r.view}|${r.spec ?? ""}|${r.cellId ?? ""}|${r.output ?? ""}`;
  const samePage = page === lastPage;
  lastPage = page;
  try {
    if (r.view === "spec") renderSpec(r.spec);
    else if (r.view === "cell") {
      // A new zoom on the same result redraws in place: no refetch, no flash.
      if (samePage && app.redrawCell) app.redrawCell(r.zoom);
      else renderCell(r.cellId, r.output, r.zoom);
    }
    else renderOverview();
  } catch (e) {
    console.error(e);
    setStatus(String(e.message ?? e), "error");
  }
  if (!samePage) scrollTo(0, 0);
}

addEventListener("hashchange", route);

(async function start() {
  app.deployed = await listDeployed();
  renderHeader();
  if (!app.deployed.length) {
    return setStatus(
      "No results are deployed with this page. Download and extract a results archive from the project’s GitHub releases, then use Open downloaded results to select manifest.json and all Parquet files together.",
      "empty");
  }
  try {
    await load(urlSource(app.deployed[0].dir));
    route();
  } catch (e) {
    console.error(e);
    setStatus(String(e.message ?? e), "error");
  }
})();
