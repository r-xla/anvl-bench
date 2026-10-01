// Charts, as hand-written SVG. There is no charting library here on purpose:
// the binade chart is a few thousand rectangles with a click handler, which is
// not a library's problem, and staying dependency-free keeps the page small.

import { MINUS, num, decadeTick } from "./fmt.js";

const NS = "http://www.w3.org/2000/svg";

const el = (name, attrs = {}, kids = []) => {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) if (v !== null) n.setAttribute(k, v);
  for (const k of [].concat(kids)) n.append(k);
  return n;
};

/** Behaviour drives colour; the bar's height carries the size of the error. */
export const BEHAVIOUR = {
  "all identical": { cls: "b-identical", label: "All match the reference" },
  "identical + differ": { cls: "b-some", label: "Some match; some differ" },
  "all differ": { cls: "b-differ", label: "All differ from the reference" },
  "mixed, some non-finite": { cls: "b-nonfinite", label: "Some errors are undefined or infinite" },
  "no finite error": { cls: "b-none", label: "No finite relative errors" },
};
const behaviourClass = (b) => BEHAVIOUR[b]?.cls ?? "b-other";

const W = 1000;
const H = 260;
const PAD = { l: 58, r: 12, t: 26, b: 34 };
const PLOT_W = W - PAD.l - PAD.r;
const PLOT_H = H - PAD.t - PAD.b;

/**
 * A band's key along the axis: sign and binade, or sign and "z" for the band
 * holding the sweep's ±0 samples, which the harness keeps apart from binade
 * 0's subnormals. The same key names the same interval for any backend at one
 * precision, which is what lets a comparator and a zoom line up.
 */
export const bandKey = (b) => (b.zero ? `${b.sign}:z` : `${b.sign}:${b.binade}`);

/**
 * Order the bands along the real line: most negative first (sign -1, highest
 * binade) down through the negative subnormals to -0, then +0, the positive
 * subnormals and up to the largest positive. The top exponent field is not an
 * interval -- it holds both infinities and every NaN -- so it is pulled out and
 * shown as a marker at each end rather than as part of the axis.
 */
export function orderBands(bands) {
  const finite = bands.filter((b) => !b.special);
  const rank = (b) => (b.zero ? -1 : b.binade); // zero sits nearest the sign change
  const neg = finite.filter((b) => b.sign < 0).sort((a, b) => rank(b) - rank(a));
  const pos = finite.filter((b) => b.sign > 0).sort((a, b) => rank(a) - rank(b));
  return {
    columns: [...neg, ...pos],
    zeroAt: neg.length,
    special: bands.filter((b) => b.special),
  };
}

/**
 * Reduce a run of bands to one bar by taking the worst of them. Summarising
 * an accuracy chart by the maximum is the only reduction that cannot hide a
 * finding: a bucket is as bad as its worst member.
 */
function bucketOf(columns, from, to) {
  let err = null; // log10 of the worst relative error in the bucket
  let behaviour = null;
  let mixed = false;
  let worst = null;
  let n = 0;
  for (let i = from; i < to; i++) {
    const b = columns[i];
    // Height comes from the continuous worst_rel_err, not from the integer
    // decade: a whole sweep often sits inside a single decade, and rounding to
    // it would draw one flat block over real structure.
    const e = b.worst_rel_err;
    if (typeof e === "number" && e > 0 && Number.isFinite(e)) {
      const l = Math.log10(e);
      if (err === null || l > err) { err = l; worst = b; }
    }
    if (behaviour === null) behaviour = b.behaviour;
    else if (behaviour !== b.behaviour) mixed = true;
    n += (b.n_identical ?? 0) + (b.n_differ ?? 0) + (b.n_nonfinite ?? 0);
  }
  return {
    from,
    to,
    err,
    behaviour: mixed ? null : behaviour,
    worstBand: worst ?? columns[from],
    samples: n,
  };
}

/** Worst log10 relative error of a comparator over the same binades. */
function compareOf(columns, from, to, byKey) {
  let present = false;
  let err = null;
  for (let i = from; i < to; i++) {
    const c = byKey.get(bandKey(columns[i]));
    if (!c) continue;
    present = true;
    const e = c.worst_rel_err;
    if (typeof e === "number" && e > 0 && Number.isFinite(e)) {
      const l = Math.log10(e);
      if (err === null || l > err) err = l;
    }
  }
  return { present, err };
}

/**
 * The binade chart. `view` is the index range currently shown. Dragging across
 * the plot selects a range and calls `onZoom([from, to))` with column indices;
 * the caller owns the view (it lives in the URL) and re-renders. Hovering a bar
 * still gives that binade's figures.
 *
 * `compare`, when given, is another backend's bands for the same cell (today,
 * JAX). It is drawn as an ink line over anvl's bars -- a different mark, not
 * just a different colour -- on the same axis, since both are the same
 * quantity: relative error against the same base R reference.
 */
export function binadeChart({ bands, view, onZoom, compare = null, domain = null }) {
  const { columns, zeroAt, special } = orderBands(bands);
  const wrap = document.createElement("div");
  wrap.className = "chart";

  if (!columns.length) {
    wrap.innerHTML = '<p class="muted">No band data for this result.</p>';
    return wrap;
  }

  const i0 = Math.max(0, view?.[0] ?? 0);
  const i1 = Math.min(columns.length, view?.[1] ?? columns.length);
  const n = i1 - i0;
  const nBuckets = Math.min(n, Math.floor(PLOT_W / 2));
  const buckets = [];
  const cmpByKey = new Map(
    (compare?.bands ?? []).filter((b) => !b.special).map((b) => [bandKey(b), b]));
  for (let j = 0; j < nBuckets; j++) {
    const from = i0 + Math.floor((j * n) / nBuckets);
    const to = Math.max(i0 + Math.floor(((j + 1) * n) / nBuckets), from + 1);
    const b = bucketOf(columns, from, to);
    b.cmp = cmpByKey.size ? compareOf(columns, from, to, cmpByKey) : null;
    buckets.push(b);
  }

  // Scale to the errors actually present, so an f64 result whose errors all sit
  // near 1e-16 still fills the chart instead of flattening against the axis.
  // The comparator shares the axis, so its errors count toward the range too.
  const present = buckets.flatMap((b) => [b.err, b.cmp?.err ?? null]).filter((d) => d !== null);
  const dMax = present.length ? Math.ceil(Math.max(...present)) : 0;
  let dMin = present.length ? Math.floor(Math.min(...present)) : -1;
  if (dMax - dMin < 1) dMin = dMax - 1;
  const span = dMax - dMin;
  const y = (v) => PAD.t + PLOT_H - ((v - dMin) / span) * PLOT_H;

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "binade-svg",
    role: "img",
    "aria-label": "Maximum relative error across input ranges",
  });

  // y axis. Integer decades are too coarse when a whole sweep lives inside one
  // decade, which is the common case, so fall back to five even steps labelled
  // with the value itself.
  const ticks = [];
  if (span >= 3) {
    const every = Math.max(1, Math.ceil(span / 8));
    for (let d = dMax; d >= dMin; d -= every) ticks.push([d, decadeTick(d)]);
  } else {
    for (let i = 0; i <= 4; i++) {
      const v = dMin + (span * i) / 4;
      ticks.push([v, num(10 ** v, 2)]);
    }
  }
  for (const [v, label] of (present.length ? ticks : [])) {
    svg.append(
      el("line", { class: "grid", x1: PAD.l, x2: W - PAD.r, y1: y(v).toFixed(1), y2: y(v).toFixed(1) }),
      el("text", { class: "tick", x: PAD.l - 6, y: (y(v) + 3.5).toFixed(1), "text-anchor": "end" },
        document.createTextNode(label)),
    );
  }
  svg.append(el("text", { class: "axis-title", x: 6, y: 12 },
    document.createTextNode(present.length
      ? "Maximum finite relative error against reference"
      : "No finite relative errors recorded")));

  // Input classes, as backgrounds behind the bars (input_class() in the
  // harness): the ±0 column, the subnormals of binade 0, and bands wholly
  // outside the valid input domain -- where the value is NaN by specification.
  // The distribution's support plays no part: a CDF below it still has an
  // ordinary answer. Shading runs of columns (not buckets) keeps the extent
  // exact at any zoom.
  const classOf = (c) =>
    c.zero ? "zero"
      : c.binade === 0 ? "sub"
        : domain && (c.x_to < domain[0] || c.x_from > domain[1]) ? "oos" : null;
  svg.append(el("defs", {}, el("pattern", {
    id: "hatch-oos", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)",
  }, el("line", { class: "hatch", x1: 0, y1: 0, x2: 0, y2: 6 }))));
  let runStart = i0;
  for (let i = i0; i <= i1; i++) {
    const cls = i < i1 ? classOf(columns[i]) : undefined;
    const prev = runStart < i1 ? classOf(columns[runStart]) : null;
    if (i === i1 || cls !== prev) {
      if (prev) {
        const x0 = PAD.l + ((runStart - i0) / n) * PLOT_W;
        const w = Math.max(((i - runStart) / n) * PLOT_W, 2);
        svg.append(el("rect", { class: `bg bg-${prev}`, x: x0.toFixed(2), width: w.toFixed(2), y: PAD.t, height: PLOT_H }));
      }
      runStart = i;
    }
  }
  const CLASS_NAME = { zero: "±0", sub: "subnormal", oos: "outside the domain" };
  const inputsOf = (b) => {
    const seen = new Set();
    for (let i = b.from; i < b.to; i++) seen.add(CLASS_NAME[classOf(columns[i])] ?? "normal");
    return [...seen].join(" + ");
  };

  // bars
  const bw = PLOT_W / nBuckets;
  const g = el("g", { class: "bars" });
  const hits = el("g", { class: "hits" });
  buckets.forEach((b, j) => {
    const x = PAD.l + j * bw;
    const base = PAD.t + PLOT_H;
    const top = b.err === null ? base - 1.5 : y(b.err);
    const rect = el("rect", {
      class: `bar ${behaviourClass(b.behaviour)}`,
      x: x.toFixed(2),
      width: Math.max(bw - 0.15, 0.4).toFixed(2),
      y: top.toFixed(2),
      height: Math.max(base - top, 1.5).toFixed(2),
    });
    const wb = b.worstBand;
    const spanTxt = (b.to - b.from > 1 ? `${b.to - b.from} binades` : wb.zero ? `${wb.sign < 0 ? MINUS : "+"}0 exactly` : `binade ${wb.binade}`) +
      ` \u2014 inputs: ${inputsOf(b)}`;
    const tip =
      `${spanTxt}\n${wb.sign < 0 ? MINUS : "+"} ${num(Math.abs(wb.x_from))} \u2026 ${num(Math.abs(wb.x_to))}\n` +
      `anvl maximum relative error ${b.err === null ? "not available" : num(10 ** b.err, 3)}` +
      ` (${BEHAVIOUR[b.behaviour]?.label ?? "Several comparison outcomes"})` +
      (b.cmp
        ? `\n${compare.label} maximum relative error ${!b.cmp.present ? "not swept" : b.cmp.err === null ? "not available" : num(10 ** b.cmp.err, 3)}`
        : "");
    rect.append(el("title", {}, document.createTextNode(tip)));
    g.append(rect);

    // A bar is under two pixels wide at full range, so hovering the bar itself
    // is a test of aim. The hover target is the full-height column behind it.
    const hit = el("rect", {
      class: "hit",
      x: x.toFixed(2),
      width: Math.max(bw, 2).toFixed(2),
      y: PAD.t,
      height: PLOT_H,
    });
    hit.append(el("title", {}, document.createTextNode(tip)));
    hits.append(hit);
  });
  svg.append(g);

  // The comparator's line: a step across each bucket, broken wherever it has no
  // band at all, and resting on the baseline where it has one with no error.
  if (buckets.some((b) => b.cmp?.present)) {
    const base = PAD.t + PLOT_H;
    let d = "";
    let open = false;
    let last = null;
    buckets.forEach((b, j) => {
      if (!b.cmp?.present) { open = false; return; }
      const yv = (b.cmp.err === null ? base : y(b.cmp.err)).toFixed(2);
      const x0 = (PAD.l + j * bw).toFixed(2);
      const x1 = (PAD.l + (j + 1) * bw).toFixed(2);
      d += open ? `L${x0},${yv}L${x1},${yv}` : `M${x0},${yv}L${x1},${yv}`;
      open = true;
      if (b.cmp.err !== null) last = { x: Number(x1), y: Number(yv) };
    });
    svg.append(
      el("path", { class: "cmp-halo", d }),
      el("path", { class: "cmp-line", d }),
    );
    // Label the line directly, so identity never rests on the legend alone.
    if (last) {
      svg.append(el("text", {
        class: "cmp-label",
        x: Math.min(last.x, W - PAD.r - 2).toFixed(1),
        y: Math.max(last.y - 5, PAD.t + 9).toFixed(1),
        "text-anchor": "end",
      }, document.createTextNode(compare.label)));
    }
  }
  // Hover targets last, so the line never intercepts the pointer.
  svg.append(hits);

  // Drag to zoom. The brush is in viewBox units; converting from the pointer
  // needs only the rendered width, because the svg keeps its aspect ratio.
  const brush = el("rect", { class: "brush", y: PAD.t, height: PLOT_H, x: 0, width: 0, visibility: "hidden" });
  svg.append(brush);
  const toX = (e) => {
    const r = svg.getBoundingClientRect();
    return Math.min(Math.max(((e.clientX - r.left) * W) / r.width, PAD.l), PAD.l + PLOT_W);
  };
  const bucketAt = (x) => Math.min(nBuckets - 1, Math.max(0, Math.floor((x - PAD.l) / bw)));
  let start = null;
  svg.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    start = toX(e);
    svg.setPointerCapture(e.pointerId);
    brush.setAttribute("x", start.toFixed(2));
    brush.setAttribute("width", "0");
    brush.setAttribute("visibility", "visible");
    e.preventDefault();
  });
  svg.addEventListener("pointermove", (e) => {
    if (start === null) return;
    const x = toX(e);
    brush.setAttribute("x", Math.min(start, x).toFixed(2));
    brush.setAttribute("width", Math.abs(x - start).toFixed(2));
  });
  const finish = (e, commit) => {
    if (start === null) return;
    const x = toX(e);
    const [a, b] = [Math.min(start, x), Math.max(start, x)];
    start = null;
    brush.setAttribute("visibility", "hidden");
    // A press without a real drag is not a zoom; neither is a range that
    // already is the whole view.
    if (!commit || b - a < 4) return;
    const from = buckets[bucketAt(a)].from;
    const to = buckets[bucketAt(b)].to;
    if (from === i0 && to === i1) return;
    onZoom([from, to]);
  };
  svg.addEventListener("pointerup", (e) => finish(e, true));
  svg.addEventListener("pointercancel", (e) => finish(e, false));

  // the sign change, when it is inside the view
  if (zeroAt > i0 && zeroAt < i1) {
    const x = PAD.l + ((zeroAt - i0) / n) * PLOT_W;
    svg.append(
      el("line", { class: "zero", x1: x.toFixed(1), x2: x.toFixed(1), y1: PAD.t, y2: PAD.t + PLOT_H }),
      el("text", { class: "tick", x: (x - 4).toFixed(1), y: H - 20, "text-anchor": "end" },
        document.createTextNode("x < 0")),
      el("text", { class: "tick", x: (x + 4).toFixed(1), y: H - 20 },
        document.createTextNode("x > 0")),
    );
  }

  // x axis: the lower bound of the leftmost binade and the upper bound of the
  // rightmost. x_from < x_to on either sign, so this holds zoomed or not.
  svg.append(
    el("line", { class: "axis", x1: PAD.l, x2: W - PAD.r, y1: PAD.t + PLOT_H, y2: PAD.t + PLOT_H }),
    el("text", { class: "tick", x: PAD.l, y: H - 8 }, document.createTextNode(num(columns[i0].x_from, 2))),
    el("text", { class: "tick", x: W - PAD.r, y: H - 8, "text-anchor": "end" },
      document.createTextNode(num(columns[i1 - 1].x_to, 2))),
  );

  wrap.append(svg);

  // The non-interval top exponent field, kept out of the axis on purpose.
  if (special.length) {
    const s = document.createElement("div");
    s.className = "special-row";
    s.append(Object.assign(document.createElement("span"), {
      className: "special-label",
      textContent: "top exponent field (±∞ and every NaN, not an interval):",
    }));
    for (const b of special) {
      const chip = document.createElement("span");
      chip.className = `chip ${behaviourClass(b.behaviour)}`;
      chip.textContent = `sign ${b.sign < 0 ? MINUS : "+"}: ${BEHAVIOUR[b.behaviour]?.label ?? b.behaviour}` +
        (b.n_nonfinite ? `, ${b.n_nonfinite} non-finite` : "");
      s.append(chip);
    }
    wrap.append(s);
  }

  return { wrap, columns, zeroAt, nColumns: columns.length, view: [i0, i1] };
}

/**
 * The error histogram: counts by decade, on a log scale. A comparator, when
 * given, is drawn as an ink step outline over the bars, on the same axes --
 * both are counts of the same number of samples, so they compare directly.
 */
export function histChart(rows, compare = null) {
  const wrap = document.createElement("div");
  wrap.className = "chart";
  const cmpBy = new Map((compare?.rows ?? []).map((d) => [d.decade, d.count]));
  const decades = [...new Set([...rows.map((d) => d.decade), ...cmpBy.keys()])].sort((a, b) => a - b);
  const mine = new Map(rows.map((d) => [d.decade, d.count]));
  const data = decades.map((decade) => ({
    decade, count: mine.get(decade) ?? 0, cmp: cmpBy.has(decade) ? cmpBy.get(decade) : null,
  }));
  const any = (d) => d.count > 0 || d.cmp > 0;
  if (!data.some(any)) {
    wrap.innerHTML = '<p class="muted">No finite errors recorded for this result.</p>';
    return wrap;
  }
  // Trim empty decades at both ends so the occupied range fills the chart.
  let lo = data.findIndex(any);
  let hi = data.length - 1 - [...data].reverse().findIndex(any);
  lo = Math.max(0, lo - 1);
  hi = Math.min(data.length - 1, hi + 1);
  const shown = data.slice(lo, hi + 1);

  const h = 150;
  const pad = { l: 52, r: 12, t: 10, b: 30 };
  const pw = W - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const max = Math.max(...shown.map((d) => Math.max(d.count, d.cmp ?? 0)));
  const scale = (c) => (c <= 0 ? 0 : (Math.log10(c + 1) / Math.log10(max + 1)) * ph);

  const svg = el("svg", { viewBox: `0 0 ${W} ${h}`, class: "hist-svg", role: "img",
    "aria-label": "distribution of relative error by decade" });
  svg.append(el("text", { class: "axis-title", x: 6, y: pad.t + 4 },
    document.createTextNode("samples (log)")));
  const bw = pw / shown.length;
  const hits = [];
  shown.forEach((d, j) => {
    const bh = scale(d.count);
    const x = pad.l + j * bw;
    const rect = el("rect", {
      class: "hbar", x: (x + bw * 0.1).toFixed(2), width: (bw * 0.8).toFixed(2),
      y: (pad.t + ph - bh).toFixed(2), height: Math.max(bh, d.count > 0 ? 1 : 0).toFixed(2),
    });
    const tip =
      `Relative error 1e${String(d.decade).replace("-", MINUS)} – 1e${String(d.decade + 1).replace("-", MINUS)}\n` +
      `anvl: ${d.count.toLocaleString("en-US")} samples` +
      (compare ? `\n${compare.label}: ${(d.cmp ?? 0).toLocaleString("en-US")} samples` : "");
    rect.append(el("title", {}, document.createTextNode(tip)));
    svg.append(rect);
    // Hover works on the whole decade, not only where anvl has a bar.
    const hit = el("rect", { class: "hhit", x: x.toFixed(2), width: bw.toFixed(2), y: pad.t, height: ph });
    hit.append(el("title", {}, document.createTextNode(tip)));
    hits.push(hit);
    if (shown.length <= 14 || j % 2 === 0) {
      svg.append(el("text", { class: "tick", x: (x + bw / 2).toFixed(1), y: h - 10,
        "text-anchor": "middle" }, document.createTextNode(decadeTick(d.decade))));
    }
  });
  svg.append(el("line", { class: "axis", x1: pad.l, x2: W - pad.r,
    y1: pad.t + ph, y2: pad.t + ph }));

  if (compare && shown.some((d) => d.cmp !== null)) {
    let d = "";
    let last = null;
    shown.forEach((s, j) => {
      const yv = (pad.t + ph - scale(s.cmp ?? 0)).toFixed(2);
      const x0 = (pad.l + j * bw).toFixed(2);
      const x1 = (pad.l + (j + 1) * bw).toFixed(2);
      d += `${j ? "L" : "M"}${x0},${yv}L${x1},${yv}`;
      if (s.cmp > 0) last = { x: Number(x1), y: Number(yv) };
    });
    svg.append(el("path", { class: "cmp-halo", d }), el("path", { class: "cmp-line", d }));
    if (last) {
      svg.append(el("text", {
        class: "cmp-label", "text-anchor": "end",
        x: Math.min(last.x, W - pad.r - 2).toFixed(1), y: Math.max(last.y - 5, pad.t + 9).toFixed(1),
      }, document.createTextNode(compare.label)));
    }
  }
  svg.append(...hits);
  wrap.append(svg);
  return wrap;
}
