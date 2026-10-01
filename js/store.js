// The domain layer: what the site knows about an artifact, and how little of
// it has to be fetched to answer a question.
//
// Read strategy, in three tiers:
//
//   eager    manifest.json + summary, runs, categories and validations: small,
//            and enough to render the overview and every function page
//            without touching another byte.
//   whole    hist, ranges and validation_samples, read once on the first leaf
//            page and kept.
//   by cell  detail, bands, points and disputes are sorted by cell with a row
//            group boundary at each cell, and the footer carries exact
//            min/max statistics on cell_id. So the row group holding a cell
//            is found from the footer alone and read on its own: measured at
//            ~2% of detail.parquet for one cell.

import { parquetMetadataAsync, parquetReadObjects } from "./hyparquet.js";

const EAGER = ["summary", "runs", "categories", "validations"];
const WHOLE = ["hist", "ranges", "validation_samples"];
const BY_CELL = ["detail", "bands", "points", "disputes"];

/** The oldest artifact layout this site reads (the harness's SCHEMA_VERSION). */
export const SCHEMA = 7;

const SEP = "␟"; // never appears in a cell_id or an output name

/** Parquet statistics come back as bytes or as a string depending on writer. */
function asText(v) {
  if (v === null || v === undefined) return undefined;
  if (typeof v === "string") return v;
  if (v instanceof Uint8Array) return new TextDecoder().decode(v);
  if (ArrayBuffer.isView(v)) return new TextDecoder().decode(new Uint8Array(v.buffer));
  return String(v);
}

export async function openStore(source) {
  const manifest = source.manifest;
  // Every figure on this site rests on how the harness classified it, and that
  // changed shape between versions: reading an older artifact with this code
  // would mislabel regions rather than fail. Say so instead.
  if (!(Number(manifest.schema_version) >= SCHEMA)) {
    throw new Error(
      `This artifact was exported with schema version ${manifest.schema_version ?? "unknown"}; this site reads ` +
      `version ${SCHEMA} and later. Re-export it with the current harness (Rscript run.R export).`);
  }
  const meta = new Map();
  const whole = new Map();

  const metadata = async (table) => {
    if (!meta.has(table)) meta.set(table, parquetMetadataAsync(await source.buffer(table)));
    return meta.get(table);
  };

  const readAll = async (table, columns) =>
    parquetReadObjects({ file: await source.buffer(table), columns });

  // The small tables in one request. They are written with a row group per
  // cell like the large ones, and reading them through the range reader cost
  // one round trip per row group -- over a hundred for `hist` on a single page.
  const readWhole = async (table) => {
    const buf = await source.buffer(table);
    return parquetReadObjects({ file: await buf.slice(0, buf.byteLength) });
  };

  // An artifact says what it holds in its manifest; older ones predate the
  // per-class table, and asking for a file that is not there would fetch a 404
  // page and try to parse it as Parquet.
  const listed = (t) =>
    source.has(t) && (manifest.files ?? []).some((f) => f.table === t && f.rows > 0);

  const [summary, runs, categories, validations] = await Promise.all([
    readAll("summary"),
    // The fingerprint is nice to have, not load-bearing: a partial artifact
    // (just manifest + summary) should still open.
    source.has("runs") ? readAll("runs").catch(() => []) : [],
    // Figures by input class: a few rows per result, small enough to read up
    // front, which is what lets the overview show normal-input accuracy.
    listed("categories") ? readAll("categories").catch(() => []) : [],
    // Every validation of a reference these results were scored with or
    // disputed by. Keyed by reference identity, not by cell: one validation
    // serves an anvl cell and its JAX twin alike.
    listed("validations") ? readAll("validations").catch(() => []) : [],
  ]);

  /**
   * Row range covering one cell, from the footer statistics. Returns null if
   * the writer left no statistics, in which case the caller falls back to
   * reading the table whole rather than returning a wrong answer.
   */
  async function cellRange(table, cellId) {
    const md = await metadata(table);
    let row = 0;
    let lo = null;
    let hi = null;
    for (const g of md.row_groups) {
      const n = Number(g.num_rows);
      const col = g.columns.find(
        (c) => (c.meta_data?.path_in_schema ?? []).join(".") === "cell_id",
      );
      const st = col?.meta_data?.statistics;
      const min = asText(st?.min_value);
      const max = asText(st?.max_value);
      if (min === undefined || max === undefined) return null;
      // A cell occupies one row group today, but tolerate it spanning several.
      if (min <= cellId && cellId <= max) {
        if (lo === null) lo = row;
        hi = row + n;
      } else if (lo !== null) {
        break;
      }
      row += n;
    }
    return lo === null ? { rowStart: 0, rowEnd: 0 } : { rowStart: lo, rowEnd: hi };
  }

  /** Every row of `table` belonging to one cell (all of its outputs). */
  async function cellRows(table, cellId, columns) {
    if (!source.has(table) || !listed(table)) return [];
    if (WHOLE.includes(table)) {
      if (!whole.has(table)) whole.set(table, readWhole(table));
      return (await whole.get(table)).filter((r) => r.cell_id === cellId);
    }
    const range = await cellRange(table, cellId);
    if (range === null) {
      // No statistics to seek with. Correct, just slow, and says so.
      console.warn(`${table}.parquet has no cell_id statistics; reading it whole`);
      if (!whole.has(table)) whole.set(table, readAll(table, columns));
      return (await whole.get(table)).filter((r) => r.cell_id === cellId);
    }
    if (range.rowEnd === range.rowStart) return [];
    const rows = await parquetReadObjects({
      file: await source.buffer(table),
      columns,
      rowStart: range.rowStart,
      rowEnd: range.rowEnd,
    });
    // The row group may hold neighbours if a future writer packs cells
    // together, so filter rather than trust the boundary.
    return rows.filter((r) => r.cell_id === cellId);
  }

  // --- indexes over the summary, which is the whole navigable structure ---

  const results = summary.map((r) => ({ ...r, key: r.cell_id + SEP + r.output }));
  const byKey = new Map(results.map((r) => [r.key, r]));

  // anvl is what this site is about; any other backend (today, JAX) is a
  // comparator, measured against the same base R reference on the same inputs.
  // Everything navigable and every aggregate is built from the subject rows
  // only -- otherwise a JAX result would be counted as one of anvl's -- and a
  // comparator is reached solely as the twin of an anvl result.
  const backends = [...new Set(results.map((r) => r.backend))].sort();
  const primary = backends.includes("anvl") ? "anvl" : backends[0];
  const comparators = backends.filter((b) => b !== primary);
  const subject = results.filter((r) => r.backend === primary);

  /** cell_id is spec/backend/...; a twin differs only in that segment. */
  const twinId = (cellId, backend) => {
    const parts = String(cellId).split("/");
    parts[1] = backend;
    return parts.join("/");
  };
  /** The comparator's result for the same cell and output, if it has one. */
  const twin = (r, backend = comparators[0]) =>
    backend === undefined ? undefined : byKey.get(twinId(r.cell_id, backend) + SEP + r.output);

  const specs = [...new Set(subject.map((r) => r.spec))].sort();
  const bySpec = new Map(specs.map((s) => [s, subject.filter((r) => r.spec === s)]));
  const runById = new Map(runs.map((r) => [r.run_id, r]));

  const byClass = new Map();
  for (const c of categories) {
    const k = c.cell_id + SEP + c.output;
    if (!byClass.has(k)) byClass.set(k, {});
    byClass.get(k)[c.input_class] = c;
  }

  /** The validations of one reference identity (for a gradient, one output). */
  const validationsOf = (refId, output = null) =>
    refId ? validations.filter((v) => v.ref_id === refId && (output === null || v.output === output)) : [];

  /** The recorded samples of one reference in one validation run -- a run
   * covers every reference, so the identity and output select this one's. */
  const samplesOf = async (validationId, refId, output) => {
    if (!listed("validation_samples")) return [];
    if (!whole.has("validation_samples")) whole.set("validation_samples", readWhole("validation_samples"));
    return (await whole.get("validation_samples"))
      .filter((s) => s.validation_id === validationId && s.ref_id === refId && s.output === output);
  };

  return {
    source,
    manifest,
    summary: subject,
    validationsOf,
    samplesOf,
    has: listed,
    runs,
    runById,
    specs,
    primary,
    comparators,
    twin,
    twinId,
    bySpec: (s) => bySpec.get(s) ?? [],
    /** A result's figures by input class, or null for an older artifact. */
    classes: (r) => byClass.get(r.cell_id + SEP + r.output) ?? null,
    hasClasses: byClass.size > 0,
    result: (cellId, output) => byKey.get(cellId + SEP + output),
    cellRows,
    tables: { EAGER, WHOLE, BY_CELL },
  };
}

