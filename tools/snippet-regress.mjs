// Check that a change to js/snippet.js leaves the existing snippets alone.
//
// Generates every reproduction snippet the site can make for an export's
// results, once with js/ as committed at a git ref and once with js/ as it is
// in the working tree, and compares the two as text. For each result: its two
// worst recorded inputs (by relative error), plus fixed probes for the special
// values, so the code paths for ±0, NaN-producing and infinite inputs run too.
//
//   node tools/snippet-regress.mjs <export dir or .zip> [--base <ref>] [--out <dir>]
//
// --base defaults to HEAD. --out writes both sides in full (base.txt, new.txt)
// for diff(1). Exits 1 if any snippet that the base made has changed or gone;
// snippets only the working tree makes (a new spec) are counted, not failed.
//
// The export's own specs are the only ones exercised: a spec the export has no
// results for (one not yet swept) produces nothing here, so check its snippets
// by running them. The Parquet reader is the site's own, at the version pinned
// in js/hyparquet.js, fetched from the CDN.

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
};
const base = opt("--base", "HEAD");
const outDir = opt("--out", null);
if (args.length !== 1) {
  console.error("usage: node tools/snippet-regress.mjs <export dir or .zip> [--base <ref>] [--out <dir>]");
  process.exit(2);
}

const tmp = mkdtempSync(join(tmpdir(), "snippet-regress-"));
process.on("exit", () => rmSync(tmp, { recursive: true, force: true }));

// --- the export -----------------------------------------------------------

let exportDir = resolve(args[0]);
if (!statSync(exportDir).isDirectory()) {
  const dir = join(tmp, "export");
  mkdirSync(dir);
  execFileSync("unzip", ["-q", exportDir, "summary.parquet", "detail.parquet", "runs.parquet", "-d", dir]);
  exportDir = dir;
}

// js/hyparquet.js imports from a CDN URL, which Node cannot import directly
const hpUrl = /from\s+"(https:[^"]+)"/.exec(readFileSync(join(ROOT, "js/hyparquet.js"), "utf8"))[1];
const hpSrc = await (await fetch(hpUrl)).text();
const { parquetReadObjects } = await import(`data:text/javascript;base64,${Buffer.from(hpSrc).toString("base64")}`);

const table = (name, columns) => {
  const b = readFileSync(join(exportDir, `${name}.parquet`));
  return parquetReadObjects({ file: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), columns });
};
const results = await table("summary", ["cell_id", "spec", "backend", "output", "ref_params", "run_id"]);
const detail = await table("detail", ["cell_id", "output", "bits", "rel_err"]);
const runs = new Map((await table("runs")).map((r) => [r.run_id, r]));

// worst first; NaN and missing errors last, as R's order(-rel_err) puts them
const errKey = (e) => (e === null || e === undefined || Number.isNaN(e) ? -Infinity : e);
const worst = new Map();
for (const d of detail) {
  const k = `${d.cell_id}|${d.output}`;
  if (!worst.has(k)) worst.set(k, []);
  worst.get(k).push(d);
}
for (const [k, rows] of worst) {
  worst.set(k, rows.sort((a, b) => errKey(b.rel_err) - errKey(a.rel_err)).slice(0, 2).map((d) => d.bits));
}

const fromBits = (b) => {
  const h = b.replace(/^0x/i, "");
  const dv = new DataView(new ArrayBuffer(8));
  if (h.length <= 8) {
    dv.setUint32(0, parseInt(h, 16));
    return dv.getFloat32(0);
  }
  dv.setBigUint64(0, BigInt(`0x${h}`));
  return dv.getFloat64(0);
};
const PROBES = [0, -0, 0.5, -1, 3, 1e-300, Infinity];

// --- both sides -----------------------------------------------------------

/** Every snippet `js/snippet.js` under `jsDir` makes, keyed by result and input. */
async function generate(jsDir) {
  const { snippet } = await import(pathToFileURL(join(jsDir, "snippet.js")).href);
  const out = new Map();
  const sorted = [...results].sort((a, b) => `${a.cell_id}|${a.output}`.localeCompare(`${b.cell_id}|${b.output}`));
  for (const r of sorted) {
    const inputs = [
      ...(worst.get(`${r.cell_id}|${r.output}`) ?? []).map((b) => [fromBits(b), b]),
      ...PROBES.map((x) => [x, null]),
    ];
    for (const [x, bits] of inputs) {
      const s = snippet({
        result: r, run: runs.get(r.run_id), x, bits,
        recorded: [["value", 1.5], ["rel err", 1e-3]], where: "worst",
      });
      if (s) out.set(`${r.cell_id} ${r.output} ${bits ?? (Object.is(x, -0) ? "-0" : x)}`, `${s.lang}\n${s.text}`);
    }
  }
  return out;
}

const baseJs = join(tmp, "base");
mkdirSync(baseJs);
const tar = execFileSync("git", ["-C", ROOT, "archive", base, "js"], { maxBuffer: 1 << 28 });
execFileSync("tar", ["-x", "-C", baseJs], { input: tar });
const before = await generate(join(baseJs, "js"));
const after = await generate(join(ROOT, "js"));

// --- compare --------------------------------------------------------------

const specOf = (k) => k.split("/")[0];
const changed = [...before.keys()].filter((k) => after.has(k) && after.get(k) !== before.get(k));
const gone = [...before.keys()].filter((k) => !after.has(k));
const added = [...after.keys()].filter((k) => !before.has(k));

if (outDir) {
  mkdirSync(outDir, { recursive: true });
  const dump = (m) => [...m].map(([k, t]) => `==== ${k}\n${t}`).join("");
  writeFileSync(join(outDir, "base.txt"), dump(before));
  writeFileSync(join(outDir, "new.txt"), dump(after));
}

const bySpec = (keys) => {
  const n = new Map();
  for (const k of keys) n.set(specOf(k), (n.get(specOf(k)) ?? 0) + 1);
  return [...n].map(([s, c]) => `${s} ${c}`).join(", ");
};
console.log(`${results.length} results; base ${base}: ${before.size} snippets, working tree: ${after.size}`);
console.log(`unchanged: ${before.size - changed.length - gone.length}`);
if (added.length) console.log(`new (no snippet at ${base}): ${added.length} (${bySpec(added)})`);
if (changed.length) console.log(`CHANGED: ${changed.length} (${bySpec(changed)}); first: ${changed[0]}`);
if (gone.length) console.log(`GONE: ${gone.length} (${bySpec(gone)}); first: ${gone[0]}`);
if (changed.length || gone.length) {
  if (!outDir) console.log("rerun with --out <dir> and diff base.txt new.txt to see the changes");
  process.exit(1);
}
