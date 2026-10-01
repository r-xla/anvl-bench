// Write data/index.json, the list of artifacts published alongside the page.
//
// Each artifact is a directory under data/ holding one export (manifest.json
// plus the Parquet tables). The index is what fills the artifact dropdown, so
// the label is built from the manifest rather than from the directory name.
//
//   node tools/build-data-index.mjs [dataDir]
//
// With a single artifact the index is optional -- the page falls back to
// data/ -- but writing it anyway keeps one code path.

import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const dataDir = process.argv[2] ?? "data";

const label = (m) => [
  m.platforms?.join("+"),
  m.devices?.length && !m.platforms?.some((p) => p.includes(m.devices[0])) ? m.devices.join("+") : null,
  m.anvl_version?.length ? `anvl ${m.anvl_version.join("/")}` : null,
  m.depths?.join("+"),
].filter(Boolean).join(" · ");

const artifacts = [];
for (const name of readdirSync(dataDir).sort()) {
  const dir = join(dataDir, name);
  if (!statSync(dir).isDirectory()) continue;
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  } catch {
    console.warn(`skipping ${dir}: no readable manifest.json`);
    continue;
  }
  artifacts.push({
    id: name,
    dir: `${dataDir}/${name}`,
    label: label(manifest) || name,
    schema_version: manifest.schema_version ?? null,
    exported_at: manifest.exported_at ?? null,
    n_results: manifest.n_results ?? null,
  });
}

if (!artifacts.length) {
  console.error(`no artifacts found under ${dataDir}/`);
  process.exit(1);
}

writeFileSync(join(dataDir, "index.json"), JSON.stringify({ artifacts }, null, 2) + "\n");
console.log(`indexed ${artifacts.length} artifact(s):`);
for (const a of artifacts) console.log(`  ${a.id}  ${a.label}`);
