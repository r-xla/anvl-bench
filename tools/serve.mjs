// Minimal static server for local development.
//
// It exists because Python's http.server does not implement Range requests,
// and the whole read strategy here depends on them: hyparquet fetches a few
// row groups out of a 46 MB Parquet file rather than the file.
//
// It also reproduces two GitHub Pages behaviours that together once broke the
// deployed site while everything passed locally:
//
//   * Pages gzips these responses, including application/octet-stream, so a
//     HEAD reports the COMPRESSED length. Anything that takes that as the file
//     size reads the Parquet footer from the wrong offset.
//   * A ranged request is nonetheless answered from the UNCOMPRESSED bytes,
//     and content-range states the real total.
//
// Serving uncompressed here would hide that asymmetry, so this does not.
//
//   node tools/serve.mjs [port]

import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { gzipSync } from "node:zlib";

const root = new URL("..", import.meta.url).pathname;
const port = Number(process.argv[2] ?? 8099);

// Compressing a 46 MB file on every request would make the server useless, so
// keep the result until the file changes.
const gzipCache = new Map();
function gzipped(path, stat) {
  const key = `${path}:${stat.mtimeMs}:${stat.size}`;
  let hit = gzipCache.get(path);
  if (!hit || hit.key !== key) {
    hit = { key, body: gzipSync(readFileSync(path)) };
    gzipCache.set(path, hit);
  }
  return hit.body;
}

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".parquet": "application/vnd.apache.parquet",
  ".svg": "image/svg+xml",
};

createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let rel = decodeURIComponent(url.pathname);
  if (rel.endsWith("/")) rel += "index.html";
  // Reject traversal before it reaches the filesystem.
  const path = join(root, normalize(rel).replace(/^(\.\.[/\\])+/, ""));

  let stat;
  try {
    stat = statSync(path);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    return res.end("404");
  }
  if (stat.isDirectory()) {
    res.writeHead(404, { "content-type": "text/plain" });
    return res.end("404");
  }

  const type = types[extname(path)] ?? "application/octet-stream";
  const base = {
    "content-type": type,
    "accept-ranges": "bytes",
    "cache-control": "no-cache",
  };
  const compressible = !/^(image|video|audio)\//.test(type);

  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
  if (range) {
    const [, rawStart, rawEnd] = range;
    const start = rawStart === "" ? stat.size - Number(rawEnd) : Number(rawStart);
    const end = rawStart === "" || rawEnd === "" ? stat.size - 1 : Number(rawEnd);
    if (!(start >= 0 && end < stat.size && start <= end)) {
      res.writeHead(416, { ...base, "content-range": `bytes */${stat.size}` });
      return res.end();
    }
    res.writeHead(206, {
      ...base,
      "content-range": `bytes ${start}-${end}/${stat.size}`,
      "content-length": end - start + 1,
    });
    return createReadStream(path, { start, end }).pipe(res);
  }

  // Unranged, and the client takes gzip: compress, exactly as Pages does. The
  // content-length that goes out is therefore the compressed size.
  const wantsGzip = /\bgzip\b/.test(req.headers["accept-encoding"] ?? "");
  if (compressible && wantsGzip) {
    const body = gzipped(path, stat);
    res.writeHead(200, {
      ...base,
      "content-encoding": "gzip",
      "vary": "Accept-Encoding",
      "content-length": body.length,
    });
    return res.end(req.method === "HEAD" ? undefined : body);
  }

  res.writeHead(200, { ...base, "content-length": stat.size });
  if (req.method === "HEAD") return res.end();
  createReadStream(path).pipe(res);
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}`));
