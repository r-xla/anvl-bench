// The only third-party dependency, pinned here so the version lives in one
// place. hyparquet is ~10 kB of pure JS with no WASM: it reads Parquet over
// HTTP range requests, which is what lets this site open one cell out of a
// 46 MB file. DuckDB-WASM was rejected as ~30 MB, larger than the data.
export {
  parquetReadObjects,
  parquetMetadataAsync,
  asyncBufferFromUrl,
} from "https://cdn.jsdelivr.net/npm/hyparquet@1.17.1/+esm";
