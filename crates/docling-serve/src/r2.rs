//! `POST /v1/convert/r2`: convert a document that sits in an object store
//! (Cloudflare R2, or any S3-compatible store) and upload the outputs there,
//! all through pre-signed URLs the caller minted — this server holds no
//! credentials and can touch nothing but the objects it was handed.
//!
//! The flow it serves: a user uploads a book to the bucket, the upload event
//! starts a worker, the worker (which knows the page count) mints one GET URL
//! for the book and one PUT URL per output, and calls this endpoint. The
//! answer is a 2xx once every output is uploaded; anything else means nothing
//! can be relied on and the worker retries.
//!
//! ```jsonc
//! POST /v1/convert/r2
//! { "file": "biology.pdf",                    // the source's name (logs, `source`)
//!   "source_url": "<GET  …/biology.pdf>",
//!   "markdown_url": "<PUT …/rag/biology.md>",   // the `llm` export
//!   "pages_per_chunk": 20,
//!   "page_count": 392,                          // optional: checked before any upload
//!   "chunk_urls": {                             // one per window, named by its pages
//!     "0001_0020": "<PUT …/chunks/0001_0020_chunk.json>",
//!     …
//!     "0381_0392": "<PUT …/chunks/0381_0392_chunk.json>" },
//!   "options": { "do_formula_enrichment": true } }  // docling-serve's option names
//! ```
//!
//! Each chunk is the `reader` export (docs/READER_FORMAT.md) cut to a window
//! of `pages_per_chunk` pages — the word boxes make a whole book's reader
//! JSON too large for a client to parse at once — with the window's 1-based
//! `firstPage` / `lastPage` added. Window names are the two page numbers
//! zero-padded to four digits (more when the book has 10,000 pages or more),
//! the last window ending at the last page. Every window's URL is checked
//! before anything is uploaded; URLs for windows the book does not have are
//! simply not used. A PDF's page count is read from its page tree before the
//! conversion, so a request without enough URLs is turned away (`422`, naming
//! the missing windows) before any GPU time is spent on it.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    check_outbound_url, compat, convert_document, fetch_url, redact_url, require_outbound,
    trim_heap, ApiError, AppState, ConvertOptions,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct R2Request {
    #[serde(default)]
    file: Option<String>,
    source_url: String,
    markdown_url: String,
    pages_per_chunk: usize,
    #[serde(default)]
    page_count: Option<usize>,
    chunk_urls: BTreeMap<String, String>,
    #[serde(default)]
    options: Map<String, Value>,
}

/// The window names of a `pages`-page book in `per`-page windows:
/// `(name, first, last)`, 1-based and inclusive.
pub(crate) fn windows(pages: usize, per: usize) -> Vec<(String, usize, usize)> {
    let width = pages.to_string().len().max(4);
    (0..pages.div_ceil(per.max(1)))
        .map(|k| {
            let first = k * per + 1;
            let last = ((k + 1) * per).min(pages);
            (format!("{first:0width$}_{last:0width$}"), first, last)
        })
        .collect()
}

/// The `reader` export cut to pages `first..=last` (1-based): the same
/// shape, with `firstPageNumber` the window's first page (0-based, as the
/// export counts) and the window's bounds spelled out.
pub(crate) fn window_json(reader: &Value, first: usize, last: usize) -> Value {
    let offset = reader["firstPageNumber"].as_u64().unwrap_or(0) as usize;
    let slice = |key: &str| -> Value {
        let all = reader[key].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let from = (first - 1).saturating_sub(offset).min(all.len());
        let to = last.saturating_sub(offset).min(all.len());
        Value::Array(all[from..to].to_vec())
    };
    let mut out = Map::new();
    if let Some(schema) = reader.get("schema") {
        out.insert("schema".into(), schema.clone());
    }
    if let Some(source) = reader.get("source") {
        out.insert("source".into(), source.clone());
    }
    out.insert("firstPage".into(), json!(first));
    out.insert("lastPage".into(), json!(last));
    out.insert("firstPageNumber".into(), json!(first - 1));
    out.insert("pageSizes".into(), slice("pageSizes"));
    out.insert("pages".into(), slice("pages"));
    Value::Object(out)
}

/// `POST /v1/convert/r2`.
pub(crate) async fn convert_r2(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let req: R2Request =
        serde_json::from_slice(&body).map_err(|e| ApiError::Bad(format!("bad JSON body: {e}")))?;
    require_outbound(&state, "object-store inputs and uploads")?;
    if req.pages_per_chunk == 0 {
        return Err(ApiError::Bad("pages_per_chunk must be at least 1".into()));
    }
    check_outbound_url(&req.source_url)?;
    check_outbound_url(&req.markdown_url)?;
    for url in req.chunk_urls.values() {
        check_outbound_url(url)?;
    }
    let (options, _) = compat::parse_fields(&compat::json_fields(&req.options))?;
    if let Some(msg) = state.overloaded() {
        return Err(ApiError::Overloaded(msg));
    }
    let permit = state
        .permits
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| ApiError::Internal(format!("semaphore: {e}")))?;
    let st = state.clone();
    let answer = tokio::task::spawn_blocking(move || {
        let out = run(&st, req, &options);
        trim_heap();
        out
    })
    .await
    .map_err(|e| ApiError::Internal(format!("convert task: {e}")))?;
    drop(permit);
    answer.map(|v| Json(v).into_response())
}

/// Whether the request can hold a `pages`-page book: the page count it
/// states, and a URL for every window. Checked before the conversion when
/// the source's page count is known up front (a PDF's, from its page tree,
/// in milliseconds) — a scanned book's GPU time is not spent on a request
/// that cannot be answered — and again on the converted document.
fn check_windows(req: &R2Request, pages: usize) -> Result<Vec<(String, usize, usize)>, ApiError> {
    if let Some(expected) = req.page_count {
        if expected != pages {
            return Err(ApiError::Unsupported(format!(
                "the document has {pages} pages, the request says {expected}"
            )));
        }
    }
    let wins = windows(pages, req.pages_per_chunk);
    let missing: Vec<&str> = wins
        .iter()
        .filter(|(name, ..)| !req.chunk_urls.contains_key(name))
        .map(|(name, ..)| name.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(ApiError::Unsupported(format!(
            "{pages} pages in windows of {} need chunk URLs for {} window(s) the request \
             does not have: {}",
            req.pages_per_chunk,
            missing.len(),
            missing.join(", ")
        )));
    }
    Ok(wins)
}

fn run(state: &AppState, req: R2Request, options: &ConvertOptions) -> Result<Value, ApiError> {
    let source = fetch_url(&req.source_url, req.file.as_deref(), &[]).map_err(|e| {
        ApiError::Upstream(format!(
            "reading the source: {}",
            crate::api_error_message(e)
        ))
    })?;
    if source.format == docling::InputFormat::Pdf {
        // An unreadable page tree is left to the conversion to report.
        if let Ok(pages) =
            docling::pdf_page_count(&source.bytes, options.convert.password.as_deref())
        {
            check_windows(&req, pages)?;
        }
    }
    let converted = convert_document(state, source, options)?;
    let document = converted.document;
    let reader = document.export_to_reader_value();
    let first = reader["firstPageNumber"].as_u64().unwrap_or(0) as usize;
    let pages = first + reader["pages"].as_array().map_or(0, Vec::len);
    // Every window must have its URL before anything is written: a partial
    // upload would look like a finished book with pages missing.
    let wins = check_windows(&req, pages)?;
    let agent = docling::fetch::guarded_agent(
        ureq::Agent::config_builder()
            .max_redirects(0)
            .timeout_connect(Some(std::time::Duration::from_secs(10)))
            .timeout_global(Some(std::time::Duration::from_secs(300)))
            .build(),
        false,
    );
    let put = |url: &str, content_type: &str, body: &[u8]| -> Result<(), ApiError> {
        agent
            .put(url)
            .header("content-type", content_type)
            .send(body)
            .map(|_| ())
            .map_err(|e| ApiError::Upstream(format!("uploading to {}: {e}", redact_url(url))))
    };
    let mut uploaded = Vec::with_capacity(wins.len());
    for (name, first, last) in &wins {
        let body = serde_json::to_vec(&window_json(&reader, *first, *last))
            .expect("the reader export is always serializable");
        put(&req.chunk_urls[name], "application/json", &body)?;
        uploaded.push(name.clone());
    }
    drop(reader);
    put(
        &req.markdown_url,
        "text/markdown; charset=utf-8",
        document.export_to_llm_markdown().as_bytes(),
    )?;
    Ok(json!({
        "status": if converted.errors.is_empty() { "success" } else { "partial_success" },
        "pages": pages,
        "chunks": uploaded,
        "markdown": true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_named_by_their_pages() {
        let w = windows(392, 20);
        assert_eq!(w.len(), 20);
        assert_eq!(w[0], ("0001_0020".to_string(), 1, 20));
        assert_eq!(w[19], ("0381_0392".to_string(), 381, 392));
        assert_eq!(windows(20, 20), vec![("0001_0020".to_string(), 1, 20)]);
        assert_eq!(windows(12345, 10000)[1].0, "10001_12345");
        assert!(windows(0, 20).is_empty());
    }

    #[test]
    fn a_window_is_the_reader_export_cut_to_its_pages() {
        let reader = json!({
            "schema": "docling-rs/reader/1",
            "source": "b.pdf",
            "firstPageNumber": 0,
            "pageSizes": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            "pages": [["p1"], ["p2"], ["p3"], ["p4"], ["p5"]],
        });
        let w = window_json(&reader, 3, 5);
        assert_eq!(w["firstPage"], 3);
        assert_eq!(w["lastPage"], 5);
        assert_eq!(w["firstPageNumber"], 2);
        assert_eq!(w["pages"], json!([["p3"], ["p4"], ["p5"]]));
        assert_eq!(w["pageSizes"], json!([[3, 3], [4, 4], [5, 5]]));
        assert_eq!(w["source"], "b.pdf");
    }
}
