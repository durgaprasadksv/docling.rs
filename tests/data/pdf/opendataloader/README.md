# OpenDataLoader's synthetic page fixtures

The single-page PDFs that [OpenDataLoader-PDF](https://github.com/opendataloader-project/opendataloader-pdf)'s
issue tests build with PDFBox and convert — a table of contents drawn column by
column, two bulleted sections, numbered steps beside another column, a wrapped
list item, a footer under the left column — each with the behaviour its test
asserts on the converted output. They are a corpus of small, hard, *fully
specified* layout cases for the reading-order and structure decisions a
read-aloud client depends on, independent of docling's own fixtures.

- `<issue-dir>/<name>.pdf` — the fixture, as the upstream test wrote it
  (`target/test-samples/` after `mvn test` in `java/opendataloader-pdf-core`).
  Synthetic: Helvetica / Times text placed by coordinates, no third-party
  content. Upstream commit `5a7976a` (2026-11-18), Apache-2.0 (Hancom Inc.).
- `<issue-dir>/<name>.odl.md` — OpenDataLoader's own Markdown for it, the
  oracle for a per-fixture word recall (the same measure the TalkingText
  client's golden test uses against the Java output).
- `cases.json` — one case per upstream test method whose assertions could be
  expressed as checks on our output: `order` (substrings in this order in the
  Markdown), `contains` / `not_contains`, `line_starts` (`[needle, prefix]`),
  `same_line`, and, on the `reader` export, `one_list` (items share one list),
  `separate_lists` (two items are in different lists),
  `in_list` / `not_in_list`, `list_items` (count). Each case records its
  `status` — `pass`, `xfail` (fails today, kept as a measured gap) or
  `unchecked` (the upstream assertion is on OpenDataLoader's own JSON shape and
  has no counterpart here; the fixture still counts for word recall) — and
  each fixture its measured `recall`. `crates/docling/tests/opendataloader.rs`
  re-runs every case and fails on a `pass` that fails, an `xfail` that passes
  (update the status), or a recall that drops more than five points.

Out of scope, not copied: upstream tests of its hybrid backend (mocked server
responses), file and resource handling, tagged-PDF structure trees and form
fields, the content sanitizer, two-namespace repair, and inline images in
Markdown.

Regenerate: clone upstream, `mvn -pl opendataloader-pdf-core -am -DskipTests
install && (cd opendataloader-pdf-core && mvn test -Dtest='Issue*Test')`, then
`scripts/conformance/opendataloader_cases.py` (the extractor + evaluator that
wrote `cases.json`) against `target/test-samples`.
