"""Offline pipeline: dump -> manifest -> parse -> chunk -> tag -> embed -> Chroma (SPEC §4).

Spoiler tagging (M4) isn't implemented yet: every chunk gets the default level 5.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk, chunk_page, count_tokens, validate_chunks  # noqa: E402
from src.config import ROOT, load_config, resolve_path  # noqa: E402
from src.index import open_client, rebuild_collection, upsert_chunks  # noqa: E402
from src.ingest.dump import load_wiki  # noqa: E402
from src.ingest.fetch import download_dump  # noqa: E402
from src.ingest.parse import Renderer, parse_page, to_json  # noqa: E402
from src.ingest.select import select_pages  # noqa: E402
from src.providers import get_embedder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--rebuild-manifest", action="store_true",
                   help="regenerate data/manifest.json (overwrites hand edits)")
    p.add_argument("--reindex", action="store_true",
                   help="skip parsing; re-embed data/chunks.jsonl into the configured embedder's collection")
    p.add_argument("--no-embed", action="store_true", help="stop after writing data/chunks.jsonl")
    args = p.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    chunks_path = resolve_path(cfg, "chunks")
    if args.reindex:
        chunks = [Chunk(**json.loads(line)) for line in open(chunks_path)]
        print(f"chunks: loaded {len(chunks)} from {cfg['paths']['chunks']}")
    else:
        chunks = parse_and_chunk(cfg, args.rebuild_manifest)
        with open(chunks_path, "w") as out:
            for c in chunks:
                out.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    validate_chunks(chunks)

    if args.no_embed:
        return
    embedder = get_embedder(cfg)
    col = rebuild_collection(open_client(resolve_path(cfg, "chroma")), embedder)
    upsert_chunks(col, embedder, chunks, batch=cfg["embeddings"]["batch_size"])
    print(f"index: {col.count()} chunks in {col.name} "
          f"({embedder.usage.embedding_tokens:,} tokens, {embedder.usage.requests} requests)")


def parse_and_chunk(cfg: dict, rebuild_manifest: bool) -> list[Chunk]:
    xml = download_dump(cfg["dump"]["url"], ROOT / cfg["dump"]["raw_dir"], cfg["wiki"]["user_agent"])
    print(f"dump: {xml.relative_to(ROOT)}")
    wiki = load_wiki(xml)
    print(f"wiki: {len(wiki.articles)} articles, {len(wiki.redirects)} redirects")

    manifest_path = resolve_path(cfg, "manifest")
    if rebuild_manifest or not manifest_path.exists():
        pages = select_pages(wiki, cfg["selection"])
        manifest = {"source": cfg["dump"]["url"], "pages": pages}
        manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        print(f"manifest: wrote {len(pages)} pages")
    else:
        manifest = json.loads(manifest_path.read_text())
        print(f"manifest: using existing {len(manifest['pages'])} pages (--rebuild-manifest to regenerate)")

    renderer, aliases = Renderer(), wiki.aliases()
    missing, n_sections, chunks = [], 0, []
    with open(resolve_path(cfg, "pages"), "w") as out:
        for entry in manifest["pages"]:
            page = wiki.articles.get(entry["title"])
            if page is None:
                missing.append(entry["title"])
                continue
            parsed = parse_page(page, wiki, cfg["wiki"]["base_url"], aliases.get(page.title, []), renderer)
            n_sections += len(parsed.sections)
            page_json = to_json(parsed)
            out.write(json.dumps(page_json, ensure_ascii=False) + "\n")
            chunks += chunk_page(page_json, cfg["chunking"])
    report = {"pages": len(manifest["pages"]) - len(missing), "sections": n_sections,
              "missing_titles": missing, "dropped_templates": dict(renderer.dropped.most_common())}
    report_path = resolve_path(cfg, "parse_report")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"parse: {report['pages']} pages, {n_sections} sections -> {cfg['paths']['pages']}"
          f" (report: {cfg['paths']['parse_report']})")
    if missing:
        print(f"warning: {len(missing)} manifest titles not in dump: {missing[:5]}")

    sizes = sorted(count_tokens(c.text) for c in chunks)
    print(f"chunk: {len(chunks)} chunks, tokens p50/p90/max {sizes[len(sizes) // 2]}/"
          f"{sizes[int(len(sizes) * .9)]}/{sizes[-1]}, total {sum(sizes):,} -> {cfg['paths']['chunks']}")
    return chunks


if __name__ == "__main__":
    main()
