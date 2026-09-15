"""Command-line interface.

Three commands: ``ask`` (the pipeline), ``cache`` (inspect/clear the local
layer), and ``eval`` (the offline harness). Failures print a one-line message
and exit non-zero rather than dumping a traceback, since the common causes -
missing API key, missing extra, no network - are user-fixable.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from eris import __version__
from eris.cache import Cache, NullCache
from eris.config import load_settings
from eris.errors import ErisError
from eris.eval import EvalError, run_eval_file
from eris.llm import EchoCitationClient
from eris.models import AskResult
from eris.pipeline import Eris

_EXIT_ERROR = 1


def _configure_logging(verbose: bool, level: str | None) -> None:
    """Send Eris logs to stderr so stdout stays pipeable."""
    resolved = level or ("INFO" if verbose else "WARNING")
    logging.basicConfig(
        level=getattr(logging, resolved.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _fail(message: str) -> None:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(_EXIT_ERROR)


def _render_ask(result: AskResult, *, verbose: bool, show_scores: bool) -> None:
    click.echo(result.answer.text)

    if result.answer.citations:
        click.echo("\nSources:")
        for citation in result.answer.citations:
            click.echo(f"  [{citation.n}] {citation.title}\n      {citation.url}")
    elif result.chunks:
        click.echo("\nRetrieved (uncited):")
        for position, chunk in enumerate(result.chunks, start=1):
            click.echo(f"  [{position}] {chunk.title}\n      {chunk.url}")

    if result.answer.invalid_citations:
        click.secho(
            f"\nwarning: model cited non-existent sources: {list(result.answer.invalid_citations)}",
            fg="yellow",
            err=True,
        )

    if show_scores and result.chunks:
        click.echo("\nChunk scores:")
        for position, chunk in enumerate(result.chunks, start=1):
            click.echo(f"  [{position}] {chunk.score:.4f}  {chunk.source_id}")

    if verbose:
        timings = "  ".join(str(t) for t in result.timings)
        click.echo(
            f"\nstages: {timings}\n"
            f"total: {result.total_ms:.1f}ms  "
            f"search_results={result.n_search_results}  "
            f"documents={result.n_documents}  "
            f"chunks={len(result.chunks)}  "
            f"offline={result.from_cache}"
        )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="eris")
def cli() -> None:
    """Eris - an LLM augmented with real-time web search and local retrieval."""


@cli.command("ask")
@click.argument("question", required=True)
@click.option("--top-k", type=click.IntRange(1, 100), help="Context chunks to retrieve.")
@click.option("--max-results", type=click.IntRange(1, 50), help="Search results to request.")
@click.option(
    "--search-backend",
    type=click.Choice(["duckduckgo", "fake"]),
    help="Search backend to use.",
)
@click.option("--model", help="Override the Claude model id.")
@click.option(
    "--no-web",
    is_flag=True,
    help="Answer only from the local cache: no search, no fetch, no network.",
)
@click.option("--no-cache", is_flag=True, help="Bypass the cache for reads and writes.")
@click.option("--json", "as_json", is_flag=True, help="Emit the full result as JSON.")
@click.option("--scores", is_flag=True, help="Show retrieval scores per chunk.")
@click.option(
    "--strip-hallucinated",
    is_flag=True,
    help="Remove citation markers that do not resolve to a source.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show per-stage timings.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def ask_command(
    question: str,
    top_k: int | None,
    max_results: int | None,
    search_backend: str | None,
    model: str | None,
    no_web: bool,
    no_cache: bool,
    as_json: bool,
    scores: bool,
    strip_hallucinated: bool,
    verbose: bool,
    log_level: str | None,
) -> None:
    """Answer QUESTION using live web sources (or the cache with --no-web)."""
    _configure_logging(verbose, log_level)
    try:
        settings = load_settings(
            top_k=top_k,
            max_results=max_results,
            search_backend=search_backend,
            model=model,
            log_level=log_level,
        )
    except ErisError as exc:
        _fail(str(exc))
        return

    cache = NullCache() if no_cache else None
    try:
        with Eris(settings, cache=cache) as eris:
            result = eris.ask(
                question,
                top_k=top_k,
                max_results=max_results,
                use_web=not no_web,
                strip_hallucinated=strip_hallucinated,
            )
    except ErisError as exc:
        _fail(str(exc))
        return
    except ValueError as exc:
        _fail(str(exc))
        return

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render_ask(result, verbose=verbose, show_scores=scores)


@cli.command("cache")
@click.option("--stats", is_flag=True, help="Show cache statistics (default).")
@click.option("--clear", "do_clear", is_flag=True, help="Delete every cache entry.")
@click.option("--purge", is_flag=True, help="Delete only expired entries.")
@click.option("--json", "as_json", is_flag=True, help="Emit statistics as JSON.")
def cache_command(stats: bool, do_clear: bool, purge: bool, as_json: bool) -> None:
    """Inspect or maintain the local page/search cache."""
    try:
        settings = load_settings()
    except ErisError as exc:
        _fail(str(exc))
        return

    try:
        cache = Cache(settings.cache_path, ttl_s=max(settings.cache_ttl_s, 1))
        try:
            if do_clear:
                removed = cache.clear()
                click.echo(f"Cleared {removed} cache entr{'y' if removed == 1 else 'ies'}.")
            if purge:
                removed = cache.purge_expired()
                click.echo(f"Purged {removed} expired entr{'y' if removed == 1 else 'ies'}.")
            if stats or not (do_clear or purge):
                snapshot = cache.stats()
                if as_json:
                    click.echo(json.dumps(snapshot.to_dict(), indent=2))
                else:
                    click.echo(f"path:      {snapshot.path}")
                    click.echo(f"ttl:       {settings.cache_ttl_s}s")
                    click.echo(
                        f"pages:     {snapshot.fresh_pages} fresh, "
                        f"{snapshot.expired_pages} expired"
                    )
                    click.echo(
                        f"searches:  {snapshot.fresh_searches} fresh, "
                        f"{snapshot.expired_searches} expired"
                    )
                    click.echo(f"size:      {snapshot.size_mb:.2f} MB")
        finally:
            cache.close()
    except ErisError as exc:
        _fail(str(exc))


@cli.command("eval")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--top-k", type=click.IntRange(1, 100), help="Context chunks to retrieve.")
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON.")
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero unless hit-rate and citation validity are both 100%.",
)
def eval_command(path: Path, top_k: int | None, as_json: bool, strict: bool) -> None:
    """Run the offline eval harness over a JSONL file of cases."""
    _configure_logging(False, None)
    try:
        settings = load_settings(search_backend="fake", cache_ttl_s=0, top_k=top_k)
        report = run_eval_file(
            path, settings=settings, llm_client=EchoCitationClient(), top_k=top_k
        )
    except (EvalError, ErisError) as exc:
        _fail(str(exc))
        return

    click.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.to_table())
    if strict and (report.hit_rate < 1.0 or report.citation_validity < 1.0 or report.errors):
        raise SystemExit(_EXIT_ERROR)


def main() -> None:
    """Console-script entry point."""
    cli()


if __name__ == "__main__":
    main()
