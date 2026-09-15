"""CLI behaviour via Click's runner, plus a real subprocess smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from eris.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_QA = REPO_ROOT / "examples" / "qa.jsonl"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestTopLevel:
    def test_help_lists_every_command(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("ask", "cache", "eval"):
            assert command in result.output

    def test_version_is_reported(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_short_help_flag_works(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["-h"]).exit_code == 0

    def test_unknown_command_fails(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["nonsense"]).exit_code != 0


class TestAskCommand:
    def test_help_documents_the_offline_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--no-web" in result.output
        assert "--top-k" in result.output

    def test_missing_question_fails(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["ask"]).exit_code != 0

    def test_missing_api_key_is_a_clear_error(self, runner: CliRunner) -> None:
        """No traceback: the cause is user-fixable, so say so in one line."""
        result = runner.invoke(cli, ["ask", "a question", "--search-backend", "fake"])
        assert result.exit_code == 1
        assert "ERIS_ANTHROPIC_API_KEY" in result.output
        assert "Traceback" not in result.output

    def test_invalid_top_k_is_rejected(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["ask", "q", "--top-k", "0"]).exit_code != 0

    def test_invalid_backend_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "q", "--search-backend", "bing"])
        assert result.exit_code != 0

    def test_offline_mode_with_empty_cache_refuses(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "sk-ant-test-not-used")
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["ask", "anything", "--no-web"])
        assert result.exit_code == 0
        assert "cannot answer" in result.output

    def test_offline_mode_answers_from_a_seeded_cache(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline mode must retrieve real chunks out of the cache.

        The LLM is stubbed at the pipeline's default-client seam, so this
        exercises the cache and retrieval stages without a credential.
        """
        from eris.cache import Cache
        from eris.llm import EchoCitationClient
        from eris.models import Document
        from eris.pipeline import Eris

        cache_dir = tmp_path / "cache"
        with Cache(cache_dir / "eris.sqlite3", ttl_s=600) as cache:
            cache.put_page(
                Document(
                    url="https://docs.example/limits",
                    title="Service limits",
                    text=(
                        "The maximum request payload size is 10 MiB per request. "
                        "Larger uploads must use the multipart endpoint. "
                    )
                    * 6,
                )
            )
        monkeypatch.setenv("ERIS_CACHE_DIR", str(cache_dir))
        monkeypatch.setattr(
            Eris, "_default_llm", staticmethod(lambda settings: EchoCitationClient())
        )

        result = runner.invoke(cli, ["ask", "maximum request payload size", "--no-web", "-v"])
        assert result.exit_code == 0, result.output
        assert "https://docs.example/limits" in result.output
        assert "Sources:" in result.output
        assert "cache=" in result.output

    def test_verbose_reports_stage_timings(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "sk-ant-test-not-used")
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["ask", "q", "--no-web", "--verbose"])
        assert "retrieve=" in result.output
        assert "total:" in result.output

    def test_json_output_is_valid_json(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_ANTHROPIC_API_KEY", "sk-ant-test-not-used")
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["ask", "q", "--no-web", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["question"] == "q"
        assert "timings_ms" in payload


class TestCacheCommand:
    def test_help_documents_the_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0
        assert "--stats" in result.output
        assert "--clear" in result.output

    def test_stats_is_the_default_action(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["cache"])
        assert result.exit_code == 0
        assert "pages:" in result.output
        assert "searches:" in result.output

    def test_stats_reports_seeded_entries(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from eris.cache import Cache
        from eris.models import Document

        cache_dir = tmp_path / "cache"
        with Cache(cache_dir / "eris.sqlite3", ttl_s=600) as cache:
            cache.put_page(Document(url="https://e.com/a", title="A", text="body"))
        monkeypatch.setenv("ERIS_CACHE_DIR", str(cache_dir))
        result = runner.invoke(cli, ["cache", "--stats"])
        assert "1 fresh" in result.output

    def test_stats_as_json(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["cache", "--json"])
        assert json.loads(result.output)["pages"] == 0

    def test_clear_empties_the_cache(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from eris.cache import Cache
        from eris.models import Document

        cache_dir = tmp_path / "cache"
        with Cache(cache_dir / "eris.sqlite3", ttl_s=600) as cache:
            cache.put_page(Document(url="https://e.com/a", title="A", text="body"))
        monkeypatch.setenv("ERIS_CACHE_DIR", str(cache_dir))

        result = runner.invoke(cli, ["cache", "--clear"])
        assert result.exit_code == 0
        assert "Cleared 1 cache entry" in result.output
        assert json.loads(runner.invoke(cli, ["cache", "--json"]).output)["pages"] == 0

    def test_purge_reports_a_count(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ERIS_CACHE_DIR", str(tmp_path / "cache"))
        result = runner.invoke(cli, ["cache", "--purge"])
        assert result.exit_code == 0
        assert "Purged 0 expired entries" in result.output


class TestEvalCommand:
    def test_help_documents_the_arguments(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--strict" in result.output

    def test_runs_the_shipped_fixtures(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", str(SHIPPED_QA)])
        assert result.exit_code == 0
        assert "hit_rate@" in result.output
        assert "PASS" in result.output

    def test_runs_without_an_api_key(self, runner: CliRunner) -> None:
        """The whole point of the harness: no credential, no network."""
        result = runner.invoke(cli, ["eval", str(SHIPPED_QA)])
        assert result.exit_code == 0
        assert "ERIS_ANTHROPIC_API_KEY" not in result.output

    def test_json_output_carries_metrics(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", str(SHIPPED_QA), "--json"])
        payload = json.loads(result.output)
        assert payload["hit_rate"] == 1.0
        assert payload["citation_validity"] == 1.0

    def test_top_k_override_is_honoured(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", str(SHIPPED_QA), "--top-k", "2", "--json"])
        assert json.loads(result.output)["top_k"] == 2

    def test_strict_passes_on_perfect_fixtures(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["eval", str(SHIPPED_QA), "--strict"]).exit_code == 0

    def test_missing_file_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["eval", str(tmp_path / "absent.jsonl")])
        assert result.exit_code != 0

    def test_malformed_file_reports_the_line(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        result = runner.invoke(cli, ["eval", str(path)])
        assert result.exit_code == 1
        assert "line 1" in result.output

    def test_strict_fails_when_retrieval_misses(self, runner: CliRunner, tmp_path: Path) -> None:
        case = {
            "question": "Where is the payload limit documented?",
            "expected_url": "https://e.com/unrelated",
            "sources": [
                {
                    "url": "https://e.com/limits",
                    "title": "Limits",
                    "text": "The maximum request payload size is 10 MiB per request. " * 8,
                },
                {
                    "url": "https://e.com/unrelated",
                    "title": "Gardening",
                    "text": "Marigolds deter nematodes from root vegetables. " * 8,
                },
            ],
        }
        path = tmp_path / "miss.jsonl"
        path.write_text(json.dumps(case), encoding="utf-8")
        result = runner.invoke(cli, ["eval", str(path), "--top-k", "1", "--strict"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestConsoleScript:
    """Real subprocess runs, which also prove the entry point is installed."""

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "eris", *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )

    def test_module_entry_point_reports_version(self) -> None:
        result = self._run("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_module_entry_point_shows_help(self) -> None:
        result = self._run("--help")
        assert result.returncode == 0
        assert "web search" in result.stdout

    def test_eval_runs_end_to_end_in_a_subprocess(self) -> None:
        result = self._run("eval", "examples/qa.jsonl")
        assert result.returncode == 0, result.stderr
        assert "hit_rate@6=100.0%" in result.stdout

    def test_console_script_is_installed(self) -> None:
        """The `eris` entry point declared in pyproject must be executable."""
        script = Path(sys.executable).parent / "eris"
        if not script.exists():
            pytest.skip("console script not present in this environment")
        result = subprocess.run(
            [str(script), "--version"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
