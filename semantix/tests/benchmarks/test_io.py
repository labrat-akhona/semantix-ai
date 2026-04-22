from pathlib import Path

from benchmarks.common.io import Row, write_csv, write_summary_md


def _rows() -> list[Row]:
    return [
        Row(
            example_id="ex-1",
            experiment="agreement",
            judge="semantix",
            intent="polite",
            text="hello",
            score=0.9,
            latency_ms=15,
            cost_usd=0.0,
            paid_equivalent_usd=0.0,
            raw=None,
            error=None,
        ),
        Row(
            example_id="ex-1",
            experiment="agreement",
            judge="groq-llama-3.3-70b",
            intent="polite",
            text="hello",
            score=0.85,
            latency_ms=300,
            cost_usd=0.0,
            paid_equivalent_usd=0.0001,
            raw="0.85",
            error=None,
        ),
    ]


def test_write_csv_roundtrip(tmp_path: Path):
    path = tmp_path / "raw.csv"
    write_csv(_rows(), path)
    content = path.read_text()
    assert "example_id,experiment,judge" in content
    assert "ex-1,agreement,semantix" in content


def test_summary_md_includes_headline_table(tmp_path: Path):
    path = tmp_path / "summary.md"
    write_summary_md(_rows(), path, task_name="customer_support_qa")
    content = path.read_text()
    assert "# customer_support_qa" in content
    assert "| Judge |" in content
    assert "semantix" in content
    assert "groq-llama-3.3-70b" in content
