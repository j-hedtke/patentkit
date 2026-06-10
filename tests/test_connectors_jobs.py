"""Job/checkpoint-level connector tests — no network."""

from __future__ import annotations

import json

from patentkit.connectors.infra.google_patents_bulk import BulkScrapeJob
from patentkit.connectors.infra.progress import FileProgressTracker
from patentkit.connectors.infra.ptab import IprDocument, IprProceeding, PtabClient
from patentkit.connectors.inference.file_wrapper import FileWrapperDocument
from patentkit.connectors.training.examiner_logs import ExaminerLogBuilder
from patentkit.connectors.training.ipr_datasets import (
    build_ipr_eval_dataset,
    load_ipr_eval_dataset,
)
from patentkit.models.patent import Citation, Patent, PatentNumber, SourceRecord

## FileProgressTracker ####################################################


def test_progress_tracker_roundtrip_and_resume(tmp_path):
    path = tmp_path / "job.progress.json"
    tracker = FileProgressTracker(path)
    assert tracker.resume_after() is None

    tracker.record_success("US1")
    tracker.record_failure("US2")
    tracker.record_success("US3")
    tracker.save()
    assert path.exists()
    # no stray temp files left behind
    assert list(tmp_path.iterdir()) == [path]

    reloaded = FileProgressTracker(path)
    assert reloaded.processed == 2
    assert reloaded.failed == ["US2"]
    assert reloaded.resume_after() == "US3"


def test_progress_tracker_ignores_corrupt_checkpoint(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    tracker = FileProgressTracker(path)
    assert tracker.processed == 0 and tracker.resume_after() is None


## BulkScrapeJob ##########################################################


class FakeScraper:
    """Stands in for GooglePatentsScraper; no network."""

    def __init__(self, fail: set[str] | None = None):
        self.fail = fail or set()
        self.calls: list[str] = []

    def fetch(self, number: str) -> Patent:
        self.calls.append(number)
        if number in self.fail:
            raise RuntimeError(f"boom: {number}")
        return Patent(
            patent_number=PatentNumber.parse(number),
            title=f"Title {number}",
            sources=[SourceRecord(source="google_patents", fidelity=3)],
        )


def test_bulk_scrape_job_writes_jsonl_and_collects_failures(tmp_path):
    out = tmp_path / "patents.jsonl"
    tracker = FileProgressTracker(tmp_path / "ck.json")
    scraper = FakeScraper(fail={"US2222222"})

    job = BulkScrapeJob(
        ["US1111111", "US2222222", "US3333333"],
        out,
        tracker=tracker,
        batch_size=2,
        scraper=scraper,
    )
    stats = job.run()

    assert stats.requested == 3
    assert stats.scraped == 2
    assert stats.failed == 1
    assert stats.failed_ids == ["US2222222"]

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["patent_number"]["number"] == "1111111"
    assert first["title"] == "Title US1111111"
    assert first["sources"][0]["source"] == "google_patents"

    assert tracker.resume_after() == "US3333333"
    assert tracker.failed == ["US2222222"]


def test_bulk_scrape_job_resumes_from_checkpoint(tmp_path):
    out = tmp_path / "patents.jsonl"
    checkpoint = tmp_path / "ck.json"
    numbers = ["US1111111", "US2222222", "US3333333"]

    job = BulkScrapeJob(
        numbers, out, tracker=FileProgressTracker(checkpoint), scraper=FakeScraper()
    )
    job.run()

    # restart with one new number appended; only it should be fetched
    scraper = FakeScraper()
    resumed = BulkScrapeJob(
        numbers + ["US4444444"],
        out,
        tracker=FileProgressTracker(checkpoint),
        scraper=scraper,
    )
    stats = resumed.run()
    assert scraper.calls == ["US4444444"]
    assert stats.skipped == 3 and stats.scraped == 1
    assert len(out.read_text().strip().splitlines()) == 4


## ExaminerLogBuilder #####################################################

SRNT_TEXT = (
    'L1 12 ("widget" and coupler).clm. USPAT OR ON 2018/06/12\n'
    "L2 4 L1 and shaft.ti. USPAT\n"
)


class FakeFileWrapper:
    """Stands in for FileWrapperClient; no network, no key."""

    def app_number_for_patent(self, number: str) -> str | None:
        return "15543210" if number != "US0000000" else None

    def get_documents_by_codes(self, app_number, codes):
        assert "SRNT" in tuple(codes)
        return [
            FileWrapperDocument(
                code="SRNT",
                description="Search information",
                date="2018-06-12T00:00:00.000Z",
                pdf_urls=["https://api.uspto.gov/fake/srnt.pdf"],
            )
        ]

    def download_pdf(self, url: str) -> bytes:
        return b"%PDF-fake"

    def get_examiner_cited_art(self, app_number):
        return [Citation(patent_number=PatentNumber.parse("US7654321"), is_examiner=True)]


def test_examiner_log_builder_writes_records(tmp_path):
    out = tmp_path / "queries.jsonl"
    tracker = FileProgressTracker(tmp_path / "ck.json")
    builder = ExaminerLogBuilder(
        FakeFileWrapper(), pdf_text_extractor=lambda pdf: SRNT_TEXT
    )
    stats = builder.build_for_patents(
        ["US10123456", "US0000000"], out, tracker=tracker
    )
    assert stats.written == 1
    assert stats.failed == 1 and stats.failed_ids == ["US0000000"]

    [line] = out.read_text().strip().splitlines()
    record = json.loads(line)
    assert record["patent_number"] == "US10123456"
    assert record["application_number"] == "15543210"
    assert record["queries"] == [
        '("widget" and coupler).clm.',
        "L1 and shaft.ti.",
    ]
    assert record["references_cited"] == ["US7654321"]
    assert record["source_doc_date"].startswith("2018-06-12")
    assert tracker.resume_after() == "US10123456"


## IPR eval dataset build #################################################


class FakePtab(PtabClient):
    def __init__(self):
        super().__init__(min_interval_s=0.0)

    def iter_ipr_proceedings(self, filed_from=None, filed_to=None, status=None):
        yield IprProceeding(
            proceeding_number="IPR2020-00001",
            patent_number="10123456",
            status="FWD Entered",
            filing_date="2020-01-15",
            petitioner="Globex",
            patent_owner="Acme Corp",
        )
        yield IprProceeding(  # not final — should be skipped
            proceeding_number="IPR2020-00002",
            patent_number="10999888",
            status="Terminated-Settled",
        )

    def list_proceeding_documents(self, proceeding_number):
        assert proceeding_number == "IPR2020-00001"
        return [
            IprDocument(document_id="111", title="Petition", category="Paper"),
            IprDocument(
                document_id="222",
                title="Final Written Decision",
                category="final",
                type_name="final decision",
            ),
        ]

    def download_document(self, doc_id, dest=None):
        assert doc_id == "222"
        return b"%PDF-fake-final-decision"


def test_build_and_load_ipr_eval_dataset(tmp_path):
    out = tmp_path / "ipr_eval.jsonl"
    decision_text = (
        "Claims 1-9 of U.S. Patent No. 10,123,456 are unpatentable over "
        "U.S. Patent No. 7,654,321 (Jones) in view of US 8123456 (Smith)."
    )
    records = build_ipr_eval_dataset(
        FakePtab(),
        "2020-01-01",
        "2020-12-31",
        out,
        pdf_text_extractor=lambda pdf: decision_text,
    )
    assert len(records) == 1
    record = records[0]
    assert record.proceeding_number == "IPR2020-00001"
    assert record.challenged_patent == "US10123456"
    # the challenged patent itself is excluded from prior art
    assert record.prior_art_references == ["US7654321", "US8123456"]
    assert record.outcome == "FWD Entered"
    assert record.metadata["petitioner"] == "Globex"

    loaded = load_ipr_eval_dataset(out)
    assert loaded == records
