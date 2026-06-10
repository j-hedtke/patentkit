"""Parsing-level connector tests — no network."""

from __future__ import annotations

import datetime
import zipfile

import pytest

from patentkit.connectors.infra.sep import (
    load_etsi_declarations_csv,
    sep_patents_for_standard,
)
from patentkit.connectors.infra.uspto_bulk import (
    iter_patents_from_archive,
    parse_redbook_xml,
    weekly_archive_urls,
)
from patentkit.connectors.training.examiner_logs import parse_search_queries
from patentkit.connectors.training.ipr_datasets import extract_us_patent_numbers
from patentkit.models.patent import PatentNumber

## Redbook XML -> Patent ##################################################

REDBOOK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE us-patent-grant SYSTEM "us-patent-grant-v45-2014-04-03.dtd" [ ]>
<us-patent-grant lang="EN" dtd-version="v4.5">
<us-bibliographic-data-grant>
<publication-reference><document-id><country>US</country><doc-number>10123456</doc-number><kind>B2</kind><date>20181113</date></document-id></publication-reference>
<application-reference appl-type="utility"><document-id><country>US</country><doc-number>15543210</doc-number><date>20170105</date></document-id></application-reference>
<priority-claims><priority-claim sequence="1"><country>US</country><doc-number>62123456</doc-number><date>20160104</date></priority-claim></priority-claims>
<invention-title id="d2e53">Widget coupling system</invention-title>
<classifications-cpc><main-cpc><classification-cpc>
<section>H</section><class>04</class><subclass>L</subclass><main-group>65</main-group><subgroup>403</subgroup>
</classification-cpc></main-cpc></classifications-cpc>
<us-parties>
<inventors><inventor sequence="001"><addressbook><last-name>Smith</last-name><first-name>Jane</first-name></addressbook></inventor></inventors>
</us-parties>
<assignees><assignee><addressbook><orgname>Acme Corp</orgname></addressbook></assignee></assignees>
</us-bibliographic-data-grant>
<abstract id="abstract"><p>A widget coupling system with improved torque transfer.</p></abstract>
<description id="description"><heading>FIELD</heading><p>This disclosure relates to widgets.</p></description>
<claims id="claims">
<claim id="CLM-00001" num="00001"><claim-text>1. A widget comprising:<claim-text>a coupler; and</claim-text><claim-text>a shaft.</claim-text></claim-text></claim>
<claim id="CLM-00002" num="00002"><claim-text>2. The widget of <claim-ref idref="CLM-00001">claim 1</claim-ref>, wherein the shaft is hollow.</claim-text></claim>
</claims>
</us-patent-grant>
"""

DESIGN_XML = REDBOOK_XML.replace("10123456", "D0812345").replace(
    'kind>B2', 'kind>S1'
)


def test_parse_redbook_xml_to_canonical_patent():
    patent = parse_redbook_xml(REDBOOK_XML, raw_ref="test.zip:ipg181113.xml")
    assert patent is not None
    assert str(patent.patent_number) == "US10123456B2"
    assert patent.title == "Widget coupling system"
    assert "torque transfer" in (patent.abstract or "")
    assert "relates to widgets" in (patent.specification or "")
    assert patent.application_number == "15543210"
    assert patent.filing_date == datetime.date(2017, 1, 5)
    assert patent.publication_date == datetime.date(2018, 11, 13)
    assert patent.grant_date == datetime.date(2018, 11, 13)
    assert patent.priority_date == datetime.date(2016, 1, 4)
    assert [i.name for i in patent.inventors] == ["Smith, Jane"]
    assert [a.name for a in patent.assignees] == ["Acme Corp"]
    assert patent.cpc_codes == ["H04L65/403"]
    # claims and dependency detection
    assert len(patent.claims) == 2
    claim1, claim2 = patent.claims
    assert claim1.number == 1 and claim1.is_independent
    assert "coupler" in claim1.text and "shaft" in claim1.text
    assert claim2.number == 2 and claim2.depends_on == 1
    # provenance
    [source] = patent.sources
    assert source.source == "uspto_bulk"
    assert source.fidelity == 2
    assert source.raw_ref == "test.zip:ipg181113.xml"


def test_parse_redbook_skips_design_and_reissue():
    assert parse_redbook_xml(DESIGN_XML) is None
    reissue = REDBOOK_XML.replace("10123456", "RE046789")
    assert parse_redbook_xml(reissue) is None


def test_iter_patents_from_archive(tmp_path):
    # one concatenated multi-document file, as in real redbook archives
    blob = REDBOOK_XML + DESIGN_XML
    zip_path = tmp_path / "ipg181113.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ipg181113.xml", blob)
    patents = list(iter_patents_from_archive(zip_path))
    assert len(patents) == 1  # design patent skipped
    assert str(patents[0].patent_number) == "US10123456B2"


def test_weekly_archive_urls():
    urls = weekly_archive_urls(2023, kind="grant")
    assert 51 <= len(urls) <= 53
    assert all(url.endswith(".zip") and "/grant/" in url for url in urls)
    assert "ipg230103.zip" in urls[0]  # first Tuesday of 2023
    app_urls = weekly_archive_urls(2023, kind="application")
    assert "ipa230105.zip" in app_urls[0]  # first Thursday of 2023
    with pytest.raises(ValueError):
        weekly_archive_urls(2023, kind="design")


## Examiner query parsing #################################################

SRNT_TEXT = """\
EAST Search History

Ref # Hits Search Query DBs Default Operator Plurals Time Stamp
L1 423 ("widget" and (coupler or coupling)).clm. US-PGPUB; USPAT OR ON 2018/06/12 10:01
L2 57 L1 and shaft.ti. USPAT OR ON 2018/06/12 10:03
S3 12 ("hollow shaft" near3 widget$2).ab,ti. US-PGPUB
(torque adj transfer) same coupler
This is ordinary prose and should not match.
Examiner: John Doe
"""


def test_parse_search_queries_heuristics():
    queries = parse_search_queries(SRNT_TEXT)
    assert '("widget" and (coupler or coupling)).clm.' in queries
    assert "L1 and shaft.ti." in queries
    assert '("hollow shaft" near3 widget$2).ab,ti.' in queries
    assert "(torque adj transfer) same coupler" in queries
    # prose and headers excluded
    assert all("ordinary prose" not in q for q in queries)
    assert all("Examiner" not in q for q in queries)
    assert all("Time Stamp" not in q for q in queries)
    # trailing DB/timestamp columns stripped
    assert all("USPAT" not in q and "2018/06/12" not in q for q in queries)


def test_parse_search_queries_dedupes():
    text = "L1 5 widget.clm. USPAT\nL2 5 widget.clm. USPAT\n"
    assert parse_search_queries(text) == ["widget.clm."]


## ETSI CSV loader ########################################################


def test_load_etsi_declarations_csv(tmp_path):
    csv_path = tmp_path / "etsi_export.csv"
    csv_path.write_text(
        "Declaring Company,Application Number,Specification Number,Project,Declaration Date\n"
        "Acme Corp,US10123456,ETSI EN 301 908-1,5G/NR,2020-03-01\n"
        "Globex,EP1234567,ETSI TS 138 211,5G/NR,2021-07-15\n"
        "Acme Corp,US10999888,ETSI EN 301 908-13,4G/LTE,2019-01-20\n"
    )
    declarations = load_etsi_declarations_csv(csv_path)
    assert len(declarations) == 3
    first = declarations[0]
    assert first.declaring_company == "Acme Corp"
    assert first.patent_number == "US10123456"
    assert first.standard == "ETSI EN 301 908-1"
    assert first.project == "5G/NR"
    assert first.declaration_date == "2020-03-01"
    assert first.raw["Specification Number"] == "ETSI EN 301 908-1"

    numbers = sep_patents_for_standard(declarations, "EN 301 908")
    assert numbers == ["US10123456", "US10999888"]


def test_load_etsi_csv_alternate_headers(tmp_path):
    csv_path = tmp_path / "alt.csv"
    csv_path.write_text(
        "Company,Patent,Standard\n"
        "Initech,US7654321,EN 302 567\n"
    )
    [declaration] = load_etsi_declarations_csv(csv_path)
    assert declaration.declaring_company == "Initech"
    assert declaration.patent_number == "US7654321"
    assert declaration.standard == "EN 302 567"


## IPR reference extraction ###############################################


def test_extract_us_patent_numbers():
    text = (
        "Petitioner relies on U.S. Patent No. 7,654,321 (“Jones”) in view of "
        "US 8123456 (Smith). The '321 patent and 7,654,321 are the same. "
        "Claims 1-9 of U.S. Patent No. 10,111,222 are unpatentable."
    )
    assert extract_us_patent_numbers(text) == ["US7654321", "US8123456", "US10111222"]


def test_extract_us_patent_numbers_ignores_noise():
    assert extract_us_patent_numbers("filed on 12/31/2020, page 1,234 of record") == []


## PatentNumber round-trips ###############################################


def test_patent_number_round_trips():
    pn = PatentNumber.parse("US 10,123,456 B2")
    assert (pn.country_code, pn.number, pn.kind_code) == ("US", "10123456", "B2")
    assert str(pn) == "US10123456B2"
    # str round-trip
    assert PatentNumber.parse(str(pn)) == pn

    ep = PatentNumber.parse("EP1234567A1")
    assert (ep.country_code, ep.number, ep.kind_code) == ("EP", "1234567", "A1")

    bare = PatentNumber.parse("10123456")
    assert bare.country_code == "US" and bare.kind_code is None
    assert bare.equivalent(pn)  # kind code ignored

    app = PatentNumber.parse("US 2020/0123456 A1")
    assert str(app) == "US20200123456A1"

    with pytest.raises(ValueError):
        PatentNumber.parse("not a patent")


## Google Patents HTML parsing (requires bs4; skipped otherwise) ##########

GOOGLE_HTML = """<html><body>
<span itemprop="title">Widget coupling system </span>
<div class="abstract">A widget coupling system.</div>
<dd itemprop="inventor">Jane Smith</dd>
<dd itemprop="assigneeCurrent">Acme Corp</dd>
<dd itemprop="legalStatusIfi">Active</dd>
<time itemprop="priorityDate">2016-01-04</time>
<time itemprop="filingDate">2017-01-05</time>
<time itemprop="publicationDate">2018-11-13</time>
<dd itemprop="events"><time itemprop="date" datetime="2018-11-13">2018-11-13</time>
<span itemprop="title">Application granted</span><span itemprop="type">granted</span></dd>
<dd itemprop="applicationNumber">US15/543,210</dd>
<section itemprop="description"><heading>FIELD</heading>
<div class="description-line">This disclosure relates to widgets.</div></section>
<section itemprop="claims"><div class="claims">
<div class="claim" num="1"><div class="claim-text">1. A widget comprising: a coupler and a shaft.</div></div>
<div class="claim" num="2"><div class="claim-text">2. The widget of <claim-ref>claim 1</claim-ref>, wherein the shaft is hollow.</div></div>
</div></section>
<section><h2>Classifications</h2>
<ul itemprop="classifications"><li>
<span itemprop="Code">H04L65/403</span><span itemprop="Description">Arrangements</span>
<meta itemprop="Leaf" content="true"/></li></ul></section>
<table><tr itemprop="backwardReferences">
<span itemprop="publicationNumber">US7654321B2</span><span itemprop="examinerCited">*</span></tr>
<tr itemprop="backwardReferences"><span itemprop="publicationNumber">US6543210A</span></tr></table>
</body></html>"""


def test_google_patents_html_parsing():
    pytest.importorskip("bs4")
    from patentkit.connectors.inference.google_patents import GooglePatentsScraper

    patent = GooglePatentsScraper().parse_html("US10123456B2", GOOGLE_HTML)
    assert patent.title == "Widget coupling system"
    assert patent.abstract == "A widget coupling system."
    assert [i.name for i in patent.inventors] == ["Jane Smith"]
    assert [a.name for a in patent.assignees] == ["Acme Corp"]
    assert patent.status == "Active"
    assert patent.priority_date == datetime.date(2016, 1, 4)
    assert patent.filing_date == datetime.date(2017, 1, 5)
    assert patent.grant_date == datetime.date(2018, 11, 13)
    assert patent.application_number == "US15/543,210"
    assert "relates to widgets" in (patent.specification or "")
    assert patent.spec_sections and patent.spec_sections[0].heading == "FIELD"
    assert len(patent.claims) == 2
    assert patent.claims[0].is_independent
    assert patent.claims[1].depends_on == 1
    assert patent.cpc_codes == ["H04L65/403"]
    examiner, applicant = patent.citations
    assert str(examiner.patent_number) == "US7654321B2" and examiner.is_examiner
    assert not applicant.is_examiner
    assert patent.examiner_cited_numbers == {"US7654321B2"}
    [source] = patent.sources
    assert source.source == "google_patents" and source.fidelity == 3


def test_google_scraper_helpful_import_error_without_bs4():
    try:
        import bs4  # noqa: F401

        pytest.skip("bs4 installed; ImportError path not reachable")
    except ImportError:
        pass
    from patentkit.connectors.inference.google_patents import GooglePatentsScraper

    with pytest.raises(ImportError, match=r"patentkit\[scrape\]"):
        GooglePatentsScraper().parse_html("US10123456B2", "<html></html>")
