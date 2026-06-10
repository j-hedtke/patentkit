# patentkit Drafting — Word add-in

A minimal Office.js task pane that drives patentkit's drafting skills from
Microsoft Word: **Draft Claims**, **Check Antecedent Basis**, and **Insert
Spec Section**. The pane talks to a local stdlib HTTP server (`serve.py`)
which calls `patentkit.analysis.drafting`.

## Architecture

```
Word task pane (taskpane.html/js, served over https://localhost:3000)
        |  POST JSON
        v
serve.py (http://localhost:8756)  ->  patentkit.analysis.drafting  ->  LLM
```

## Run the backend

```bash
pip install patentkit[anthropic]        # or patentkit[openai]
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY
python serve.py                         # listens on http://localhost:8756
```

## Serve the task pane

Office add-ins must load over HTTPS. Any static HTTPS server on port 3000
works; the easiest is the Office dev tooling:

```bash
npx office-addin-https-reverse-proxy --url http://localhost:3001  # or:
npx http-server . -p 3000 --ssl --cert localhost.crt --key localhost.key
```

Use `npx office-addin-dev-certs install` to generate and trust the
`localhost` certificates first.

## Sideload the manifest

**Word on Windows:** share a folder containing `manifest.xml`, add it as a
trusted catalog under *File > Options > Trust Center > Trusted Add-in
Catalogs*, restart Word, then *Insert > My Add-ins > Shared Folder >
patentkit Drafting*.

**Word on Mac:** copy `manifest.xml` to
`~/Library/Containers/com.microsoft.Word/Data/Documents/wef/` (create the
folder if needed), restart Word, then *Insert > Add-ins > My Add-ins >
patentkit Drafting*.

**Word on the web:** *Insert > Add-ins > Upload My Add-in* and pick
`manifest.xml`.

## Use it

1. Open the task pane (ribbon button "patentkit Drafting").
2. **Draft Claims** — paste an invention disclosure, choose counts, click;
   numbered claims are inserted at the cursor.
3. **Check Antecedent Basis** — select claims text in the document, click;
   a pure-python heuristic report is inserted after the selection (no LLM
   call, works offline).
4. **Insert Spec Section** — pick a section, click; the drafted section is
   inserted at the cursor.
