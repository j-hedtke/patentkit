/* patentkit Drafting task pane.
 *
 * Each button POSTs JSON to the local helper server (serve.py, port 8756)
 * and inserts the returned text at the current Word selection.
 */

const API_BASE = "http://localhost:8756";

function setStatus(message, isError) {
  const el = document.getElementById("status");
  el.textContent = message;
  el.className = isError ? "error" : "";
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Server error ${response.status}: ${detail}`);
  }
  const data = await response.json();
  return data.text || "";
}

function insertAtSelection(text) {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.insertText(text, Word.InsertLocation.replace);
    await context.sync();
  });
}

function getSelectionText() {
  return Word.run(async (context) => {
    const selection = context.document.getSelection();
    selection.load("text");
    await context.sync();
    return selection.text;
  });
}

async function draftClaims() {
  try {
    setStatus("Drafting claims...");
    const text = await postJson("/api/draft-claims", {
      disclosure: document.getElementById("disclosure").value,
      n_independent: parseInt(document.getElementById("n-independent").value, 10) || 1,
      n_dependent: parseInt(document.getElementById("n-dependent").value, 10) || 5,
    });
    await insertAtSelection(text);
    setStatus("Claims inserted.");
  } catch (err) {
    setStatus(String(err), true);
  }
}

async function checkBasis() {
  try {
    setStatus("Checking antecedent basis...");
    const claimsText = await getSelectionText();
    if (!claimsText || !claimsText.trim()) {
      setStatus("Select the claims text in the document first.", true);
      return;
    }
    const text = await postJson("/api/check-basis", { claims_text: claimsText });
    await Word.run(async (context) => {
      const selection = context.document.getSelection();
      selection.insertText("\n" + text + "\n", Word.InsertLocation.after);
      await context.sync();
    });
    setStatus("Antecedent basis check inserted after selection.");
  } catch (err) {
    setStatus(String(err), true);
  }
}

async function draftSection() {
  try {
    setStatus("Drafting section...");
    const text = await postJson("/api/draft-section", {
      disclosure: document.getElementById("disclosure").value,
      section: document.getElementById("section").value,
    });
    await insertAtSelection(text);
    setStatus("Section inserted.");
  } catch (err) {
    setStatus(String(err), true);
  }
}

Office.onReady((info) => {
  if (info.host !== Office.HostType.Word) {
    setStatus("This add-in only works in Word.", true);
    return;
  }
  document.getElementById("btn-draft-claims").addEventListener("click", draftClaims);
  document.getElementById("btn-check-basis").addEventListener("click", checkBasis);
  document.getElementById("btn-draft-section").addEventListener("click", draftSection);
  setStatus("Ready.");
});
