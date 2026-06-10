"""patentkit quickstart — a fully offline invalidity search demo.

Builds five tiny synthetic patents, indexes them in the in-memory BM25
store, and runs the invalidity search agent in keys-free degraded mode
(no LLM, no vector store, no network): a single keyword pass with the
default exclusions. With an LLM configured the same agent runs a pure
agentic search — the model generates and refines its own queries via tool
use and returns a full reasoning trace. Run it with:

    python examples/quickstart.py
"""

from datetime import date

from patentkit.agents import InvaliditySearchAgent
from patentkit.models import Citation, Claim, Patent, PatentNumber
from patentkit.search.bm25 import BM25Store


def make_patent(number: str, title: str, claim: str, spec: str,
                priority: date, **kwargs) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(number),
        title=title,
        abstract=spec[:160],
        claims=[Claim(number=1, text=claim)],
        specification=spec,
        priority_date=priority,
        **kwargs,
    )


# --- 1. A tiny synthetic corpus -------------------------------------------
corpus = [
    make_patent(
        "US7000001B1", "Wireless soil moisture sensor network",
        "1. A soil moisture sensor node transmitting moisture readings over a "
        "wireless mesh network to an irrigation controller.",
        "Each sensor node measures soil moisture with a capacitive probe and "
        "relays readings over a low-power wireless mesh network. The "
        "irrigation controller aggregates moisture readings to schedule "
        "watering of individual zones.",
        date(2001, 3, 1),
    ),
    make_patent(
        "US7000002B1", "Capacitive soil probe calibration",
        "1. A method of calibrating a capacitive soil moisture probe using a "
        "reference resistor network.",
        "Capacitive probes drift with temperature. The disclosed calibration "
        "method uses a reference resistor network to normalize soil moisture "
        "measurements across sensor nodes before wireless transmission.",
        date(2002, 6, 15),
    ),
    make_patent(
        "US7000003B1", "Drip irrigation valve controller",
        "1. An irrigation controller actuating drip valves according to a "
        "schedule derived from weather forecast data.",
        "The controller receives weather forecast data and computes a "
        "watering schedule, actuating drip irrigation valves per zone. "
        "Moisture sensors may optionally refine the schedule.",
        date(2003, 1, 20),
    ),
    make_patent(
        "US7000004B1", "Greenhouse climate telemetry",
        "1. A greenhouse telemetry system reporting temperature and humidity "
        "over a radio link to a central logger.",
        "Temperature and humidity sensors in a greenhouse report readings "
        "over a radio link. The central logger charts climate trends; no "
        "irrigation control is performed.",
        date(2000, 9, 5),
    ),
    make_patent(
        # Filed AFTER the target's priority date -> must be filtered out
        # by the prior-art date cutoff.
        "US9000001B2", "Machine-learning irrigation scheduling",
        "1. Training a model on soil moisture sensor data to predict "
        "irrigation demand per zone.",
        "A machine learning model is trained on historical soil moisture "
        "sensor readings and weather data to predict irrigation demand.",
        date(2014, 5, 1),
    ),
]

# --- 2. The patent we want to invalidate ----------------------------------
target = make_patent(
    "US8123456B2", "Sensor-driven zone irrigation system",
    "1. An irrigation system comprising: wireless soil moisture sensor nodes; "
    "and a controller that schedules watering of individual zones based on "
    "the transmitted moisture readings.",
    "The system combines wireless soil moisture sensor nodes with a zone "
    "irrigation controller scheduling watering from transmitted readings.",
    date(2008, 4, 10),
    # Examiner already cited US7000003B1 -> excluded from results by default.
    citations=[Citation(patent_number=PatentNumber.parse("US7000003B1"), is_examiner=True)],
)

# --- 3. Index and search (keys-free degraded mode) -------------------------
store = BM25Store()
store.index(corpus)
print(f"Indexed {len(store)} patents.\n")

# Plug in providers here for the full agentic search (the LLM generates and
# refines its own queries via tool use; result.trace holds the reasoning):
#   llm=get_llm("high")                          # needs ANTHROPIC_API_KEY / OPENAI_API_KEY
#   vector_store=InMemoryVectorStore(VoyageEmbeddings())   # needs VOYAGE_API_KEY
#   file_wrapper=FileWrapperClient(...)          # recovers examiner citations
agent = InvaliditySearchAgent(keyword_store=store, vector_store=None, llm=None)

result = agent.search(target, claims=[1], final_k=5, progress=lambda msg: print(f"  [{msg}]"))

# --- 4. Report -------------------------------------------------------------
print(f"\nTarget: {result.target}  (claims {result.claims})")
print(f"Prior-art cutoff: {result.plan_or_params['before_date']}")
print("Excluded:", {reason: nums for reason, nums in result.excluded.items()})
print(f"\nTop {len(result.results)} references:")
for i, ref in enumerate(result.results, start=1):
    print(f"\n{i}. {ref['patent_number']} — {ref['title']}  (score {ref['score']:.3f})")
    for passage in ref["passages"][:2]:
        print(f"   [{passage['field']}] …{passage['text'][:120]}…")

print("\nNote: US9000001B2 (post-priority) and US7000003B1 (examiner-cited) were excluded.")
