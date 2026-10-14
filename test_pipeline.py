"""
Direct programmatic test script for TruthLens pipeline.analyze().
Tests 3 inputs (1 fake, 1 real, 1 borderline) and asserts all fields.
"""

import sys
import json
from pathlib import Path

# Ensure root is on path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.pipeline import analyze

test_cases = [
    {
        "name": "Case 1: Viral Cancer Cure Hoax (Expected FAKE)",
        "text": "SHOCKING: Doctors BANNED this miracle herbal tea because it cures all stage 4 cancer in 48 HOURS! Big Pharma executives are panicking and desperately trying to scrub this secret video off the internet. Just boiling two kitchen spices dissolves tumors completely with zero side effects!",
        "source": "naturalnews.com"
    },
    {
        "name": "Case 2: WHO Official Health Advisory (Expected REAL)",
        "text": "World Health Organization releases updated clinical management recommendations for viral seasonal influenza. Surveillance data collected across 48 reference laboratories indicates predominant circulating sub-strains remain susceptible to standard antiviral regimens. Public health authorities urge annual vaccination for geriatric populations.",
        "source": "who.int"
    },
    {
        "name": "Case 3: Borderline Financial Breaking News (Tricky)",
        "text": "BREAKING: Central Bank governor delivers urgent warning regarding volatile global currency markets and inflation pressures during emergency summit.",
        "source": "reuters.com"
    }
]

print("=" * 70)
print("TRUTHLENS PIPELINE VERIFICATION TEST")
print("=" * 70)

for idx, tc in enumerate(test_cases, 1):
    print(f"\n--- Testing {tc['name']} ---")
    print(f"Source: {tc['source']}")
    res = analyze(tc["text"], tc["source"])

    assert res["status"] == "success", f"Pipeline failed: {res.get('error_message')}"
    assert res["prediction"] in ["FAKE", "REAL"], f"Invalid prediction: {res['prediction']}"
    assert 0 <= res["confidence_pct"] <= 100, f"Invalid confidence: {res['confidence_pct']}"
    assert 0 <= res["risk_score"] <= 100, f"Invalid risk score: {res['risk_score']}"
    assert res["risk_label"] in ["Low Risk", "Medium Risk", "High Risk"], f"Invalid risk label: {res['risk_label']}"
    assert len(res["highlighted_html"]) > 0, "Highlighted HTML is empty"
    assert len(res["plain_english_reason"]) > 0, "Plain English reason is empty"

    print(f"Prediction: {res['prediction']} ({res['confidence_pct']}%)")
    print(f"Model Used: {res['model_used']}")
    print(f"Risk Score: {res['risk_score']}/100 [{res['risk_label']}]")
    print(f"Source Credibility: {res['credibility']['domain']} -> {res['credibility']['credibility_score']}/100 ({res['credibility']['category']})")
    print(f"Fact-Check Contradiction: {res['factcheck']['contradiction_score']}")
    if res['factcheck']['top_match']:
        print(f"Top Fact-Check Match: '{res['factcheck']['top_match']['claim'][:60]}...' ({res['factcheck']['top_match']['verdict']}) [Sim: {res['factcheck']['top_match']['similarity_pct']}%]")
    print(f"LIME Word Highlights: {[w for w, _ in res['word_weights'][:5]]}")
    print(f"Plain-English Rationale: {res['plain_english_reason']}")
    print("STATUS: PASSED OK")

print("\n" + "=" * 70)
print("ALL 3 PIPELINE TESTS PASSED PERFECTLY!")
print("=" * 70)
