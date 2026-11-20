"""
Project 6: Clinical Trial Protocol & Pharmacovigilance Assistant
Main Execution Script: Evaluates 5 assessment questions and outputs results.
"""

import json
from rag_pipeline import answer

QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the minimum required estimated Glomerular Filtration Rate (eGFR) threshold for a patient to be included in the CardioVex Phase III trial?",
        "expected_answer": "at least 45 mL/min/1.73m2",
        "expected_sources": ["Page 14"],
    },
    {
        "id": "Q2",
        "question": "Within how many hours of becoming aware of a Serious Adverse Event (SAE) must a site investigator report it to the pharmacovigilance safety desk?",
        "expected_answer": "within 24 hours",
        "expected_sources": ["Page 28"],
    },
    {
        "id": "Q3",
        "question": "What is the wholesale or retail price of CardioVex in commercial pharmacies?",
        "expected_answer": "[not found]",
        "expected_sources": [],
    },
    {
        "id": "Q4",
        "question": "Within what exact temperature range, in degrees Celsius, must reconstituted CardioVex vials be stored before administration?",
        "expected_answer": "2°C to 8°C",
        "expected_sources": ["Page 42"],
    },
    {
        "id": "Q5",
        "question": "What is the standard recommended dosage of amoxicillin for treating a paediatric ear infection?",
        "expected_answer": "[not found]",
        "expected_sources": [],
    },
]


def main():
    print("=" * 80)
    print("PROJECT 6: CLINICAL TRIAL PROTOCOL & PHARMACOVIGILANCE ASSISTANT")
    print("Evaluating 5 Assessment Questions against cardiovex_clinical_trial_protocol_50p.pdf")
    print("=" * 80)
    print()

    outputs = []

    for item in QUESTIONS:
        q_id = item["id"]
        q_text = item["question"]

        print(f"[{q_id}] Question: {q_text}")
        result = answer(q_text)

        actual_answer = result.get("answer", "")
        actual_sources = result.get("sources", [])

        print(f"      Answer : {actual_answer}")
        print(f"      Sources: {actual_sources}")
        print("-" * 80)

        outputs.append(
            {
                "id": q_id,
                "question": q_text,
                "answer": actual_answer,
                "sources": actual_sources,
                "expected_answer": item["expected_answer"],
                "expected_sources": item["expected_sources"],
            }
        )

    # Save to outputs.json
    output_path = "outputs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"\n[Execution Complete] Results successfully saved to {output_path}")


if __name__ == "__main__":
    main()
