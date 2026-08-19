#!/usr/bin/env python3
"""Verify the committed publication contract for this repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = (
    "PARTIAL_C1_C2_C3_VERIFIED_C4_SOURCE_AUDITED_C5_WEATHER_DEFERRED_C6_SWISS_ROLL_PENDING_NO_CURRENT_SCORE"
)
EXPECTED_BRANCHES = {
    "audit/loss-inverse-eot-equivalence",
    "audit/swiss-roll-baselines",
    "audit/swiss-roll-benchmark",
    "audit/swiss-roll-conditional-flow",
    "audit/swiss-roll-linear-calibration",
    "audit/weather-tables",
    "historical/judged-baseline",
    "main",
}
EXPECTED_COMMITS = 24
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
CLAIM_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load(name: str):
    return json.loads((ROOT / name).read_text())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    remote = {
        name.removeprefix("origin/")
        for name in git(
            "for-each-ref", "refs/remotes/origin", "--format=%(refname:short)"
        ).splitlines()
        if name.startswith("origin/") and name != "origin/HEAD"
    }
    if remote:
        return remote
    return set(git("for-each-ref", "refs/heads", "--format=%(refname:short)").splitlines())


def main() -> None:
    claims = load("claims.json")
    verdicts = load("reproduction_verdicts.json")
    manifest = load("EVIDENCE_MANIFEST.json")
    state = load("AUTONOMOUS_STATE.json")
    claim_1 = load(".openresearch/artifacts/claim_1/formal_raw.json")
    claim_2 = load(".openresearch/artifacts/baseline/claim_2/formal_raw.json")
    claim_3 = load(".openresearch/artifacts/baseline/claim_3/formal_raw.json")
    source = load(".openresearch/artifacts/baseline/claim_2/claim_contract.json")

    expected_statuses = {
        "C1": "VERIFIED_SCOPED",
        "C2": "VERIFIED_SCOPED",
        "C3": "VERIFIED_SCOPED",
        "C4": "SOURCE_AUDITED",
        "C5": "DEFERRED_EXTERNAL_DATA",
        "C6": "PENDING_ROUTE",
    }
    require(claims["overall_status"] == EXPECTED_STATUS, "claims overall status")
    require(state["overall_status"] == EXPECTED_STATUS, "state overall status")
    require(verdicts["claim_statuses"] == expected_statuses, "verdict statuses")
    require([claim["id"] for claim in claims["claims"]] == CLAIM_IDS, "claim ordering")
    require({claim["id"]: claim["status"] for claim in claims["claims"]} == expected_statuses, "claim statuses")
    require(all((ROOT / path).exists() for path in manifest["required_paths"]), "manifest paths")
    require(claims["paper"]["source_sha256"] == manifest["source"]["source_sha256"], "source hash")
    require(claim_1["verdict"] == "VERIFIED" and claim_1["controls_failed_as_intended"], "Claim 1 accepted evidence")
    require(claim_2["verdict"] == "VERIFIED" and claim_2["control_failed_as_intended"], "Claim 2 accepted evidence")
    require(claim_3["verdict"] == "VERIFIED" and claim_3["control_failed_as_intended"], "Claim 3 accepted evidence")
    require(source["source_anchor"].startswith("arXiv:2410.02628v5 Proposition 3.1"), "Claim 2 source anchor")
    require("Formal result pending" in (ROOT / ".openresearch/artifacts/claim_4/EVAL.md").read_text(), "weather/practical pending boundary")
    require("Formal result pending" in (ROOT / ".openresearch/artifacts/claim_6/EVAL.md").read_text(), "Swiss Roll pending boundary")
    require("DineshAI/0p617sK4Z4" in (ROOT / ".openresearch/protected/judged-space-3c31d94f29e79228a5d3ea9c9e1ea575ebb70e32/logbook.json").read_text(), "Space identity")
    require(verdicts["historical_external_result"]["score_recorded"] is False, "historical score boundary")
    require(verdicts["historical_external_result"]["current_score_claim"] is False, "current score claim")
    require(verdicts["publication"]["publication_allowed"] is False, "publication state")
    require(verdicts["publication"]["author_endorsement_claimed"] is False, "author endorsement state")

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branches")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>\n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical commit identity")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(branches)} commits={EXPECTED_COMMITS} "
        "claims=C1:C3_verified_scoped,C4_source_audited,C5_weather_deferred,C6_swiss_roll_pending "
        "current_score_claim=false publication_allowed=false"
    )


if __name__ == "__main__":
    main()
