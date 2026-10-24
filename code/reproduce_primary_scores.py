#!/usr/bin/env python3
"""Read-only, standard-library replay of KineLoopBench 225/150 cohort totals.

Run from repository root: python code/reproduce_primary_scores.py
Uses derived flags from original scored responses, not the model responses.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "primary_225_per_case_scored_outcomes.csv"
SUMMARY = ROOT / "results" / "non_singular_cohort_results.csv"
PRIMARY = ROOT / "subsets" / "primary_225.csv"
FEASIBLE = ROOT / "subsets" / "feasible_150.csv"
SOURCE_SHA256 = "29fb39e6ae34c79c351f6918c8921766ab743a0d31eb7ccdf760b441bc87cc23"
MODELS = ("qwen", "gemma", "gemini", "gpt")


def records(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def check(cond, message):
    if not cond:raise AssertionError(message)


def main():
    check(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256,
          'Primary per-case score outcomes SHA-256 mismatch')
    rows=records(SOURCE)
    primary={x['case_uid'] for x in records(PRIMARY)}
    feasible={x['case_uid'] for x in records(FEASIBLE)}
    check(len(primary)==225 and len(feasible)==150 and feasible<=primary,'Bad source subset IDs')
    check(len(rows)==900,'Expected four models x 225 cases')
    check(Counter(x['model'] for x in rows)=={m:225 for m in MODELS},'Unequal model rows')
    check(Counter((x['model'],x['benchmark_state']) for x in rows)==
          {(m,s):75 for m in MODELS for s in ('regular','near_singular','infeasible')},
          'Primary state distribution mismatch')
    summary={(r['model'],r['cohort']):r for r in records(SUMMARY)}
    for model in MODELS:
        entries=[r for r in rows if r['model']==model]
        check({r['case_uid'] for r in entries}==primary, 'Primary IDs differ for '+model)
        for cohort,expected_ids in [('non_singular_225',primary),('regular_near_150',feasible)]:
            subset=[r for r in entries if r['case_uid'] in expected_ids]
            expected=summary[(model,cohort)]
            check(len(subset)==int(expected['n']),'Unexpected cohort size')
            for cond in ('direct','assisted'):
                for metric,col in [('physical_exact_success','physical'),('configuration_count_correct','count_correct')]:
                    totals=sum(int(r[cond+'_'+metric]) for r in subset)
                    check(totals==int(expected[cond+'_'+col]), f'{model} {cohort} {cond} {metric}: {totals} differs from archived {expected[cond+"_"+col]}')
            gain=round(100*(int(expected['assisted_physical'])-int(expected['direct_physical']))/len(subset),4)
            check(abs(gain-float(expected['assisted_minus_direct_pp']))<0.0001,'Archived delta mismatch')
    print('PASS: 225 primary and 150 feasible case IDs validated for each of four models.')
    print('PASS: All 32 archived aggregate score totals and eight assisted–direct changes reproduced.')
    print('No model requests, no dataset modifications.')


if __name__=='__main__':main()
