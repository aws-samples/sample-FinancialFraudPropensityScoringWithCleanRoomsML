# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate synthetic demo data for AWS Clean Rooms ML FSI Fraud Propensity Scoring.

Scenario: Cross-Institution Fraud Propensity Scoring
- Party A (Bank): account behavior data
- Party B (Payment Processor): transaction signal data
- Shared key: customer_id (overlapping customer space)

Design principles for realistic ML performance:
- The is_suspicious label is derived from a COMBINED score using signals from
  BOTH parties, so neither party alone can predict fraud well.
- Each feature has moderate predictive power with significant noise.
- Feature importance is distributed across many features rather than concentrated.
- Target class balance is ~30% suspicious (realistic for a pre-filtered population).
"""

import csv
import random
import math
import os
from datetime import datetime, timedelta

random.seed(42)

NUM_CUSTOMERS = 50000
SHARED_CUSTOMERS = int(NUM_CUSTOMERS * 0.8)
BANK_ONLY = int(NUM_CUSTOMERS * 0.1)
PROCESSOR_ONLY = int(NUM_CUSTOMERS * 0.1)

shared_ids = [f"cust_{i:06d}" for i in range(SHARED_CUSTOMERS)]
bank_only_ids = [f"cust_{SHARED_CUSTOMERS + i:06d}" for i in range(BANK_ONLY)]
processor_only_ids = [f"cust_{SHARED_CUSTOMERS + BANK_ONLY + i:06d}" for i in range(PROCESSOR_ONLY)]

bank_customer_ids = shared_ids + bank_only_ids
processor_customer_ids = shared_ids + processor_only_ids

ACCOUNT_TYPES = ["checking", "savings", "business", "joint", "premium"]
CARD_TYPES = ["credit", "debit", "prepaid", "virtual"]

BASE_DATE = datetime(2025, 1, 1)
END_DATE = BASE_DATE + timedelta(days=180)

# Per-customer latent traits (set during generation)
CUSTOMER_BANK_SCORE = {}      # bank-side risk signal (0-1)
CUSTOMER_PROCESSOR_SCORE = {} # processor-side risk signal (0-1)


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


def assign_latent_traits():
    """
    Assign each customer two independent latent risk scores:
    - bank_risk: drives bank-side features (auth failures, device count, etc.)
    - proc_risk: drives processor-side features (chargebacks, declines, etc.)

    The is_suspicious label is derived from a WEIGHTED COMBINATION of both,
    so neither party alone has enough signal to predict well.
    """
    all_ids = set(bank_customer_ids) | set(processor_customer_ids)
    for cid in all_ids:
        # Two semi-independent risk dimensions with moderate correlation (r~0.3)
        shared_component = random.gauss(0, 1)
        bank_independent = random.gauss(0, 1)
        proc_independent = random.gauss(0, 1)

        # Mix shared + independent components
        bank_raw = 0.3 * shared_component + 0.7 * bank_independent
        proc_raw = 0.3 * shared_component + 0.7 * proc_independent

        # Normalize to 0-1 range via sigmoid
        CUSTOMER_BANK_SCORE[cid] = 1.0 / (1.0 + math.exp(-bank_raw))
        CUSTOMER_PROCESSOR_SCORE[cid] = 1.0 / (1.0 + math.exp(-proc_raw))


def compute_is_suspicious(cid):
    """
    Determine if a customer is suspicious based on BOTH risk scores.
    This ensures the model needs data from both parties to predict well.

    Formula: combined_score = 0.45 * bank_risk + 0.45 * proc_risk + 0.10 * noise
    Threshold at 0.55 → ~30% suspicious rate
    """
    bank_r = CUSTOMER_BANK_SCORE.get(cid, 0.5)
    proc_r = CUSTOMER_PROCESSOR_SCORE.get(cid, 0.5)
    noise = random.gauss(0, 0.08)
    combined = 0.45 * bank_r + 0.45 * proc_r + 0.10 * noise
    return int(combined > 0.55)


def generate_bank_data():
    """
    Generate Party A (Bank) account behavior data.

    Features are driven by bank_risk score but with substantial noise,
    so no single feature is a strong predictor on its own.
    """
    rows = []
    for cid in bank_customer_ids:
        risk = CUSTOMER_BANK_SCORE.get(cid, 0.5)

        num_accounts = random.randint(1, min(5, 2 + int(3 * risk + random.random())))
        for acct_type in random.sample(ACCOUNT_TYPES, num_accounts):
            account_label = f"acc_{cid}_{acct_type[:3]}"

            # login_count: moderate correlation with risk (more logins = more activity)
            login_count = max(1, int(random.gauss(20 + 25 * risk, 12)))

            # failed_auth_attempts: weak-moderate signal
            fail_rate = clamp(random.gauss(0.03 + 0.08 * risk, 0.04), 0, 0.4)
            failed_auth_attempts = max(0, int(login_count * fail_rate))

            # account_age_days: newer accounts slightly riskier, but lots of overlap
            account_age_days = max(30, int(random.gauss(900 - 250 * risk, 350)))

            # linked_devices: moderate signal
            linked_devices = max(1, int(random.gauss(2 + 3 * risk, 1.5)))

            # avg_transaction_value: higher risk → more variance, slightly higher
            avg_transaction_value = round(max(15.0, random.gauss(
                400 + 600 * risk, 250 + 200 * risk)), 2)

            # geo_spread_score: moderate signal (fraudsters transact from more places)
            geo_spread_score = round(clamp(random.gauss(
                0.15 + 0.35 * risk, 0.18)), 4)

            # NEW: night_activity_ratio — fraction of logins between 11pm-5am
            night_activity_ratio = round(clamp(random.gauss(
                0.05 + 0.20 * risk, 0.10)), 4)

            # NEW: avg_session_duration_min — suspicious users have shorter sessions
            avg_session_duration_min = round(max(0.5, random.gauss(
                12 - 5 * risk, 4)), 2)

            # NEW: ip_change_frequency — how often IP changes per login session
            ip_change_frequency = round(clamp(random.gauss(
                0.05 + 0.25 * risk, 0.12)), 4)

            # NEW: dormant_reactivation — 1 if account was dormant >90 days then reactivated
            dormant_prob = clamp(0.05 + 0.25 * risk + random.gauss(0, 0.1))
            dormant_reactivation = int(random.random() < dormant_prob)

            observation_date = random_date(BASE_DATE, END_DATE).strftime("%Y-%m-%d")

            rows.append({
                "customer_id": cid,
                "account_id": account_label,
                "login_count": login_count,
                "failed_auth_attempts": failed_auth_attempts,
                "account_age_days": account_age_days,
                "linked_devices": linked_devices,
                "avg_transaction_value": avg_transaction_value,
                "geo_spread_score": geo_spread_score,
                "night_activity_ratio": night_activity_ratio,
                "avg_session_duration_min": avg_session_duration_min,
                "ip_change_frequency": ip_change_frequency,
                "dormant_reactivation": dormant_reactivation,
                "observation_date": observation_date,
            })
    return rows


def generate_payment_processor_data():
    """
    Generate Party B (Payment Processor) transaction signal data.

    Features are driven by proc_risk score but with substantial noise.
    The is_suspicious label uses BOTH risk scores.
    """
    rows = []
    for cid in processor_customer_ids:
        risk = CUSTOMER_PROCESSOR_SCORE.get(cid, 0.5)
        is_suspicious = compute_is_suspicious(cid)

        num_cards = random.randint(2, min(4, 2 + int(2 * risk + random.random())))
        for card_type in random.sample(CARD_TYPES, num_cards):

            # chargeback_count: moderate signal, lots of noise
            chargeback_count = max(0, int(random.gauss(0.5 + 3.0 * risk, 1.8)))

            # declined_transactions: moderate signal
            declined_transactions = max(0, int(random.gauss(2 + 8 * risk, 4)))

            # transaction_velocity: txns per day, moderate signal
            transaction_velocity = round(max(0.1, random.gauss(
                3 + 6 * risk, 3)), 2)

            # merchant_category_diversity: weak-moderate signal
            merchant_category_diversity = max(1, int(random.gauss(
                4 + 8 * risk, 4)))

            # cross_border_ratio: moderate signal
            cross_border_ratio = round(clamp(random.gauss(
                0.05 + 0.25 * risk, 0.12)), 4)

            # days_since_last_dispute: lower = riskier, but noisy
            days_since_last_dispute = max(1, int(random.gauss(
                200 - 120 * risk, 60)))

            # NEW: avg_txn_amount: higher risk → more extreme amounts
            avg_txn_amount = round(max(5.0, random.gauss(
                150 + 400 * risk, 180)), 2)

            # NEW: txn_amount_stddev: suspicious users have higher variance
            txn_amount_stddev = round(max(0.0, random.gauss(
                50 + 200 * risk, 80)), 2)

            # NEW: weekend_txn_ratio: fraction of transactions on weekends
            weekend_txn_ratio = round(clamp(random.gauss(
                0.25 + 0.15 * risk, 0.10)), 4)

            # NEW: rapid_succession_count: txns within 1 min of each other
            rapid_succession_count = max(0, int(random.gauss(
                1 + 5 * risk, 2.5)))

            # NEW: unique_country_count: number of distinct countries
            unique_country_count = max(1, int(random.gauss(
                1.5 + 4 * risk, 2)))

            last_dispute_date = (
                END_DATE - timedelta(days=min(days_since_last_dispute, 180))
            ).strftime("%Y-%m-%d")

            rows.append({
                "customer_id": cid,
                "card_type": card_type,
                "chargeback_count": chargeback_count,
                "declined_transactions": declined_transactions,
                "transaction_velocity": transaction_velocity,
                "merchant_category_diversity": merchant_category_diversity,
                "cross_border_ratio": cross_border_ratio,
                "days_since_last_dispute": days_since_last_dispute,
                "last_dispute_date": last_dispute_date,
                "avg_txn_amount": avg_txn_amount,
                "txn_amount_stddev": txn_amount_stddev,
                "weekend_txn_ratio": weekend_txn_ratio,
                "rapid_succession_count": rapid_succession_count,
                "unique_country_count": unique_country_count,
                "is_suspicious": is_suspicious,
            })
    return rows


def write_csv(filename, rows, fieldnames):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows to {filename}")


if __name__ == "__main__":
    print("Assigning latent customer traits...")
    assign_latent_traits()

    print("Generating payment processor data (Party B)...")
    proc_data = generate_payment_processor_data()
    write_csv(
        "data/payment_processor_transactions.csv",
        proc_data,
        ["customer_id", "card_type", "chargeback_count", "declined_transactions",
         "transaction_velocity", "merchant_category_diversity", "cross_border_ratio",
         "days_since_last_dispute", "last_dispute_date",
         "avg_txn_amount", "txn_amount_stddev", "weekend_txn_ratio",
         "rapid_succession_count", "unique_country_count", "is_suspicious"],
    )

    print("Generating bank data (Party A)...")
    bank_data = generate_bank_data()
    write_csv(
        "data/bank_account_behavior.csv",
        bank_data,
        ["customer_id", "account_id", "login_count", "failed_auth_attempts",
         "account_age_days", "linked_devices", "avg_transaction_value",
         "geo_spread_score", "night_activity_ratio", "avg_session_duration_min",
         "ip_change_frequency", "dormant_reactivation", "observation_date"],
    )

    bank_custs = set(r["customer_id"] for r in bank_data)
    proc_custs = set(r["customer_id"] for r in proc_data)
    overlap = bank_custs & proc_custs
    print(f"\nStats:")
    print(f"  Bank customers:              {len(bank_custs)}")
    print(f"  Payment processor customers: {len(proc_custs)}")
    print(f"  Overlapping:                 {len(overlap)}")
    print(f"  Bank rows:                   {len(bank_data)}")
    print(f"  Payment processor rows:      {len(proc_data)}")

    # Suspicious rate
    total_suspicious = sum(1 for r in proc_data if r["is_suspicious"])
    total_rows = len(proc_data)
    print(f"  Overall suspicious rate:     {total_suspicious}/{total_rows} ({100*total_suspicious/total_rows:.1f}%)")

    # Per-customer suspicious rate
    cust_suspicious = {}
    for r in proc_data:
        cust_suspicious[r["customer_id"]] = r["is_suspicious"]
    shared_susp = sum(v for k, v in cust_suspicious.items() if k in overlap)
    shared_total = sum(1 for k in cust_suspicious if k in overlap)
    print(f"  Shared customer suspicious:  {shared_susp}/{shared_total} ({100*shared_susp/max(1,shared_total):.1f}%)")
