# Decision Log

Non-obvious decisions made during this project, and why. Chronological.

## Data & Brand Selection

1. **Chose AmericanAir over Delta/British_Airways** despite similar domain, because it had the
   highest brand-reply volume among airline candidates (36,764) with comparable thread-depth
   (39.6% multi-turn) — more room for golden-set sampling without scraping the bottom of the barrel.

2. **Switched to chunked two-pass CSV reading** (200k rows/chunk) after hitting a `MemoryError`
   loading the full 516MB file with all columns at once on a 7.2GB RAM machine. Kept peak memory
   low by only ever holding filtered subsets, never the full 2.8M-row dataset.

3. **Capped OpenBLAS threads to 1** in the venv activation script after numpy imports failed with
   memory allocation errors — root cause was 8 logical cores trying to spin up a thread pool on a
   low-RAM machine. Fixed once, permanently, rather than as a per-session workaround.

## Taxonomy

4. **Iterated the taxonomy from two real samples** (n=30, then n=100) rather than designing it
   upfront. Added `onboard_technical_issue` as a distinct category after the second sample
   revealed it needs a different resolution path (crew workaround) than `flight_disruption`.

5. **Used `praise_non_actionable` instead of a general sentiment score.** ~25-30% of real inbound
   tweets were pure praise/chatter with no actionable ask; a dedicated intent captured this more
   precisely and cheaply than sentiment analysis would.

6. **Labeling rule: classify by the actionable ask, not the emotional framing.** A tweet combining
   a complaint with a concrete request (e.g. "racist airline... refund my flight") is labeled by
   the ask (`billing_fee_dispute`), with tone/severity handled separately by escalation logic.

## Golden Set

7. **Sampled 300 candidates** (oversampling above the 150-250 target) with a fixed seed
   (`random_state=2024`), filtering out tweets under 15 characters, to absorb labeling attrition
   without needing to re-sample later.

8. **Golden set contained 1 duplicate `tweet_id`** (2757715) from random sampling — both instances
   had identical hand-applied labels, so it was collapsed post-labeling with no data loss. Final
   golden set: 258 unique examples.

## Baselines & Evaluation Methodology

9. **Keyword baseline's 44% headline accuracy is partly inflated by its fallback default**
   (`praise_non_actionable` catches anything with no keyword match, giving it 0.82 recall but only
   0.29 precision). The number looks solid in aggregate but hides that the fallback bucket is
   absorbing other categories' misses — flagged explicitly rather than left implicit.

10. **Retrieval corpus explicitly excludes all golden-set tweet_ids** before indexing, preventing
    retrieval from ever surfacing the exact ground-truth answer for an example under evaluation.
    Caught a real bug here: the exclusion initially compared the wrong ID column (customer's
    tweet_id instead of the brand reply's own tweet_id) and silently excluded zero rows — caught
    via a sanity check that the exclusion count shouldn't be zero, not by trusting the first result.

## Model & Infrastructure

11. **Switched intent classification from `gemini-3.6-flash` to `gemini-3.1-flash-lite`** after
    hitting a 20-requests/day free-tier cap on the newer model. Flash-Lite tiers carry
    substantially higher free quotas, and classification is a simple-enough task that the lighter
    model is an appropriate choice, not just a workaround.

12. **Added explicit 30s HTTP timeouts to every Gemini API call** after a request hung indefinitely
    with no error during escalation decisioning. The SDK doesn't set one by default.

13. **Every LLM-calling script uses a checkpoint/resume pattern** (save every 10 rows, skip
    already-done tweet_ids on re-run) specifically to survive free-tier daily quota walls (500
    requests/day) without losing progress or re-spending quota on already-completed work.

## Agent Behavior

14. **Reply drafter initially fabricated a "speedy recovery" wish for a praise tweet** (no actual
    medical emergency occurred — customer was thanking AA for a fee waiver) because intent
    classification wasn't passed into the drafting prompt. Fixed by conditioning generation on
    the classified intent, with a distinct non-actionable path for praise. Verified via regression
    test on the exact failing example before scaling to the full golden set.

15. **Escalation false negatives (should escalate, didn't) are surfaced explicitly**, not just
    aggregated into accuracy. In a real support system a missed escalation is costlier than an
    unnecessary one, so recall on the "yes" class matters more than overall accuracy — this feeds
    directly into the business-framing section (false-auto-handle rate).

## Judge Validation

16. **Reported quadratic weighted Cohen's kappa alongside raw agreement, not kappa alone.**
    Groundedness and overall-quality kappa came out near-zero or undefined despite 94-100% raw
    agreement, because both score distributions were heavily skewed toward high scores (low
    variance). Kappa corrects for chance agreement, and with almost no variance, "chance
    agreement" is already near-total — so kappa can't distinguish real agreement from coincidence.
    Reporting kappa alone here would misleadingly suggest poor agreement when raw agreement was
    actually very high.

17. **The one major judge/human disagreement** (a highly ambiguous, low-context tweet: "it
    couldn't be gotten for me") had the judge score the reply 5/5 for "not making assumptions,"
    while human judgment scored it 3/5 for being unhelpfully vague. This suggests the judge may
    over-reward technical correctness at the expense of actual usefulness when the input itself
    is too sparse to act on meaningfully.

## CI/Eval Gate

18. **CI eval-gate runs only the free, instant baselines** (trivial + keyword) on every
    prompt/retrieval change, not the LLM-based checks. Running LLM calls on every PR would burn
    API quota and add latency for marginal signal, since baseline regressions already catch most
    accidental breakage (e.g. a broken import, a corrupted golden set) cheaply and fast. Full
    LLM-based evaluation stays a manual, deliberate step.

19. **Demo app wraps `run_agent()` in a try/except with a clean `st.error()` fallback**, rather
    than letting exceptions crash the app. Verified this works correctly when the daily API
    quota was hit mid-testing — the app showed a readable error instead of a stack trace, which
    matters for a live demo shown to a reviewer.
