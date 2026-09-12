# Report: AmericanAir Twitter Support Agent

## 1. Problem Framing

**What "good" means for this agent.** For AmericanAir's Twitter support volume, a good agent
does three things reliably: (1) correctly identifies what the customer actually wants, even when
buried in venting or ambiguous phrasing; (2) drafts a reply that's factually grounded in how
similar cases were actually resolved, never inventing circumstances the customer didn't state;
(3) reliably escalates anything with real stakes (safety, discrimination, chronic failure,
high-value customers) rather than confidently auto-handling something it shouldn't. Of these
three, escalation reliability matters most for trust — a wrong intent label is an inconvenience;
a missed escalation is the failure mode that actually damages the business or a customer.

**What I chose NOT to build, and why:**

- **No multi-turn conversation tracking.** Only 39.6% of AmericanAir's own replies received a
  further customer response (measured directly from the data, see `src/explore_brands.py`
  output) — meaning roughly 60% of threads are effectively single-turn from the agent's
  perspective. Modeling conversation state would have added real complexity (session tracking,
  context window management, multi-turn golden-set labeling) for a minority of cases, and would
  have made the evaluation harder to trust in the time available. This is a good candidate for
  "next week" (see below), not this submission.

- **No fine-tuned model or heavier retrieval (embeddings).** The brief explicitly rewards
  evaluation rigor over system sophistication. TF-IDF retrieval already produced strongly
  relevant historical matches (0.68-0.74 cosine similarity on real test queries) — reaching for
  embeddings or a fine-tuned classifier would have added engineering surface area without
  evidence that the simpler approach was actually failing.

- **No dedicated sentiment-analysis module.** Roughly 25-30% of real inbound tweets are pure
  praise/chatter with no actionable ask. A single `praise_non_actionable` intent category
  captured this more precisely and far more cheaply than a general sentiment score would have,
  since sentiment alone doesn't tell you whether something is actionable.

- **No full-dataset processing.** The brief explicitly discourages this. All work uses a golden
  set of 258 hand-labeled examples, sampled from AmericanAir's ~36,764 historical replies.

## 2. Results vs. Baselines

| Method                        | Accuracy  | 95% CI                   |
| ----------------------------- | --------- | ------------------------ |
| Trivial (majority class)      | 21.6%     | —                        |
| Keyword/rule-based            | 44.0%     | —                        |
| **LLM intent classification** | **79.1%** | **± 5.0%** (74.1%–84.1%) |

The LLM classifier clears the strongest baseline (keyword) by ~35 points and the trivial baseline
by nearly 4x. The confidence interval is computed via normal approximation on n=258
(`p ± 1.96·√(p(1−p)/n)`) — worth noting explicitly since 258 examples carries real sampling
uncertainty that a bare percentage hides.

**Escalation decisioning**: 84.1% overall accuracy (± 4.5% CI), but only **66% recall on true
escalations** (± 17.2% CI — this subgroup is only 29 examples, so this interval is wide and
should be read as directional, not precise). See Section 4 for why the second number is the one
that actually matters.

## 3. Business Framing (what a Hiver support team lead would actually care about)

Translating the escalation numbers into ticket-handling terms: of 258 messages, the agent would
auto-handle **208 (80.6%)** and escalate 50 (19.4%). Of the 208 auto-handled, **10 (4.8%) were
messages that a human labeler determined actually needed escalation** — meaning roughly **5 out
of every 100 auto-handled tickets would need human rework that the system didn't flag**. This is
the number that should sit next to any ticket-deflection headline, not instead of it: an 80.6%
deflection rate sounds strong on its own, but it's only meaningful alongside its 4.8%
false-auto-handle rate.

## 4. Failure Analysis (Top 5)

**1. Escalation recall gap (66%) — the costliest failure mode.** Real missed escalations from
the golden set include a mechanical issue ("plane is broken and there's no part to fix it"), a
$4,200 customer threatening to leave, and a customer asking about filing a formal complaint for
poor service at a specific gate. Hypothesis: the escalation prompt's trigger criteria (safety,
discrimination, chronic failure, high-value customer) are somewhat literal-keyword-dependent —
messages that imply seriousness through context or dollar amounts rather than explicit trigger
language get under-flagged.

**2. Billing disputes classified by surface topic instead of embedded ask.** Real example: _"Nice
that one of your agents let another friend on with MUCH more than .9 lbs over limit"_ — labeled
`billing_fee_dispute` (a fee-enforcement-consistency complaint) but predicted
`service_staff_complaint` (reasoning: "dissatisfaction... without making a specific request").
The model's own stated reasoning shows it's applying the "needs an explicit ask" heuristic too
literally — missing that inconsistent fee enforcement _is_ the billing complaint, even phrased
as an observation rather than a demand.

**3. `praise_non_actionable` over-prediction on ambiguous, non-negative-worded messages.** Real
example: _"On the bright side, AA1111 arrived early. So early that we're waiting for the previous
plane to depart before we can approach the gate"_ — labeled `flight_disruption`, predicted
`praise_non_actionable` (reasoning: "sharing a positive observation... without making a
complaint"). The model treats absence of explicit complaint language as presence of praise,
missing that this is actually describing an ongoing wait/disruption in a resigned tone.

**4. Reply drafter fabricated facts when intent wasn't passed into the prompt (found and fixed
during development).** Before intent-conditioning, a pure praise tweet ("thanks for waiving fees
so I could see a family member in an emergency") produced a drafted reply wishing the family
member "a speedy recovery" — a fabricated detail with no basis in the message. Root cause: the
drafting prompt had no signal that this was praise, not a problem. Fixed by passing predicted
intent into the prompt with distinct handling for `praise_non_actionable`; verified via
regression test on the exact failing example (see `DECISIONS.md` #14).

**5. LLM-judge over-rewards technical correctness over practical helpfulness on sparse inputs.**
The one major judge/human disagreement (judge=5, human=3) involved an extremely low-context tweet
("it couldn't be gotten for me") where the judge scored the vague reply highly for "not making
assumptions," while human judgment penalized it as unhelpfully vague. This suggests the judge's
rubric rewards avoiding fabrication more than it penalizes unhelpfulness — a real, if narrow, gap
between what the judge values and what a human considers quality service.

## 5. What Is Misleading About My Headline Number?

The 79.1% classification accuracy and 84.1% escalation accuracy are both, individually, correct
— and both are also incomplete on their own:

- **79.1% hides which categories the classifier struggles with.** `billing_fee_dispute` recall is
  only 62% — nearly 4 in 10 real billing disputes get classified elsewhere (Section 4, failure
  #2). A stakeholder reading "79% accuracy" would not know that one specific, revenue-relevant
  category is meaningfully weaker than the average.

- **84.1% escalation accuracy is inflated by class imbalance.** 229 of 258 golden examples don't
  need escalation; a model that said "never escalate" would already score 88.8% — _higher_ than
  the actual system's 84.1%. Accuracy alone is actively misleading here; recall on the "yes"
  class (66%) and the resulting false-auto-handle rate (4.8%) are the numbers that matter for
  trust, and neither is visible in the headline accuracy figure.

- **The keyword baseline's 44% accuracy is itself partly a `praise_non_actionable` fallback
  artifact** (0.82 recall, 0.29 precision on that class) — meaning the "beat by 35 points"
  comparison is fair, but the baseline's own number was already somewhat propped up by its
  catch-all default, not pure keyword signal (`DECISIONS.md` #9).

## 6. What I'd Do Next With One More Week

1. **Multi-turn context tracking** for the ~40% of threads that are genuinely multi-turn — likely
   the single highest-value addition, since the current agent treats every message independently.
2. **Expand the escalation golden set specifically** — only 29 of 258 examples are true
   escalations, giving a wide confidence interval (±17.2%) on the most important metric in the
   whole project. A targeted additional sample of escalation-worthy cases would tighten this
   considerably.
3. **Fix the two named failure modes directly**: strengthen the classification prompt's handling
   of embedded billing asks (failure #2) and implicit-tone disruption complaints (failure #3),
   then re-run against the same golden set to measure the delta.
4. **Fine-tune the escalation trigger criteria** away from literal keyword dependence toward
   contextual severity (dollar amounts, repeated-failure framing without the word "again").
5. **Language-matching in reply drafting** — the one Spanish-language golden example got a
   sensible English reply; matching customer language would be a natural, low-risk improvement.

---

_Dataset used under CC-BY-NC-SA-4.0, non-commercial academic/evaluation use. See `DECISIONS.md`
for the full non-obvious decision log and `README.md` for reproduction instructions._
