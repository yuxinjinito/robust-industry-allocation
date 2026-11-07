# Addendum: what the robust model adds, and how strong the signal must be

This addendum extends the CS 524 report "Asset Allocation When Expected Returns Are Unreliable".
The report left three things open. Its robust book beat the single-industry nominal book on
drawdown but had no edge over equal weight (report 4.F). It listed correlated estimation
errors, means that move over time, and missing transaction costs as limitations (report 4.H).
And it could not say why equal weight was so hard to beat. We ran two rounds of experiments on
the same data and the same 31 rolling windows. For each round we wrote down the predictions
before running it, and we report every prediction as held or refuted.

Numbers carry an anchor in brackets: [T1] to [T5] are tables in `results/tables.md` (round 1),
[I] is `results/injection_tables.md` (round 2), [R] is the original report.

## 1. Six ways to handle unreliable means

### Question

Given estimated means that are too noisy to trust, an investor can trust them anyway (the
nominal book), defend against them (a robust uncertainty set), repair them before optimizing
(shrink them toward a model), or ignore them (equal weight). We asked which of these does best
against equal weight when every rule is tuned by the same validation step on the same windows.

### Rules compared

- NOM: the report's LP at Γ = 0. EW: 1/30 in each industry.
- BOX: the report's Bertsimas-Sim LP, Γ chosen on validation.
- ELL: an ellipsoidal uncertainty set built from the covariance of the mean estimate. Its robust
  counterpart is a second-order cone program, solved with Clarabel. It is the same family as
  mean-variance with a validation-chosen risk aversion.
- RBOX and RELL: BOX and ELL applied to means shrunk toward a CAPM-implied target, with the
  shrinkage weight chosen on validation.
- DRIFT: BOX with an uncertainty width built from the spread of yearly means, so that a mean
  that moves between years widens the set.

Every rule picks its one parameter by the report's rule (best validation Sharpe, smallest value
on ties) and is scored on the same 31 non-overlapping three-year test windows, 1934 to 2026.
Before scoring anything, the new code had to reproduce all 53 printed lines of the original
notebook exactly, and it did.

### Results

| rule | median Sharpe | median max drawdown % | windows with higher Sharpe than EW, of 31 | windows with smaller drawdown than EW, of 31 |
|---|---|---|---|---|
| NOM | 0.437 | 37.5 | 6 | 2 |
| EW | 0.987 | 22.0 | | |
| BOX | 0.957 | 26.8 | 10 | 8 |
| ELL | 0.957 | 24.5 | 11 | 12 |
| RBOX | 0.999 | 24.8 | 12 | 9 |
| RELL | 0.974 | 23.4 | 15 | 14 |
| DRIFT | 0.872 | 27.2 | 12 | 10 |

[T1]. No rule beats equal weight on Sharpe in more than 15 of 31 windows, and no sign test
against equal weight survives the Holm correction (every adjusted p is 1). The fit windows
overlap, so these counts are checks, not significance claims. Every rule except NOM and DRIFT
sits within about 0.03 of equal weight's median Sharpe. Against the all-in book, every robust rule
has a far smaller drawdown: 26.8% for BOX and 24.5% for ELL, against 37.5% [T1, T5].

Transaction costs do not change the picture. At 10 basis points per unit of turnover, no
rule's median Sharpe falls by more than 0.0104, and no two rules swap order [T1, P5 line].

### What we expected and what happened

- Held: no rule beats equal weight in more than 20 of 31 windows (largest: 15).
- Refuted: we expected ELL, the only rule that uses the covariance, to have the most wins. RELL
  had more (15 against 11).
- Refuted: we expected ELL to beat equal weight on drawdown in at least 16 of 31 windows. It
  did in 12. Using the covariance helped against BOX (8) but not against equal weight.
- Held: ELL is more concentrated than BOX (median 4 holdings against 9). Its selection often
  pushed the risk aversion to the top of its grid (12 of 31 windows), which moves it toward the
  long-only minimum-variance book, and that book sits in a few low-volatility industries.
- Held: costs move no median Sharpe by more than 0.05.
- Refuted: we expected the CAPM repair to be used in most windows. RBOX chose a nonzero
  shrinkage in only 8 of 31, and both repair rules have a median paired Sharpe difference of 0
  against their parents.
- Held: the yearly means of industries vary more than independent daily noise would produce in
  17 of 31 training blocks, mostly before 1990, and DRIFT then chooses a different Γ than BOX
  in 16 windows. Accounting for this drift gave the lowest median Sharpe of the robust rules.

[T1, T4, Predictions section of `results/tables.md`.]

### A structural result

Shrinking every mean toward one common value cannot repair a long-only, fully invested
portfolio. With weights summing to one, the shrunk objective (1 - w) μ̂'x + w c differs from
the original only by a constant and a positive scale. So Bayes-Stein shrinkage of the Jorion
type is the same as widening the uncertainty set by 1/(1 - w). A repair rule only changes the
portfolio if its target differs across industries, which is why RBOX and RELL use a
CAPM-implied target. We found this while checking the design, before any result existed. We
have not checked whether it appears in the literature.

### What round 1 says

The robust model's benefit is diversification. Against the book that trusts its estimates
completely, it cuts the median drawdown by about 30% (37.5% to 26.8% for BOX) at no cost in
return. Against equal weight it is
comparable, and none of the five ways we tried to give it more information (covariance, a
pricing model, moving means, costs) turned comparable into better. This matches DeMiguel,
Garlappi and Uppal (2009) on a different set of assets and a century of data.

## 2. How strong must the signal be?

### Question

A tie with equal weight has two readings. Either the data contain no usable difference between
industries, or the difference is there and six years of data cannot find it. The real data
cannot tell these apart, because we never see the true premia. So we built worlds where we do.

### Method

- Take the real daily returns of 1927 to 2025 and remove each industry's average return
  relative to the others. The correlations, fat tails, volatility clusters and changes between
  eras all stay. Only the long-run difference between industries is gone.
- Add back a known difference: each industry gets a fixed extra return per year. Its spread,
  the standard deviation of those extras across the 30 industries, is the signal strength s.
  The extras sum to zero, so equal weight is unaffected.
- Build 200 histories of 11 distinct years each (6 to fit, 2 to validate, 3 to test, no year
  shared), run NOM, BOX and ELL exactly as in round 1, and record how often each beats equal
  weight. Repeat for s from 0 to 10% per year.
- The threshold s* is the smallest s at which a rule beats equal weight in more than half the
  histories.
- Separately, estimate s in the real data from a century of yearly returns, correcting for the
  noise in those averages.

Each of the eight checks in the run passed before any number was used, including one that the
noise correction is unbiased in a world where the true s is known [I, gates].

### Results

| pattern of the injected difference | NOM s* | BOX s* | ELL s* |
|---|---|---|---|
| dense: every industry gets a different extra | 6% | 2% (0.5 to 3) | 2% (1 to 2) |
| sparse: one industry gets all of it | 2% | 2% | 2% |
| real direction: extras point the way the real data's premia point | 6% | 4% (3 to 4) | 3% (2 to 4) |

[I]. Brackets are 90% intervals from resampling the histories. At s = 10% (dense), BOX and ELL
beat equal weight in 94% of histories, so the pipeline works when the signal is strong.

The real data's estimate is s = 0.72% per year, with a 90% interval of 0 to 2.02% [I]. The
interval is conservative: in the known-s check it covered the truth about 99% of the time
instead of 90%. Allowing for persistence between years changes little (0.64%, 0 to 2.24%; the
lag-1 autocorrelation of yearly relative returns is 0.076).

### What we expected and what happened

- Held: with no injected difference, NOM beats equal weight in only 25% of histories.
- Refuted: we expected the threshold for BOX and ELL to be at least 3%. It is 2%. The pipeline
  is more sensitive than we guessed.
- Held: both rules beat equal weight in more than half the histories at 10%.
- Inconclusive (the primary comparison): the real estimate, 0.72%, is about a third of the 2%
  threshold, but the upper end of its interval reaches 2.02%, so we cannot rule out a real
  difference near the threshold.
- Refuted: our simulator should reproduce the real win rates at the estimated s. For BOX it is
  close (0.42 against the real 10 of 31), but for ELL it is too kind (0.52 against 11 of 31). In
  the zero-signal world built from consecutive years, ELL already beats equal weight in 52% of
  histories, so ELL's curves should be read with that discount.
- Undecidable for ELL for the same reason; BOX's threshold moves by one grid step between the
  two ways of building histories.
- Held: all-in detects a concentrated difference (2%) much earlier than a spread-out one (6%).

### What round 2 says

Robust rules start to beat equal weight once industries differ by about 2% a year in long-run
return. The real data's best estimate is about 0.7% a year. That is consistent with the tie in
round 1, but the uncertainty in the estimate is wide enough that we report it as an estimate
below the threshold, not a proof. In the direction the real premia actually point, the
threshold is higher: its point values (3% for ELL, 4% for BOX) lie above the whole interval of
the real estimate, although ELL's own interval reaches down to 2%. That last result is
descriptive; we did not predict it in advance.

## 3. Limitations

- One data source and one universe of 30 industries. The 31 windows share fit years, and the
  200 histories share calendar years, so neither set is independent. The interval on s* is
  simulation noise given this one century, not the sampling error of a threshold.
- The noise correction for the real estimate assumes years are independent. The block version
  agrees, but a longer persistence would widen the interval.
- The simulator is too favourable to ELL, as the calibration check showed.
- Every Sharpe uses a zero risk-free rate, and every model is single-period and long-only, as
  in the report.

## How the results were checked

Each round followed the same steps: a written plan, an independent read of the plan, predictions
fixed before any code ran, the code, an adversarial review of the code, then the run. The reviews changed the design several times before any result
existed. Three examples: the first repair rule turned out to be the defend rule in disguise
(the structural result above); the first way of building the null world let test years leak
into the fit; and a scoping bug in the table code would have reported two predictions as held
whatever the data said. Two checks stopped runs on their own: a solver that stalled on badly
scaled inputs, and an interval that covered the truth more often than intended. Every deviation from a plan is reported in this addendum where it affects a
number.
