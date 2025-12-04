# Predicting FIRST Robotics Competition Match Outcomes  
**Author:** Brian K.

## Introduction

This project asks a simple but important question: *given two alliances in a FIRST Robotics Competition (FRC) match, how reliably can we predict who will win using only historical performance?*  

Using match data from 2016-2025, I built a family of supervised models that estimate the probability that the **red alliance** wins each match. After moving from a single train/test split to a rolling, "live" evaluation that predicts each match using only information that would have been known before it was played, the best models reach **75-76% accuracy** on 2025 matches.  

Across the season, the models are remarkably stable: weeks 1-6 all sit in a narrow accuracy band around the mid 70% range, with week 0 as an expected outlier due to its small size and preseason nature. At the event level, most regionals and district events land between **70% and 85% accuracy**, with a handful above that. These results are competitive with widely used community tools such as Statbotics, despite using a relatively simple feature set and generic machine learning models.

---

## 1. Why Match Prediction Matters

Match outcome prediction is not just a leaderboard guessing game. Accurate, well calibrated forecasts can change how teams and organizers operate:

- **Scouting and strategy.** If a model can say "your alliance has a 78% win probability" with reasonable reliability, drive teams can make smarter trade offs (e.g., how risky to be in teleop, whether to play defense, focusing on drive practice vs dominating the match).
- **Equity and competitive balance.** Systematically higher or lower predicted win rates for specific regions, weeks, or event types can signal structural inequities (travel costs, resource disparities, district vs. regional formats) that are otherwise hard to quantify.
- **Education and transparency.** Teams already rely on opaque "EPA style" metrics from community tools. A transparent analysis that shows *what features* drive predictive power helps students understand where performance actually comes from. Specifically for future years in understanding what designs work best and what to avoid when comparing matchups and robot designs.

---

## 2. High Level Modeling Setup

At a high level, every match is represented by:

- **Alliance level scoring statistics** for auto, teleop, and total score based on historical performance of the three teams in each alliance.
- **Distribution features** such as max, top 2 average, min, and standard deviation for each scoring phase, capturing whether an alliance is carried by one powerhouse team or is more balanced.
- **Short term form features**: rolling averages over the **last five matches** for each team (including win rate), then aggregated to the alliance.
- **Match context**: whether the match is a qualification, semifinal, or final.

These are derived from team level stats precomputed across all seasons and then merged into each match record.  

For modeling, I focused on three classifiers that work well on structured tabular data:

- **Logistic Regression**
- **XGBoost**
- **LightGBM**

I experimented with KNN and Random Forest earlier but they were consistently weaker and were dropped from the final comparison.

Instead of a single train/test split, I use a **rolling, live evaluation**: to predict 2025 matches, the models are trained on all matches *strictly earlier* in time, then rolled forward match by match. This mimics how a real "live scout" model would behave at an event.

---

## 3. Weekly Accuracy and Season-Long Stability

Figure 1 shows the accuracy of the three main models across the 2025 season using the rolling evaluation.

> **Figure 1.** Model accuracy across weeks (rolling evaluation).  
> ![`model_comparison_accuracy.png`](outputs/rolling_match/model_comparison_accuracy.png)

Numerically, the weekly accuracies look like this:

| Week | Logistic Reg. | XGBoost | LightGBM |
|------|---------------|---------|----------|
| 0    | 0.581         | 0.581   | 0.581    |
| 1    | 0.739         | 0.749   | 0.754    |
| 2    | 0.735         | 0.745   | 0.755    |
| 3    | 0.743         | 0.753   | 0.758    |
| 4    | 0.771         | 0.772   | 0.762    |
| 5    | 0.754         | 0.773   | 0.773    |
| 6    | 0.743         | 0.739   | 0.738    |
| 7    | 0.721         | 0.693   | 0.699    |

A few key observations:

1. **Week 0 is noisy.** All three models only reach ~58% accuracy in week 0. This makes sense: week 0 is a small, pre-season event where teams are still debugging robots and playing early versions of their strategies. There is almost no same year data, so the models are relying on historical performance that may not reflect the new game yet.

2. **Weeks 1-6 are remarkably consistent.** Once the real season starts, accuracy jumps into a band between **~0.74 and ~0.77** and stays there. This suggests:
   - The features capture relatively stable aspects of team strength that transfer across weeks.
   - The game’s scoring structure and alliance formation process do not introduce wild swings in predictability from week to week.

3. **Championships behave like a "normal" week.** Week 7 (championship divisions and Einstein) drops slightly to the low 70% range, but it does not collapse. Even at the highest stakes events, a model trained on prior weeks can still get roughly 3 out of 4 matches correct.

4. **Model differences are small but real.**
   - **LightGBM** has a slight edge in weeks 1-3.
   - **XGBoost** pulls ahead in weeks 4-6.
   - **Logistic regression** is consistently only 1-2 percentage points behind, despite being much simpler and faster.

From a practical standpoint, this means event organizers or teams do not have to worry that "week 3 is unpredictable" or "district championships are a different universe." Despite Championship accuracy dropping slightly it is still fairly consistent. This drop can be explained due to it being a higher level and having more evenly matched teams. Furthermore, it is a stressful event and there are many external factors that can cause upsets. Overall though, a single model trained in a rolling fashion generalizes well across the entire regular season. 

---

## 4. Event-Level Accuracy: Where Predictions Shine and Struggle

To understand how these weekly averages break down, I computed per-event accuracy for each model. Each row in `event_accuracy.csv` aggregates all matches from a single event (e.g., `2025casj`) and computes the fraction of matches correctly predicted.

Some patterns stand out:

- **Most events fall between 70% and 85% accuracy.**  
  For example:
  - At **2025 CASJ**, LightGBM hits ~84% and logistic regression ~83%.
  - At **2025 TXWAC**, accuracies cluster around 72-74%.
  - District championship style events such as **2025 ONCMP2** reach ~86-87% accuracy.

- **Outliers are small events or special cases.**  
  Events with very few matches (e.g., `2025necmp` with 8 matches, or tiny "cmp" subsets with 2 matches) show extreme accuracy values (0%, 50%, or 100%) simply because the sample size is so small based on how I grouped the competition keynames. (I should probably look into grouping better with non standard naming formats)

- **No single model dominates every event.**  
  In many regionals, XGBoost edges out LightGBM; in others the order reverses. Logistic regression is often within 1-3 percentage points of the best model. This is important: it suggests that *most* of the predictive signal is captured by the features themselves, not by a specific fancy algorithm.

Taken together, the event level results say: **for a typical regional or district event, a reasonable model can correctly call the winner in about 3 out of 4 matches using only pre-event data**. For scouting purposes, this is strong enough to be actionable, but not so strong that humans can stop paying attention.

---

## 5. What Drives Predictive Power?

Rather than obsessing over the exact coefficients or tree splits, the feature experiments point to a few broad themes:

- **Overall scoring strength matters most.**  
  Alliance total score averages (historical) are among the strongest predictors. This lines up with intuition: past ability to score tends to translate into future ability to score.

- **Distribution within the alliance is informative.**  
  Features like `max_total`, `top2_total`, and `std_total` help distinguish:
  - alliances carried by a single elite team (high max, high std),
  - from well balanced alliances (high top-2, lower std).  
  Both can win, but they behave differently, and the models exploit that.

- **Recent form is worth tracking but not a silver bullet.**  
  Rolling "last 5 matches" features add a modest amount of signal, especially early in the season when same year performance is scarce. However, they do not dramatically change overall accuracy and instead only helps to pick up on signals such as robot malfunctions that are rare, which suggests that long term performance history remains the dominant factor.

- **Some features are noise or leak information.**  
  After feature importance analysis, variables like **week**, **year**, **endgame points**, and **foul points** cotributied little. Removing them simplified the model without hurting accuracy.

A key takeaway for teams is that **consistent scoring and alliance balance** matter more than any one "flashy" component score. A robot that is solid across the whole game and fits well with partners is easier for the model to trust and likely easier for human scouts to trust as well.

---

## 6. Comparison to External Benchmarks

Community tools like **Statbotics** report event-level accuracies typically ranging from **70% to 95%**, depending on the event and the maturity of their EPA metrics. My best models sit around **75-76% overall**, with many events in the 75-85% band and a few edging higher.

This means:

- The approach is **competitive with the lower to mid range** of current community tools, despite:
  - using generic metrics and models,
  - not hand tuning per year game specific features,
  - and enforcing a strict rolling evaluation that mimics live predictions.

- There is **clear room for improvement** if I incorporate richer features: game specific auto/teleop breakdowns, schedule strength, or more nuanced measures of alliance play styles.

This shows that despite being slightly worse than a community model being worked on for a couple of years with extensive rsearch and analysis, a simple model can match the pace and that these additional game specific features do not contribute a significant amount.

---

## 7. How Teams and Organizers Could Use This

Given a model that achieves ~75% live accuracy with event level breakdowns, here are some concrete applications:

1. **Pre event prep for teams.**
   - Before traveling, a team could simulate likely qualification schedules and estimate the distribution of win probabilities they will face.
   - The model can highlight which alliances they are heavily favored in and which ones are coin flips, guiding where to focus limited practice time.

2. **In event scouting augmentation.**
   - A simple dashboard could show, for each upcoming match, the model’s predicted win probability alongside human scout notes.
   - Large discrepancies ("model says 80% win, scouts are nervous") are red flags that something qualitative is missing perhaps an unreliable mechanism or a newly fixed robot.

3. **Post event analysis and coaching.**
   - Teams could review events where the model confidently predicted wins that turned into losses (and vice versa) to understand how on field execution and strategy deviated from statistical expectations.

4. **Event level diagnostics.**
   - Organizers can identify events with unusually low predictability (e.g., lots of upsets) and investigate whether that reflects healthy parity, structural imbalance, or issues like field faults and schedule quirks.

5. **Educational tooling.**
   - Because the models are based on interpretable features, they can be used in workshops to teach students about regression, classification, and evaluation in a similar context of competition styled datasets.

---

## 8. Limitations and Future Directions

This analysis is a solid first pass, but several limitations remain:

- **Single season focus.**  
  The rolling evaluation centers on 2025. Extending the same methodology backward (e.g., training on 2013-2018 and predicting 2019) would test how robust the approach is to game changes.

- **No explicit modeling of schedule strength or alliance selection.**  
  The current features treat each match independently. Incorporating opponent strength, draft position, and schedule balance could sharpen predictions and reveal additional sources of inequity.

- **Feature engineering left on the table.**  
  I deliberately avoided more exotic but potentially powerful features team postal codes, stability/volatility metrics, or competitiveness measures because of time and data leakage concerns. These are promising directions for future iterations if I can resolve those issues.

- **Runtime and complexity.**  
  Even after major optimizations, full match by match rolling evaluation with gradient boosting still takes on the order of about an hour on my device. Of course with complete version where it can quickly download the recent event match data to add into the dataset and only have to predict the next few matches will only take a fraction of the full runtime of an entire season.

Despite these limitations, the core message is clear: **FRC match outcomes are substantially predictable using transparent, data driven models, and that predictability can be harnessed to improve scouting, strategy, and understanding of the competition as a whole.**

---

## 9. Conclusion

This project shows that with carefully engineered team and alliance features, a relatively simple modeling pipeline can:

- Achieve **75-76% live prediction accuracy** across the 2025 season,
- Deliver **stable performance** across weeks and events,
- Provide **actionable insights** for teams and organizers,
- And stay competitive with established community tools.

The real value is not just in calling who wins, but in turning years of FRC data into a lens on how competitive robotics actually works and giving students a hands on example of applied data science in a domain they care about.
