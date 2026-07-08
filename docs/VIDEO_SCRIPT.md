# Demo Video Script — 2:30 target runtime

Goal: a recruiter or reviewer who has 150 seconds and no context should
finish this video understanding what you built, that it actually runs,
and one specific finding worth remembering. Screen recording + voiceover;
no editing skill required beyond cuts.

Record with OBS Studio (free) at 1080p60. Record each shot separately —
much easier to redo one bad take than the whole video.

---

## Shot 1 — Hook (0:00–0:12)
**Screen:** `benchmarks/plots/robustness_comparison.png` full-frame.
**Say:**
> "I benchmarked five path planners, five controllers, and three state
> estimators for a ground robot — and found that below a certain point,
> the controller barely matters. Here's why."

## Shot 2 — The pipeline (0:12–0:30)
**Screen:** `architecture_flow.png`, slow scroll top to bottom.
**Say:**
> "The full pipeline: noisy sensors, sensor fusion, path planning, path
> tracking control, and a learning layer that adapts the controller
> online — all open source, all with measured numbers, no simulation
> hand-waving."

## Shot 3 — Live terminal run (0:30–0:55)
**Screen:** terminal, `make bench` running, scrolling output.
**Say (over the top, don't wait for it to finish — cut when the table appears):**
> "Everything here is reproducible with one command. No cherry-picked
> screenshots — this is the actual run producing the actual numbers."
**Cut to:** the printed results table in the terminal, hold 2 seconds.

## Shot 4 — Estimation result (0:55–1:15)
**Screen:** `estimation_comparison.png`.
**Say:**
> "First finding: EKF and UKF tie at 8.7 centimeters of position error.
> The fancier factor-graph smoother actually loses here — it needs loop
> closures to pay for its extra compute, and this scenario doesn't have
> any. Knowing when NOT to use the expensive method is half of good
> engineering."

## Shot 5 — Planner result (1:15–1:35)
**Screen:** `planner_comparison.png`, point cursor at the Hybrid A* trajectory.
**Say:**
> "For planning, Hybrid A* wins for car-like robots specifically — it
> searches in position AND heading, so every path segment is actually
> drivable. Plain A* is faster but produces paths a real steering system
> can't follow smoothly."

## Shot 6 — Controller + learning result (1:35–2:00)
**Screen:** `full_controller_dashboard.png`, point at the velocity panel.
**Say:**
> "This is the core result: a controller that learned, in under two
> minutes of training, to slow down before curves and speed up on
> straights — beating a hand-tuned MPC by twelve percent on accuracy and
> forty percent on smoothness. That dotted line is the learned speed
> schedule."

## Shot 7 — The robustness finding (2:00–2:20)
**Screen:** back to `robustness_comparison.png`.
**Say:**
> "But here's the twist. Once I fed every controller realistic sensor
> noise instead of perfect ground truth, MPC, Stanley, and the learned
> controller all converge to the same six-and-a-half-centimeter error
> floor — set by localization, not control. Past a certain point, a
> better controller buys you nothing. Better localization does."

## Shot 8 — Close (2:20–2:30)
**Screen:** GitHub repo README, scrolled to the top.
**Say:**
> "Full code, Docker image, ROS2 launch files, and the technical report
> are linked below. Thanks for watching."

---

## Recording checklist
- [ ] Regenerate all plots fresh right before recording (`make report`) —
      never show stale/manually-edited figures
- [ ] Terminal font size ≥ 18pt, dark theme, clean prompt (no clutter)
- [ ] Mute notifications, close unrelated tabs
- [ ] Record voiceover as ONE continuous read per shot, not sentence by
      sentence — sounds far more natural
- [ ] Export captions (auto-generate on YouTube/LinkedIn, then correct
      technical terms manually: "EKF", "MPC", "Hybrid A*" are commonly
      mis-transcribed)

## Where to post
1. **LinkedIn native video** (not a YouTube link) — 3–5x the reach of a
   link-out post per LinkedIn's own algorithm behavior
2. **YouTube** (unlisted is fine) as the canonical long-lived link for
   your resume/portfolio site
3. Pin the GitHub repo link in the first comment, not the caption —
   captions with links get deprioritized by LinkedIn's algorithm
