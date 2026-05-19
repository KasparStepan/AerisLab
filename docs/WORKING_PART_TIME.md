# Working Part-Time on AerisLab — A Reality Check

**Context:** notes captured from a working session on 2026-05-14 about whether 1-2 days a week is enough for a project of this scope, and how to make part-time work actually work. For Štěpán, a PhD student with limited weekly time available.

---

## The headline answer

**No, it's not a problem.** Part-time is the normal mode for PhD code, not the exception. Most PhD students work on their core code 1-3 days a week alongside teaching, classes, FSI work, supervisor meetings, and life. The work plan's calendar estimates already assume part-time pace.

But part-time has specific challenges that full-time doesn't, and being aware of them changes how you should work.

---

## What's actually different about part-time

### The big one: context loss between sessions
When you work daily, the problem is in your head. When you work Tuesday and again the following Tuesday, six days have passed — you've forgotten where you were, what you were debugging, what you decided yesterday. The first 30-60 minutes of every session goes to *re-warming-up* before any real work happens.

This is the single biggest tax on part-time work. A 5-hour session has maybe 4 hours of effective work. **Two 5-hour sessions a week ≠ one 10-hour session a week ≠ five 2-hour sessions a week** — they all give different amounts of usable progress.

### Slower compound learning
Skills build faster with daily exposure. Daily Python users get fluent in patterns much faster than once-a-week users — not because they're smarter, but because the synapses are reinforced before they fade. At 1-2 days a week, you'll plateau slower. Not impossible to learn, just slower.

### Fewer "shower thought" breakthroughs
Background processing — the "I figured it out in the shower" mechanism — happens better when the problem is actively in your head. With week-long gaps, the problem fades from background processing.

### Momentum is fragile
Skip two weeks (illness, conference, holiday) and motivation tanks hard. Easy to slide from "I work on it twice a week" to "I haven't touched it in a month."

---

## What part-time gives you (real advantages)

- **Forced prioritization.** You can't do everything. You're forced to pick what matters. Full-time devs often spend weeks on things that turn out not to matter; you can't afford to.
- **Sustainability.** PhDs are marathons (4-5+ years). Burnout kills more PhDs than time pressure does. Working sustainably matters more than working maximum hours per week.
- **Time for ideas to mature.** A week away from the code sometimes surfaces better architectural insights than a week buried in it. The work plan's "Move 1" came partly from that kind of stepping-back.
- **Reduced perfectionism.** You don't have time to over-engineer. You ship the thing that works and move on. Often this is the *right* call.

---

## Tactics that matter much more for part-time work

These are the difference between "1-2 days/week works fine" and "1-2 days/week is frustrating."

### End-of-session ritual is non-negotiable
Before you stop for the week:
- Commit everything (even WIP, even broken — `git commit -m "WIP: P0-T1, halfway through reproducer"`).
- Write 3-5 lines in a `next_session.md` file at the repo root: "Where I am, what's the next concrete step, what's blocked." Not your learning log — a short *to-do for next week's you*.
- Push to GitHub.
- Close all editor tabs cleanly.

The 10 minutes you spend doing this saves an hour next session.

### Pick tasks that fit your session length
The work plan tags tasks XS/S/M/L/XL. For 1-day sessions, target XS or S tasks. For 2-day sessions, XS, S, or M. **Don't start an L task on the last hour of a session** — you'll lose more to context-switching than you'll gain by starting early.

### One task per session, no parallel work
Full-time devs can have 2-3 things in flight. You can't — context-switching cost is too high. Pick one task at the start of the session, work it to a stopping point, ship it.

### Front-load the heavy thinking
Hardest cognitive task in the first two hours of your session. Documentation / cleanup / commit messages in the last hour. The order matters more than the total time.

### The learning log is more important for you, not less
When sessions are spaced apart, written notes carry more weight. Re-read last week's entry at the start of every session.

### Don't trust your memory across sessions
What feels obvious on Tuesday will be forgotten next Tuesday. **Write everything down.** Decisions, why-I-chose-X-over-Y, what-I-tried-that-didn't-work. The cost of writing is small; the cost of re-deriving the same conclusion is large.

### Use AI to re-load context fast
Start of session: paste the last few commit messages and `next_session.md` into Claude, ask it to summarize where you left off. Cheap, fast, recovers your mental state in 5 minutes instead of 60.

---

## What your timeline actually looks like

Translating the work plan to your pace (assume 1.5 sessions/week, ~5 hours each = 7-8 hours/week of real work):

| Phase | FTE estimate | Your calendar |
|---|---|---|
| Phase 0 (hotfix critical bugs) | ~3 days | **2-3 weeks** |
| Phase 1 (hygiene + atmosphere) | ~2 weeks | **2-3 months** |
| Phase 2 (the 5 architecture moves) | ~6 weeks | **6-9 months** |
| Phase 3 (joints, Parquet, YAML, CLI) | ~3 weeks | **3-4 months** |
| Phase 4 (ML/FSI integration) | ~10 weeks | **10-14 months** |

Phases 0-3 (the engine becomes a real platform): roughly **12-18 months** at your pace.

Phase 4 (ML, the thesis novelty): another **10-14 months** on top of that.

**Total to a thesis-grade engine + working ML pipeline: 2-2.5 years at 1-2 days/week.** For a 4-5 year PhD, that fits with room to spare for actually *running* the experiments, writing papers, and dealing with life.

This is *exactly the timeline you should expect* for a PhD code project done right alongside everything else a PhD student does. **Don't measure yourself against a hypothetical full-time developer** — measure against a PhD student doing similar work, and you're on track.

---

## When to worry

Part-time is fine. **Inconsistency is the killer.** A few warning signs:

- "I haven't touched it in 3+ weeks" → momentum is gone; harder to restart than to maintain.
- "I keep starting tasks but never finishing them" → tasks too big for your sessions; pick smaller ones.
- "Every session I waste an hour figuring out where I was" → end-of-session ritual broken; tighten it.
- "I'm avoiding the project because it feels overwhelming" → the work plan feels too long; ignore it for a month and just do anything you find interesting in the codebase. Reignite the spark first; structure later.

If those happen, don't push harder — change your approach. The cost of bad weeks is much lower than the cost of burning out.

---

## One more thing: the FTE fiction

Your "full-time-equivalent week" is a fiction. Even a "full-time" PhD developer gets ~3-4 hours of deep coding per day after meetings, email, lunch, breaks, and the unavoidable context-switching costs. So a 5-day full-timer gets maybe 15-20 hours of real coding work per week. **You at 1-2 days getting 7-10 hours focused is closer to ½ that, not ⅕ of that.** The gap is smaller than it looks.

---

## Short version

- Not a problem.
- Be deliberate about end-of-session rituals.
- Pick tasks that fit your session length.
- One task per session.
- Front-load the heavy thinking.
- Trust the timeline; ~2-2.5 years to thesis-ready is the right calibration.
- Worry about inconsistency, not insufficient hours.
