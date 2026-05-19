# Learning Priorities for an Engineer-Who-Codes

**Context:** notes captured from a working session on 2026-05-14, about whether coding is worth investing in for a mechanical/aerospace engineer, and what specifically to focus on. For Štěpán, who wants to stay in mechanical/aerospace engineering long-term but doesn't know yet which industry.

---

## Is coding worth learning if you're an aerospace engineer?

**Yes — and not as a "side skill." It's a fundamental tool now, like math.**

30 years ago, an aerospace engineer who couldn't operate a CAD program was at a disadvantage. 15 years ago, someone who couldn't use Excel/MATLAB was. Today, an engineer who can't drop into Python to wrangle CFD outputs, run Monte Carlo studies, build a quick digital twin, or process test data is increasingly handicapped. Not blocked from a career — but at a real disadvantage versus peers who can.

The reason: engineering has *become* computational. The daily work of modern aerospace/mechanical engineering — CAD, FEA, CFD, optimization, digital twins, ML surrogates, control systems, data processing from test campaigns — is all mediated by code. Even when the engineer doesn't write the code, they're using tools written by people who do, and the engineers who can dip into Python *when the GUI doesn't do what they need* are dramatically more effective.

**You don't call math a "side skill" for an engineer. Coding has moved into the same category** — it's a tool of the trade, not a separate discipline.

### Where it pays back across possible futures

Pretty much wherever you end up:

- **Aerospace primes** (Airbus, Boeing, Lockheed, ESA, NASA, SpaceX): every one uses computational tools heavily. The engineers who can code get on the interesting projects.
- **New space** (rocket startups, satellite companies): they explicitly hire people who blur the line between aerospace engineer and software engineer. Engineers-who-code are essentially the entire workforce at companies like Rocket Lab, Relativity, Astranis.
- **Defense / national labs**: scripts and simulations are the daily work.
- **Automotive** (especially anything autonomous, EVs, controls): coding is non-negotiable.
- **Energy** (wind, nuclear, fusion startups): simulation + control = code.
- **Consulting**: ability to build a custom analysis quickly is gold.
- **Academia**: obviously.

You'll find very few interesting aerospace roles in the next 20 years where coding skill is wasted.

### Don't become a software engineer

Important caveat. You want to stay a mechanical/aerospace engineer — **don't drift into being a programmer.** The career rewards for engineers-who-also-code are higher (in their domain) than for engineers who pivot fully to SWE roles. Stay engineering-first; let coding be the multiplier.

This means: don't spend 5 years learning category theory, data structures, system design at FAANG-interview level. You don't need it. The aerospace engineer's coding bar is different — you need numpy fluency, scipy, plotting, basic git, debugging skill, comfort reading and modifying others' code, ability to use the command line without fear. That's most of it. It's a much lower bar than CS-major bar, and it pays back enormously in the engineering context.

**Get good enough to be dangerous; don't optimize for being a programmer.**

### The AI angle

With LLMs, the practical bar for "can this person code" has shifted. You don't need to memorize syntax anymore — that's table stakes. What you need is **engineering judgment about what to build, whether the code is right, and how to debug it when it isn't**. The aerospace engineer who can prompt AI well, read its output critically, and verify the physics — that person is the new force multiplier.

Your domain knowledge is the moat. AI can write Python all day; it can't know what an aerospace engineer actually needs to compute, what's physically reasonable, what assumptions are dangerous, what failure modes matter. Combining that domain judgment with the ability to translate it into running code (with AI as your typist) is a combination that's currently rare and increasingly valuable.

### Your specific case

You're already past the "should I learn coding" question, whether you realize it or not. Your PhD project — building a 6-DOF simulator with constraint solvers, ML-FSI coupling, validation against experiments — is fundamentally a coding project that happens to be about aerospace. The papers you'll be able to write, the experiments you'll be able to run, the collaborations you'll be able to do — all gated by your code skill.

After you finish this PhD, you'll be in a small group: aerospace engineers who have shipped a real simulation tool from scratch. **That's a differentiator on a CV that opens doors that pure-domain-expertise wouldn't.**

---

## Calibrating what to learn

You initially listed: architecture, security, verification, testing.

That direction is mostly right (high-level engineering thinking is the durable, transferable axis). But the list itself needs adjustment.

### Drop "security" from your priority list

For scientific simulation code, security is a near-zero concern. It matters when you're writing:
- Web apps that take user input
- APIs that handle credentials
- Anything that processes untrusted data
- User-facing software with personal data

For your situation — a Python package that runs simulations from your own scripts, on your own machine, producing CSVs and plots — security is essentially irrelevant. Spending time on it is time not spent on things that *do* matter.

If you ever build a web dashboard for results, or an API that other people query, *then* learn enough security to not embarrass yourself. Until then, ignore it.

### Don't add fashionable web/SWE topics either

Not high-leverage for you:
- Microservices, container orchestration in anger
- Distributed systems
- Database design (beyond basics)
- Web frameworks
- Mobile / UI development
- High-performance LAPACK internals, GPU programming at scale

These belong on someone else's curriculum, not yours.

---

## The 10 high-leverage skills for an aerospace engineer-who-codes

Ranked roughly by payoff per hour invested.

### 1. Reading code well
The 80/20 reality. You'll spend more time reading code than writing it for the rest of your career. Get fluent.
- **Where to learn:** HOW_TO_WORK_EXPANDED §8.

### 2. Debugging skill — specifically scientific debugging
Not "find the missing semicolon" debugging. The harder kinds: NaN propagation, numerical drift, off-by-one in array indices, "the test passes but the physics is wrong." Scientific code has its own bug taxonomy. Fluency here is rare and valuable.
- **Where to learn:** HOW_TO_WORK_EXPANDED Part VI; doing every CRIT-* fix in the work plan.

### 3. Testing discipline
Especially for numerical code. The pattern is different from web-app testing — you care less about mocking and more about: does the simulation conserve energy? Does it match the analytical solution? Does the new commit reproduce the old commit's force-time curve to 4 decimals? The verification suite in your existing tests is exactly this style — learn it deeply.
- **Where to learn:** HOW_TO_WORK_EXPANDED Part VII; tests/verification/ in the codebase.

### 4. Architecture (the small kind)
How to organize a 5,000-line project so that 5 years from now you can still find what you need and add features without breaking everything. *Not* how to design distributed microservices. The whole "Move 1-5" section of your review is this kind of architecture.
- **Where to learn:** *A Philosophy of Software Design* (John Ousterhout) — short, practical, exactly your level. Plus doing the Phase 2 architecture work.

### 5. Verification & Validation (the V&V scientific sense)
Different from "testing." V is "did I solve the equations correctly?" V is "did I solve the right equations?" Your verification tests (energy conservation, pendulum period, Dzhanibekov) are V; reproducing experimental drop-test data is V. This is *the* methodology of computational science. Learn the distinction; learn the standards (NASA-STD-7009A, ASME V&V 10/20). Scarce skill in academia and industry.
- **Where to learn:** any V&V chapter in a scientific computing textbook; the work plan tasks P1-T15 (convergence) and P3-T11 (literature replication).

### 6. LLM fluency
The skill compounds — every month it gets more important.
- **Where to learn:** HOW_TO_WORK_EXPANDED Part IX; daily use with Claude.

### 7. Numerical methods literacy
Not "implement an ODE solver" — but enough to know *why* Radau is different from RK45, what "stiff" means, when to worry about quaternion drift, what symplectic integration actually buys you.
- **Where to learn:** see `NUMERICAL_METHODS_LEARNING.md` for the dedicated path.

### 8. Reproducibility
Manifest.json, seeded RNG, pinned dependencies, container per published result. Hugely undervalued. The day a reviewer asks "can you regenerate Figure 4.7?" and you can answer "yes, here's the script and the seed and the version" is the day you've separated yourself from 80% of researchers.
- **Where to learn:** doing the work plan tasks (P1-T14 manifest, P3-T8 YAML, eventual containerization).

### 9. Performance / profiling — but only when it matters
Don't learn until you actually need it. When you do need it, the specific skill is "I can read a flame graph and identify what to fix."
- **Where to learn:** HOW_TO_WORK_EXPANDED Part X.

### 10. Git fluency beyond the basics
Reflog, bisect, rebase, cherry-pick. Once a quarter you'll have a problem where these skills save you a day.
- **Where to learn:** HOW_TO_WORK_EXPANDED Part VIII; survive a few git disasters.

---

## The meta-skill: engineering judgment

Knowing *which* of those 10 to apply when. You don't need to be world-class at all 10. You need to recognize "ah, this is a numerical drift problem, time to add constraint-violation logging" or "ah, this needs a regression test, not more debugging." That recognition is what you're really building.

The meta-skill of engineering judgment can't be taught directly — it builds from doing the work, screwing up, and noticing the patterns. The work plan is structured to give you that — every task is a chance to practice making the call.

---

## How to learn these on this project

**Don't take courses.** The work plan already exercises every one of those 10 skills. Specifically:

- **Reading**: Phase 0/1 read all the existing code.
- **Debugging**: every CRIT-* fix.
- **Testing**: every "write a regression test" task.
- **Architecture**: the 5 architectural moves in Phase 2.
- **V&V**: convergence tests (P1-T15), literature replication (P3-T11).
- **LLM**: every session you sit with Claude.
- **Numerical methods**: implementing the integrators, dealing with stiff-solver behavior, atmosphere model formulas.
- **Reproducibility**: manifest.json (P1-T14), seeded RNG, eventual Docker.
- **Performance**: Phase 3 benchmarks.
- **Git**: every commit.

Two years of doing the work plan will make you genuinely skilled in all 10. **No course covers what doing the work covers.**

---

## Short version

- Drop security.
- Don't add fashionable web/SWE topics.
- The right list is up there ↑. Compact and high-leverage.
- Don't learn from courses; learn by doing this project deliberately.
- Engineering judgment is the meta-skill — that's what you're really building.
- Treat coding as fluency in a tool, not as a separate career.
- Stop describing it as a "side skill" — that framing undervalues it and may unconsciously cap your investment.
