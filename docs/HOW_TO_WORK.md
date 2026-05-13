# How to Work on AerisLab — A Practical Guide

**Who this is for:** you, sitting down to actually do the work. Not a tutorial on programming generally — a guide on how to *work through this project*, step by step, building good habits as you go.

**What this is not:** a programming textbook. There are 10,000 of those. This is the stuff you only learn by working with someone more experienced for a few months, condensed into one document.

**Companion files:**
- `AERISLAB_REVIEW_PLAIN.md` — the *why* (what's broken, what should be better)
- `WORK_PLAN.md` — the *what* (the ordered to-do list)
- This file — the *how* (the process and mindset)

---

## Table of Contents

1. [Mindset — how to think about this](#1-mindset--how-to-think-about-this)
2. [Setting up your day](#2-setting-up-your-day)
3. [The work loop — every task follows this pattern](#3-the-work-loop--every-task-follows-this-pattern)
4. [How to read code (yours or anyone else's)](#4-how-to-read-code-yours-or-anyone-elses)
5. [How to debug when something goes wrong](#5-how-to-debug-when-something-goes-wrong)
6. [How to write and run tests](#6-how-to-write-and-run-tests)
7. [Git for solo work — your safety net](#7-git-for-solo-work--your-safety-net)
8. [Writing good commit messages](#8-writing-good-commit-messages)
9. [Using Claude Code (or any AI assistant) well](#9-using-claude-code-or-any-ai-assistant-well)
10. [Keeping a learning log](#10-keeping-a-learning-log)
11. [Common traps and how to escape them](#11-common-traps-and-how-to-escape-them)
12. [Worked example: doing task P0-T1 from the work plan, step by step](#12-worked-example-doing-task-p0-t1-from-the-work-plan-step-by-step)
13. [What "good" looks like — habits to build over the next year](#13-what-good-looks-like--habits-to-build-over-the-next-year)

---

## 1. Mindset — how to think about this

A few attitudes that separate productive engineers from frustrated ones. None of these are unique to coding — they're how craftspeople work.

### Be a detective, not a magician
Before you change anything, **understand what's actually happening**. Magicians wave their hands and hope. Detectives gather evidence. When a test fails or a simulation gives a weird number, your first move is *not* to start changing code — it's to ask "what is the code actually doing right now, and what did I expect it to do?"

### Small steps beat big steps
Especially when you're learning. Make one small change, run the tests, see what happened. Then the next small change. This is slower per change but much faster overall, because when something breaks, you know it was the last thing you did.

The opposite — making 10 changes at once and then trying to figure out which one broke the tests — costs hours.

### Reading is 80% of the job
Beginners think coding is about writing code. It isn't. It's about reading code (yours, other people's, your own from 6 months ago) and understanding it well enough to change one thing without breaking ten others. Get comfortable reading.

### Don't be afraid to break things
Git makes almost everything reversible. The worst case for "I changed something and now nothing works" is `git checkout .` (throw away everything since the last commit). The worst case for `git reset --hard HEAD~1` is "I lost the last commit," which only matters if it was important — and even then, `git reflog` can usually rescue it.

The thing you should fear is **uncommitted work that you might lose**. Commit often. Even half-finished things, in a branch, are safe.

### "I don't understand this yet" is fine
You're allowed to write `# I'm not sure why this works, come back later` in a comment. You're allowed to do a task with help and then redo it without help to internalize it. You're allowed to spend an hour reading instead of writing. The PhD-time clock is real, but skipping understanding to "save time" creates 10× more time-loss later when bugs surface from things you didn't understand.

### Slow is smooth, smooth is fast
Borrowed from soldiers and surgeons. Rushing creates mistakes that take longer to fix than the time you saved. Take the extra two minutes.

---

## 2. Setting up your day

Before you sit down to write any code, do this little ritual. It takes two minutes and saves you hours.

```bash
cd /home/kaspar/Projects/AerisLab
git status                              # Where are we?
git log --oneline -5                    # What did we do recently?
venv/bin/pytest --tb=short             # Do the tests pass right now?
```

**Why:** You want to start every session in a known-good state. If the tests are already failing before you change anything, you'll spend an hour debugging a problem that wasn't caused by what you're about to do.

If the tests *are* failing when you start, **fix that first** (or revert to the last commit where they passed). Don't build on a broken foundation.

Then, before each task:

```bash
git checkout ClaudeHelp                    # or main, depending on your flow
git pull                                   # if there's a remote
git checkout -b crit-1-scenario-connect    # new branch for this task
```

Each task gets its own branch. The branch name = the task ID from `WORK_PLAN.md`. This is your safety net — if the task goes sideways, you can `git checkout ClaudeHelp` and try again with no consequences.

---

## 3. The work loop — every task follows this pattern

This is the rhythm. Memorize it. Every single task in `WORK_PLAN.md` should be done this way.

```
1. Pick ONE task from the work plan
2. Create a branch named after the task
3. Read the task description twice — slowly
4. Read the file(s) it touches
5. Form a hypothesis: "I think the change is X, in line Y of file Z"
6. Make the smallest possible change that implements your hypothesis
7. Run the tests
8. Run the verification command from the task
9. If both pass: commit. If not: investigate, then back to step 6.
10. Mark the task done in WORK_PLAN.md (change [ ] to [x])
11. Merge the branch back to ClaudeHelp
12. Take a 10-minute break before the next task
```

**Two rules that look strict but save you grief:**

- **One task per branch.** Don't bundle "Fix Bug #1" with "while I'm there, also clean up the whitespace." Bundling makes the diff hard to review and the commit message hard to write.
- **Don't skip step 7 (run the tests).** Even if you "only changed a comment." You'll be surprised how often "only" turns out to be wrong.

---

## 4. How to read code (yours or anyone else's)

Reading code is a skill. Here's how to get better at it.

### Start at the entry point and follow the call chain

When you want to understand "what does this script actually do," start at the very top:

```python
if __name__ == "__main__":
    run_example()
```

OK, what's `run_example()`? Find it. Read it top to bottom. When it calls something (`Scenario(name="01_simple_drop")`), pause. Don't dive in immediately. First, ask yourself: "what do I think this does, based on the name and the arguments?" Then verify by jumping to the definition.

In your editor, "go to definition" (often `F12` or `Ctrl+Click`) is your friend. So is `grep`:

```bash
grep -rn "def Scenario" src/    # find where Scenario is defined
grep -rn "Scenario(" src/       # find everywhere Scenario is used
```

### When you hit something you don't understand, write a note

Don't try to understand everything in one pass. Make a note ("come back to this — `quat_normalize` magic"), keep going to get the overall shape, then come back and dig in.

I keep a `notes.md` file open during reading sessions. It looks like:

```
- World.add_system: registers bodies + constraints from the system. But constraints
  is captured by reference at this point, not later? Need to check.
- assemble_system: returns Minv, J, F, rhs, v. What's `v`? oh — generalized
  velocities, stacked [v_lin, omega] per body.
- Why does ParachuteDrag have its own activation logic AND Parachute the component
  also has activation logic? Are they coordinated? <-- check this
```

This is exactly how I found Bug #1 — by writing a note that didn't make sense, then chasing it down.

### Read in three passes, not one

1. **Skim** — read the docstrings, the function signatures, the class names. Get the shape.
2. **Trace** — pick one realistic input and follow it through the code, mentally executing.
3. **Detail** — when you understand the shape, dig into the bits that surprised you.

---

## 5. How to debug when something goes wrong

A specific recipe. Use this every time, even if you think you know what's wrong.

### Step 1 — reproduce reliably

You cannot fix what you cannot reproduce. If a bug happens "sometimes," your first job is to find a recipe that makes it happen 100% of the time. This might mean writing a 5-line script that triggers it.

```python
# crash_repro.py — a temporary file, throw away later
from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload, Parachute

p = Payload(name="cap", mass=50.0, radius=0.4, position=[0,0,2000])
c = Parachute(name="main", mass=5.0, diameter=12.0, model="knacke",
              activation_altitude=1500, position=[0,0,2000.5])
sc = Scenario("repro").add_system([p, c]).connect(p, c, "tether", 10.0)
sc.run(duration=60.0)
print("payload final z:", sc.world.bodies[0].p[2])
print("payload final v:", sc.world.bodies[0].v[2])
```

If the bug shows up here, you have a reproducer. If not, change the script until it does.

### Step 2 — read the error message *all the way to the bottom*

Python tracebacks look scary. The actual error is at the **bottom**. The lines above it are the call chain that led there.

```
Traceback (most recent call last):
  File "crash_repro.py", line 8, in <module>
    sc.run(duration=60.0)
  File "src/aerislab/api/scenario.py", line 172, in run
    self.world.integrate_to(solver, t_end=duration, log_interval=log_interval)
  ...
ValueError: Cannot invert inertia tensor: Singular matrix
```

The error is `ValueError: Cannot invert inertia tensor: Singular matrix`. The traceback tells you it happened during `world.integrate_to`. Now you know where to look.

### Step 3 — narrow down with `print` statements

The simplest debugging tool. Add a print right before the line where you think the bug is:

```python
print(f"DEBUG: about to integrate, body.q = {body.q}, |q| = {np.linalg.norm(body.q)}")
```

Run it. Read the output. Did your guess match what's actually happening?

Once the bug is fixed, **delete the print statements**. Don't leave debug prints in committed code.

### Step 4 — when prints aren't enough, use a debugger

```python
import pdb; pdb.set_trace()       # Python pauses here, gives you an interactive prompt
# in the prompt: type variable names to inspect them, type 'n' for next line, 'c' to continue
```

For a serious debugger experience, `ipdb` is nicer (`pip install ipdb`, then `import ipdb; ipdb.set_trace()`).

### Step 5 — bisect when nothing else works

If a long script "works at the start and breaks somewhere," cut it in half. Does the first half work? Then it's in the second half. Cut that half in half. Repeat until you've narrowed the problem to one or two lines.

This sounds slow but takes ~6 cuts to narrow down a 100-line script. Faster than staring at the whole thing.

### What NOT to do when debugging

- **Don't change multiple things at once.** Change one. Test. Change another. Test. If you change three things and one of them fixes the bug, you don't know which one — and you've maybe introduced two new bugs.
- **Don't refactor while debugging.** "While I'm in here, let me also clean up this function" is the path to disaster. Fix the bug. Commit. Then refactor in a separate commit.
- **Don't guess.** Verify. Every time you think "it must be because…" — go check.

---

## 6. How to write and run tests

A test is just a small program that runs your code with known inputs and checks the answer is what you expect. That's it.

### Why bother?

Two reasons:
1. **You can change code without fear.** A passing test suite says "the things I checked still work." Without it, every change is a roll of the dice.
2. **It forces you to think about correctness.** When you write the test, you're forced to ask "what's the *right* answer?" — which is half the battle.

### The structure of a test

```python
# tests/test_my_thing.py
def test_payload_falls_due_to_gravity():
    # Arrange — set up the world
    payload = Payload(name="p", mass=10, radius=0.5, position=[0, 0, 100])
    sc = Scenario("test").add_system([payload])

    # Act — do the thing
    sc.run(duration=1.0)

    # Assert — check the answer
    final_z = payload.body.p[2]
    expected_z = 100 - 0.5 * 9.81 * 1.0**2          # z = z₀ + v·t + 0.5·g·t²
    assert abs(final_z - expected_z) < 0.5, \
        f"expected ~{expected_z}, got {final_z}"
```

The pattern is **Arrange → Act → Assert** (sometimes called "AAA"). Almost every test follows this shape.

### Running tests

```bash
venv/bin/pytest                                          # all tests
venv/bin/pytest tests/test_body.py                       # one file
venv/bin/pytest tests/test_body.py::test_quaternion_normalization  # one test
venv/bin/pytest -v                                       # verbose
venv/bin/pytest -k "drag"                                # only tests with "drag" in name
venv/bin/pytest -m "not slow"                            # skip slow tests
venv/bin/pytest --tb=short                               # shorter tracebacks on failure
venv/bin/pytest -x                                       # stop at the first failure
```

Run the *full* suite (`venv/bin/pytest`) before every commit. It takes ~36 seconds — short enough that there's no excuse.

### What makes a good test?

- **One thing per test.** Don't write a test that checks 10 things. If it fails, you don't know which one broke.
- **Specific failure messages.** `assert x == 5, f"x should be 5, got {x}"` is much better than just `assert x == 5`.
- **Independent.** Test 2 should not depend on test 1 having run. Each test sets up its own world.
- **Tests the *behavior*, not the *implementation*.** Bad: "after add_system, world.constraints has length 1." Good: "after add_system + connect, the bodies stay tethered when simulated."

### When to write a test

- **Before fixing a bug.** Write a failing test that demonstrates the bug. Then fix the bug, and now the test passes. Now you know it's actually fixed, *and* you've added a regression test that catches the bug if it ever returns.
- **When you add a feature.** Test that the feature does what it's supposed to.
- **When you refactor.** Tests prove the refactor didn't change behavior.

---

## 7. Git for solo work — your safety net

Even working alone, git is your most important tool. Not for collaboration — for **time travel**. Every commit is a save point you can return to.

### The five commands you'll use 95% of the time

```bash
git status              # what's changed?
git diff                # show me what changed (line by line)
git add <file>          # stage this file for commit
git commit -m "..."     # save a snapshot
git log --oneline       # show recent commits
```

### The two commands that save you when things go wrong

```bash
git checkout .          # throw away ALL uncommitted changes (be sure!)
git checkout <file>     # throw away changes to one file
```

### Workflow for one task

```bash
# Before starting
git status                                   # confirm clean
git checkout -b p0-t1-scenario-connect       # new branch

# Work on the task
# ... edit files ...
git status                                   # what did I change?
git diff                                     # show me the changes

# Commit a logical chunk
git add src/aerislab/api/scenario.py
git commit -m "Fix CRIT-1: propagate constraint to World in Scenario.connect"

# Run tests, run verification command from work plan
venv/bin/pytest

# Merge back when done
git checkout ClaudeHelp
git merge p0-t1-scenario-connect
git branch -d p0-t1-scenario-connect         # delete the branch (it's merged in)
```

### When you've made a mess

If you've been hacking around and want to reset:

```bash
git status              # check what's about to be lost
git checkout .          # throw away all uncommitted changes
```

If you've committed something you shouldn't have, but you haven't pushed yet:

```bash
git reset --soft HEAD~1   # undo the last commit, but keep the changes staged
git reset HEAD~1          # undo the last commit, keep changes unstaged
git reset --hard HEAD~1   # undo the last commit AND throw away the changes (careful!)
```

If you've completely lost track and need to see what you've done:

```bash
git reflog              # shows EVERY action git has done — even "lost" commits
                        # → find the SHA you want, then `git checkout <sha>`
```

### One ground rule

**Never lose work.** If you're about to do something that scares you (a big rewrite, a `git reset --hard`, a force-push), commit first. Even garbage commits with messages like "WIP, about to try a refactor." A bad commit is infinitely better than lost work.

---

## 8. Writing good commit messages

The commit message is a letter to your future self. In six months, when you wonder "why did I change this?", the message is your only clue.

### The shape

```
Subject line: short summary of what changed (≤72 characters)

Optional body: WHY the change was made. What problem does it solve?
What did you consider and reject? Anything tricky to remember?
Wrap at 72 columns.

Multiple paragraphs are fine. Bullet lists are fine.
```

### Subject line conventions

- **Imperative mood.** "Fix the parachute bug," not "Fixed the parachute bug." Think of it as completing the sentence "If applied, this commit will…"
- **Capitalize the first letter.**
- **No trailing period.**
- **Be specific.** "Fix CRIT-1: propagate constraint to World in Scenario.connect" is better than "fix bug" or "scenario fix."

### Body — the "why"

The diff already shows *what* changed. The body explains *why*. Write the body if:
- The reason isn't obvious from the diff.
- You considered an alternative and rejected it (write down why).
- The change is part of a larger plan (reference the work plan task).
- There's a gotcha future-you needs to know about.

### Examples

**Bad:**
```
fix
```

**Slightly better:**
```
Fix tether bug
```

**Good:**
```
Fix CRIT-1: propagate constraint to World in Scenario.connect

Scenario.connect was adding the constraint to current_system.constraints
after world.add_system had already snapshotted that list. Result: the
KKT solver never saw the tether, so all parachute simulations through
the Scenario API ran with the parachute and payload as independent
falling bodies.

Hotpatch: explicitly call self.world.add_constraint(constraint) at the
end of connect(). The proper structural fix (deriving World.constraints
from systems) is task P2-T11 in the work plan.

Verified: smoke run of examples/scenarios/02_parachute_system.py now
shows touchdown velocity of -8 m/s instead of the bare-payload -56 m/s.
```

That last example is what a "good" commit message looks like for a real bug fix. It will save you 30 minutes of confusion in 8 months.

---

## 9. Using Claude Code (or any AI assistant) well

You're using me to help with this project. A few notes on doing that productively, since AI assistance is a real skill of its own.

### Be specific

- Bad: "fix my code"
- OK: "fix bug 1 in scenario.py"
- Good: "I'm working on P0-T1 from WORK_PLAN.md. The bug is that Scenario.connect doesn't propagate the constraint to World. Read scenario.py and tell me the smallest change that fixes it without breaking the other tests."

The more context you give, the better the answer. Reference the file. Paste the error. State your goal.

### Always paste the actual error

Don't summarize. Don't paraphrase. Copy-paste the entire traceback. The exact wording matters.

### Don't accept code you don't understand

This is the most important rule for learning. If I write something and you can't explain it back in plain words to a hypothetical other person, **don't merge it**. Ask me to explain. Ask me to simplify. Ask me to add comments. The point of working with an assistant isn't to get the work done as fast as possible — it's to learn while getting the work done.

A useful exercise: **do a task with help, then close everything and redo it from scratch.** The second time you'll find what you actually understood vs. what you just copy-pasted.

### Use me for explanations, not just code

Some good prompts:
- "Explain what `quat_derivative(q, omega)` is doing — I get the idea but the math feels magical."
- "Walk me through what happens when I call `Scenario("foo").add_system([p]).run(60)`. Step by step."
- "Why is semi-implicit Euler called 'symplectic'? What does that mean for me practically?"
- "I'm reading `solver.py` and don't understand why we compute `Minv` instead of `M`. Why?"

Use me as a senior coworker you can ask anything without feeling stupid.

### Push back on me

I'm wrong sometimes. If something I suggest doesn't make sense to you, say so. "I don't see how that fixes the bug — when I trace through the code, X happens." Half the time I'll see your point and correct myself. The other half I'll explain better and you'll learn something.

### Don't let me skip understanding for you

If I say "do X," ask "why X?" If the answer is "because reasons," push for the actual reason. The goal isn't to get a working `connect()` method — it's to know how to fix the next bug yourself.

---

## 10. Keeping a learning log

Get a notebook (paper or digital — Notion, Obsidian, plain markdown file, doesn't matter). At the end of each work session, write **two sentences**:

- What was the hardest thing today?
- What's one thing that clicked?

Examples from the kind of session you might have:

> *2026-05-14:* Fixing Bug #1 was easier than I expected once I traced the code with Claude. The hard part was understanding why `world.add_system` captures constraints by value — it's because Python lists are passed by reference but then `for` loop only iterates the current contents. Need to read up on Python references.

> *2026-05-15:* Spent two hours trying to figure out why my new test was failing. Turned out I was modifying a shared fixture between tests — pytest fixtures with `scope="module"` are reused. Lesson: keep fixtures `scope="function"` (the default) unless I have a strong reason. Felt frustrated but I won't make this mistake again.

This habit feels silly for the first two weeks. After two months, you have a record of every concept you struggled with and how you got past it. After a year, you have a personal textbook calibrated to *your* learning. It also helps you notice patterns ("I keep messing up git rebases — maybe I should sit down and learn rebase properly").

A separate `learning_log.md` in your home directory is fine. Don't put it in the project repo.

---

## 11. Common traps and how to escape them

Things that have eaten weeks of other people's lives. Each has a one-line escape.

### "I'll just refactor while I'm fixing this bug"
**Trap:** the bug fix gets entangled with the refactor; tests fail and you can't tell why.
**Escape:** finish the bug fix in one commit. Refactor in a separate commit (ideally a separate branch).

### "I changed something and now nothing works, but I'm not sure what I changed"
**Trap:** you've been editing for 2 hours without committing, can't remember the last working state.
**Escape:** `git diff` shows what's changed since the last commit. `git checkout .` resets everything. Future-proofing: commit every 30 minutes, even if it's "WIP".

### "The tests pass on my machine but fail in CI"
**Trap:** your local environment has something CI doesn't (e.g., a stale install, an environment variable, a test that depends on file order).
**Escape:** `pip install -e . --force-reinstall`; check `pyproject.toml` is right; run tests in a *fresh* venv to mimic CI.

### "I'm 4 levels deep in a debugging rabbit hole and I forget what I was trying to do"
**Trap:** you set out to fix bug A, which led to investigating function B, which led to question C about library D…
**Escape:** when you start, write down on a sticky note "I am trying to fix bug A." Look at it every 20 minutes. If you've been off-track for an hour, stop — make a note of what you found, **commit any progress**, return to the original task. The rabbit hole is now a separate task.

### "This problem is too big, I don't know where to start"
**Trap:** task feels overwhelming, so you do nothing.
**Escape:** make it smaller. "Implement the Model layer" is too big. "Make a stub `AtmosphereModel` class with a single `density(altitude)` method that returns 1.225" is doable in 10 minutes. Start there.

### "I'll just copy this code from somewhere similar"
**Trap:** copied code carries assumptions from its original context. Often it doesn't fit, and the bugs are subtle.
**Escape:** copy is fine. Then read line by line and ask "is this actually right *here*?" Modify until it fits.

### "Let me read all the code first, then I'll know enough to start"
**Trap:** reading code without a goal is endless. There's always more to read. You read for a week and then can't remember any of it.
**Escape:** read with a question in mind ("how does the IVP solver call the parachute model?"). Stop when you've answered the question. Trust that you can come back when you have the next question.

### "Nobody else would write this badly, what's wrong with me?"
**Trap:** everyone else's code looks beautiful in the highlight reels (their final commits). You're seeing your work in real time, including all the messy in-progress stages.
**Escape:** look at git history of any famous open-source project. They have commits like `fix typo`, `wip`, `actually fix it this time`. Programming is messy. Smooth code is the result of iteration, not the initial draft.

### "I should learn more before I attempt this"
**Trap:** "I'll learn X then do Y" turns into never doing Y because there's always more X.
**Escape:** the best way to learn X is to need X for Y. Start Y. When you hit something you don't know, learn it. The motivation of needing it makes learning stick.

---

## 12. Worked example: doing task P0-T1 from the work plan, step by step

Let's actually walk through Bug #1 (P0-T1 in the work plan), narrating the thought process at every step. This is the level of detail you want when you're learning. As you get experienced, you'll skip steps without thinking — but right now, do all of them.

### Step 1 — Open the task description and read it

Go to `WORK_PLAN.md`, find P0-T1. Read it twice. Slowly.

What I want to absorb:
- **What** is the bug (constraint not propagated).
- **Where** it lives (`api/scenario.py`, `core/simulation.py`).
- **Why** it's a bug (the System's constraints are added after World already copied them).
- **The acceptance criteria** (`world.constraints` length matches `system.constraints` length after `connect()`).
- **The verification command** (the bash script with `assert len(...) == 1`).

If anything in the description is unclear, that's where I stop and either re-read the bug explanation in the review, or ask Claude.

### Step 2 — Prep my workspace

```bash
git status                                   # confirm clean
venv/bin/pytest --tb=short                   # confirm tests pass before I start
git checkout -b p0-t1-scenario-connect       # new branch
```

If tests don't pass before I start, that's a *different* problem. I deal with that first or revert to a known-good commit.

### Step 3 — Reproduce the bug myself

I want to see the bug with my own eyes before I fix it. So I run the verification command from the task:

```bash
venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload, Parachute
p = Payload(name='cap', mass=50.0, radius=0.4, position=[0,0,2000])
c = Parachute(name='main', mass=5.0, diameter=12.0, model='knacke',
              activation_altitude=1500, position=[0,0,2000.5])
sc = (Scenario(name='check').add_system([p, c])
      .connect(p, c, type='tether', length=10.0))
print('System constraints:', len(sc.current_system.constraints))
print('World constraints: ', len(sc.world.constraints))
"
```

Output:
```
System constraints: 1
World constraints:  0
```

Good. The bug is real and I can reproduce it.

### Step 4 — Read the code with the bug in mind

I open `src/aerislab/api/scenario.py` and read `connect()`:

```python
def connect(self, comp1, comp2, type="tether", length=0.0):
    ...
    constraint = joint.attach(self.current_system.get_bodies())
    self.current_system.add_constraint(constraint)
    return self
```

OK — it adds the constraint to `current_system`, not to `self.world`. So that confirms the diagnosis from the review.

Now I open `src/aerislab/core/simulation.py` and read `add_system()`:

```python
def add_system(self, system):
    self.systems.append(system)
    for component in system.components:
        self.add_body(component.body)
    for constraint in system.constraints:
        self.add_constraint(constraint)
```

Yes — at the moment `add_system` runs, `system.constraints` is empty (because `connect()` hasn't been called yet). The `for` loop iterates zero times. `world.constraints` stays empty.

I now understand the bug 100%. Time to fix.

### Step 5 — Decide on the smallest possible change

The work plan says: hotpatch is to add `self.world.add_constraint(constraint)` at the end of `Scenario.connect`. The proper architectural fix comes later (task P2-T11).

For now, smallest change = the hotpatch. I won't try to do the architectural fix here — that's later, in its own task.

### Step 6 — Write a test that fails before the fix

This is the discipline that catches regressions. Open `tests/` and add a new test file or a new test in an existing file:

```python
# tests/test_scenario_api.py (or similar)
def test_scenario_connect_propagates_constraint_to_world():
    """Regression test for CRIT-1 / Bug #1.

    Scenario.connect must register the constraint with World, not only
    with the active System, otherwise the KKT solver never sees it and
    the tether is silently inactive.
    """
    import sys; sys.path.insert(0, 'src')
    from aerislab.api.scenario import Scenario
    from aerislab.components.standard import Payload, Parachute

    payload = Payload(name='cap', mass=50.0, radius=0.4, position=[0,0,2000])
    chute = Parachute(name='main', mass=5.0, diameter=12.0, model='knacke',
                      activation_altitude=1500, position=[0,0,2000.5])
    sc = (Scenario(name='test').add_system([payload, chute])
          .connect(payload, chute, type='tether', length=10.0))

    # The bug: world.constraints stayed empty because add_system snapshotted
    # the System's constraints list before connect() added to it.
    assert len(sc.current_system.constraints) == 1
    assert len(sc.world.constraints) == 1, \
        "World.constraints should include the tether"
```

Run just this test:

```bash
venv/bin/pytest tests/test_scenario_api.py::test_scenario_connect_propagates_constraint_to_world -v
```

It fails. Good — the test is correctly catching the bug.

### Step 7 — Make the fix

Open `src/aerislab/api/scenario.py`. The change:

```python
def connect(self, comp1, comp2, type="tether", length=0.0):
    ...
    constraint = joint.attach(self.current_system.get_bodies())
    self.current_system.add_constraint(constraint)

    # Hotpatch for CRIT-1: also register with World so the solver sees it.
    # Proper fix (deriving World.constraints from systems) is task P2-T11.
    self.world.add_constraint(constraint)

    return self
```

That's the entire change. One added line. Two comment lines.

### Step 8 — Run the test

```bash
venv/bin/pytest tests/test_scenario_api.py::test_scenario_connect_propagates_constraint_to_world -v
```

It passes.

### Step 9 — Run the *full* test suite

This is the step beginners skip. Don't.

```bash
venv/bin/pytest
```

All 116 tests pass (your new one plus the existing 115). Good.

### Step 10 — Verify the smoke test (the original symptom)

The bug was visible in `examples/scenarios/02_parachute_system.py` showing -56 m/s touchdown. Let's confirm the symptom is gone:

```bash
venv/bin/python examples/scenarios/02_parachute_system.py 2>&1 | tail -20
```

If touchdown is now in the -8 to -15 m/s range, the parachute is actually working. Bug fixed.

### Step 11 — Commit

```bash
git status                                 # what changed?
git diff                                   # actually look at the diff
git add src/aerislab/api/scenario.py tests/test_scenario_api.py
git commit
```

In the editor that opens, write a real commit message:

```
Fix CRIT-1: propagate constraint to World in Scenario.connect

Scenario.connect was adding the tether constraint to current_system.constraints
after world.add_system had already snapshotted that list, so the KKT solver
never saw the tether. Result: every parachute simulation through the
Scenario API ran with the parachute and payload as independent bodies.

Confirmed by smoke-running examples/scenarios/02_parachute_system.py:
before fix, payload touchdown velocity = -56 m/s (= bare payload terminal
velocity). After fix, touchdown velocity in expected range with parachute
acting.

Hotpatch: explicit world.add_constraint() call in connect(). The proper
structural fix (deriving World.constraints from systems, eliminating the
out-of-sync class entirely) is task P2-T11 in WORK_PLAN.md.

Added regression test in tests/test_scenario_api.py.
```

### Step 12 — Mark the task done

Open `WORK_PLAN.md`, find P0-T1, change `[ ]` to `[x]`, optionally fill in `<sha>`:

```
- [x] Done — commit `abc1234`
```

Commit this change too:

```bash
git add docs/WORK_PLAN.md
git commit -m "Mark P0-T1 done"
```

### Step 13 — Merge back

```bash
git checkout ClaudeHelp
git merge p0-t1-scenario-connect
git branch -d p0-t1-scenario-connect       # safe to delete; it's merged in
```

### Step 14 — Take a break

Seriously. 10 minutes. Coffee, walk, anything not screen-related. Then start the next task fresh.

---

That whole process for a one-line fix took maybe 45 minutes the first time. With practice it'll take 15. The discipline is what makes the difference between "I think it's fixed" and "I know it's fixed and there's a test that proves it stays fixed."

---

## 13. What "good" looks like — habits to build over the next year

Over the year of work in `WORK_PLAN.md`, try to internalize these. Don't try to adopt all at once — pick one a month.

| Habit | Why |
|---|---|
| Run the full test suite before every commit | Prevents 90% of regressions |
| One task per branch, one logical change per commit | Makes the diff readable in 6 months |
| Read 10 lines of code for every line you write | The 80/20 reality of coding |
| Write a regression test for every bug you fix | The bug doesn't come back |
| Commit at least every hour while working | Never lose more than an hour |
| Take a 5-minute break every 45 minutes | Bug rate goes up exponentially when tired |
| Write a learning log entry at the end of each session | Compound learning |
| Re-read your own code the next day before merging | Catches obvious mistakes |
| Push back on AI suggestions you don't understand | The point is to learn |
| Plan the work *before* opening the editor | "Editor open" mode is for typing, not thinking |
| Do the simplest thing that could possibly work, then improve | Avoids over-engineering |
| When stuck for >30 minutes, write down what you've tried | Sometimes the writing itself unblocks you |
| When stuck for >2 hours, stop and ask for help | You've usually missed something obvious |
| Update documentation in the same commit as the code change | Docs that drift from code are worse than no docs |

You won't do all of these consistently. Nobody does. But the *direction* matters. Every month, pick one thing on the list you're not doing and try to do it for two weeks.

---

## A closing note

The single biggest thing I can tell you, having watched many people learn to code: **the difference between people who get good and people who don't is not talent. It's volume of deliberate practice and willingness to sit with confusion.**

You will spend hours stuck. You will write bad code. You will introduce bugs and not understand why. You will read code that feels like it was written in another language. **This is the work.** It's not a sign you're failing — it's a sign you're learning. The people who got good went through exactly the same phase, they just kept going.

The work plan in `WORK_PLAN.md` is going to take you a year of part-time work. That's fine. At the end of that year you will be a meaningfully different engineer than you are today, *and* you will have a working tool to show for it. Both happen at the same time, by doing the same work.

Don't rush. Don't skip understanding. Don't be too proud to ask. Don't be afraid to make a mess.

Good luck.

---

*This guide is for Štěpán Kaspar, kaspar.stepan.cz@gmail.com. Updated 2026-05-13.*
