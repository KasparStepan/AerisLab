# How to Work on AerisLab — Expanded Practical Guide

**Companion to:** `HOW_TO_WORK.md` (the shorter original). This is the expanded version with more depth on debugging, refactoring, scientific Python, vibecoding patterns, performance, and daily/weekly rhythm. Same audience (you, learning while doing); same purpose (practical, not a textbook).

**How to use this:** skim once now to know what's here. Come back to specific sections when you need them — when stuck debugging, before a big refactor, when you want to use AI better. First read it'll feel like a lot. After three months of doing the work it'll feel obvious.

**Companion files:**
- `AERISLAB_REVIEW_PLAIN.md` — the *why* (what's broken, what should be better)
- `WORK_PLAN.md` — the *what* (the ordered to-do list)
- `HOW_TO_WORK.md` — the *how* (shorter version of this doc)
- This file — the *how* (deeper)

---

## Table of Contents

### Part I — Mindset
1. [Core attitudes](#1-core-attitudes)
2. [Specific traps PhD researchers fall into](#2-specific-traps-phd-researchers-fall-into)
3. [Energy management — when to do hard vs easy tasks](#3-energy-management--when-to-do-hard-vs-easy-tasks)

### Part II — Setup
4. [Your dev environment](#4-your-dev-environment)
5. [Tools that pay back over time](#5-tools-that-pay-back-over-time)

### Part III — The work loop
6. [The basic rhythm](#6-the-basic-rhythm)
7. [Estimating time, and why you'll always be wrong](#7-estimating-time-and-why-youll-always-be-wrong)

### Part IV — Reading code
8. [How to read code (yours or anyone else's)](#8-how-to-read-code-yours-or-anyone-elses)
9. [Reading scientific Python specifically](#9-reading-scientific-python-specifically)
10. [Reading code that uses libraries you don't know](#10-reading-code-that-uses-libraries-you-dont-know)

### Part V — Writing code
11. [The minimum viable change principle](#11-the-minimum-viable-change-principle)
12. [Naming things](#12-naming-things)
13. [Comments — when to write them](#13-comments--when-to-write-them)
14. [Refactoring without breaking things](#14-refactoring-without-breaking-things)
15. [Working with numpy — vectorization mindset](#15-working-with-numpy--vectorization-mindset)
16. [Working with floats — NaN, precision, comparison](#16-working-with-floats--nan-precision-comparison)
17. [Type hints — when they help](#17-type-hints--when-they-help)

### Part VI — Debugging
18. [The debugging recipe](#18-the-debugging-recipe)
19. [Print debugging done well](#19-print-debugging-done-well)
20. [Using pdb / ipdb properly](#20-using-pdb--ipdb-properly)
21. [Logging-based debugging for long simulations](#21-logging-based-debugging-for-long-simulations)
22. [When to suspect the framework vs your code](#22-when-to-suspect-the-framework-vs-your-code)
23. [Bisecting with git](#23-bisecting-with-git)

### Part VII — Testing
24. [Why and what — the basics](#24-why-and-what--the-basics)
25. [Pytest fixtures](#25-pytest-fixtures)
26. [Parametrized tests](#26-parametrized-tests)
27. [Property-based testing with hypothesis](#27-property-based-testing-with-hypothesis)
28. [Mocking — when and how](#28-mocking--when-and-how)
29. [Coverage — how to read it](#29-coverage--how-to-read-it)
30. [Snapshot / regression tests for science](#30-snapshot--regression-tests-for-science)

### Part VIII — Git
31. [The basic flow](#31-the-basic-flow)
32. [Branching strategy for solo work](#32-branching-strategy-for-solo-work)
33. [Stash](#33-stash)
34. [Rebase vs merge](#34-rebase-vs-merge)
35. [Reflog — your time machine](#35-reflog--your-time-machine)
36. [Recovering from disasters](#36-recovering-from-disasters)
37. [Writing good commit messages](#37-writing-good-commit-messages)

### Part IX — Vibecoding (working with AI)
38. [The mental model — AI as a senior coworker](#38-the-mental-model--ai-as-a-senior-coworker)
39. [Prompt patterns that work](#39-prompt-patterns-that-work)
40. [Prompt patterns that fail](#40-prompt-patterns-that-fail)
41. [Reviewing AI output — the most important vibecoding skill](#41-reviewing-ai-output--the-most-important-vibecoding-skill)
42. [Different models for different tasks](#42-different-models-for-different-tasks)
43. [Context windows](#43-context-windows)
44. [Workflow patterns](#44-workflow-patterns)
45. [Avoiding learned helplessness](#45-avoiding-learned-helplessness)
46. [The "explain it back" test](#46-the-explain-it-back-test)

### Part X — Performance & profiling
47. [When NOT to optimize (almost always)](#47-when-not-to-optimize-almost-always)
48. [How to know if optimization is needed](#48-how-to-know-if-optimization-is-needed)
49. [Profiling tools](#49-profiling-tools)
50. [Common Python performance wins](#50-common-python-performance-wins)

### Part XI — Working alone
51. [Code review for one](#51-code-review-for-one)
52. [Documentation as future-self insurance](#52-documentation-as-future-self-insurance)
53. [Talking to people anyway — online communities](#53-talking-to-people-anyway--online-communities)
54. [Keeping a learning log](#54-keeping-a-learning-log)

### Part XII — Rhythm & sustainability
55. [Daily rhythm](#55-daily-rhythm)
56. [Weekly rhythm](#56-weekly-rhythm)
57. [Monthly rhythm](#57-monthly-rhythm)
58. [When you hit a wall](#58-when-you-hit-a-wall)

### Part XIII — Worked examples
59. [Example 1: fixing a bug (P0-T1)](#59-example-1-fixing-a-bug-p0-t1)
60. [Example 2: implementing a feature (atmosphere model)](#60-example-2-implementing-a-feature-atmosphere-model)
61. [Example 3: doing a refactor safely](#61-example-3-doing-a-refactor-safely)
62. [Example 4: hunting a mysterious test failure](#62-example-4-hunting-a-mysterious-test-failure)

### Part XIV — Reference
63. [Common traps and how to escape them](#63-common-traps-and-how-to-escape-them)
64. [The "I'm stuck" decision tree](#64-the-im-stuck-decision-tree)
65. [Habits to build over the next year](#65-habits-to-build-over-the-next-year)
66. [A closing note](#66-a-closing-note)

---

# Part I — Mindset

## 1. Core attitudes

A handful of attitudes that have nothing to do with talent and everything to do with outcome.

### Be a detective, not a magician
Before you change anything, **understand what's actually happening**. Magicians wave their hands and hope. Detectives gather evidence. When a test fails or a simulation gives a weird number, your first move is *not* to start changing code — it's to ask "what is the code actually doing right now, and what did I expect it to do?" The gap between those two answers is where the bug lives.

### Small steps beat big steps
Especially when you're learning. Make one small change, run the tests, see what happened. Then the next small change. This is slower per change but much faster overall, because when something breaks, you know it was the last thing you did. The opposite — making 10 changes at once and then trying to figure out which one broke the tests — costs hours.

### Reading is 80% of the job
Beginners think coding is about writing code. It isn't. It's about reading code (yours, other people's, your own from 6 months ago) and understanding it well enough to change one thing without breaking ten others. Get comfortable reading. The rest follows.

### Don't be afraid to break things
Git makes almost everything reversible. The worst case for "I changed something and now nothing works" is `git checkout .` (throw away everything since the last commit). The worst case for `git reset --hard HEAD~1` is "I lost the last commit," which only matters if it was important — and even then, `git reflog` can usually rescue it. The thing you should fear is **uncommitted work that you might lose**. Commit often. Even half-finished things, in a branch, are safe.

### "I don't understand this yet" is fine
You're allowed to write `# I'm not sure why this works, come back later` in a comment. You're allowed to do a task with help and then redo it without help to internalize it. You're allowed to spend an hour reading instead of writing. The PhD-time clock is real, but skipping understanding to "save time" creates 10× more time-loss later when bugs surface from things you didn't understand.

### Slow is smooth, smooth is fast
Borrowed from soldiers and surgeons. Rushing creates mistakes that take longer to fix than the time you saved. Take the extra two minutes.

### The bug is in your code, not in the framework
99% of the time. When you think "this must be a numpy bug," it's almost certainly a misunderstanding of what numpy does. The library has been used by millions of people; your code has been used by you for an hour. Lead with humility.

### Finished beats perfect
A working ugly solution that you can iterate on is more valuable than a perfect solution you never finish. Especially in research code. Get to working first; pretty later.

### When in doubt, do less
The instinct of an inexperienced coder is to add more code to fix a problem. Often the right move is to remove code, simplify, or change a design choice. "What's the simplest thing that could possibly work?" should be your first question on every task.

---

## 2. Specific traps PhD researchers fall into

You're not just a coder — you're a researcher. The combination has its own failure modes.

### The perfectionism trap
Academic culture rewards thoroughness. For prose this is correct. For research code, it's poison. Code that works on the example you actually need to run, even if it's ugly, beats elegant code that handles every edge case but isn't done. **Ship 0.1 versions; iterate.**

### The "I should learn properly first" trap
"I'll learn category theory before I touch this functional code." "I'll read the entire numpy docs before I use it." You won't. The best way to learn X is to need X for Y. Start Y. When you hit a wall, learn the X you need. The motivation of needing it makes learning stick.

### The reinventing-everything trap
"I should write this from scratch to understand it." Sometimes true — for the parts of your engine that *are* your contribution. False — for ISO standards, file formats, atmospheric models, plotting libraries, web servers. Reuse aggressively where the wheel is well-made; build only what's genuinely yours.

### The "this isn't novel enough" trap
Engineering work that supports research is often invisible in your final paper but enables it. Don't apologise for spending two weeks on infrastructure. The validation case you reproduced from a published paper is *more* impressive than another ML model variant.

### The lone-wolf trap
You don't have a team. Easy to think this means no one can help. False — Stack Overflow exists, the scientific Python community is friendly, your supervisor's other students probably have related tools, GitHub issues on the libraries you use have answers. Reaching out costs you 5 minutes; staying stuck costs you days.

### The "I'm not a real coder" impostor trap
You don't need to be. You need to be an engineer-scientist who can write working code that produces correct numbers. That's a different (and more useful) thing than "software developer." The bar is functional correctness, not employability at Google.

### The notebook-only trap
Jupyter notebooks are great for exploration. They are bad for code that other things depend on. The trap: prototype in a notebook, never move it to a `.py` file, end up with a 3000-line notebook that's the canonical version of your simulator. Convert to `.py` modules early. Notebooks are scaffolding, not architecture.

### The "I'll write tests later" trap
You won't. Write the test as you write the code. Even one assertion is better than none. The test isn't tax — it's the thing that lets you change code with confidence in 6 months when you've forgotten what it does.

---

## 3. Energy management — when to do hard vs easy tasks

You have ~3-4 hours of high-cognitive-load work in you per day. Maybe 5 on a great day. Beyond that, code quality drops and bug rate climbs exponentially. Acknowledge this and plan around it.

### Hard tasks (need your peak focus)
- New code that requires understanding new concepts
- Debugging a tricky bug
- Refactoring across multiple files
- Designing a new abstraction
- Reading dense unfamiliar code

Save these for your peak hours. For most people that's morning, but check your own pattern (a learning-log entry per day with "energy 1-10" tracked over a month makes this obvious).

### Easy tasks (great for low-energy hours)
- Writing or improving documentation
- Fixing whitespace, lint warnings, type errors
- Reading code (without changing it)
- Running tests, profiling, looking at results
- Updating comments and docstrings
- Cleaning up notebooks
- Writing commit messages for work already done

Mix-and-match: spend the morning on a hard refactor, the afternoon writing documentation for what you just did. Both contribute, neither competes for the same mental resource.

### Signs you've gone past your limit
- Re-reading the same line of code 3 times without absorbing it
- Making typo-level mistakes in your code
- Forgetting what you were trying to do mid-task
- Frustration spike when a small thing fails
- Wanting to "just push through one more thing"

When you notice these, **stop**. Take a real break (walk, eat, leave the screen). Coming back at 70% capacity is more productive than continuing at 30%. The work you push through at low energy will need re-doing at higher energy anyway.

### The "shower thought" mechanism is real
A surprising fraction of debugging breakthroughs happen away from the keyboard. Your brain processes problems in the background when you stop actively working on them. **Schedule walks.** Genuinely.

---

# Part II — Setup

## 4. Your dev environment

You don't need a fancy setup. You need a few things to work reliably.

### Editor / IDE
- **VS Code** with the Python extension. Free, popular, integrates well with WSL. Default recommendation. Has terminal, debugger, git GUI, Jupyter support.
- **PyCharm** (Community edition is free). Heavier, more "IDE-like," better for refactoring across files.
- **Vim/Neovim** with appropriate plugins. Steep curve, fastest once known. Not recommended unless you already use vim.

The only must-haves:
- Syntax highlighting for Python
- Linting feedback inline (ruff or pylint)
- "Go to definition" (jump to where a function is defined)
- "Find all references" (where is this function used?)
- A built-in terminal so you don't context-switch to another window

### Virtual environment hygiene
Your repo has a `venv/` directory. Always use it:

```bash
# Activate (then your shell uses the venv's python and pip)
source venv/bin/activate

# OR call binaries directly without activating:
venv/bin/python script.py
venv/bin/pytest
venv/bin/pip install foo
```

Either works. Pick one and be consistent. **Never `pip install` outside the venv** — you'll pollute your system Python and create reproducibility issues.

When you start a new project, create a fresh venv. Don't share venvs across projects.

### Terminal aliases
Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias gs='git status'
alias gd='git diff'
alias gco='git checkout'
alias gl='git log --oneline -20'
alias pyt='venv/bin/pytest'
alias py='venv/bin/python'
```

These shave seconds off every command, and you run them hundreds of times per day.

### Project layout to keep in your head
```
AerisLab/
├── src/aerislab/        ← the package itself
├── tests/               ← unit tests
├── examples/            ← example scripts users would run
├── docs/                ← markdown docs (you're reading one)
├── output/              ← simulation outputs (gitignored)
├── venv/                ← virtual environment (gitignored)
└── pyproject.toml       ← project metadata, dependencies, lint config
```

Knowing where things live without thinking is what "knowing a codebase" means.

---

## 5. Tools that pay back over time

Each takes <30 minutes to set up and pays back daily.

### `ripgrep` (`rg`) — fast searching
```bash
sudo apt install ripgrep
rg "Scenario" src/        # find every "Scenario" in src/
rg -l "import torch"      # files that import torch
rg --type py "def apply"  # only in .py files
```

10× faster than `grep -r` and ignores binary/`venv`/`.git` automatically.

### `fd` — better find
```bash
sudo apt install fd-find
alias fd=fdfind
fd "test_.*\.py"
fd "" src/
```

### `bat` — better cat
```bash
sudo apt install bat
bat src/aerislab/dynamics/body.py    # syntax-highlighted, paged output
```

### `gh` — GitHub from the command line
```bash
sudo apt install gh
gh auth login
gh pr create
gh repo view --web
```

Stops you from context-switching to the browser for trivial GitHub things.

### `pre-commit` — automatic checks before each commit
```bash
pip install pre-commit
pre-commit install
```

Now every `git commit` automatically runs ruff and mypy. **Prevents 90% of "oh no, CI is failing" moments.**

### `nbstripout` — strips notebook outputs before commit
```bash
pip install nbstripout
nbstripout --install
```

Notebook diffs become readable. Unconditional win.

### A note-taking app (any)
Notion, Obsidian, plain markdown files, paper notebook — doesn't matter. **Use it.** Every session, write 2 sentences (see §54). Compound interest on learning.

---

# Part III — The work loop

## 6. The basic rhythm

Every task in `WORK_PLAN.md` should be done this way. Memorize it. The discipline is what separates "I think I fixed it" from "I know I fixed it."

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
11. Merge the branch back to main
12. Take a 10-minute break before the next task
```

### Pre-work ritual (every session)

```bash
cd ~/Projects/AerisLab
git status              # any uncommitted work from last time?
git log --oneline -5    # what did I do recently?
pyt --tb=short          # do tests pass right now?
```

You want to start every session in a known-good state. If the tests are already failing before you change anything, you'll spend an hour debugging a problem that wasn't caused by what you're about to do. **Fix that first.**

### Two ground rules

- **One task per branch.** Don't bundle "Fix Bug #1" with "while I'm there, also clean up the whitespace." Two tasks → two branches → two commits.
- **Don't skip step 7 (run the tests).** Even if you "only changed a comment."

### What "make the smallest change" means

When fixing a bug, affect as few lines as possible. **Fix the bug, not the surroundings.** Resist:
- Renaming variables while you're in there
- Reformatting code while you're in there
- "While I'm here, let me also..."

If you find related issues during the work, write them down as new tasks. Don't bundle.

---

## 7. Estimating time, and why you'll always be wrong

You will, every time, underestimate how long a task takes. Universal fact of software development. Estimates in `WORK_PLAN.md` are calibrated for part-time PhD pace, but expect to be wrong by a factor of 2-3 in either direction on individual tasks.

### Why
- The time isn't in the change itself — it's in the surrounding work (reading, debugging, testing).
- "I'll just quickly..." sentences are wrong about 80% of the time.
- New unforeseen complications surface during the work.
- You context-switch (lunch, meeting, email) — switching costs 15+ minutes of warm-up.

### What to do
- **Track your time per task** in your learning log. After 20 tasks you'll calibrate yourself.
- **Don't promise dates.** "I'll have this done by Friday" → stress + shoddy work. "I'll work on this Friday" is fine.
- **Add a 1.5× buffer** when you tell anyone when something will be done.
- **Stop work cleanly** at the end of a session, even mid-task. Leave a note in your branch ("WIP, next step is X").

---

# Part IV — Reading code

## 8. How to read code (yours or anyone else's)

Most beginners try to read code like prose — start at the top, read down. That doesn't work.

### Three-pass reading

1. **Skim** — read the docstrings, function signatures, class names. Get the *shape*.
2. **Trace** — pick one realistic input, follow it through the code mentally.
3. **Detail** — when you understand the shape, dig into the bits that surprised you.

### Start at the entry point and follow the call chain

When you want to understand "what does this script do," start at the very top:

```python
if __name__ == "__main__":
    run_example()
```

Find `run_example()`. Read it top to bottom. When it calls something (`Scenario(name="01_simple_drop")`), pause. Ask yourself: "what do I think this does, based on the name and arguments?" Then verify by jumping to the definition.

In your editor, "go to definition" (often `F12` or `Ctrl+Click`) is your friend. So is `rg`:

```bash
rg "def Scenario" src/    # find where Scenario is defined
rg "Scenario\(" src/      # find everywhere it's used
```

### When you hit something you don't understand, write a note

Don't try to understand everything in one pass. Make a note ("come back to this — `quat_normalize` magic"), keep going to get the overall shape, then come back and dig in.

I keep a `notes.md` file open during reading sessions:

```
- World.add_system: registers bodies + constraints from the system. But constraints
  is captured by reference at this point, not later? Need to check.
- assemble_system: returns Minv, J, F, rhs, v. What's `v`? oh — generalized
  velocities, stacked [v_lin, omega] per body.
- Why does ParachuteDrag have its own activation logic AND Parachute the component
  also has activation logic? <-- check this
```

This is exactly how I found Bug #1 — by writing a note that didn't make sense, then chasing it down.

### Read with a question

Reading code "to understand it" is endless. Reading code to answer a specific question terminates: when you have the answer, you stop. Always have a question:

- "How does the IVP solver call the parachute model?"
- "Where does the simulation decide it's done?"
- "What happens to the constraint forces after they're computed?"

### Time-box first reads

Set a 30-minute timer for "first read of new file." When it goes off, write down (a) what you understood, (b) what you're still confused about. Don't keep reading past the timer — retention drops sharply.

### Reading your own old code

The hardest reading task. Code from 6 months ago feels like a stranger's. Same technique:
- Start at the entry point that calls the part you've forgotten.
- Read the docstring (you wrote one, right?).
- Trace one realistic input.

If you find yourself asking "why did I do this?" — that's a sign the code needs a comment. Add one.

---

## 9. Reading scientific Python specifically

Scientific Python has its own idioms. Knowing them means you can read code 5× faster.

### `np.array` operations are vectorized

```python
v = body.v                              # shape (3,)
speed = np.linalg.norm(v)              # scalar — magnitude
direction = v / speed                   # shape (3,) — unit vector
force = -0.5 * rho * Cd * A * speed * v  # shape (3,) — vectorized arithmetic
```

There's no explicit loop because `*`, `/`, `-` operate element-wise on arrays. The expression `-0.5 * rho * Cd * A * speed * v` does scalar-times-vector arithmetic and produces a `(3,)` vector. **Train your eye to see this.**

### `@` is matrix multiplication

```python
F_world = body.rotation_world() @ F_body    # rotation matrix @ vector
M = R @ I_body @ R.T                        # rotate inertia tensor
```

`@` (added in Python 3.5) means matmul for numpy arrays. `R.T` is transpose. `R @ x` does standard matrix-vector multiplication.

### Indexing patterns
```python
y[off:off+3]         # slice — elements off, off+1, off+2
y[off+3:off+7]       # 4-element slice (quaternion)
A[0:3, 0:3]          # sub-matrix, top-left 3×3
A[:, 0]              # entire first column
A[i, :]              # entire i-th row
v[v > 0]             # boolean indexing — all positive elements
```

### Common numpy gotchas

- **`a = b` doesn't copy.** `b` and `a` point to the same array. Mutating one mutates the other. Use `b.copy()` if you need a real copy.

- **`a + b` creates a new array, but `a += b` doesn't.** In-place operations save memory but mutate.

- **Shape mismatches don't error early.** Numpy "broadcasts" arrays of different shapes silently. `np.array([1,2,3]) * 2.0` works fine, but `np.array([1,2,3]) * np.array([1,2])` fails. Keep mental track of shapes.

- **Floats from numpy are not Python floats.** `np.float64(0.5)` looks like `0.5` but is a numpy scalar. Usually fine; occasionally surprising in `isinstance(x, float)` checks.

- **`np.array([1])` has shape `(1,)`, not `()` or `(1, 1)`.** Empty trailing dimensions matter. The shape is the most important property of any array — when in doubt, `print(arr.shape)`.

### The "axis" argument
Many numpy functions take an `axis` parameter. It tells them which dimension to operate over.

```python
A.sum()              # sum of all elements, scalar
A.sum(axis=0)        # sum down columns, shape (n_cols,)
A.sum(axis=1)        # sum across rows, shape (n_rows,)
```

For 2D arrays: `axis=0` is "down" (collapse rows), `axis=1` is "across" (collapse columns). You'll get this wrong sometimes; the fix is always `print(result.shape)` to see what came out.

### scipy patterns

```python
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as ScR

# solve_ivp returns an OdeResult object
sol = solve_ivp(rhs, t_span=(0, 10), y0=y_initial, method="Radau")
print(sol.t)         # array of time points the solver chose
print(sol.y.shape)   # (n_state, n_timepoints)
print(sol.success)   # did integration succeed?

# Rotation handles 3D rotations with proper math
R = ScR.from_quat([0, 0, 0, 1])      # identity, scalar-last quaternion
R.as_matrix()                         # convert to 3×3 matrix
R.as_euler('xyz', degrees=True)       # convert to Euler angles
```

Scipy is well-documented. When you don't know what `solve_ivp` returns, look at https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html — the docs are excellent.

### Pandas patterns
```python
import pandas as pd
df = pd.read_csv("simulation.csv")
df.columns                                    # list of column names
df["payload.p_z"]                             # one column, a Series
df[["payload.p_z", "payload.v_z"]]            # two columns, a DataFrame
df.iloc[0]                                    # first row
df[df["payload.p_z"] < 100]                   # filter rows
df.plot(x="t", y="payload.p_z")               # quick plot
```

---

## 10. Reading code that uses libraries you don't know

You'll need to read code that uses something unfamiliar — `pyarrow`, `streamlit`, `astropy`. You don't need to learn the whole library to read the code; you need enough context.

### The 5-minute library skim

When code imports something you don't know:
1. Open the library's docs in a browser.
2. Read the *front page* (often a 60-second pitch).
3. Look at the "quick start" or "tutorial" — first 2-3 minutes.
4. Now you know what kind of library it is and roughly what its API looks like.

You don't have to retain it. You're building enough context to read the code in front of you.

### When the docs are bad
- Look at the library's `examples/` folder on GitHub.
- Search "<library> tutorial" — third-party blog posts are often clearer.
- Ask Claude to summarize what the library does and how it's typically used.
- Read the docstrings of the functions actually being called.

### When you're really stuck on what a function does
The Python REPL is your best friend:

```bash
$ venv/bin/python
>>> import scipy.integrate
>>> help(scipy.integrate.solve_ivp)            # full docstring
>>> sol = scipy.integrate.solve_ivp(lambda t, y: -y, (0, 1), [1])
>>> sol                                         # see the return type
>>> sol.t                                       # actual values
```

Trying things in the REPL teaches you faster than reading.

---

# Part V — Writing code

## 11. The minimum viable change principle

For every change you make, ask: "what's the smallest change that solves the problem in front of me?"

### Why
- Smaller changes have smaller chances of bugs.
- Smaller changes are easier to review.
- Smaller changes are easier to revert if they go wrong.
- Smaller changes ship sooner, which means you find out sooner if the approach was right.

### Examples

You want to add a wind model. You could:

- **Bad (over-engineered):** Design a `WindField` Protocol, three implementations (constant, profile, turbulent), refactor `Drag` to take a wind field, refactor every test, write benchmarks. **Two weeks of work.**

- **Good (minimum viable):** Add a `wind=np.array([0,0,0])` argument to `Drag.__init__`. Use `body.v - wind` inside `apply()`. Default keeps existing behavior. Write a test that drag with `wind=[10,0,0]` produces a horizontal force. **One hour of work.**

The minimum viable version is shippable today. The fancy version may emerge later when you actually need different wind models — and you'll design it better with the experience of having built one already.

### When MVP is not enough
- When the simple version doesn't actually solve the problem.
- When the simple version pollutes the public API in a way you can't undo.

When in doubt: **start with MVP**. You can always add more.

---

## 12. Naming things

### Names should describe what, not how
- `compute_drag_force` ✓ (describes what)
- `apply_quadratic_formula_for_drag` ✗ (describes implementation)

### Names should be the same length as their scope
- A loop variable in a 3-line for-loop: `i`, `b`, `f` are fine.
- A class attribute that lives forever: `payload_mass`, not `pm`.
- A module-level function: `compute_atmospheric_density`, not `atm`.

### Use domain language
For aerospace code, name things the way aerospace engineers name them. `Cd`, `alpha`, `Mach`, `rho`, `omega` are recognizable. Don't substitute them with `drag_coefficient_dimensionless`. The lint rule `N806` is intentionally disabled in your `pyproject.toml` for this reason.

### Avoid stuttering
- `Payload.payload_mass` ✗ — say `Payload.mass`
- `parachute.parachute_diameter` ✗ — say `parachute.diameter`

The class context already provides the prefix.

### Be ruthless about renaming when it's wrong
If `t` started as "time" but evolved into "iteration counter," rename it to `i` or `step`. Code drifts. Names lose meaning. Renaming in your editor is cheap.

---

## 13. Comments — when to write them

Default to writing fewer comments, not more.

### Don't comment what the code does

```python
# WRONG — restates the code
i += 1   # increment i

# WRONG — code is self-explanatory
mass = m1 + m2   # add the two masses
```

If the code needs a comment to explain *what* it does, the code is unclear — fix the code (rename, refactor) instead of adding a comment.

### Do comment WHY when it's not obvious

```python
# Use exponential map for stiff cases — Euler integration loses
# orthogonality after ~10⁴ steps with large omega.
self.q = quat_integrate_exponential_map(self.q, self.w, dt)
```

The "why" — the hidden constraint, the reference to a paper — is what comments are for.

### Comment surprises and gotchas

```python
# Note: scipy uses scalar-last [x,y,z,w] quaternions, NOT [w,x,y,z].
# If you see something weird with rotations, check this first.
```

### Don't use comments to track tasks

```python
# WRONG — describes a moment in time
# Added this for the FSI integration, see PR #42
def predict(self, state):
    ...
```

PR descriptions and commit messages are for *the change*; comments are for *the code*.

### Docstrings are different
Public functions and classes should have docstrings. Use NumPy style (Parameters, Returns, Notes, Examples). The codebase already does this consistently — match the existing style.

---

## 14. Refactoring without breaking things

Refactoring = changing the structure of code without changing its behavior. The Phase 2 work in `WORK_PLAN.md` is full of these.

### The refactoring rules

1. **Tests must pass before you start.** If they don't, fix that first.

2. **Tests must pass after every step.** Not "after the whole refactor" — after every small step.

3. **Refactor in tiny steps.** Bad: "split World into 4 classes." Good: 50 small steps that, taken together, split World into 4 classes.

4. **Don't change behavior during a refactor.** If you spot a bug while refactoring, write it down — fix it in a separate commit, *after* the refactor. Bug fixes mixed with refactors cause "the tests broke and I don't know whether it's the refactor or the fix."

5. **Commit at every green state.** Tests pass → commit. You can squash later if you want a clean history.

### A concrete pattern: extract function

You see a 30-line function with two responsibilities. Split into steps:

1. **Step 1**: Move the second half into a new function, but keep calling it.
   ```python
   def compute_force_and_log(body, t):
       f = -0.5 * body.rho * ...
       _log_force(t, f)
       return f
   def _log_force(t, f):
       print(f"force = {f}")
       log.append((t, f))
   ```
   Tests pass? Commit.

2. **Step 2**: Rename `compute_force_and_log` to `compute_force`. Update all callers. Tests pass? Commit.

3. **Step 3**: Move the `_log_force` call out to the callers. Tests pass? Commit.

You've done the refactor in 3 commits, each verified green. If anything breaks, you know exactly which step did it.

### When refactoring goes wrong

Symptom: tests start failing in unrelated places.

Cause: usually you've changed behavior, not just structure. Common ways:
- A method that mutated state now doesn't.
- A function that returned an object now returns a copy.
- A side effect (file write, print, log) was lost.

Fix: bisect with git to find the commit that started the breakage. Look at *only* that diff.

---

## 15. Working with numpy — vectorization mindset

The biggest performance win in scientific Python is replacing Python loops with numpy operations.

### Slow (Python loop)
```python
result = np.zeros(1000)
for i in range(1000):
    result[i] = math.sin(x[i]) * math.cos(y[i])
```

### Fast (vectorized)
```python
result = np.sin(x) * np.cos(y)    # operates on the whole arrays at once
```

The fast version is ~50× faster. Why: the loop runs in C inside numpy, not in Python.

### How to "think vectorized"

Stop thinking "for each element, do X." Start thinking "do X to the whole array."

If you find yourself writing a loop over a numpy array, ask: "is there a numpy function that does this?" Usually yes:
- Sum: `arr.sum()`, `arr.sum(axis=0)`
- Mean: `arr.mean()`
- Element-wise math: `np.sin`, `np.exp`, `np.log`, etc.
- Conditional: `np.where(condition, val_if_true, val_if_false)`
- Cumulative: `np.cumsum`, `np.cumprod`
- Sorting: `np.sort`, `np.argsort`
- Linear algebra: `np.linalg.solve`, `np.linalg.inv`, `np.linalg.eig`

### When NOT to vectorize

When each iteration depends on the previous one (like a time-stepping integrator). You can't vectorize across time. Vectorize *within* each iteration instead.

### The shape-printing reflex

When working with multi-dimensional arrays, **print `.shape` constantly**. Most numpy bugs come from confusion about array shapes.

```python
print(f"A.shape = {A.shape}, B.shape = {B.shape}")
result = A @ B
print(f"result.shape = {result.shape}")
```

Once it works, delete the prints.

---

## 16. Working with floats — NaN, precision, comparison

Floating-point arithmetic is *not* the same as real-number arithmetic.

### Don't compare floats with `==`

```python
# WRONG — almost never True due to floating-point error
if x == 0.5:
    ...

# RIGHT
if abs(x - 0.5) < 1e-9:
    ...

# RIGHT for arrays
if np.allclose(x, 0.5, atol=1e-9):
    ...
```

### NaN propagates silently

```python
x = np.array([1.0, 2.0, np.nan, 4.0])
x.sum()             # → nan (a single nan poisons the whole sum)
x.mean()            # → nan
x.max()             # → nan
np.nansum(x)        # → 7.0 (skips NaNs)
```

If a NaN appears in your simulation output, the bug happened *before* the NaN appeared. NaN is a symptom of a bad arithmetic operation upstream — usually a division by zero, sqrt of a negative, or operations on uninitialized data.

### Common sources of NaN
- `0 / 0` → NaN (silent! no exception)
- `sqrt(negative)` → NaN
- `log(non-positive)` → NaN
- Operations involving an existing NaN
- Reading uninitialized memory

### Detecting NaN
```python
np.isnan(x).any()                # is there at least one NaN?
np.where(np.isnan(x))            # indices of NaNs
```

In long simulations, **add a NaN check after the integration loop**:
```python
if np.isnan(body.p).any() or np.isnan(body.v).any():
    raise RuntimeError(f"NaN appeared at t = {world.t}")
```

This stops the simulation immediately when something goes wrong.

### Precision matters at scale

Floats have ~15 decimal digits of precision. But:
- Subtracting two nearly-equal numbers loses precision rapidly.
- Long simulations accumulate roundoff error.
- Constraint drift is a real phenomenon (it's why Baumgarte stabilization exists).

If you're hitting precision issues, the fix is usually a math redesign, not switching to higher precision.

---

## 17. Type hints — when they help

Python type hints (`: int`, `-> str`) are optional but useful.

### When they help
- Public APIs (function signatures other code depends on).
- Code with complex data flow.
- Auto-completion in your editor.

### When they get in the way
- Simple internal helper functions.
- Highly dynamic code.
- During rapid prototyping. Add hints later when the API stabilizes.

### Modern syntax (3.10+)

```python
def foo(x: int, y: list[float] | None = None) -> dict[str, int]:
    ...
```

- `int`, `float`, `str`, `bool` — basic types.
- `list[X]`, `dict[K, V]`, `tuple[X, Y]` — collections.
- `X | None` — "X or None" (replaces `Optional[X]`).
- `X | Y` — "X or Y" (replaces `Union[X, Y]`).

For numpy:
```python
from numpy.typing import NDArray
import numpy as np

def foo(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    ...
```

### Type-checking
```bash
venv/bin/mypy src/aerislab/
```

mypy reads your type hints and tells you when they're inconsistent. Treat warnings as bugs.

---

# Part VI — Debugging

## 18. The debugging recipe

A specific procedure. Use it every time, even if you think you know what's wrong.

### Step 1 — reproduce reliably

You cannot fix what you cannot reproduce. If a bug happens "sometimes," your first job is to find a recipe that makes it happen 100% of the time. This might mean writing a 5-line script that triggers it.

```python
# crash_repro.py — temporary, throw away later
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

**Make the reproducer as small as possible.** Smaller reproducer → easier debugging → smaller fix → tighter test.

### Step 2 — read the error message *all the way to the bottom*

Python tracebacks look scary. The actual error is at the **bottom**. The lines above it are the call chain that led there.

```
Traceback (most recent call last):
  File "crash_repro.py", line 8, in <module>
    sc.run(duration=60.0)
  File "src/aerislab/api/scenario.py", line 172, in run
    self.world.integrate_to(solver, t_end=duration, ...)
  ...
ValueError: Cannot invert inertia tensor: Singular matrix
```

Read the *bottom* first, then the line right above the error (which is the line of code that produced it), then trace back through the call chain.

### Step 3 — narrow down with print statements

Add a print right before the line where you think the bug is:

```python
print(f"DEBUG: about to integrate, body.q = {body.q}, |q| = {np.linalg.norm(body.q)}")
```

Run it. Read the output. Did your guess match what's actually happening?

### Step 4 — when prints aren't enough, use a debugger

```python
import pdb; pdb.set_trace()       # or: import ipdb; ipdb.set_trace()
```

Python pauses there. You get an interactive prompt where you can inspect any variable.

### Step 5 — bisect when nothing else works

If a long script "works at the start and breaks somewhere," cut it in half. Does the first half work? Then it's in the second half. Cut that half in half. Repeat until you've narrowed the problem to one or two lines.

### Step 6 — the rubber duck

When you've been stuck for >30 minutes, explain the problem out loud (or to a duck, or in writing, or to Claude). Articulating the problem often surfaces the answer. The act of having to make the explanation coherent forces you to organize your understanding.

### What NOT to do when debugging

- **Don't change multiple things at once.** Change one. Test. Change another. Test.
- **Don't refactor while debugging.** Disaster.
- **Don't guess.** Verify. Every time you think "it must be because…" — go check.
- **Don't blame the framework.** It's almost always your code.

---

## 19. Print debugging done well

Print debugging is unglamorous and effective. The key is doing it systematically.

### The print template

Always include:
1. A label (so you can find it in a wall of output).
2. The variable name.
3. The variable value.
4. Optionally, the type or shape if relevant.

```python
print(f"[debug:integrate_step] t={t}, body.q={body.q}, |q|={np.linalg.norm(body.q):.6f}")
```

### Use a unique prefix
Prefix all debug prints with something searchable like `[DEBUG]` or `[dbg]`. After fixing the bug:

```bash
rg "\[DEBUG\]" src/        # find all debug prints
```

Now you can clean them all up. Without a prefix, hard to remember where you put prints.

### Print *before* and *after* the suspect line
```python
print(f"[dbg] before: body.v = {body.v}")
body.v += a_lin * dt
print(f"[dbg] after:  body.v = {body.v}")
```

Often the issue isn't on the line you suspect.

### Print arrays compactly
```python
print(f"a.shape={a.shape}, a.min={a.min():.3g}, a.max={a.max():.3g}, a[:5]={a[:5]}")
```

Printing a 10000-element array fills your terminal. Print summary stats and a slice instead.

---

## 20. Using pdb / ipdb properly

`pdb` is Python's built-in debugger. `ipdb` is the same with tab-completion and syntax highlighting (`pip install ipdb`).

### Drop into the debugger at a specific point

```python
import pdb; pdb.set_trace()        # or ipdb
```

Run your script. When Python hits this line, it pauses and gives you a prompt:

```
(Pdb) 
```

### Essential pdb commands
- `n` (next) — execute current line, move to next.
- `s` (step) — step *into* a function call.
- `c` (continue) — resume execution until next breakpoint or end.
- `l` (list) — show source code around current line.
- `p variable_name` — print a variable.
- `pp variable_name` — pretty-print.
- `w` (where) — show current call stack.
- `u` (up), `d` (down) — move up/down the call stack.
- `q` (quit) — stop execution.

### Conditional breakpoints
You only want to pause on iteration 1000:

```python
for i, step in enumerate(steps):
    if i == 1000:
        import pdb; pdb.set_trace()
    # ...
```

### Inspecting numpy arrays in pdb
```
(Pdb) p arr.shape
(3, 1000)
(Pdb) p arr[:, 0]
array([1.0, 2.0, 3.0])
(Pdb) p arr.dtype
float64
```

Same as in normal Python — `arr` is just a variable.

---

## 21. Logging-based debugging for long simulations

For simulations that run for minutes or hours, prints are too noisy. Use Python's `logging` module instead.

### Setup
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
```

### Use it
```python
logger.debug("Detailed info, usually only when investigating")
logger.info("General progress messages")
logger.warning("Something looks off but we can continue")
logger.error("Something failed")
```

### Switch verbosity at runtime
```python
logging.getLogger("aerislab").setLevel(logging.DEBUG)    # show everything
logging.getLogger("aerislab").setLevel(logging.WARNING)  # only warnings+
```

This means you can leave `logger.debug(...)` calls in your committed code. They're silent in normal runs but visible when you opt in.

### Log to a file for long runs
```python
logging.basicConfig(filename="run.log", level=logging.INFO)
```

Then `tail -f run.log` in another terminal to watch progress.

---

## 22. When to suspect the framework vs your code

Almost always: **your code**.

### When to start suspecting the framework
- The behavior matches a known bug filed against the library (search GitHub issues).
- The minimal reproducer shows the issue with no involvement of your code.
- You've thoroughly debugged your code and ruled out every reasonable cause.
- The library is young / experimental.

### Order of suspicion (default)
1. Your latest change.
2. Your old code that touches the relevant area.
3. Your understanding of how the library works.
4. Your environment (wrong version, wrong venv, etc.).
5. The library itself.
6. Hardware / OS issues.

Move down the list one item at a time. Don't skip.

### When you do report a library bug
File a GitHub issue with:
- Minimal reproducer (smallest possible script).
- Expected behavior.
- Actual behavior.
- Versions: `python --version`, `pip show <library>`, OS.

---

## 23. Bisecting with git

You haven't worked on the code in a week. Run the tests — three failures. Last week they passed. What changed?

`git bisect` finds the offending commit automatically.

### The recipe

```bash
git bisect start
git bisect bad                       # current state is bad
git bisect good <known-good-sha>     # last commit you know was good

# git checks out a commit halfway between. Run your tests:
venv/bin/pytest tests/test_problem.py

# If they pass:
git bisect good

# If they fail:
git bisect bad

# Repeat. After ~log2(N) iterations, git names the bad commit.

git bisect reset                     # done, return to your branch
```

For 100 commits, this takes 7 steps. Hugely faster than reading every commit.

### Automating with a script

```bash
git bisect start HEAD <known-good-sha>
git bisect run venv/bin/pytest tests/test_problem.py
```

Git runs the tests on each candidate automatically. Comes back with the answer.

---

# Part VII — Testing

## 24. Why and what — the basics

A test is a small program that runs your code with known inputs and checks the answer is what you expect.

### Why bother?
1. **You can change code without fear.** A passing test suite says "the things I checked still work."
2. **It forces you to think about correctness.** When you write the test, you ask "what's the *right* answer?"
3. **Tests are documentation.** A well-named test tells the reader what the code is supposed to do.

### The structure of a test

```python
def test_payload_falls_due_to_gravity():
    # Arrange — set up the world
    payload = Payload(name="p", mass=10, radius=0.5, position=[0, 0, 100])
    sc = Scenario("test").add_system([payload])

    # Act — do the thing
    sc.run(duration=1.0)

    # Assert — check the answer
    final_z = payload.body.p[2]
    expected_z = 100 - 0.5 * 9.81 * 1.0**2
    assert abs(final_z - expected_z) < 0.5, \
        f"expected ~{expected_z}, got {final_z}"
```

The pattern is **Arrange → Act → Assert** (AAA).

### Running tests
```bash
venv/bin/pytest                                          # all tests
venv/bin/pytest tests/test_body.py                       # one file
venv/bin/pytest tests/test_body.py::test_q_normalization # one test
venv/bin/pytest -v                                       # verbose
venv/bin/pytest -k "drag"                                # only tests with "drag" in name
venv/bin/pytest -m "not slow"                            # skip slow tests
venv/bin/pytest --tb=short                               # shorter tracebacks
venv/bin/pytest -x                                       # stop at first failure
venv/bin/pytest --lf                                     # only re-run last failures
```

The `--lf` (last failed) flag is incredibly useful — fix a failing test, run `--lf`, see if it's actually fixed without re-running the whole suite.

### What makes a good test?

- **One thing per test.** Don't write a test that checks 10 things.
- **Specific failure messages.** `assert x == 5, f"x should be 5, got {x}"` >> `assert x == 5`.
- **Independent.** Each test sets up its own world.
- **Tests behavior, not implementation.** Bad: "after add_system, world.constraints has length 1." Good: "after add_system + connect, the bodies stay tethered when simulated."
- **Fast** (ideally <1s per test).

---

## 25. Pytest fixtures

A fixture is reusable test setup. Pytest finds them by name.

### Basic fixture

```python
import pytest

@pytest.fixture
def gravity():
    """Standard Earth gravity vector."""
    return np.array([0.0, 0.0, -9.81])

def test_freefall_velocity(gravity):    # pytest sees parameter name, provides fixture
    body = RigidBody6DOF(name="p", mass=1.0, ...)
    body.apply_force(body.mass * gravity)
    # ...
```

### Fixture with cleanup

```python
@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates temporary output directory; pytest cleans it up after the test."""
    output = tmp_path / "output"
    output.mkdir()
    yield output             # the test runs here
    # cleanup goes after yield (rarely needed; tmp_path auto-cleans)
```

### Fixtures in conftest.py
A `tests/conftest.py` file holds fixtures shared across multiple test files. Pytest finds them automatically.

### Common pitfalls

- **Fixture scope.** By default fixtures run once per test (`scope="function"`). If you set `scope="module"` or `scope="session"`, the fixture is shared. Sharing mutable objects between tests creates bugs that take hours to find. **Default to function scope.**

---

## 26. Parametrized tests

Often you want to run the same test with different inputs.

```python
@pytest.mark.parametrize("mass,expected_terminal_v", [
    (1.0, 7.07),
    (10.0, 22.4),
    (100.0, 70.7),
])
def test_terminal_velocity(mass, expected_terminal_v):
    body = make_falling_sphere(mass=mass)
    sc = Scenario("test").add_system([body])
    sc.run(duration=30.0)
    assert abs(body.body.v[2] - expected_terminal_v) < 0.5
```

Pytest runs this as 3 tests, one per row. Each shows up separately. If only `mass=10.0` fails, you know exactly which one.

### Cross-product
```python
@pytest.mark.parametrize("mass", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("Cd", [0.47, 1.0, 1.5])
def test_drag(mass, Cd):
    ...    # runs 3 × 3 = 9 times
```

---

## 27. Property-based testing with hypothesis

Sometimes you don't have specific examples — you have a property that should hold for *any* input. The `hypothesis` library generates random inputs.

```python
from hypothesis import given, strategies as st

@given(
    mass=st.floats(min_value=0.1, max_value=1e6),
    h0=st.floats(min_value=10, max_value=1e5),
)
def test_freefall_energy_is_conserved(mass, h0):
    """Total energy should be constant during free fall (no drag)."""
    body = RigidBody6DOF(name="p", mass=mass, ...)
    body.p[2] = h0
    initial_energy = mass * 9.81 * h0
    sc = Scenario("test").add_system([Payload(body=body)]).run(duration=1.0)
    final_KE = 0.5 * mass * np.dot(body.v, body.v)
    final_PE = mass * 9.81 * body.p[2]
    final_energy = final_KE + final_PE
    assert abs(final_energy - initial_energy) / initial_energy < 0.001
```

Hypothesis tries hundreds of combinations, including edge cases you'd never think of. When it finds a failure, it shrinks the input to the minimum that still triggers the bug.

Heavyweight but extraordinarily powerful for physical invariants.

---

## 28. Mocking — when and how

A "mock" is a fake version of something, used in tests to avoid the real thing's side effects.

### When mocking is appropriate
- The real thing is slow (network call, big computation).
- The real thing is unreliable (hits an external API).
- The real thing has side effects (writes files, sends emails).
- You're testing how your code *uses* something, not what that thing does.

### When mocking is wrong (the trap)
- Mocking your own pure functions. Just call them.
- Mocking complex internal classes — test ends up testing the mock.
- Mocking the database in a way that doesn't match real DB behavior.

For AerisLab, you should rarely need mocks. The simulation is deterministic, fast, no external dependencies.

### If you do need mocks
```python
from unittest.mock import MagicMock

def test_logger_called_after_step():
    world = World(...)
    world.logger = MagicMock()
    world.step(solver, dt=0.01)
    world.logger.log.assert_called_once_with(world)
```

`MagicMock` records every call made on it. You assert against the calls.

---

## 29. Coverage — how to read it

Coverage = "what fraction of lines did the tests run?" Higher is better, but the goal is not 100%.

### Running coverage
```bash
venv/bin/pytest --cov=aerislab --cov-report=term-missing
```

You see:
```
src/aerislab/dynamics/body.py            109     19    83%   45-50, 116-131, ...
```

109 statements total, 19 not run, 83% covered. The line numbers tell you which lines were missed.

### How to use coverage
- **Find untested code.** The "Missing" column points at lines no test touched. Some are dead code, some are bug-friendly.
- **Find tested code that isn't tested well.** A line being executed doesn't mean the test would catch a bug in it.

### Don't obsess
- 100% doesn't mean bug-free. It means every line was *run*, not that every behavior was *checked*.
- 70-80% on critical paths > 95% on trivial getters.
- Adding a test "to bump coverage" produces useless tests.

For AerisLab: 75% overall, 0% on `scenario.py` and `standard.py` — that's where the bugs went undetected. Aim to raise *those specific* numbers, not the average.

---

## 30. Snapshot / regression tests for science

For numerical code, a useful pattern: capture the output of your code today, save it as a baseline, compare future runs to it.

```python
import json
from pathlib import Path

def test_continuous_inflation_matches_baseline():
    para = create_parachute(diameter=10.0, model="continuous_inflation")
    forces = []
    for t in np.arange(0, 5, 0.05):
        body = make_test_body(velocity=[0, 0, -50], altitude=1000)
        forces.append(np.linalg.norm(para.compute_force(body, t=t)))
    
    baseline = json.loads(Path("tests/baselines/continuous_inflation.json").read_text())
    np.testing.assert_allclose(forces, baseline, rtol=1e-4)
```

The first time you run, the test fails because the baseline doesn't exist. You manually save the current output as the baseline. From then on, the test catches *any* change — silent regressions impossible.

Genuinely powerful for the parachute models. If you change `_force_continuous_inflation` next year, you know immediately whether behavior changed.

---

# Part VIII — Git

## 31. The basic flow

The five commands you'll use 95% of the time:

```bash
git status              # what's changed?
git diff                # show me what changed
git add <file>          # stage this file for commit
git commit -m "..."     # save a snapshot
git log --oneline       # show recent commits
```

The two recovery commands:

```bash
git checkout .          # throw away ALL uncommitted changes
git checkout <file>     # throw away changes to one file
```

The two "what's the situation" commands:

```bash
git branch -v                           # all branches with last commit
git log --oneline --all --graph -20     # visualize branches
```

---

## 32. Branching strategy for solo work

Even working alone, branches help. The rule:

**One task per branch. Branch name = task ID.**

```bash
git checkout main
git pull
git checkout -b p0-t1-scenario-connect
# ... work ...
git checkout main
git merge p0-t1-scenario-connect
git branch -d p0-t1-scenario-connect
```

### Why bother with branches when working alone?

- **Isolation.** If the task goes sideways, `git checkout main` returns you to known-good.
- **Cleaner history.** `git log` shows logical units of work, not "fix typo, oops, that didn't work, ok now."
- **Squashing.** If you made 10 commits during a task, squash them into one before merging:
  ```bash
  git checkout main
  git merge --squash p0-t1-scenario-connect
  git commit -m "Fix CRIT-1: ..."
  ```

### Naming branches
- Task ID for tasks: `p0-t1-scenario-connect`.
- For ad-hoc work: `fix-NaN-in-quaternion`, `add-isa-atmosphere`.
- Prefix experimental/throwaway branches with `try-`: `try-jax-backend`.

---

## 33. Stash

You're mid-edit. Tests are failing. Suddenly you need to fix something on `main`. You don't want to commit the half-done work.

```bash
git stash                  # save current changes; revert to clean
git checkout main
# fix the urgent thing
git checkout your-branch
git stash pop              # restore your changes
```

`git stash` is your "shelf for in-progress work."

### Stash multiple things
```bash
git stash push -m "WIP atmosphere refactor"
git stash push -m "WIP test scaffolding"
git stash list             # show all stashes
git stash pop stash@{1}    # restore a specific one
```

### Don't leave stashes hanging
For anything you might want to keep more than a day, **commit instead** — even with a `WIP` message. You can squash later.

---

## 34. Rebase vs merge

Both combine work of two branches. They produce different histories.

### Merge
```bash
git checkout main
git merge feature-branch
```

Produces a "merge commit" that has two parents. History shows both lines of work.

### Rebase
```bash
git checkout feature-branch
git rebase main
```

Re-applies your feature commits on top of `main`. The history looks linear, as if you'd never branched.

### When to use which (solo work)

- **Merge** when you want to preserve the fact that work happened on a branch (e.g., a multi-week refactor).
- **Rebase** when the branch is short-lived and you want a clean linear history (e.g., a one-day bug fix).
- **Either** is fine — pick a style and stick with it.

### Don't rebase work you've pushed to a shared remote

Rebasing rewrites commit hashes. If others have pulled the old commits, rebase causes confusion. As a solo dev this rarely matters.

### Conflict resolution
Both merge and rebase can have conflicts. Git stops, marks the conflicting files with `<<<<<<<`, `=======`, `>>>>>>>` markers. You edit the files to resolve, then:

```bash
git add <resolved-files>
git commit                # if merging
git rebase --continue     # if rebasing
```

If conflicts feel overwhelming:
```bash
git merge --abort         # back out the merge
git rebase --abort        # back out the rebase
```

---

## 35. Reflog — your time machine

`git reflog` shows every action git has done. It's how you recover when "the commit is gone."

```bash
git reflog
# Output:
# 2bea5f8 HEAD@{0}: commit: Post-merge cleanup
# c4a1526 HEAD@{1}: merge main: Merge made by the 'ort' strategy.
# dcc405b HEAD@{2}: commit: Add plain-language review and a how-to-work guide
```

Every state your repo has been in is here.

### Recover a "lost" commit
```bash
git reset --hard <sha>     # WARNING: discards current state
git checkout <sha>         # detached HEAD; safe to look around
git checkout -b recovery   # create a branch from this state
```

### Reflog has limits
By default, reflog entries expire after 30-90 days. Don't use it as backup — use commits/branches for things you actually want to keep.

---

## 36. Recovering from disasters

A short list of "I screwed up, what now."

### "I committed to the wrong branch"
```bash
git log -1                          # note the SHA of your wrong commit
git reset --hard HEAD~1             # remove the commit from this branch
git checkout right-branch
git cherry-pick <sha>               # apply the commit to the right branch
```

### "I committed something that should have been in two commits"
```bash
git reset HEAD~1                    # undo the commit, keep changes unstaged
git add file1
git commit -m "First logical change"
git add file2
git commit -m "Second logical change"
```

### "I committed sensitive data (password, key)"
1. **Don't push** if you haven't already.
2. Remove the file: `git rm --cached <file>`
3. Add to `.gitignore`.
4. Amend or reset.
5. If you already pushed: rotate the credential immediately.

### "I want to throw away everything I've done since this morning"
```bash
git reflog                          # find the morning's SHA
git reset --hard <morning-sha>
```

### "I deleted a branch I needed"
```bash
git reflog                          # find the SHA of the deleted branch's tip
git checkout -b recovered <sha>
```

The pattern: **commits aren't really gone for 30-90 days**. You can always recover. Don't panic.

---

## 37. Writing good commit messages

The commit message is a letter to your future self.

### The shape

```
Subject line: short summary of what changed (≤72 characters)

Optional body: WHY the change was made. What problem does it solve?
Wrap at 72 columns.
```

### Subject line conventions

- **Imperative mood.** "Fix the parachute bug," not "Fixed the parachute bug." Completes the sentence "If applied, this commit will…"
- **Capitalize the first letter.**
- **No trailing period.**
- **Be specific.**

### Body — the "why"

Write the body if:
- The reason isn't obvious from the diff.
- You considered an alternative and rejected it.
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

### Conventional commits (optional)

Prefix the subject with a type:
- `fix:` — bug fix
- `feat:` — new feature
- `refactor:` — code change with no behavior change
- `docs:` — documentation only
- `test:` — test changes only
- `chore:` — maintenance

Example: `fix(scenario): propagate constraint to World in connect()`

---

# Part IX — Vibecoding (working with AI)

Vibecoding — using AI assistants like Claude Code as part of your workflow — is a real skill. Done well, it accelerates you 2-3×. Done badly, you produce code you don't understand and learn nothing. This section is the longest because the difference between "well" and "badly" matters most here.

## 38. The mental model — AI as a senior coworker

Treat the AI like a senior coworker you can ask anything without feeling stupid:

- **Smart.** Understands code, math, libraries, patterns.
- **Patient.** Will re-explain something 5 different ways if needed.
- **Confident, sometimes wrongly so.** Sounds authoritative even when wrong. *Verify what matters.*
- **No long-term memory.** Doesn't remember last week unless you remind it.
- **Contextually limited.** Knows your codebase only as much as you've shown it.
- **Imperfect at math.** Better than ever, but still occasionally arithmetic-wrong.
- **Up-to-date but not omniscient.** Training cutoff exists.

### What this means in practice
- Ask broad questions when exploring; ask narrow questions when fixing.
- Always verify when correctness matters.
- Provide context (file paths, error messages, the relevant code).
- Don't expect it to remember; restate context each session if needed.
- Push back when something feels off.

### What this is NOT
- Not a replacement for understanding.
- Not a black box that produces answers — it's a collaborator that produces drafts.
- Not always right.

---

## 39. Prompt patterns that work

Specificity beats brevity. Context beats cleverness.

### Pattern 1: state the goal, the context, and the constraint

Bad:
> fix my code

Better:
> fix the bug in scenario.py

Good:
> I'm working on task P0-T1 from `docs/WORK_PLAN.md` (fix CRIT-1: Scenario.connect doesn't propagate the tether constraint to World). Read `src/aerislab/api/scenario.py` and `src/aerislab/core/simulation.py`. Tell me the smallest change that fixes the bug without breaking existing tests, and write a regression test in `tests/test_scenario_api.py`.

The good version states **goal**, **context**, **constraint**.

### Pattern 2: ask for explanation, not just code

Instead of "implement an ISA atmosphere model," try:

> Walk me through what an ISA atmosphere model is, why we need it for AerisLab, and what API would fit best with the existing `Drag` and `ParachuteDrag` classes (I've attached `src/aerislab/dynamics/forces.py`). Then propose an implementation plan with files to add, before you write any code.

Plan before code, you can correct misunderstandings before they're baked in.

### Pattern 3: ask for tradeoffs, not decisions

> I'm considering two designs for the wind model: (a) a `Wind` Protocol with multiple implementations, (b) a single `Wind` class with a `mode=` parameter. What are the trade-offs given the rest of the codebase's style? Don't decide for me; lay out the considerations.

You stay in control; you learn from the analysis.

### Pattern 4: paste the actual error

> I got this error when running tests:
>
> ```
> [paste the full traceback here, exactly as it appeared]
> ```
>
> The relevant source file is `src/aerislab/...`. What's the most likely cause?

Don't summarize. Don't paraphrase. Copy-paste exactly.

### Pattern 5: ask for the smallest possible thing

> Show me the absolute minimum change needed to make this test pass. I'll improve it later.

Forces the AI to focus rather than over-engineer.

### Pattern 6: ask for a review of code you wrote

> Here's the function I wrote to fix Bug #2. Review it: are there edge cases I missed? Is it consistent with the rest of `logger.py`? Is the error message useful?
>
> ```python
> [paste your code]
> ```

Use the AI as a code reviewer. Cheap, instant, often catches real issues.

### Pattern 7: explain a concept

> Explain what "symplectic integrator" means in plain English, then explain why semi-implicit Euler is symplectic. I have a basic understanding of ODEs but not numerical methods.

Calibrate the explanation to your level. The AI tunes accordingly.

---

## 40. Prompt patterns that fail

### Anti-pattern 1: vague goals
"Make this better." → produces generic suggestions you could have generated yourself.

### Anti-pattern 2: no context
"Fix this:" followed by 5 lines of code with no explanation of what it's supposed to do.

### Anti-pattern 3: yes-or-no questions when you want analysis
"Is X a good idea?" → AI tends to say yes. Better: "What are the downsides of X?"

### Anti-pattern 4: chained instructions in one mega-prompt
"Read X, then refactor Y, then test Z, then deploy W." → AI tries to do all of it, half-asses each step. Break into smaller turns.

### Anti-pattern 5: accepting code without reading it
You ask for code; AI produces it; you commit. You learn nothing. Six months later you can't fix bugs in your own codebase. **The single biggest vibecoding failure mode.**

### Anti-pattern 6: using AI to skip thinking on the design
"Design the architecture for me." → produces something plausible-looking that won't fit your specific needs. Architecture is your call.

### Anti-pattern 7: re-asking the same question when you don't like the answer
The AI's next answer probably won't be much better. Change the question or add context.

---

## 41. Reviewing AI output — the most important vibecoding skill

If you take one thing from this whole document, take this: **never accept code you can't explain back in plain words**.

### The review checklist

For every AI-generated change, ask:

1. **Does it actually do what I asked?** Read it. Don't trust the AI's summary.
2. **Does it make sense given the rest of the codebase?** Same patterns? Same naming? Same style?
3. **Are there obvious issues?**
   - Hardcoded values that should be parameters
   - Missing edge cases (empty list, None, NaN)
   - Variables shadowing builtins (`list`, `type`, `id`)
   - Off-by-one errors in indexing or loops
   - Mutable default arguments (`def foo(x=[])`)
4. **Does it use APIs that actually exist?** AI sometimes invents methods.
5. **Are the imports right?** AI sometimes imports from the wrong module.
6. **Did it write tests?** If the task warranted them, are they testing the right thing?
7. **Is there commented-out code, debug prints, or TODO markers it should have removed?**

Run this checklist for every change. After a month it becomes automatic.

### Run the tests
Even if the code looks right. Always.

```bash
venv/bin/pytest --tb=short
```

If tests fail, tell the AI what failed. Don't guess.

### Read the diff
`git diff` after the AI has made changes. Read every line. If a line confuses you, ask "explain this line." Don't accept lines you can't explain.

### Specific red flags
- The AI changed more files than you asked it to.
- Comments like `# Auto-generated, don't modify`.
- Imports from packages that aren't in `pyproject.toml`.
- Use of features from a Python version newer than yours.
- "Just trust me" energy in the explanation.

---

## 42. Different models for different tasks

Different AI models have different strengths.

### Opus — the heavy thinker
- Hardest tasks: complex reasoning, multi-file analysis, architecture design.
- Slower, more expensive per token.
- Use when you need real depth.

### Sonnet — the workhorse
- Most day-to-day work.
- Good balance of capability and speed.
- Default for most tasks.

### Haiku — the speed demon
- Fast, cheap.
- Quick answers, simple edits, formatting tasks.

### When to switch
- Stuck on a hard problem with Sonnet → switch to Opus, restate the problem fresh.
- Iterating quickly on small fixes → switch to Haiku.
- Need detailed code review → Opus.
- Need a quick rename across files → Haiku.

---

## 43. Context windows

The AI's "memory" of your conversation has a limit. Manage it.

### What to keep in context
- The current task description.
- Files relevant to the current change.
- Recent code you've written.
- Error messages from the current debugging session.

### What to discard
- Old completed tasks.
- Detailed explanations of things you already understand.
- Files you're no longer touching.
- Old conversation that's not relevant anymore.

### Tactics for managing context
- **Start a new conversation when switching tasks.** Especially in long sessions, old context becomes noise.
- **Summarize key decisions in a `notes.md` file** that you can re-paste at the start of a new session.
- **Reference files by path** rather than pasting content when the AI can read them directly.
- **Trim long error logs** to the most relevant 20 lines before pasting.

### Conversation length
A conversation that runs for hours accumulates noise. Don't be afraid to start fresh. Anything important is in your code, your commits, your `notes.md`, or the work plan — not in the chat scrollback.

---

## 44. Workflow patterns

A few specific ways to actually use AI day-to-day.

### Pattern A: pair programming
You drive. AI is the navigator.

```
You: "I'm about to add a wind model. I think it should be a Protocol with
      Constant and AltitudeProfile implementations. The Protocol method is
      velocity(p, t). Sound right?"
AI:  "Yes, plus consider..."
You: "Good point. Let me write the Protocol first."
... (you write code) ...
You: "OK what do you think of this?"
AI:  "Looks good, one suggestion: ..."
```

You keep agency. AI keeps you on track.

### Pattern B: code review
You wrote code. AI reviews.

```
You: "Review this function. Look for bugs, edge cases I missed, and
      inconsistency with the rest of forces.py."
AI:  "I noticed three things: ..."
```

### Pattern C: exploration
You don't know what you want yet.

```
You: "I want to add atmospheric drag at altitude. What are the options for
      the atmosphere model? What do real aerospace simulators use? What
      level of fidelity do I need for my use case (parachute drops up to 5 km)?"
AI:  [walks through ISA, COESA, NRLMSISE, etc.]
You: "OK, ISA is enough. What's the complexity to implement?"
```

Use AI to explore the design space before committing.

### Pattern D: documentation
You did the work. Now you need a docstring or commit message.

```
You: "Here's the function I just wrote. Write a NumPy-style docstring for it,
      matching the style of the other docstrings in this file."
```

Tedious work AI is genuinely good at.

### Pattern E: rubber-duck-with-feedback

```
You: "I'm trying to figure out why this test fails after I added X. The error
      is Y. I've checked Z and W. What am I missing?"
AI:  "Have you checked V?"
You: "Oh, I didn't think of that."
```

Articulating the problem often surfaces the answer. AI sometimes adds the missing piece.

### Pattern F: AI as a search engine for your codebase
"Where does ISA atmosphere get applied?" "Which test covers the deployment state machine?"

For navigation in an unfamiliar codebase, AI is faster than grep when you don't know the exact term.

---

## 45. Avoiding learned helplessness

The biggest long-term risk of AI assistance is **learned helplessness** — losing the ability to do things yourself.

Symptoms:
- Can't read a stack trace without pasting it to AI.
- Can't write a function without AI scaffolding.
- Don't remember syntax for things you do daily.
- Frustration spike when AI is unavailable.
- "I don't know if this is right" feeling about your own code.

### Counter-tactics

- **Once a week, do a task with no AI.** Force yourself. Read documentation. Use the REPL. Write the code yourself.
- **Re-do a task you did with AI, alone.** Pick a finished task. Throw away the work. Try to do it again from scratch without AI.
- **Type out AI's answers instead of copy-pasting.** Slower, but builds muscle memory.
- **Periodically ask AI not to write code, only to advise.** "Tell me what to change but don't show me the code; I'll write it."
- **Read code before asking AI about it.** Don't paste a 100-line file with "what does this do?" Read it for 5 minutes first; ask only the parts that are still unclear.

### The right ratio
Early on, ~70% AI assistance is fine — you're learning the codebase and patterns. After a year, aim for 30-50% AI assistance. After two years, AI should be a productivity multiplier on tasks you could do without it.

---

## 46. The "explain it back" test

Before merging any code, AI-written or otherwise: **explain it back, in plain words, to a hypothetical other person**.

If you can:
> "This function takes the body's velocity and the wind vector, computes the airspeed, and uses it to calculate drag force. The wind defaults to zero so existing tests still work."

You understand it. Merge.

If you can only manage:
> "Uh, it does drag stuff. Adds wind I think. I asked for it."

You don't understand it. **Don't merge.** Ask the AI to walk you through, line by line, until you can explain it.

This test is the difference between vibecoding-as-learning and vibecoding-as-cargo-culting.

---

# Part X — Performance & profiling

## 47. When NOT to optimize (almost always)

The most common performance mistake in scientific Python is optimizing too early.

- **Choosing data structures based on imagined performance** before any measurement.
- **Switching to a "faster" library** because someone on the internet said it was faster.
- **Vectorizing code** that runs once at startup.
- **Caching results** that are only computed once.
- **Switching to Numba/Cython/C** before profiling shows Python is the bottleneck.

The cost of premature optimization isn't just wasted time — it's *worse code*. Optimized code is harder to read, harder to debug, and harder to change.

### The rule

Don't optimize until:
1. You've measured.
2. The measurement shows a real bottleneck.
3. You've identified *which line* is the bottleneck.
4. You've confirmed that fixing it would meaningfully change the user experience.

For research code, the bar is even higher: don't optimize until something is *too slow to use*.

---

## 48. How to know if optimization is needed

AerisLab's parachute simulation runs in seconds. A 60-second simulated drop takes 5-30 wall-clock seconds.

Is that a problem? Probably not — for prototyping, you run it occasionally. **It's not slow.**

Becomes a problem when:
- You want 10,000 Monte Carlo runs (then 5s × 10,000 = 14 hours).
- You want gradient-based design optimization (1000 evaluations × 5s = 1.5 hours per step).
- You want real-time (need <0.01 s).

Until you have a use case like one of those, **don't optimize**. Move on.

### The performance benchmark

When you do start caring:

```bash
venv/bin/pytest tests/perf/ --benchmark-only
```

(Once `pytest-benchmark` benchmarks exist — task P3-T10.) This gives you a baseline.

---

## 49. Profiling tools

When you've decided to optimize, profile first. Don't guess.

### cProfile — built into Python

```bash
venv/bin/python -m cProfile -o profile.out -s cumulative my_script.py
```

Then:
```bash
venv/bin/python -m pstats profile.out
> sort cumulative
> stats 30                  # top 30 slowest functions
```

Or visually:
```bash
pip install snakeviz
snakeviz profile.out
```

cProfile adds overhead. Use it for relative comparison.

### py-spy — sampling profiler, low overhead

```bash
pip install py-spy
sudo py-spy record -o flamegraph.svg -- venv/bin/python my_script.py
```

Produces a flame graph. Wide bars = slow functions. Click to zoom. **My favorite tool.**

### scalene — measures CPU AND memory AND GPU

```bash
pip install scalene
scalene my_script.py
```

Newer. Particularly useful when memory allocation might be the issue.

### Reading a profile output

What you're looking for:
- **Hot functions** — what fraction of total time is in each function?
- **Hot lines** — within a hot function, which line dominates?
- **Allocation rate** — are we creating millions of temporary arrays?
- **Surprises** — anything taking >5% of time you didn't expect.

The bottleneck is almost never where you think.

---

## 50. Common Python performance wins

In rough order of frequency:

### 1. Stop recomputing the same thing
The single biggest source of slowness in scientific Python.

```python
# SLOW — recomputes Minv every loop
for step in steps:
    Minv = body.inv_mass_matrix_world()
    a = Minv @ F

# FAST — compute once, reuse
Minv = body.inv_mass_matrix_world()
for step in steps:
    a = Minv @ F
```

For AerisLab: the inverse mass matrix only changes when orientation changes. Caching it (with proper invalidation) is a 2-5× speedup. Task P2-T5.

### 2. Vectorize loops
A Python loop over 10000 elements doing scalar math is ~50× slower than the numpy equivalent.

### 3. Use the right numpy operation
- `np.dot(a, b)` faster than `(a*b).sum()`.
- `np.einsum` sometimes faster than chained `@` for complex contractions.
- `np.where` faster than `if`/`else` in a loop.

### 4. Avoid creating unnecessary arrays
```python
# SLOW — creates a temporary
result = (a + b) * c

# FAST — in-place
result = a.copy()
result += b
result *= c
```

For tight loops with large arrays, this matters.

### 5. Use scipy's solvers, not your own
`scipy.linalg.solve` is much faster than implementing Gaussian elimination yourself.

### 6. The order of operations matters

```python
# SLOW — creates huge intermediate
A @ b @ c    # (n×n) @ (n×n_small) → big intermediate

# FAST — small intermediate
A @ (b @ c)  # (n×n_small) → small intermediate
```

Math is associative; performance isn't.

### When to leave Python entirely

- **Numba** — JIT-compile Python to machine code with a decorator. Often 10-100× speedup.
- **Cython** — write Python-like code that compiles to C. More effort.
- **JAX** — rewrite in JAX for GPU + autodiff. Big effort.
- **C/C++/Fortran extension** — ultimate, painful.

For your project, you're nowhere near needing these.

---

# Part XI — Working alone

## 51. Code review for one

In a team, someone else reviews your PRs. Solo, you have no one. Simulate the review:

### Tactic 1: time delay

Don't merge a branch the same day you wrote it. Wait at least overnight. Re-read with fresh eyes the next morning.

### Tactic 2: the "stranger walking by" mental check

Imagine a colleague walking by your desk. They glance at the diff. Do they say "this looks fine" or "wait, what's going on here"? If you can't be sure they'd say "fine," ask why and fix it.

### Tactic 3: AI as code reviewer

```
You: "Review this diff for any bugs, code-smell, or inconsistency with the
      rest of the codebase. Be critical."
[paste git diff]
```

### Tactic 4: rubber-duck the change

Before committing, write down (in your `notes.md`):
- What did I change and why?
- What edge cases did I think about?
- What could break?

The act of articulation surfaces gaps.

### Tactic 5: read the full diff before merging

`git diff main..feature-branch` shows the entire difference. Read every hunk. **Don't merge things you haven't read.**

---

## 52. Documentation as future-self insurance

Documentation = letters to your future self. The single most valuable kind: **why** documentation.

### Inline (in code)
- Docstrings on public functions/classes.
- Comments on the *why* of unusual decisions.
- Module-level docstrings explaining what the file is for.

### Top-level project docs
- `README.md` — what the project does, how to install, how to run an example.
- `CHANGELOG.md` — what changed in each version.
- The `docs/` directory — any extended notes.

### Architectural Decision Records (ADRs)
For big decisions, write a 1-page ADR:

```markdown
# ADR 001: Use numpy state vectors instead of object-mutation in the IVP solver

## Context
The current IVP solver mutates RigidBody6DOF objects inside its RHS...

## Decision
Refactor World to expose state(), set_state(), state_derivative()...

## Consequences
- Pro: enables ML surrogate plumbing
- Pro: eliminates double-RHS-for-logging anti-pattern
- Con: 1-2 weeks of refactoring effort
- Con: temporarily breaks any user code that called the old internal API

## Date
2026-05-13
```

After 30 ADRs, you have a record of every major choice and *why*. Invaluable when writing the thesis chapter on architecture.

### "Future me will figure it out" is a lie

You won't. Either you'll be working on something else, or you'll have forgotten the context, or both. Write it down now.

---

## 53. Talking to people anyway — online communities

You don't have a team but you do have a community.

### Stack Overflow
Best for: specific Python / library questions with a clear answer.
- Search first. Most things are already asked.
- When asking, follow the format: minimal reproducer, expected vs actual, what you tried.
- Tag appropriately (`python`, `numpy`, `scipy`).

### GitHub issues
Best for: bugs in libraries you use, feature requests, questions about specific projects.
- Read existing issues first.
- Be specific (versions, OS, reproducer).
- Be patient — maintainers are volunteers.

### Reddit (r/Python, r/learnpython, r/scipy)
Best for: general advice, library recommendations.
- Mixed quality, treat advice critically.

### Discord / Slack communities
Many libraries have a Discord. Real-time help, friendly. Search "<library> discord."

### Don't be shy
You're not stupid for asking. Communities exist because people want to help. Cost of asking: 5 minutes. Cost of staying stuck: a day.

### Etiquette
- Ask once, in the right place. Don't cross-post.
- When you get an answer, follow up with what worked.
- Pay it forward — answer questions you know, even if your knowledge is small.

---

## 54. Keeping a learning log

Get a notebook. At the end of each work session, write **two sentences**:

- What was the hardest thing today?
- What's one thing that clicked?

Examples:

> *2026-05-14:* Fixing Bug #1 was easier than expected once I traced the code with Claude. The hard part was understanding why `world.add_system` captures constraints by value — it's because the `for` loop iterates over the snapshot of the current contents, and the loop had already finished before connect() was called.

> *2026-05-15:* Spent two hours trying to figure out why my new test was failing. Turned out I was modifying a shared fixture between tests — pytest fixtures with `scope="module"` are reused. Lesson: keep fixtures `scope="function"` (the default).

This habit feels silly for the first two weeks. After two months, you have a record of every concept you struggled with and how you got past it. After a year, you have a personal textbook calibrated to *your* learning.

### What else to track
- New tools/libraries you tried.
- Tasks completed (with actual time vs estimated time).
- Mistakes worth remembering.
- Ideas for follow-up work.
- "Aha" moments.

### Review monthly
First Sunday of each month: read the past month's entries. Patterns emerge. You've grown more than you realize.

---

# Part XII — Rhythm & sustainability

PhD work is a marathon. The biggest determinant of how you finish is not your peak speed — it's your sustainability.

## 55. Daily rhythm

A workable daily template (adjust for your life):

### Morning (peak focus)
- 10 min: pre-work ritual (`git status`, `pytest`, check work plan)
- 90 min: the hardest task of the day (deep focus)
- 15 min break
- 60 min: a second focused task
- Lunch + walk

### Afternoon (medium focus)
- 60 min: a smaller task or extension of morning work
- 30 min: documentation, learning log, code review of own work
- Long break

### Late afternoon (low focus)
- 30 min: clean up, commit messages, plan tomorrow
- 15 min: write learning log entry

### Evening
- Off. Don't work. Brain processes things in the background.

Important elements:
- **One hard task per day**, max two.
- **Breaks every 60-90 minutes**, longer ones every 3-4 hours.
- **End cleanly** with a commit and a plan-for-tomorrow note.

---

## 56. Weekly rhythm

- **Monday:** plan the week. Pick 1-3 tasks. Block calendar time.
- **Tuesday-Thursday:** execute. Most coding here.
- **Friday:** finish, document, push, plan next week. No starting new big things.
- **Weekend:** off (genuinely off — your brain needs it).

### Friday cleanup ritual

- Merge any green branches.
- Push everything to GitHub (off-machine backup).
- Update the work plan (mark done tasks `[x]`).
- Write a short weekly summary in your learning log.

This 30-minute habit prevents work from accumulating into chaos.

---

## 57. Monthly rhythm

Once a month:

- Re-read the past month's learning log entries.
- Re-read the work plan; check if priorities have shifted.
- Look at git log: what got done? Anything you've been avoiding?
- Update the review document if any of its findings are now stale.
- Take a longer break (a real day off; no thinking about the project).

Once a quarter:

- Tag a release (even pre-1.0 — `v0.2.x` etc.).
- Write a release note in `CHANGELOG.md`.
- If supervisor exists, send a short progress update.

---

## 58. When you hit a wall

### Wall #1: the impossible bug
You've been debugging for 2+ hours, no progress.
- Stop.
- Write down what you've tried.
- Write down what you know is true.
- 30-minute walk.
- Come back; explain the problem to AI as if it's an idiot.
- If still stuck after another hour: post on Stack Overflow.
- If still stuck the next day: skip this for now, do something else, come back next week.

### Wall #2: the wrong design
You've been refactoring for days, not converging.
- Stop.
- Look at git: roll back to before the refactor.
- Re-read your design notes. Was the design actually right?
- Don't be afraid to abandon a refactor.

### Wall #3: low motivation
- Acknowledge it.
- Do an easy task instead. Documentation, cleanup. Build momentum.
- Take an actual day off.
- If it persists for 2+ weeks: talk to someone.

### Wall #4: the confidence crisis
"I'm not smart enough for this." Hits everyone.
- Look at your git log. You've done dozens of things.
- Read your learning log. You've grown.
- Talk to other PhD students. They feel the same way.

---

# Part XIII — Worked examples

## 59. Example 1: fixing a bug (P0-T1)

The first task in the work plan: fix Bug #1.

### Step 1 — Open the task description

Go to `WORK_PLAN.md`, find P0-T1. Read it twice.

What I want to absorb:
- **What** is the bug.
- **Where** it lives.
- **Why** it's a bug.
- **The acceptance criteria.**
- **The verification command.**

### Step 2 — Prep workspace

```bash
git status                                   # confirm clean
venv/bin/pytest --tb=short                   # confirm tests pass
git checkout -b p0-t1-scenario-connect       # new branch
```

### Step 3 — Reproduce the bug

```bash
venv/bin/python -c "
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

Output: `System constraints: 1`, `World constraints: 0`. Confirmed.

### Step 4 — Read the code with the bug in mind

Open `scenario.py::connect` and `simulation.py::add_system`. Trace the call order. Confirm the diagnosis.

### Step 5 — Smallest change

Hotpatch: add `self.world.add_constraint(constraint)` at end of `Scenario.connect`. Proper fix later (P2-T11).

### Step 6 — Failing test first

```python
def test_scenario_connect_propagates_constraint_to_world():
    """Regression test for CRIT-1 / Bug #1."""
    from aerislab.api.scenario import Scenario
    from aerislab.components.standard import Payload, Parachute
    payload = Payload(name='cap', mass=50.0, radius=0.4, position=[0,0,2000])
    chute = Parachute(name='main', mass=5.0, diameter=12.0, model='knacke',
                      activation_altitude=1500, position=[0,0,2000.5])
    sc = (Scenario(name='test').add_system([payload, chute])
          .connect(payload, chute, type='tether', length=10.0))
    assert len(sc.current_system.constraints) == 1
    assert len(sc.world.constraints) == 1
```

Run it:
```bash
venv/bin/pytest tests/test_scenario_api.py -v
```
Fails. Good.

### Step 7 — Make the fix

```python
# scenario.py, in connect():
constraint = joint.attach(self.current_system.get_bodies())
self.current_system.add_constraint(constraint)

# Hotpatch for CRIT-1; proper fix is task P2-T11.
self.world.add_constraint(constraint)
```

### Step 8 — Test passes

```bash
venv/bin/pytest tests/test_scenario_api.py -v
```
Passes.

### Step 9 — Full test suite

```bash
venv/bin/pytest
```
All pass.

### Step 10 — Smoke test

```bash
venv/bin/python examples/scenarios/02_parachute_system.py 2>&1 | tail -10
```

Touchdown velocity now in expected range.

### Step 11 — Commit

```bash
git add src/aerislab/api/scenario.py tests/test_scenario_api.py
git commit
```

Real commit message (§37).

### Step 12 — Mark done

In `WORK_PLAN.md`, change `[ ]` to `[x]` for P0-T1.

### Step 13 — Merge

```bash
git checkout main
git merge p0-t1-scenario-connect
git branch -d p0-t1-scenario-connect
```

### Step 14 — Break

That whole process: 45 minutes the first time. With practice, 15.

---

## 60. Example 2: implementing a feature (atmosphere model)

A bigger task: implement ISA atmosphere (P1-T9).

### Step 1 — Understand the concept

If you don't know what ISA is, find out *before* opening the editor.

```
You: "Explain the International Standard Atmosphere (ISA) at the level of
      a PhD student in aerospace who hasn't worked with atmospheric models
      before. What does it model? Inputs/outputs? Layers?"
```

Read. Maybe Wikipedia. Now you know:
- ISA divides atmosphere into layers (troposphere 0-11 km, etc.).
- Each layer has a temperature lapse rate.
- Density and pressure follow from temperature.
- Standard sea-level values: T=288.15 K, P=101325 Pa, ρ=1.225 kg/m³.

### Step 2 — Sketch the API

Before any code:

```
class AtmosphereModel(Protocol):
    def density(self, altitude: float) -> float: ...
    def temperature(self, altitude: float) -> float: ...
    def pressure(self, altitude: float) -> float: ...

class ISA:
    def __init__(self):
        # standard sea-level values
        ...
    def density(self, altitude): ...
```

### Step 3 — Set up

```bash
git checkout main
git checkout -b p1-t9-isa-atmosphere
mkdir -p src/aerislab/models/atmosphere tests/models
touch src/aerislab/models/atmosphere/{__init__,base,isa,exponential}.py
touch tests/models/test_atmosphere.py
```

### Step 4 — Protocol first

```python
# base.py
from typing import Protocol

class AtmosphereModel(Protocol):
    """Atmospheric properties as a function of altitude."""
    def density(self, altitude: float) -> float: ...
    def temperature(self, altitude: float) -> float: ...
    def pressure(self, altitude: float) -> float: ...
```

### Step 5 — Implement ISA

Reference: Wikipedia, NASA SP-7012, or AI summary. Write the code carefully.

```python
# isa.py
class ISA:
    """International Standard Atmosphere (ICAO Doc 7488)."""
    LAYERS = [
        (0,     288.15, 101325.0, -0.0065),
        (11000, 216.65,  22632.1,  0.0),
        (20000, 216.65,   5474.9,  0.001),
        (32000, 228.65,    868.0,  0.0028),
    ]
    R_SPECIFIC = 287.058
    G = 9.80665
    
    def temperature(self, altitude):
        h_base, T_base, _, L = self._find_layer(altitude)
        return T_base + L * (altitude - h_base)
    
    def pressure(self, altitude):
        h_base, T_base, P_base, L = self._find_layer(altitude)
        T = self.temperature(altitude)
        if L == 0.0:
            return P_base * np.exp(-self.G * (altitude - h_base) / (self.R_SPECIFIC * T_base))
        else:
            return P_base * (T / T_base) ** (-self.G / (self.R_SPECIFIC * L))
    
    def density(self, altitude):
        return self.pressure(altitude) / (self.R_SPECIFIC * self.temperature(altitude))
    
    def _find_layer(self, altitude):
        for i in range(len(self.LAYERS) - 1):
            if altitude < self.LAYERS[i+1][0]:
                return self.LAYERS[i]
        return self.LAYERS[-1]
```

### Step 6 — Tests against tabulated values

```python
@pytest.mark.parametrize("altitude,expected_density", [
    (    0,  1.2250),
    ( 5000,  0.7361),
    (11000,  0.3639),
    (20000,  0.0880),
])
def test_isa_density_matches_tabulated(altitude, expected_density):
    isa = ISA()
    actual = isa.density(altitude)
    assert abs(actual - expected_density) / expected_density < 0.005
```

Run, iterate until passing.

### Step 7 — Commit each milestone

Protocol → commit. Basic ISA implementation → commit. Tests pass → commit. Exponential model → commit.

### Step 8 — Don't wire into Drag yet

P1-T11 is a separate task. Different branch.

That whole task: 4-6 hours including reading the standard and writing tests. **Split across two sessions if needed** — the formulas are easy to mess up when tired.

---

## 61. Example 3: doing a refactor safely

A Phase 2 task: introduce `World.state / set_state / state_derivative` (P2-T1).

### Step 1 — Plan on paper, not in editor

Sit down with markdown file. Sketch:

```
What I'm trying to achieve:
  Pure-function state derivative; no World mutation inside IVP RHS.

API:
  World.state() -> NDArray
  World.set_state(y) -> None
  World.state_derivative(t, y) -> NDArray

Migration plan (small steps):
  1. Add state() and set_state() with round-trip test.
  2. Add state_derivative(t, y) — calls existing IVP RHS internals.
  3. Modify HybridIVPSolver.integrate() to use new methods.
  4. Verify all existing tests still pass.
  5. Refactor logging path.
  6. Verify all tests still pass.

At each step: tests must pass. Commit.

What can break:
  - State packing order (must match unpacking).
  - Tests that depend on bodies being mutated during integration.
  - Logging that reads forces from body.f after integration.
```

The plan is the most important step.

### Step 2 — Write the failing test first

```python
def test_world_state_round_trip():
    world = make_test_world()
    y_before = world.state()
    world.bodies[0].p[0] = 999.0
    world.set_state(y_before)
    y_after = world.state()
    np.testing.assert_array_equal(y_before, y_after)
```

Fails because methods don't exist yet.

### Step 3 — Implement state() / set_state()

Copy packing/unpacking from `HybridIVPSolver._pack` / `_unpack_to_world`. Move to `World`. Test passes. Commit.

### Step 4 — Implement state_derivative

Build by lifting code out of `HybridIVPSolver.integrate()`'s RHS. **Leave the IVP solver still using its old RHS initially.** Two paths exist temporarily.

Test that new path produces same ydot as old path. When equal, commit.

### Step 5 — Switch the IVP solver

Now `HybridIVPSolver.integrate()` uses `world.state_derivative(t, y)`. All 115 existing tests pass. If they don't, the issue is in the new code, and you have a much smaller diff to debug.

### Step 6 — Refactor logging path

Separate task; separate branch ideally.

### Each step is committable

If you have to stop mid-task, you can resume. Commits are stepping stones.

The refactor takes longer than you expect. **Slow progress is still progress; broken progress is regression.**

---

## 62. Example 4: hunting a mysterious test failure

You haven't worked for a week. Sit down. `pytest` shows three new failures.

### Step 1 — Confirm the failure is real

```bash
git status                                          # uncommitted state?
venv/bin/pip list | grep aerislab                   # right version?
venv/bin/pip install -e . --force-reinstall         # fresh install
venv/bin/pytest --tb=short
```

Eliminating "stale install" takes 30 seconds.

### Step 2 — Read the failure messages

What's failing? The same thing? Different things? Look for a pattern.

### Step 3 — Did anything change?

```bash
git log --oneline -10
pip list --outdated
```

A scipy update could change the IVP solver's behavior. **This happens.**

### Step 4 — Bisect

```bash
git bisect start
git bisect bad
git bisect good <last-week's-sha>
git bisect run venv/bin/pytest tests/test_solver_ivp.py
```

Git names the culprit.

### Step 5 — Fix the root cause

If your code: revert it, try different approach.

If a dependency: pin the version (`scipy==1.16.0`), file an upstream issue, or adapt.

If a flaky test: there's a race condition or shared mutable state. Track it down.

### Step 6 — Add a test that would have caught this

If a dependency update broke things, your CI matrix should include "latest dependencies." If a refactor broke things, your tests should have caught it before merge — make them stricter. Every "how did this slip through" is a chance to improve the safety net.

---

# Part XIV — Reference

## 63. Common traps and how to escape them

Things that have eaten weeks of other people's lives.

### "I'll just refactor while I'm fixing this bug"
**Escape:** finish the bug fix in one commit. Refactor in a separate commit/branch.

### "I changed something and now nothing works"
**Escape:** `git diff`, `git checkout .`. Future-proof: commit every 30 minutes, even "WIP".

### "The tests pass on my machine but fail in CI"
**Escape:** `pip install -e . --force-reinstall`; check `pyproject.toml`; run tests in a fresh venv.

### "I'm 4 levels deep in a debugging rabbit hole"
**Escape:** sticky note on monitor: "I am trying to fix bug A." If off-track for an hour: stop, note findings, **commit progress**, return to original task.

### "This problem is too big, I don't know where to start"
**Escape:** make it smaller. "Implement Model layer" too big. "Make stub AtmosphereModel with density(altitude) returning 1.225" doable in 10 minutes. Start there.

### "I'll just copy this code from somewhere similar"
**Escape:** copy is fine. Then read line by line and ask "is this right *here*?" Modify until it fits.

### "Let me read all the code first, then I'll know enough to start"
**Escape:** read with a question. Stop when you have the answer.

### "Nobody else would write this badly, what's wrong with me?"
**Escape:** look at git history of any famous project. They have commits like `wip`, `fix typo`, `actually fix it`. Programming is messy.

### "I should learn more before I attempt this"
**Escape:** the best way to learn X is to need X for Y. Start Y.

### "I'll just run a quick experiment in the production code"
**Escape:** scratchpad files (`scratch/try_x.py`) or separate branch (`try-x`). Never experiment in code about to be committed.

### "AI gave me code, I'll just paste it"
**Escape:** the explain-it-back test (§46).

### "Let me start the next task before finishing this one"
**Escape:** finish-then-start is a discipline.

### "I should use the latest fancy library"
**Escape:** use boring tools. numpy, scipy, pandas, matplotlib, pytest — they're proven.

---

## 64. The "I'm stuck" decision tree

```
┌─ How long stuck on this specific thing?
│
├─ <15 minutes
│   → Keep going. Normal.
│
├─ 15-30 minutes
│   → 5-minute break.
│
├─ 30-60 minutes
│   → Write down what you've tried.
│   → Reproduce with smallest possible script.
│   → Read the error again, slowly, all the way down.
│
├─ 1-2 hours
│   → Explain out loud (or to AI, or in writing).
│   → Search Stack Overflow / GitHub issues.
│   → Re-read the relevant library docs.
│
├─ 2-4 hours
│   → Actual break (lunch, walk).
│   → Try a different approach entirely.
│   → Ask Claude with full context.
│
├─ 4+ hours / next day
│   → Post on Stack Overflow with minimal reproducer.
│   → Consider: is this task right now?
│       - Move to another task; come back later.
│       - Scope down (simpler version).
│       - Reconsider the design.
│
└─ Multiple days
    → Talk to a human.
    → Accept a worse-but-shippable solution.
    → It's OK to abandon a path.
```

The longer you've been stuck, the more aggressive your unstucking tactics should become.

---

## 65. Habits to build over the next year

Don't try to adopt all at once — pick one a month.

| Habit | Why |
|---|---|
| Run full test suite before every commit | Prevents 90% of regressions |
| One task per branch, one logical change per commit | Makes the diff readable in 6 months |
| Read 10 lines of code for every line you write | The 80/20 reality of coding |
| Write a regression test for every bug you fix | Bug doesn't come back |
| Commit at least every hour | Never lose more than an hour |
| 5-minute break every 45 minutes | Bug rate goes up exponentially when tired |
| Learning log entry at end of each session | Compound learning |
| Re-read your own code the next day before merging | Catches obvious mistakes |
| Push back on AI suggestions you don't understand | The point is to learn |
| Plan the work *before* opening the editor | "Editor open" mode is for typing, not thinking |
| Do simplest thing that could possibly work | Avoids over-engineering |
| When stuck >30 minutes, write down what you've tried | The writing itself unblocks |
| When stuck >2 hours, ask for help | You've missed something |
| Update docs in same commit as code change | Drift makes docs worse than no docs |
| Use vectorized numpy instead of Python loops | 50× speedup, often free |
| Print `.shape` of every numpy array when debugging | Half of numpy bugs are shape mismatches |
| Treat warnings as errors | Early sign of real bugs |
| Pin dependencies for reproducibility | Future-you needs to regenerate figures |
| Ship 0.x versions; iterate | Perfect is enemy of done |
| Once a week, do a task with no AI | Build independent skill |

The *direction* matters. Pick one a month. Try it for two weeks.

---

## 66. A closing note

The single biggest thing I can tell you, having watched many people learn to code: **the difference between people who get good and people who don't is not talent. It's volume of deliberate practice and willingness to sit with confusion.**

You will spend hours stuck. You will write bad code. You will introduce bugs and not understand why. You will read code that feels like it was written in another language. **This is the work.** It's not a sign you're failing — it's a sign you're learning. The people who got good went through exactly the same phase, they just kept going.

The work plan in `WORK_PLAN.md` is going to take you a year of part-time work. That's fine. At the end of that year you will be a meaningfully different engineer than you are today, *and* you will have a working tool to show for it.

A few things to remember when it gets hard:

- **Doing the work is the work.** Most "productivity advice" is procrastination in disguise. Sitting down and doing the next thing is the only thing that works.

- **You're not behind.** You're not "supposed to be" anywhere by any specific date. Compare yourself to your past self.

- **Small wins compound.** Each fixed bug, each passing test, each working feature is a step. You don't see the staircase until you turn around.

- **The code you have today is enough to start.** Pick the next task. Do it. Move on.

- **Tools and frameworks come and go. Engineering judgment lasts.** Investing in the *thinking* (debugging, design, testing, communication) pays back forever.

- **The thesis is the goal, not perfection.** A working, well-validated tool that produces correct numbers and supports your research is success.

- **Ask for help.** You don't get points for suffering alone.

- **Take care of yourself.** Sleep, exercise, time off, real meals, real friends. Code is a tool for living, not the other way around.

Don't rush. Don't skip understanding. Don't be too proud to ask. Don't be afraid to make a mess.

Good luck.

---

*Expanded guide for Štěpán Kaspar, kaspar.stepan.cz@gmail.com. Last updated 2026-05-13. Updated regularly — this document grows with you.*
