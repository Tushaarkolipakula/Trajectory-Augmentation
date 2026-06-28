# Agentic AI Usage & Integration

## Our Agentic Setup

We relied heavily on AI coding assistants to accelerate development. As the intelligence engine for our agentic coding setup, we primarily used **Gemini 3.1 Pro (High)** and **Claude Opus 4.6 (with thinking)**. The two main agentic interfaces we used were:

- **Antigravity:** This was our primary coding agent. We used it for pretty much everything — writing Python scripts, debugging MuJoCo physics errors, parsing stack traces, restructuring the repository, and even running batch dataset conversions. It could chain together terminal commands, read files, and iterate on code autonomously.
- **AlphaXiv:** We used this early on for reading and extracting key ideas from research papers related to trajectory augmentation and VLA architectures.

## Reasoning & Planning Pipelines

The most useful pattern we discovered was the **Plan-Then-Execute** workflow. Before making any major changes (like restructuring the dataset pipeline or rewriting the augmentation logic), we'd have the agent generate a structured `implementation_plan.md` first. This forced it to stop, think through the full plan, and wait for our approval before touching any code.

This was genuinely helpful because without it, the agent would sometimes jump straight into implementing something and go down a wrong path. Having that planning checkpoint saved us a lot of time.

The Antigravity IDE also generates walkthroughs and artefects 

## Agent Skills & Behavioral Controls

We configured the agent with several behavioral "skills" (from [mattpocock/skills](https://github.com/mattpocock/skills)) that shaped how it worked:

1. **`grill-with-docs`:** Made the agent document its reasoning and record bugs/decisions before writing code. This was essential for keeping track of what was happening across sessions.
2. **`tdd` (Test-Driven Development):** The agent would identify failing test cases first, then write targeted fixes. This worked really well for our inverse kinematics validation.
3. **`improve-codebase-architecture`:** Pushed the agent to keep scripts modular instead of dumping everything into one giant file.
4. **`zoom-out`:** Prevented the agent from tunnel-visioning on one file. It would read related scripts simultaneously to understand the full data flow before making changes.
5. **`caveman`:** A prompt compression technique that cut down on verbose agent responses. Reduced context usage by roughly 75%, which was important when working on large codebases.
6. **`grill-me`:** Forced the agent to question its own plans before executing. Caught a few logic errors this way.
7. **`git-guardrails`:** Safety net that required explicit human approval before any destructive git operations (deleting datasets, force pushes, etc.).


## What Worked

- **Asynchronous tool chaining:** The agent could write a script, run it in `tmux`, poll the output, and trigger the next step — all without me having to babysit each command. This was huge for the dataset conversion pipeline where we had to process ~4,500 trajectories.
- **Dynamic codebase exploration:** When we switched from `libero_object` to `libero_goal` datasets, the agent independently discovered that the state dimensionality changed from 110 to 79 dimensions by searching through the LIBERO source code. I didn't have to point it to the right file.
- **Plan-Then-Execute hooks:** Having the agent pause and show me its plan before executing was the single most valuable pattern. It caught bad ideas early.
- **Stack trace parsing:** The agent was genuinely good at reading long MuJoCo/Robosuite error traces and pinpointing the root cause. This saved hours of manual debugging.

## What Did NOT Work

- **Background execution was flaky:** The agent's built-in mechanism for running long commands in the background would frequently hang or fail to stream logs. I had to explicitly tell it to use `tmux` sessions every single time — it wouldn't default to this on its own.
- **Sycophancy / over-eager execution:** This was probably the most frustrating issue. When I'd ask the agent to fix something, it would sometimes blindly agree and keep trying the same broken approach repeatedly. For example, when a process needed to be killed, it would issue the kill command over and over in a loop trying to "please" me instead of stepping back and diagnosing why the process wouldn't die. I had to add explicit anti-sycophancy rules to the agent instructions.
- **Local instructions taken as global rules:** If I told the agent to do something specific for one situation — like "delete this folder" or "skip validation for this run" — it would sometimes internalize that as a general instruction and start doing it everywhere, even in completely unrelated contexts. A one-off command would suddenly become a recurring behavior, and I'd only notice when something downstream broke.
- **Killing processes instead of reasoning:** There were times when I'd ask the agent "why are you doing this?" or "what's happening here?" just wanting an explanation and it would interpret my question as dissatisfaction and immediately kill the running process or undo its work. Instead of giving me a clear reasoning for its actions, it would treat my question as an implicit instruction to stop. This was really counterproductive.
- **Forgetting explicit instructions across sessions:** Unless I wrote down recurring instructions in `AGENT.md`, the agent would forget them between sessions. Things like "always use `tmux`", "never delete the outputs folder", or "always run with `MUJOCO_GL=egl`" had to be told every single time until I put them in the agent file.
- **Not auto-updating documentation:** The agent never proactively updated `HISTORY.md` or `TRACKING.md` after completing work. I had to explicitly tell it "now update the tracking doc with what you just did" every time. This made handoffs between sessions harder than they needed to be.
- **Rabbit hole behavior:** The agent would sometimes latch onto one approach and just keep going deeper and deeper into it, even when it clearly wasn't working. For example, when fixing the image orientation issue, it spent a long time trying increasingly complex FFmpeg filter chains and post-processing hacks instead of stepping back and considering that the root cause might be in the rendering code itself. It couldn't "zoom out" on its own. I had to explicitly tell it to abandon its current approach and look at the problem from scratch. Agent was very reluctant to explore alternatives without explicit prompting.
- **Math across files:** The agent really struggled with tracking mathematical operations (image rotations, matrix flips) across multiple scripts. When our rendered frames came out upside-down, the agent tried to fix it with an FFmpeg post-processing hack instead of tracing the bug back to the actual `np.flip()` call in the rendering function. I had to manually force it to "zoom out" and fix the root cause. This is mainly because many of the coding agents still do not have the capabilities to ingest image media. And in this case it was needed to compare this to other images from other reference which it did not think about or had the capability to do.
- **Context loss on long sessions:** After many back-and-forth exchanges, the agent would sometimes forget earlier decisions or re-introduce bugs that we'd already fixed. The `AGENT.md` and `TRACKING.md` files helped mitigate this, but it was still an issue.
- **Committing without explicit permission:** Sometimes the agent commits changes (like adding or deleting files) without my explicit permission or asking me first if it should or not. Although for pushing and undoing commits it does ask for permission, the spontaneous commits were frustrating and I had to explicitly tell it never to run `git commit` without asking me.

## How I Managed the Agent

Since the agent wasn't perfect, I had to step in with explicit controls:

1. **Enforcing `tmux`:** I made it a rule in `AGENT.md` that all long-running scripts (physics simulations, dataset conversions) must run inside dedicated `tmux` sessions. This let me detach from the agent conversation and monitor progress independently.
2. **Anti-sycophancy prompts:** I added explicit instructions telling the agent to stop repeating failed commands and instead do a clean restart when something was broken. Without this, it would waste tokens retrying the same thing.
3. **"Zoom-out" overrides:** When the agent was band-aiding a video rendering bug with FFmpeg filters, I overrode it and forced it to delete all the corrupted outputs, go back to the physics rendering code, and fix the rotation math at the source. Agents are fast, but they need firm human direction for architectural decisions.

## Key Takeaway

Agentic coding tools dramatically sped up our development. What would have taken us weeks of manual debugging and scripting was done in days. But they're not autonomous. The best results came from treating the agent as a very fast junior engineer: great at execution, but needs clear direction, guardrails, and regular check-ins to stay on track.
