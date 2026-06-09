# Jules Roussel

**ML engineer & researcher working on the parts of LLM systems that are easy to get wrong and hard to measure: reward design, evaluation faithfulness, and the gap between "a metric went up" and "the model actually got better."**

MSc Data Science & Machine Learning, UCL (Distinction). I build evaluation harnesses that are reproducible, audited, and honest about their own limits — and I contribute the reusable pieces back upstream.

- LinkedIn: https://www.linkedin.com/in/jules-roussel-69b87a277/
- Reach me: julesroussel.de.c@gmail.com

---

## Open-source contributions

Two pieces of my recent research are open PRs to repositories I learned from:

**[`compute_metrics` for `GRPOTrainer` → Hugging Face TRL #5790](https://github.com/huggingface/trl/pull/5790)** *(open PR, fixes [#2959](https://github.com/huggingface/trl/issues/2959))*
GRPO only logged reward-based metrics during eval, so there was no way to plug in a downstream task metric — and therefore no way to *detect reward hacking* from inside the training loop. This PR adds `compute_metrics` following TRL's existing trainer interface, handling GRPO's text-generation eval path (completions buffered across eval batches, `compute_metrics` invoked from `log()` with an `(N, R)` reward tensor). Shipped with integration tests.

**[Lesson 10: Evaluating Subjective Behavior with Model-Graded Evals → Anthropic `courses` (`prompt_evaluations`)](https://github.com/anthropics/courses)** *(open PR)*
A new lesson extending the course's model-graded-eval material to *subjective* behavior — tutoring guidance, tone, answer-revealing discipline — via a fixed rubric, structured tool outputs, judge validation against human labels, and a small generate → judge → revise loop. Core idea: *subjective at the source, objective within the protocol.* Self-contained notebook, runs on Claude Haiku in ~20 API calls.

---

## Featured work

### [grpo-reward-vs-eval](https://github.com/YOUR_USERNAME/grpo-reward-vs-eval) — auditing GRPO reward faithfulness

A reproducible testbed that **separates and audits** the three things RLVR pipelines usually fuse into one regex: the **reward** a model is trained against, the **metric** it's measured by, and the **extractor** that turns a completion into the answer feeding both. When they coincide — as in most public GRPO pipelines — "reward went up" and "the model improved" become true *by construction*. This pulls them apart on GSM8K (Qwen2.5-0.5B-Instruct, GRPO via TRL) and uses an independently-audited instrument to tell capability gains from proxy-gaming.

- **A clean, controlled reward-hacking case.** A format-only reward saturates `mean_reward` and format-compliance to 1.0 while honest accuracy *collapses below its starting point* (≈0.23 → 0.03): the reward removed any incentive to produce the chain-of-thought that was load-bearing for correctness.
- **The verifier is audited, not assumed.** A Claude judge (Anthropic tool-use for structured verdicts) is validated at **96% agreement (48/50)** against hand-labels *before* it's trusted, then used to score three answer extractors by precision/recall/F1 — establishing `last_number` (F1 0.94) as the only defensibly honest ruler vs. `lenient` (over-credits) and `strict_tag` (under-counts).
- **The most faithful extractor for *measuring* is the worst for *rewarding*** — a controlled instance of the published objective-mismatch principle, agreed on by three of four independent instruments.
- **A cheap recall-convergence diagnostic** reaches the pass@k "RLVR elicits, doesn't instill" conclusion without large-k sampling. *(Single-seed signal — reported as directional, not proven.)*

> Scope is stated plainly in the repo: single seed, single 0.5B model, GSM8K, 100 steps. The robust claims and the preliminary ones are labelled separately, and the README documents a hypothesis the data overturned and a circularity bug I found and fixed in my own design. This replicates and *packages* known results with an audited verifier; it doesn't claim them as new.

### [llm-behavior-eval-harness](https://github.com/YOUR_USERNAME/llm-behavior-eval-harness) — evaluating subjective LLM behavior reproducibly

The methodology behind the Anthropic lesson, in full. An end-to-end pipeline that turns subjective tutor behavior into auditable signals: deterministic checks as cheap first gates, strict Pydantic schemas, a **frozen** Claude rubric judge validated against MRBench human annotations, class-aware zero-leakage splits, and a mini actor-critic prompt-optimization loop comparing baseline vs. optimized prompts on held-out examples. Includes a prompt-drift guard so the judge can't silently desync from its frozen prompt, and a pytest suite with mocked API behavior (no live calls in CI).

> Results are reported as a small-scale demonstration that the harness reproducibly *detects* behavior changes between prompt variants — not as a claim that any one prompt generalizes.

### Real-time multi-service LLM agent (MSc thesis) — VR conversational agent

A real-time pipeline driving an interactive VR agent, where **Unity is the timing authority**. Built the Node/TypeScript orchestrator (per-sentence blocking loop: compute → send → wait-ACK → next), Python service workers (LLM, TTS), and Unity/C# client. Designed a compact custom wire protocol (JSON header + streamed 48 kHz PCM + end-of-sentence ACKs) with back-pressure, bounded timeouts, and monotonic sequencing. Affect handling uses a **hybrid LLM + deterministic guardrail**: an LLM-proposed gesture is vetoed and replaced by a safe default if it falls outside a label-specific allow-set — reproducible, no retries. Offline, the wording is optimized with an actor-critic loop scored by a frozen GoEmotions classifier (macro-F1), with bootstrap CIs and exact paired tests for the small-N VR study.

---

## Other work

- **Compilers & language tooling** — a full pipeline for the WHILE language in Scala: derivative-based regex matcher, Sulzmann & Lu lexer, parser-combinator interpreter, and two compilers (JVM assembly + **LLVM-IR**), the LLVM target built from independent research beyond the course material.
- **Foundation-model segmentation** — a weakly-supervised pipeline using **SAM (ViT-H)** to auto-generate zero-shot pseudo-masks, with a custom multi-task U-Net (parallel segmentation + classification head) and a weighted Dice/BCE/CE loss; weak supervision cut per-epoch training time >58% for a marginal DSC drop.
- **Production RAG** — an oncology clinical-trial RAG service end-to-end: PubMed ingestion, Qdrant vector store, a TypeScript/Node backend returning structured trial stats, a React/Vite dashboard, containerized with Docker and deployed on AWS App Runner via GitHub Actions CI/CD.
- **Constraint solving** — CSP solvers (Skyscrapers + variants; Sudoku-like) using MAC / arc-consistency with MRV and custom value-ordering heuristics; >100x runtime improvement through custom propagation, extensible OOP architecture, full unit + functional test coverage.
- **Full-stack engineering** — led an 8-person Agile team building a Django/PostgreSQL microblogging app (owned core backend, ~10k LOC, CI/CD auto-deploy); shipped production Rails apps with LLM features end-to-end.

---

## What I care about

Disciplined evaluation, honest reporting of limitations (including rejecting my own hypotheses when the data says so) and writing that makes engineering decisions legible. If you work on LLM evaluation, RL, or measuring model behavior, I'd genuinely enjoy comparing notes. Open to ML engineering and research roles.

---

*Tech I reach for: Python · PyTorch · JAX · Hugging Face (TRL) · Anthropic API · TypeScript/Node · Unity/C# · Scala · SQL · Docker · AWS*
