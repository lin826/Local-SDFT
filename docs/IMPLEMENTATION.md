# Implementation notes — data & code behind the three demos

How the **QA**, **tool-use**, and **database** demos actually work: the shared
learning engine first, then what each demo changes on top of it (its data, its
reward/shaper, and which learning path it uses). Companion to
[DEMO_QUICKSTART.md](DEMO_QUICKSTART.md) (how to run them).

## The shared engine (all three reuse this)

Every demo drives one `OnlineController` (`sdft/online/controller.py`) that owns a
single LFM2.5-230M with a LoRA adapter on top. Serving and training share the
model under an `RLock`, so a training step never races a generation. The base
230M weights are frozen; only the LoRA adapter trains (`configs/*.yaml → lora`,
targets `.*(self_attn\.(q|k|v|out)_proj|feed_forward\.w[123])`, rank `r=32`,
`lr 2e-4`).

**A "demonstration" is the unit of learning** — a `(messages, demonstration_text,
weight, topic)` record: the prompt context plus the *exact target string* the
model should have produced. An update (`run_update`) pulls a batch from the replay
buffer, runs a few gradient steps, saves a **versioned adapter** (`v1`, `v2`, …)
that can be rolled back with one call, and marks the demos trained.

**The gradient step is completion-only NLL** (`TorchTrainer._sft_loss_for_demo`).
With `loss_type: "sft"` (all three demos), the model is scored only on how likely
it makes the demonstration tokens given the prompt:

```
input = [prompt_ids ... comp_ids]        # comp = target string + EOS
nll   = -mean(log p(comp_ids | prompt))  # loss on completion positions only
```

A second, more SDFT-canonical loss exists — `_loss_for_demo` — which builds a
*teacher* view by feeding the model its own context **plus the golden answer as a
prior turn** (`teacher.py`) and minimizes per-token **forward KL(teacher‖student)**
over the full vocab (`loss.py:forward_kl`). The demos deliberately avoid it:
reward-selected forward-KL collapsed the 230M at `lr 2e-4`, so they use the stable
SFT-on-the-target path above.

**The one fork that matters — where the target comes from:**

| path | who supplies the target | used by |
|---|---|---|
| **Correction** (`controller.correct`) | a human edit becomes the golden answer | **QA / correct-once** |
| **Reward-selected on-policy** (`_reward_harvest`) | the model's *own best sample*, reshaped to pass a checkable reward | **tool-use, database** |

The reward path is the RL-flavored one: each turn it samples `N` candidates
(optionally conditioned on a `coach_instruction` teacher hint so a cold model can
produce a passing one), scores them with the task's **reward function**, takes the
best, and — if a **shaper** is registered — replaces it with a guaranteed
full-marks version of the model's own content. That shaped string is the
demonstration. Rewards and shapers live in `sdft/online/reward.py`, selected by
`online.reward_fn`.

With that in place, the three demos differ only in data, reward/shaper, and path.

---

## 1. Simple QA — `scripts/demo_correct_once.py`

**Behavior taught:** answer in a single sentence — a *policy*, not a fact, so a fix
on one topic should transfer to any topic.

**Data** (inline in the script): two topic-disjoint sets.
- `COACH` = 12 **cooking** questions ("How do I boil an egg?") — the only thing
  ever corrected.
- `HELDOUT` = 6 **programming** questions ("What is a hash map?") — never
  corrected, only measured.

The cooking→programming split *is* the experiment: one-sentence success on the
programming set can't be memorized from cooking corrections, so it proves the
habit generalized.

**Reward + shaper** (`one_sentence` in `reward.py`). Reward is a checkable
predicate: non-empty, single line (no `\n`), ≤1 sentence terminator, 1–30 words →
`1.0`, else `0.0`. The shaper models how a human fixes a bad reply: keep the first
sentence, one line, ≤28 words, add a period.

**Implementation — the correction path** (this demo does *not* set
`online.reward_fn`, so reward-harvest is off):
1. `ctrl.chat(conv, q)` on a cooking question → reply.
2. Score with `one_sentence`; if it already obeys, skip. Otherwise compute the
   one-line fix and call `ctrl.correct(conv, mid, fixed)`, storing a
   `Demonstration` whose target is the fix.
3. `ctrl.maybe_update(force=True)` → a few SFT steps (`steps_per_update: 4 ×
   demos_per_step: 2`).
4. Re-measure the one-sentence rate on the **held-out programming** set.

Once cooking questions all obey, it enters **consolidation**: `maybe_update` with
*no new input*, so the replay buffer re-serves the handful of corrections
(`replay_ratio: 0.5`), strengthening the habit until it generalizes — the
`50% → 83% → 100%` climb.

**ICL/RAG baseline:** run on the *base* weights (`ctrl.rollback(0)`), the same
corrections are injected in-context (ICL = rule + all corrections; RAG = rule +
top-3 by word overlap) and both held-out accuracy and per-call token overhead are
measured. That produces the table where finetuning matches RAG's accuracy at +0
tokens/call.

---

## 2. Tool use — `scripts/demo_toolcall.py`

**Behavior taught:** answer arithmetic by emitting `<tool>calc("347 + 288")</tool>`
instead of doing mental math.

**Data** (`sdft/online/demo.py`), built so memorization is impossible:
- `COACH_CALC` = 12 problems on **small numbers** ("What is 3 + 4?").
- `HELDOUT_CALC` = 6 problems on **large numbers** ("What is 128 * 47?").

The script asserts the two number-sets are disjoint before running, so a correct
held-out answer can only come from learning *question → tool call*.

**Reward + shaper** (`calc_tool` in `reward.py`) — graded, which shapes the signal:
- `1.0` — a `calc(...)` call whose expression evaluates to the correct value;
- `0.4` — a call that runs but is wrong;
- `0.2` — call-shaped but not evaluable;
- `0.0` — no tool call (**including a correct freehand answer** — the reward is for
  tool *use*, which is what generalizes).

Truth comes from `extract_arithmetic(prompt) → safe_eval`. The shaper emits the
exact correct call for that problem as the full-marks target.

**The calculator** (`sdft/online/tools.py`) is AST-safe, never `eval`:
`parse_calc_call` extracts the expression; `safe_eval` parses with
`ast.parse(mode="eval")` and walks the tree allowing only numeric constants and a
whitelist of arithmetic operators. Anything else raises → `None`. Executing
untrusted model output here can do no more than arithmetic.

**Implementation — reward-selected path** (`online.reward_fn: calc_tool`). Each
`ctrl.chat` turn, `_reward_harvest`:
1. Samples `reward_num_samples: 4` candidates at `sample_temperature: 0.8` from a
   model conditioned on `coach_instruction` ("respond with a tool call of the form
   `<tool>calc("EXPRESSION")</tool>`…") — the hint steers *sampling only* and is
   never present at serve time.
2. Scores candidates + the served reply, keeps the best.
3. Shapes it into the exact correct call → the demonstration.

`maybe_update(force=True)` does `6×4` SFT steps. Evaluation serves **hint-free**
(`[{"role":"user","content":q}]`), parses any `calc()`, executes it, and compares
to the true value. `ctrl.rollback(0)` reproduces the base 0%.

---

## 3. Database interaction — `scripts/demo_sqlite.py` + `sdft/online/sqlenv.py`

**Behavior taught:** answer natural-language questions about a real database by
writing SQL that an actual SQLite engine executes.

**Data — a real seeded database** (`sqlenv.build_db`): three tables —
`customers(id,name,city)`, `products(id,name,category,price,stock)`,
`orders(id,customer_id,product_id,quantity,order_date)` — with deterministic seed
rows (8 customers, 10 products, 12 orders). The question sets are disjoint by
value:
- `COACH_QA` = 8 (question, gold-SQL) pairs over `Books`/`Office`, `Seattle`,
  customer `Alice`.
- `HELDOUT_QA` = 6 pairs over `Toys`/`Kitchen`, `Denver`, customer `Carol`.

Because reward is computed by **executing** against the seeded rows, a correct
held-out result set means the model wrote genuinely correct SQL for unseen values.

**Reward + shaper** (`sqlite_tool` in `reward.py`) — graded:
- `1.0` — query runs and its **result set matches the gold query's**;
- `0.4` — runs but returns different rows;
- `0.2` — query errored or was denied by the jail;
- `0.0` — no SQL.

Matching is order- and float-format-insensitive (`results_match` / `_norm_rows`),
so right-rows-wrong-order still counts. The shaper wraps the gold query in
`<sql>…</sql>`.

**The safety jail** — load-bearing, because the model's SQL is untrusted and
actually executed (`run_query`). Four independent layers:
1. Read-only connection: `sqlite3.connect("file:{path}?mode=ro", uri=True)`.
2. `PRAGMA query_only = ON`.
3. A **SQLite authorizer** returning `SQLITE_OK` only for
   `SELECT / READ / FUNCTION / RECURSIVE` and `SQLITE_DENY` for everything else —
   the primary defense; it holds even if the text parser is fooled, so
   `DROP`/`UPDATE`/`INSERT`/`ATTACH`/pragmas are refused at the engine level.
4. One statement per call (`con.execute` raises on stacked statements) + a
   progress-handler instruction budget (`100_000` VM ops) so a pathological query
   can't spin.

`parse_sql` pulls the query from `<sql>…</sql>` (or a bare leading `SELECT`/`WITH`)
and strips a trailing `;`. `tests/test_sqlenv.py` asserts each attack fails **and
leaves the DB byte-for-byte unchanged**.

**Implementation** — same reward-selected path (`online.reward_fn: sqlite_tool`,
`reward_num_samples: 6`, since a cold model often produces no runnable query). The
teacher hint (`coach_instruction`) contains the schema; served responses are
**schema-less**, so a correct query proves the schema was learned into the weights.
Two demo-specific wrinkles:
- **Best-adapter early-stopping:** coaching on ~8 questions eventually over-fits and
  drifts, so it tracks the best held-out checkpoint (`best_version`) and
  `ctrl.rollback(best_version)` before the A/B and examples.
- `show_example` prints the actual SQL the model wrote and the **real rows** the
  engine returned, checked against the gold rows.

---

## The one-paragraph summary

All three are the same loop — *sample or correct → turn it into a
guaranteed-correct target → a few completion-NLL LoRA steps → versioned adapter* —
instantiated with three different **checkable rewards** and three different
**disjoint coach/held-out splits**. QA uses human corrections and proves transfer
across topics; tool-use and database use reward-selected self-distillation and
prove transfer across unseen values, with the database demo additionally
sandboxing real, untrusted query execution.
