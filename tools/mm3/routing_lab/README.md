# Routing lab

A local page for judging routing rules by hand: chat with MiniMax M3.1 Flash, change the routing rule for each
prompt without reloading the model, look at every next-token distribution, and measure how far a rule moves the
model from the unmodified router. It tests usability and quality only. It is not a timing tool: speed claims come
from the timed tests.

The plan (`docs/minimax-wsl/README.md`) owns requirements, decisions and status. This file only says how to run
the tool.

## Run it

```
python3 tools/mm3/routing_lab/launch_lab.py start     # about seven minutes: the model load
# open http://127.0.0.1:8796/
python3 tools/mm3/routing_lab/launch_lab.py stop      # always, when done: it frees the cards
python3 tools/mm3/routing_lab/launch_lab.py status
```

`start` takes the GPU mutex the test campaigns use. If a campaign holds the cards it refuses and names the holder
(`--wait` queues behind it). While the lab is up no timed test can run, so stop it when you are done. It loads one
engine with the drafter, so one row and drafting are a per-prompt choice, runs the timed tests' setup (bootstrap
request, helper-first placement) and saves that cache state for the page to restore.

To try the page without the cards (synthetic text and numbers, for looking at the page only):

```
python3 tools/mm3/routing_lab/lab_server.py --mock --port 8797
```

Standard library only. The page is served on 127.0.0.1 and is for this machine.

The page's backend sends nothing to the engine, not even a status query, unless the lab's launcher holds the GPU
mutex. An engine on the usual port may belong to a timed campaign, and a foreign request spoils its measurements.
`lab_server.py --any-engine` switches the check off; never use it while a campaign may be running.

## What the page does

**Chat.** Rules are set on the right and apply to the next prompt only. Each answer keeps the rules it was made
with. "Try again with the current rules" answers the same prompt again, so two rules can be read side by side;
"Continue the conversation from this answer" picks which one the next turn builds on. Thinking is folded above the
answer. Tokens can be coloured by surprise, uncertainty, rank of the chosen token, router mass lost, picks moved,
or whether the token was an accepted draft. Doubled words and repeated phrases are flagged.

**Rules.** One table (`settings_schema.py`) defines every control; the engine receives them as per-request
`strata_tune` keys.

| On the page | Key | Meaning |
|---|---|---|
| Bonus, one row | `routing_prior_bonus` | added to the score of every expert already on a GPU; 0 is the unmodified router, serving uses 0.20 |
| Bonus, drafting | `route_bonus_draft` | a separate bonus for passes with more than one row |
| Rules while reading the prompt | `route_lab_prompt` | on (the page's default): the prompt's rows are routed under the same rules as generated tokens; off: the serving kernel reads the prompt with the plain bonus |
| Bonus while reading the prompt | `route_prompt_bonus` | the bonus for the prompt's rows; unset, the one-row bonus as serving does today |
| Bonus by layer | `route_layer_scale` | a multiplier on the bonus for each of the 57 layers |
| Per-row budget on lost router mass | `route_floor` (1 minus the budget) | substitutions are applied cheapest first until the next would exceed the budget |
| Protect | `route_protect` | the strongest one or two picks are never replaced |
| Price each substitution | `route_priced`, `route_price_*` | substitute only where time saved per unit of mass lost is at least this |
| Drop a layer's routed experts | `route_layer_drop` | if the row's top-four router probabilities sum below this |
| Weights | `route_weights` | renormalise (today), dampen (other picks keep their weights), amplify (the substitute takes the replaced pick's weight) |
| Drop a weak unready pick | `route_k_drop`, `route_k_drop_ranks` | run three experts |
| Widen to five experts | `route_k_widen` | top six all above the threshold, one of the top four not on a GPU, ranks 5 and 6 ready |

**Token.** Click a token: the model's next-token distribution at that position (at temperature 1, before filters),
or what was actually sampled from (after temperature and top-p), the token's rank, and how each of the 57 layers
was routed for it.

**Analysis.** Per answer: perplexity, mean probability, uncertainty, share of first choices, surprise and
uncertainty along the answer, a layers-by-tokens map of router mass lost, the least likely tokens, doubled words
and repeated phrases. "Download the records" saves everything as JSON.

**Answers.** Tick two answers to the same prompt: their totals, the divergence between their distributions over
the tokens they share, and both distributions at the token where they part. A small divergence there means the
same distribution and a different draw; a large one means the rule changed what the model wanted to say.

**Fidelity meter.** A recorded answer is replayed token by token (forced decoding) under each rule. At every
position the rule's next-token distribution is compared with the unmodified router's (bonus 0): mean, median and
95th percentile of the Kullback-Leibler divergence, share of positions with the same first choice, and the change
in the probability of the token that was forced. Running the reference twice gives the noise floor. Every run
starts from the saved cache state, so the rules are compared from the same start.

## What to keep in mind when reading results

- The divergence is a lower bound: it is computed over the candidates both records list (64 by default) plus one
  bucket for the rest. It is exactly 0 for identical distributions.
- Which experts are on the GPUs decides which substitutions a rule makes. The meter restores the saved state before
  every run; in chat that is a choice under Sampling ("Cache state before the answer").
- Text differs between runs by design (sampling). Compare meaning and damage, not wording. The meter is the tool
  for comparing rules on identical tokens.
- "Est. load cost" is the engine's priced estimate of the expert loads behind each token. It is for ordering
  rules, not a timing; in drafting each token carries its whole window's cost.
- By default the rules apply to the prompt's rows as well as to generated tokens. With "Rules while reading the
  prompt" off, the serving kernel reads the prompt with the plain bonus: then a rule that needs a high bonus (the
  priced rule runs at 1) should be given its own prompt bonus, or the prompt is read at that bonus with nothing to
  limit it.
- The engine checks its own lab routing against the serving kernel on every row. If that check ever fails, the
  answer carries a red "ROUTING CHECK FAILED" note: do not rely on it.

## Files

- `launch_lab.py`: start, stop, status.
- `lab_server.py`: the page's backend (answers, presets and meter runs are kept under the data directory).
- `engines.py`: the adapter to the Maya server (`MayaEngine`), token text from the pack's vocabulary, and the mock.
- `settings_schema.py`: the rules and sampling options.
- `compare.py`: divergence between records and per-answer summaries. `python3 compare.py` runs its self-test.
- `profiles.json`: the tune keys of the two modes, copied from the timed tests of 2026-11-10.
- `static/`: the page.

Request and record formats: `/mnt/wsl/transfer/minimax-wsl/runs/routing-lab-20261110/CONTRACT.md` and `API.md`.
