"""The routing lab's settings: one table that the page builds its controls from and the backend validates against.

Keys and meanings follow /mnt/wsl/transfer/minimax-wsl/runs/routing-lab-20261110/CONTRACT.md. Each rule is a
per-request setting; nothing here needs a model reload.
"""
from __future__ import annotations

N_LAYERS = 57

# kind: "bool" (0/1), "float", "int", "choice", "layers" (57 floats)
SCHEMA = [
    {"group": "Bonus", "key": "routing_prior_bonus", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.20,
     "label": "Bonus, one row",
     "help": "Added to the router score of every expert already on a GPU. 0 is the unmodified router. Serving uses 0.20."},
    {"group": "Bonus", "key": "route_bonus_draft", "kind": "float", "min": -1.0, "max": 1.0, "step": 0.01, "default": -1.0,
     "label": "Bonus, drafting", "optional": True,
     "help": "The bonus for passes with more than one row. Off: the one-row value is used."},
    {"group": "Bonus", "key": "route_layer_scale", "kind": "layers", "min": 0.0, "max": 3.0, "step": 0.05, "default": 1.0,
     "label": "Bonus by layer",
     "help": "A multiplier on the bonus for each of the 57 expert layers. 1 everywhere is today's rule."},

    {"group": "Selecting substitutes", "key": "route_floor", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01,
     "default": 0.0, "label": "Per-row budget on lost router mass", "optional": True, "on_value": 0.80, "invert": True,
     "help": "The most of a row's original top-four router mass that its substitutions may give up. They are applied "
             "cheapest first until the next one would go over. The engine's setting is the floor, 1 minus this: a "
             "budget of 0.20 is the floor of 0.80 screened on the replay."},
    {"group": "Selecting substitutes", "key": "route_protect", "kind": "choice", "default": 0,
     "choices": [[0, "none"], [1, "the strongest pick"], [2, "the two strongest picks"]],
     "label": "Protect", "help": "Protected picks are never replaced. If one is not on a GPU it is loaded."},
    {"group": "Selecting substitutes", "key": "route_priced", "kind": "float", "min": 0.0, "max": 20.0, "step": 0.1,
     "default": 0.0, "label": "Price each substitution", "optional": True, "on_value": 2.0,
     "help": "A substitution is made only if time saved divided by router mass lost is at least this (ms per unit of "
             "mass), taken in that order. It filters the substitutions the bonus allows, so it does nothing at bonus 0; "
             "set the bonus to 1 to let the price alone decide."},
    {"group": "Selecting substitutes", "key": "route_price_first", "kind": "float", "min": 0.0, "max": 3.0, "step": 0.01,
     "default": 0.60, "label": "Price: a pass's only load (ms)", "advanced": True,
     "help": "Time saved by removing the one load that makes a pass wait for an upload."},
    {"group": "Selecting substitutes", "key": "route_price_more", "kind": "float", "min": 0.0, "max": 3.0, "step": 0.01,
     "default": 0.32, "label": "Price: a further load (ms)", "advanced": True,
     "help": "Time saved by removing one of several loads in a pass."},
    {"group": "Selecting substitutes", "key": "route_price_staged", "kind": "float", "min": 0.0, "max": 3.0, "step": 0.01,
     "default": 0.10, "label": "Price: a copy already staged or in flight (ms)", "advanced": True,
     "help": "Time saved by replacing a pick whose copy is already on its way. A load another row needs anyway saves 0."},
    {"group": "Selecting substitutes", "key": "route_layer_drop", "kind": "float", "min": 0.0, "max": 4.0, "step": 0.01,
     "default": 0.0, "label": "Drop a layer's routed experts", "optional": True, "on_value": 0.50,
     "help": "If the sum of a row's top-four router probabilities at a layer is below this, that layer's routed experts "
             "add nothing for that row. Attention and the shared expert still run."},

    {"group": "Multipliers", "key": "route_weights", "kind": "choice", "default": 0,
     "choices": [[0, "renormalise over the executed picks (today)"],
                 [1, "dampen: leave the other picks' weights alone"],
                 [2, "amplify: the substitute takes the replaced pick's weight"]],
     "label": "Weights",
     "help": "Today every executed pick is scaled up to fill the gap a weak substitute leaves. Dampen keeps the other "
             "picks at their original weights. Amplify gives the substitute the weight of the pick it replaced."},
    {"group": "Multipliers", "key": "route_k_drop", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0,
     "label": "Drop a weak unready pick (run three)", "optional": True, "on_value": 0.15,
     "help": "An unready pick among the lowest ranks whose weight is below this is neither loaded nor replaced."},
    {"group": "Multipliers", "key": "route_k_drop_ranks", "kind": "choice", "default": 1,
     "choices": [[1, "rank 4 only"], [2, "ranks 3 and 4"]], "label": "Drop applies to", "advanced": True,
     "help": "Which of the row's picks may be dropped."},
    {"group": "Multipliers", "key": "route_k_widen", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0,
     "label": "Widen to five experts", "optional": True, "on_value": 0.10,
     "help": "If the top six router probabilities are all at least this, exactly one of the top four is not on a GPU "
             "and ranks 5 and 6 are, the three ready picks run with ranks 5 and 6: five experts."},

    {"group": "Reading the prompt", "key": "route_lab_prompt", "kind": "choice", "default": 1,
     "choices": [[1, "the rules apply to the prompt too"], [0, "generated tokens only: the prompt gets the plain bonus"]],
     "label": "Rules while reading the prompt",
     "help": "On: the prompt's rows are routed under the same rules as generated tokens, so an answer shows how the "
             "model behaves under the rule from the first token of the prompt. Off: the prompt is read by the "
             "serving kernel with the plain bonus, and only generated tokens see the rules."},
    {"group": "Reading the prompt", "key": "route_prompt_bonus", "kind": "float", "min": -1.0, "max": 1.0, "step": 0.01,
     "default": -1.0, "label": "Bonus while reading the prompt", "optional": True, "on_value": 0.20,
     "help": "Off: the prompt is read with the one-row bonus, as serving does today. On: with this value, whatever "
             "the bonus for generation is. 0 reads the prompt with the unmodified router."},
]

SAMPLING = [
    {"key": "temperature", "kind": "float", "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0, "label": "Temperature"},
    {"key": "top_p", "kind": "float", "min": 0.05, "max": 1.0, "step": 0.01, "default": 0.95, "label": "Top p"},
    # No cap by default: a cap counts the thinking too, and one that bites cuts the answer short. Off, the model
    # writes until it stops, up to the engine's context (8,192 tokens with the prompt).
    {"key": "max_tokens", "kind": "int", "min": 0, "max": 8000, "step": 16, "default": 0, "optional": True, "on_value": 2048,
     "label": "Cap on generated tokens",
     "help": "Off: the model writes until it stops, thinking included, up to the engine's context of 8,192 tokens. "
             "On: generation is cut at this many tokens, thinking included; a cut answer is marked."},
    {"key": "seed", "kind": "int", "min": 0, "max": 2 ** 31 - 1, "step": 1, "default": 20261101, "label": "Seed"},
    {"key": "reasoning_effort", "kind": "choice", "default": "high", "label": "Thinking",
     "choices": [["none", "off"], ["low", "low"], ["medium", "medium"], ["high", "high (as in the timed tests)"]],
     "help": "How much the model reasons before it answers. The thinking is shown folded above the answer and its "
             "tokens are recorded like the answer's."},
    {"key": "same_start", "kind": "choice", "default": 0, "label": "Cache state before the answer",
     "choices": [[0, "carry on from the last answer's"], [1, "restore the saved starting state first"]],
     "help": "Which experts are on the GPUs decides which substitutions a rule makes. Restoring first gives every "
             "answer the same start, so two rules can be compared on the same prompt. The fidelity meter always "
             "restores."},
    {"key": "lab_path", "kind": "choice", "default": 1, "label": "Routing path",
     "choices": [[1, "the lab's path, which records the routing"], [0, "the serving kernel when no lab rule is on"]],
     "help": "The lab's path gives the same picks as the serving kernel when no lab rule is on, and adds the routing "
             "numbers shown here. It is slower, which can leave more experts ready in time than in serving. The "
             "serving kernel has no routing numbers. The fidelity meter does not follow this choice: with no lab "
             "rule on it uses the serving kernel, so it reads the prompt layer by layer as serving does (the lab's "
             "path cannot) and shows no routing numbers; with a lab rule on it uses the lab's path, the only one "
             "that applies the rules."},
]

LAB_KEYS = [s["key"] for s in SCHEMA]


def defaults() -> dict:
    out = {}
    for s in SCHEMA:
        out[s["key"]] = [s["default"]] * N_LAYERS if s["kind"] == "layers" else s["default"]
    return out


def sampling_defaults() -> dict:
    return {s["key"]: s["default"] for s in SAMPLING}


def validate(settings: dict) -> dict:
    """Return a clean copy: known keys only, in range. Raises ValueError with a message for the page."""
    clean = defaults()
    for s in SCHEMA:
        key = s["key"]
        if key not in settings:
            continue
        v = settings[key]
        if s["kind"] == "layers":
            if not isinstance(v, list) or len(v) != N_LAYERS:
                raise ValueError(f"{key}: expected {N_LAYERS} numbers")
            vals = [float(x) for x in v]
            if any(x < s["min"] or x > s["max"] for x in vals):
                raise ValueError(f"{key}: values must lie in [{s['min']}, {s['max']}]")
            clean[key] = vals
        elif s["kind"] == "choice":
            allowed = [c[0] for c in s["choices"]]
            if v not in allowed:
                raise ValueError(f"{key}: must be one of {allowed}")
            clean[key] = v
        else:
            x = float(v)
            if x < s["min"] or x > s["max"]:
                raise ValueError(f"{key}: {x} is outside [{s['min']}, {s['max']}]")
            clean[key] = int(x) if s["kind"] == "int" else x
    return clean


def validate_sampling(sampling: dict | None) -> dict:
    """Known sampling keys only, in range. Raises ValueError with a message for the page."""
    clean = sampling_defaults()
    for s in SAMPLING:
        key = s["key"]
        if not sampling or key not in sampling:
            continue
        v = sampling[key]
        if s["kind"] == "choice":
            allowed = [c[0] for c in s["choices"]]
            if v not in allowed:
                raise ValueError(f"{key}: must be one of {allowed}")
            clean[key] = v
        else:
            x = float(v)
            if x < s["min"] or x > s["max"]:
                raise ValueError(f"{key}: {x} is outside [{s['min']}, {s['max']}]")
            clean[key] = int(x) if s["kind"] == "int" else x
    return clean


def lab_needed(settings: dict) -> bool:
    """True if any rule beyond today's single bonus is switched on, so the engine's lab path is required."""
    d = defaults()
    for key in LAB_KEYS:
        if key in ("routing_prior_bonus", "route_lab_prompt"):      # the bonus is today's key; the prompt switch only
            continue                                                # says where the other rules apply
        if settings.get(key, d[key]) != d[key]:
            return True
    return False


def describe(settings: dict) -> str:
    """A short human label for a settings dict: only what differs from today's rule."""
    d = defaults()
    parts = [f"bonus {settings.get('routing_prior_bonus', d['routing_prior_bonus']):.2f}"]
    names = {"route_bonus_draft": "draft bonus", "route_prompt_bonus": "prompt at", "route_floor": "budget", "route_protect": "protect",
             "route_priced": "priced", "route_layer_drop": "layer drop", "route_k_drop": "drop pick",
             "route_k_widen": "widen"}
    for key, name in names.items():
        v = settings.get(key, d[key])
        if v != d[key]:
            parts.append(f"{name} {round(1 - v, 4) if key == 'route_floor' else v:g}")
    w = settings.get("route_weights", 0)
    if w:
        parts.append("dampen" if w == 1 else "amplify")
    if not settings.get("route_lab_prompt", d["route_lab_prompt"]):
        parts.append("rules skip the prompt")
    ls = settings.get("route_layer_scale")
    if isinstance(ls, list) and any(abs(x - 1.0) > 1e-9 for x in ls):
        parts.append("by layer")
    return ", ".join(parts)
