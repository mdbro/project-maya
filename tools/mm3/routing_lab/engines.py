"""Where the lab's tokens come from: the Maya server, or a mock for developing the page without the GPUs.

Both yield the same events:
    {"type": "delta", "text": "..."}            text as it streams
    {"type": "token", "record": {...}}          one record per committed token (CONTRACT.md)
    {"type": "done", "summary": {...}}          the request's last record
    {"type": "error", "message": "..."}

The Maya adapter is deliberately thin and lives in one class, so it can follow the engine agents' final request and
record shapes (API.md in the lab's run folder) without touching the rest of the backend.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import settings_schema as schema


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------------------------------------------

_WORDS = ("the of and to a in is that it for as with on was be by at this from or an are not have which one all "
          "library closes early on Friday because the heating is being repaired please return books to the desk "
          "pump was inspected Monday its seal replaced Tuesday a test on Wednesday found no leakage schedule jobs "
          "def order_jobs deps return result for job in ready append while queue pop if not raise ValueError "
          "muffins boxes of six and four combinations bus leaves at seven bring a folding chair choir hall").split()
_VOCAB = sorted(set(_WORDS)) + [",", ".", ":", "(", ")", "\n"]


def _fidelity_loss(settings: dict, drafting: bool) -> float:
    """A made-up but monotone stand-in for how far a rule moves the model from the unmodified router (mock only)."""
    bonus = settings.get("routing_prior_bonus", 0.2)
    if drafting and settings.get("route_bonus_draft", -1) >= 0:
        bonus = settings["route_bonus_draft"]
    scale = settings.get("route_layer_scale") or [1.0]
    if settings.get("route_priced", 0) > 0:
        bonus = min(bonus, 0.25)              # a price filters substitutions; it does not act like a bonus of 1
    loss = (bonus * (sum(scale) / len(scale))) ** 1.6 * (1.9 if drafting else 1.0)
    floor = settings.get("route_floor", 0.0)
    if floor > 0:
        loss *= max(0.15, 1.0 - floor) * 2.2
    loss *= {0: 1.0, 1: 0.75, 2: 0.6}[settings.get("route_protect", 0)]
    if settings.get("route_priced", 0) > 0:
        loss *= 0.8
    loss *= {0: 1.0, 1: 0.85, 2: 1.15}[settings.get("route_weights", 0)]
    if settings.get("route_k_drop", 0) > 0:
        loss += 0.02 + 0.2 * settings["route_k_drop"]
    if settings.get("route_k_widen", 0) > 0:
        loss *= 0.9
    if settings.get("route_layer_drop", 0) > 0:
        loss += 0.15 * settings["route_layer_drop"]
    return loss


class MockEngine:
    """Deterministic fake tokens with distributions and routing counters that respond to the settings."""
    name = "mock"

    def info(self) -> dict:
        return {"engine": "mock", "connected": True, "route_lab_control": 1, "forced_decoding": 1,
                "note": "Mock engine: the text and numbers are synthetic. Use it to try the page, not to judge rules."}

    def restore(self) -> dict:
        return {"ok": True, "seconds": 0.0}

    def generate(self, messages, settings, sampling, drafting, force_tokens, cancel: threading.Event,
                 route_stats: int = 2, token_topk: int = 64):
        prompt = json.dumps(messages, sort_keys=True)
        seed = int(hashlib.sha256((prompt + str(sampling.get("seed", 0))).encode()).hexdigest()[:8], 16)
        text_rng = random.Random(seed)                      # the text depends on the prompt and seed only
        loss = _fidelity_loss(settings, drafting)
        noise_rng = random.Random(seed ^ int(loss * 1e6))   # the distribution's perturbation depends on the rule
        n = len(force_tokens) if force_tokens is not None else min(int(sampling.get("max_tokens", 256)), 80 + seed % 120)
        temperature = max(float(sampling.get("temperature", 1.0)), 0.05)
        bonus = settings.get("routing_prior_bonus", 0.2)
        moved_base = min(0.6, 0.02 + (0.27 if drafting else 0.21) * (bonus / 0.2) ** 0.45) if bonus > 0 else 0.0
        if settings.get("route_floor", 0) > 0:
            moved_base *= 0.7
        moved_base *= {0: 1.0, 1: 0.7, 2: 0.45}[settings.get("route_protect", 0)]
        totals = {"moved": 0, "picks": 0, "loads": 0}
        # With thinking on, the first tokens are the model's reasoning and end with MiniMax's closing tag.
        think = 0 if sampling.get("reasoning_effort", "high") == "none" else min(n - 1, 10 + seed % 14)
        think_end_id = len(_VOCAB) + 1
        for i in range(n):
            if cancel.is_set():
                raise Cancelled()
            time.sleep(0.012)
            k = 24
            cand = text_rng.sample(range(len(_VOCAB)), k)
            base = sorted((text_rng.random() ** (2.2 + 1.5 * text_rng.random()) for _ in range(k)), reverse=True)
            logits = [math.log(b + 1e-6) / temperature for b in base]
            # the rule perturbs the logits; a larger loss perturbs more, and occasionally a lot
            spike = 4.0 if noise_rng.random() < min(0.5, loss * 0.8) else 1.0
            pert = [l + noise_rng.gauss(0.0, 1.0) * loss * 2.5 * spike for l in logits]
            z = max(pert)
            w = [math.exp(p - z) for p in pert]
            rest = 0.03 * (1.0 + loss)
            tot = sum(w) / (1.0 - rest)
            probs = [x / tot for x in w]
            if force_tokens is not None:
                tok = int(force_tokens[i])
                if tok not in cand:
                    cand[min(3, k - 1)] = tok
                chosen = cand.index(tok)
            else:
                # The draw uses the rule's own distribution, so two rules part company where it matters.
                r, acc, chosen = text_rng.random() * (1.0 - rest), 0.0, 0
                for j, x in enumerate(probs):
                    acc += x
                    if r <= acc:
                        chosen = j
                        break
            if force_tokens is None and think and i == think - 1:      # the closing tag ends the thinking
                cand[chosen] = think_end_id
            word = lambda t: "</mm:think>" if t == think_end_id else _VOCAB[t % len(_VOCAB)]
            order = sorted(range(k), key=lambda j: -probs[j])
            top = [[cand[j], probs[j], word(cand[j])] for j in order[:token_topk]]
            ent = -sum(p * math.log(p) for p in probs if p > 0) - rest * math.log(rest)
            is_end = cand[chosen] == think_end_id
            piece = word(cand[chosen])
            piece = piece if piece in ",.:()\n" or is_end else " " + piece
            rows = 1 if not drafting else noise_rng.choice([1, 1, 2, 3, 4, 6])
            layers_kept, layers_moved, layers_loads, layers_k = [], [], [], []
            floor = settings.get("route_floor", 0.0)
            for li in range(schema.N_LAYERS):
                sc = (settings.get("route_layer_scale") or [1.0] * schema.N_LAYERS)[li]
                m = sum(1 for _ in range(4) if noise_rng.random() < moved_base * (0.6 + 0.8 * sc) * (1.4 if 3 <= li <= 12 else 0.9))
                kept = max(0.0, 1.0 - sum(noise_rng.random() ** 2 * 0.22 * (bonus / 0.2 if bonus else 0) for _ in range(m)))
                if floor > 0:
                    kept = max(kept, floor)
                layers_moved.append(m)
                layers_kept.append(round(kept, 4))
                layers_loads.append(1 if noise_rng.random() < max(0.02, 0.27 - bonus) else 0)
                layers_k.append(3 if settings.get("route_k_drop", 0) > 0 and noise_rng.random() < 0.08 else
                                5 if settings.get("route_k_widen", 0) > 0 and noise_rng.random() < 0.05 else 4)
            route = {"picks": 4 * schema.N_LAYERS, "moved": sum(layers_moved),
                     "mass_kept": round(sum(layers_kept) / schema.N_LAYERS, 4), "min_kept": min(layers_kept),
                     "rows_under_08": sum(1 for x in layers_kept if x < 0.8), "loads": sum(layers_loads),
                     "staged": 45 if not drafting else 9, "dropped": sum(1 for x in layers_k if x == 3),
                     "widened": sum(1 for x in layers_k if x == 5),
                     "layers_dropped": 0, "est_ms": round(26.0 + 0.6 * sum(layers_loads), 2)}
            totals["moved"] += route["moved"]; totals["picks"] += route["picks"]; totals["loads"] += route["loads"]
            rec = {"type": "token", "i": i, "id": cand[chosen], "text": piece, "p": probs[chosen], "p_raw": probs[chosen],
                   "logp": math.log(max(probs[chosen], 1e-12)), "top": top, "rest": rest, "entropy": ent,
                   "forced": force_tokens is not None, "source": "draft" if rows > 1 and noise_rng.random() < 0.6 else "target",
                   "rows": rows, "route": route}
            # What was sampled from: the nucleus of the distribution (top_p), renormalised.
            acc, keep = 0.0, []
            for j in order:
                keep.append(j)
                acc += probs[j]
                if acc >= float(sampling.get("top_p", 0.95)) * (1.0 - rest):
                    break
            if chosen not in keep:
                keep.append(chosen)
            z2 = sum(probs[j] for j in keep)
            rec.update(samp=[[cand[j], probs[j] / z2, word(cand[j])] for j in keep], samp_n=len(keep), samp_rest=0.0,
                       p=probs[chosen] / z2)
            if route_stats >= 2:
                rec["layers"] = {"kept": layers_kept, "moved": layers_moved, "k": layers_k, "loads": layers_loads}
            if force_tokens is not None:
                in_think = think_end_id in force_tokens[i:]
            else:
                in_think = i < think
            if not is_end:
                yield {"type": "delta", "text": piece, **({"channel": "reasoning"} if in_think else {})}
            yield {"type": "token", "record": rec}
        yield {"type": "done", "summary": {
            "type": "done", "tokens": n, "finish": "forced" if force_tokens is not None else "stop",
            "settings": settings, "engine": {"commit": "mock", "binary_sha256": "mock"},
            "route": {"moved_share": totals["moved"] / max(1, totals["picks"]), "loads_per_token": totals["loads"] / max(1, n)}}}


# ---------------------------------------------------------------------------------------------------------------
# Maya
# ---------------------------------------------------------------------------------------------------------------

class MayaEngine:
    """The real engine, through the Maya server's OpenAI-compatible endpoint.

    The request carries the rule settings in `strata_tune`, with `token_log` naming a JSON-lines file the engine
    writes in its token-record directory (API.md in the lab's run folder). The token records are read from that
    file while the text streams. `profiles` gives the tune keys of the two modes (one row, drafting), taken from
    the timed tests' protocols so the lab runs the engine those tests measured.
    """
    name = "maya"

    def __init__(self, url: str, profiles: dict, model: str = "maya", records_dir: Path | None = None,
                 tokenizer_dir: Path | None = None):
        self.url = url.rstrip("/")
        self.profiles = profiles
        self.model = profiles.get("model", model)
        self.records_dir = Path(records_dir) if records_dir else None
        self.tokenizer_dir = Path(tokenizer_dir or profiles.get("tokenizer") or "")
        self._reader = None
        self._text = None

    def token_text(self):
        """The id-to-text table, loaded on first use (about a second)."""
        if self._text is None:
            try:
                self._text = TokenText(self.tokenizer_dir)
            except Exception as exc:
                print(f"[lab] no token text ({type(exc).__name__}: {exc}); tokens will show as ids", flush=True)
                self._text = False
        return self._text or None

    def restore(self) -> dict:
        """Put the experts back where the launcher's snapshot had them, so the next answer starts from that state."""
        began = time.time()
        req = urllib.request.Request(self.url + "/v1/minimax/placement", data=json.dumps({"action": "restore"}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}"}
        except (urllib.error.URLError, OSError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "seconds": round(time.time() - began, 2)}

    def _status(self) -> dict:
        with urllib.request.urlopen(self.url + "/v1/status", timeout=3) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def info(self) -> dict:
        out = {"engine": "maya", "url": self.url, "connected": False}
        try:
            status = self._status()
            out["connected"] = True
            found = _find(status, "token_records_dir")
            if found and self.records_dir is None:
                self.records_dir = Path(found)
            out["token_records_dir"] = str(self.records_dir) if self.records_dir else None
            out["route_lab_control"] = _find(status, "route_lab_control")
            out["token_records_control"] = _find(status, "token_records_control")
            if not out["token_records_dir"]:
                out["note"] = ("The engine is reachable but reports no token-record directory, so answers will have text "
                               "and no per-token records. Start it with the lab launcher.")
        except Exception as exc:                                     # the page shows why it cannot reach the engine
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def reader(self):
        """The engine agents' reader, which adds token text with the server's own tokenizer. None if not present."""
        if self._reader is None:
            try:
                import importlib
                import sys
                tools = str(Path(__file__).resolve().parent.parent)
                if tools not in sys.path:
                    sys.path.insert(0, tools)
                self._reader = importlib.import_module("token_records")
            except Exception:
                self._reader = False
        return self._reader or None

    def request_body(self, messages, settings, sampling, drafting, force_tokens, log_name, route_stats, token_topk):
        tune = dict(self.profiles.get("drafting" if drafting else "one_row", {}))
        lab = schema.lab_needed(settings)
        defaults = schema.defaults()
        for key, value in settings.items():
            if key == "route_layer_scale":
                if any(abs(x - 1.0) > 1e-9 for x in value):
                    tune[key] = ",".join(f"{x:g}" for x in value)
            elif key == "routing_prior_bonus":
                tune[key] = float(value)
            elif value != defaults[key]:
                tune[key] = value
        if lab:
            tune["route_lab"] = 1
            tune["route_stats"] = int(route_stats)
        if log_name:
            tune["token_log"] = log_name
            tune["token_topk"] = int(token_topk)
        body = {"model": self.model, "messages": messages, "stream": True,
                "temperature": sampling.get("temperature", 1.0), "top_p": sampling.get("top_p", 0.95),
                "top_k": 0, "min_p": 0,
                "max_tokens": int(sampling.get("max_tokens", 1024)), "seed": int(sampling.get("seed", 0)),
                "reasoning_effort": sampling.get("reasoning_effort", "high"),
                "loop_guard": False,              # the lab shows what the model wrote, repeats included
                "strata_tune": tune}
        if force_tokens is not None:
            tune["force_tokens"] = [int(t) for t in force_tokens]
            tune["verify_segment"] = 0
            body["max_tokens"] = max(body["max_tokens"], len(force_tokens) + 8)
        return body

    def generate(self, messages, settings, sampling, drafting, force_tokens, cancel: threading.Event,
                 route_stats: int = 2, token_topk: int = 64):
        if self.records_dir is None:
            self.info()
        log_name = f"lab-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}" if self.records_dir else None
        if force_tokens is not None and not log_name:
            yield {"type": "error", "message": "the engine reports no token-record directory, so forced decoding cannot be recorded"}
            return
        body = self.request_body(messages, settings, sampling, drafting, force_tokens, log_name, route_stats, token_topk)
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        tail = _Tail(self.records_dir / f"{log_name}.jsonl", self.token_text()) if log_name else None
        done_summary = None

        def drain():
            nonlocal done_summary
            for rec in (tail.read() if tail else []):
                kind = rec.get("type", "token")
                if kind == "done":
                    done_summary = dict(rec, begin=tail.begin)
                elif kind == "token":
                    yield {"type": "token", "record": rec}

        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for raw in resp:
                    if cancel.is_set():
                        raise Cancelled()
                    yield from drain()
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in chunk:
                        yield {"type": "error", "message": json.dumps(chunk["error"])[:800]}
                        return
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            yield {"type": "delta", "text": delta["reasoning_content"], "channel": "reasoning"}
                        if delta.get("content"):
                            yield {"type": "delta", "text": delta["content"]}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            yield {"type": "error", "message": f"the engine refused the request (HTTP {exc.code}): {detail}"}
            return
        except urllib.error.URLError as exc:
            yield {"type": "error", "message": f"cannot reach the engine at {self.url}: {exc.reason}"}
            return
        deadline = time.time() + 5.0                       # the file's last lines may land just after the stream
        while tail and done_summary is None and time.time() < deadline:
            yield from drain()
            if done_summary is None:
                time.sleep(0.1)
        yield {"type": "done", "summary": done_summary or {"type": "done", "finish": "unknown",
                                                         "note": "no token records were written for this request"}}


def _find(obj, key):
    """The first value of `key` anywhere in a nested status object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find(v, key)
            if hit is not None:
                return hit
    return None


class TokenText:
    """Token ids to text, from a pack's tokenizer directory (vocab.json, token_type.json). Standard library only.

    The vocabulary is byte-level BPE in GPT-2's alphabet: every byte is one printable character, so a token's
    bytes are read back character by character (tools/strata_tokenizer.py is the reference; this is its decoding
    half). A character of several bytes can be split across tokens, so the text of committed tokens comes from an
    incremental decoder: the character appears on the token that completes it.
    """

    def __init__(self, directory: Path):
        vocab = json.loads((Path(directory) / "vocab.json").read_text(encoding="utf-8"))
        self.tokens = [""] * len(vocab)
        for text, i in vocab.items():
            self.tokens[i] = text
        types = Path(directory) / "token_type.json"
        self.types = json.loads(types.read_text()) if types.exists() else None
        bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_of = {chr(c): b for b, c in zip(bs, cs)}
        self.cache: dict[int, bytes] = {}

    def raw(self, i: int) -> bytes:
        b = self.cache.get(i)
        if b is None:
            if not 0 <= i < len(self.tokens):
                b = f"<{i}>".encode()
            elif self.types and self.types[i] in (3, 4):          # control and user-defined tokens are their text
                b = self.tokens[i].encode("utf-8")
            else:
                try:
                    b = bytes(self.byte_of[ch] for ch in self.tokens[i])
                except KeyError:
                    b = self.tokens[i].encode("utf-8")
            self.cache[i] = b
        return b

    def one(self, i: int) -> str:
        return self.raw(i).decode("utf-8", "replace")

    def stream(self):
        """A function for consecutive committed tokens: id -> the text that token completes."""
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        return lambda i: dec.decode(self.raw(i))


class _Tail:
    """Read the complete JSON lines appended to a token-record file since the last call, and add token text.

    The engine has no tokenizer, so its records carry ids only; `text` (a TokenText) fills the committed token's
    text and turns each candidate [id, p] into [id, p, text]. Without it the page shows ids.
    """

    def __init__(self, path: Path, text: TokenText | None = None):
        self.path, self.offset, self.partial, self.text = Path(path), 0, b"", text
        self.committed = text.stream() if text else None
        self.begin = None

    def read(self) -> list[dict]:
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b"\n")
        self.partial = lines.pop()
        out = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            kind = rec.get("type", "token")
            if kind == "begin":
                self.begin = rec
                continue
            if kind == "token":
                one = self.text.one if self.text else (lambda i: f"<{i}>")
                if "text" not in rec:
                    rec["text"] = self.committed(int(rec["id"])) if self.committed else f"<{rec.get('id')}>"
                for name in ("top", "samp"):
                    if name in rec:
                        rec[name] = [[t[0], t[1], t[2] if len(t) > 2 else one(int(t[0]))] for t in rec[name]]
                route = rec.get("route")
                if isinstance(route, dict) and "layers" in route and "layers" not in rec:
                    rec["layers"] = route.pop("layers")      # the page reads per-layer arrays beside the totals
            out.append(rec)
        return out
