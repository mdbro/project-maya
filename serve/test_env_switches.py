"""serve/test_env_switches.py - the environment switches' list (serve/env_switches.py) against the engine's copy
(src/mm3/mm3_env_switches.hpp) and against the code that reads switches; the server's start check (check_environment)
and its one reading at the start (env_switches.freeze / get).  No GPU, no engine.

    python -m unittest serve.test_env_switches -v
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import env_switches as E  # noqa: E402
from serve import server as S  # noqa: E402
from serve.server import check_environment  # noqa: E402
import maya_names  # noqa: E402  (tools/, on the path once serve.server is imported)

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/mm3/mm3_env_switches.hpp"

ENTRY = re.compile(r'^\{"([A-Za-z0-9_]+)",\s*(true|false)\},?\s*(?://.*)?$')


def header_blocks(text: str) -> dict[str, list[tuple[str, bool]]]:
    """kSwitches' entries by block - {"engine" | "server" | "launch": [(name, maya_alias), ...]} - each block opened by
    its comment line; any other line inside the array is an error (the parse has to see every entry)."""
    start = text.index("kSwitches[] = {")
    body = text[start:text.index("\n};", start)].splitlines()[1:]
    blocks, current = {}, None
    for raw in body:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("//"):
            low = line.lower()
            current = ("launch" if "the launch path" in low else "server" if "the server" in low else
                       "engine" if "maya-mm3" in low else None)
            if current is None or current in blocks:
                raise AssertionError(f"mm3_env_switches.hpp: a block comment not understood: {line}")
            blocks[current] = []
            continue
        m = ENTRY.match(line)
        if m is None or current is None:
            raise AssertionError(f"mm3_env_switches.hpp: a line in kSwitches not understood: {line}")
        blocks[current].append((m.group(1), m.group(2) == "true"))
    return blocks


# what the code reads switches with: the server's Python (os.environ.get, env_switches.get, an env dict's get, ...)
PY_READ = re.compile(r"""(?<![A-Za-z0-9_])(?:get|getenv|read_as)\(\s*["']((?:MM3|MAYA|STRATA)_[A-Z0-9_]+)["']"""
                     r"""|environ\[\s*["']((?:MM3|MAYA|STRATA)_[A-Z0-9_]+)["']\s*\]""")
# ... and the engine's C++: getenv / std::getenv, and mm3::env (src/mm3/mm3_env.hpp), which reads MAYA_<name> first
CPP_GETENV = re.compile(r'(?<![A-Za-z0-9_])(?:std::)?_?w?getenv\(\s*(?:L|u8)?"((?:MM3|MAYA|STRATA)_[A-Z0-9_]+)"')
CPP_ENV = re.compile(r'(?<![A-Za-z0-9_:])(?:::)?(?:mm3::)?env\(\s*"((?:MM3|MAYA|STRATA)_[A-Z0-9_]+)"')
CPP_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
CPP_SUFFIXES = {".cpp", ".cc", ".c", ".cu", ".hpp", ".h", ".cuh", ".inc", ".inl"}
CPP_ROOTS = ("src/mm3", "src/kernels/cpu")           # maya-mm3's own code and its CPU kernels, all of it
CPP_HEADERS = "include/strata"                       # ... and the shared headers they include
INCLUDE_DIRS = ("include", "src/mm3", "src")         # (CMakeLists.txt: maya-mm3's include directories)


def engine_files() -> list[Path]:
    """src/mm3 and src/kernels/cpu, and the include/strata headers they reach - also through src/ds4's headers, which
    are followed but not read (only part of src/ds4 is linked into maya-mm3).  An include/strata header that nothing of
    maya-mm3 includes (GLM's) reads switches of another engine."""
    todo = [p for d in CPP_ROOTS for p in sorted((ROOT / d).rglob("*")) if p.suffix in CPP_SUFFIXES]
    seen = set()
    while todo:
        p = todo.pop().resolve()
        if p in seen:
            continue
        seen.add(p)
        for name in CPP_INCLUDE.findall(p.read_text(encoding="utf-8", errors="replace")):
            for base in (p.parent, *(ROOT / d for d in INCLUDE_DIRS)):
                if (base / name).is_file():
                    todo.append(base / name)
                    break
    roots = [(ROOT / d).resolve() for d in (*CPP_ROOTS, CPP_HEADERS)]
    return sorted(p for p in seen if any(p.is_relative_to(r) for r in roots))


class TheLists(unittest.TestCase):
    def test_the_engine_header_lists_the_same_switches(self):
        blocks = header_blocks(HEADER.read_text(encoding="utf-8"))
        self.assertEqual(set(blocks), {"engine", "server", "launch"})
        names = [n for entries in blocks.values() for n, _ in entries]
        self.assertEqual(len(names), len(set(names)), "a name listed twice")
        self.assertEqual({n for n, _ in blocks["engine"]}, set(E.ENGINE))
        self.assertEqual({n for n, alias in blocks["engine"] if alias}, set(E.MAYA_ALIAS))
        self.assertEqual({n for n, _ in blocks["server"]}, set(E.SERVER))
        self.assertEqual({n for n, _ in blocks["launch"]}, set(E.LAUNCH))
        self.assertEqual([n for b in ("server", "launch") for n, alias in blocks[b] if alias], [])
        self.assertEqual(set(names), set(E.SWITCHES))
        self.assertLessEqual(set(E.MAYA_ALIAS), set(E.ENGINE))
        # the prefixes the start check covers: the same three
        checked = HEADER.read_text(encoding="utf-8").split("inline bool checked(", 1)[1].split("}", 1)[0]
        self.assertEqual(set(re.findall(r'"([A-Z0-9]+_)"', checked)), set(E.PREFIXES))

    def test_every_switch_the_code_reads_is_known(self):
        found, unknown = {}, {}

        def saw(name, where, maya_too=False):
            found.setdefault(name, where)
            for n in (name, "MAYA_" + name) if maya_too else (name,):
                if not E.known(n):
                    unknown.setdefault(n, where)
        for p in sorted((ROOT / "serve").glob("*.py")):
            if p.name.startswith("test_"):
                continue
            for m in PY_READ.finditer(p.read_text(encoding="utf-8")):
                saw(m.group(1) or m.group(2), p.relative_to(ROOT).as_posix())
        python_names = len(found)
        for k in (*S.SERVER_ENV, *S.RESPONSE_ENV_KEYS):   # (the config's env entries the server puts in its environment)
            saw(k, "serve/server.py SERVER_ENV")
        for p in engine_files():
            text = p.read_text(encoding="utf-8", errors="replace")
            where = p.relative_to(ROOT.resolve()).as_posix()
            for m in CPP_GETENV.finditer(text):
                saw(m.group(1), where)
            for m in CPP_ENV.finditer(text):
                saw(m.group(1), where, maya_too=True)
        # (the scans find what they should: a broken pattern would pass every name it does not see)
        self.assertGreater(python_names, 20)
        self.assertGreater(len(found) - python_names, 60)
        self.assertEqual(unknown, {}, "read by the code but in neither list (serve/env_switches.py, "
                                      "src/mm3/mm3_env_switches.hpp)")

    def test_known_follows_the_names_the_launch_path_makes(self):
        for name in ("MAYA_ENGINE_STALL_S", "STRATA_ENGINE_STALL_S", "STRATA_MM3_SIDE", "MAYA_MM3_SIDE", "MM3_SIDE",
                     "MAYA_MM3_ATTN_WIDE", "STRATA_MM3_ATTN_WIDE", "MM3_SERVE_MONITORING", "MAYA_SH",
                     "CUDA_VISIBLE_DEVICES"):
            self.assertTrue(E.known(name), name)
        for name in ("MAYA_MM3_SERVE_MONITORING", "STRATA_MM3_SERVE_MONITORING", "MM3_FOO", "MAYA_MM3_FOO",
                     "MAYA_FOO", "STRATA_FOO", "MAYA_", "STRATA_", ""):
            self.assertFalse(E.known(name), name)
        # every name maya_names gives a listed setting (alias_env: MAYA_X -> STRATA_X; export_env: both names) is known
        env = {n: "1" for n in E.SWITCHES} | {"MAYA_" + n: "1" for n in E.MAYA_ALIAS}
        maya_names.alias_env(env)
        for k in list(env):
            env.update({n: "1" for n in maya_names.both_names(k)})
        self.assertEqual([k for k in env if k.startswith(E.PREFIXES) and not E.known(k)], [])

    def test_report(self):
        found, unknown = E.report({"MM3_SIDE": "1", "MM3_FOO": "2", "HOME": "/h", "STRATA_DEBUG": "1",
                                   "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(found, [("MM3_FOO", "2"), ("MM3_SIDE", "1"), ("STRATA_DEBUG", "1")])
        self.assertEqual(unknown, ["MM3_FOO"])


class StartCheck(unittest.TestCase):
    """main's check: every MM3_ / MAYA_ / STRATA_ variable set, a line each; a MiniMax config refuses a name no list
    knows, the other families' configs (whose engines read switches the list does not cover) do not."""
    ENV = {"MM3_FOO": "1", "MAYA_MM3_ATTN_WIDE": "1", "STRATA_ENGINE_STALL_S": "0", "PATH": "/usr/bin"}

    def check(self, cfg, env):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                return check_environment(cfg, env), out.getvalue().splitlines(), None
            except SystemExit as e:
                return None, out.getvalue().splitlines(), str(e.code)

    def test_a_minimax_config_refuses_an_unknown_switch(self):
        for cfg in ({"exe": "/x/maya-mm3", "args": ["--pack", "/p", "--serve"]},
                    {"exe": "/x/maya-mm3-200031d", "args": ["--pack", "/p"]},
                    {"exe": "/x/engine", "installer": {"family": "minimax"}}):
            unknown, lines, refused = self.check(cfg, self.ENV)
            self.assertIsNone(unknown, cfg)
            self.assertIn("MM3_FOO", refused)
            self.assertIn("serve/env_switches.py", refused)
            self.assertNotIn("MAYA_MM3_ATTN_WIDE", refused)
            self.assertEqual(lines, ["[maya] environment MAYA_MM3_ATTN_WIDE=1", "[maya] environment MM3_FOO=1",
                                     "[maya] environment STRATA_ENGINE_STALL_S=0"])
        unknown, lines, refused = self.check({"exe": "/x/maya-mm3"}, {"MM3_FOO": "1", "MAYA_BAR": "2"})
        self.assertIn("MAYA_BAR, MM3_FOO are not", refused)
        unknown, lines, refused = self.check({"exe": "/x/maya-mm3"}, {k: v for k, v in self.ENV.items() if k != "MM3_FOO"})
        self.assertEqual((unknown, refused), ([], None))
        self.assertEqual(len(lines), 2)

    def test_other_families_print_only(self):
        for cfg in ({"exe": "/x/maya", "args": ["--glm-pack", "/p/pack"]}, {"exe": "/x/maya-ds4", "args": ["--pack", "/p"]},
                    {}):
            unknown, lines, refused = self.check(cfg, self.ENV)
            self.assertEqual((unknown, refused), (["MM3_FOO"], None), cfg)
            self.assertIn("[maya] environment MM3_FOO=1", lines)

    def test_a_key_is_not_shown(self):
        unknown, lines, refused = self.check({"exe": "/x/maya-mm3"}, {"STRATA_API_KEY": "sekrit", "MAYA_API_KEY": "sekrit"})
        self.assertEqual((unknown, refused), ([], None))
        self.assertEqual(lines, ["[maya] environment MAYA_API_KEY=(set, not shown)",
                                 "[maya] environment STRATA_API_KEY=(set, not shown)"])


class ReadOnce(unittest.TestCase):
    """get(): the environment as it is now until freeze(), then the reading freeze() took (main, at the start)."""

    def test_get_before_and_after_freeze(self):
        self.addCleanup(setattr, E, "_frozen", None)    # (the other tests read the environment as it is)
        E._frozen = None
        with unittest.mock.patch.dict(os.environ, {"STRATA_DRAIN_S": "7"}):
            os.environ.pop("STRATA_FAIR_SLICE_S", None)
            self.assertFalse(E.frozen())
            self.assertEqual(E.get("STRATA_DRAIN_S"), "7")
            os.environ["STRATA_DRAIN_S"] = "8"
            self.assertEqual(E.get("STRATA_DRAIN_S"), "8")              # before freeze: as it is now
            self.assertEqual(E.get("STRATA_FAIR_SLICE_S", "0"), "0")
            E.freeze()
            self.assertTrue(E.frozen())
            os.environ["STRATA_DRAIN_S"] = "9"
            os.environ["STRATA_FAIR_SLICE_S"] = "0.5"
            self.assertEqual(E.get("STRATA_DRAIN_S"), "8")              # after: the start's reading
            self.assertIsNone(E.get("STRATA_FAIR_SLICE_S"))
            self.assertEqual(E.get("STRATA_FAIR_SLICE_S", "0"), "0")
            self.assertEqual((S._drain_seconds(), S.fair_slice_seconds()), (8.0, 0.0))   # (the server's readers)
        E.freeze({"MAYA_UPDATE_CHECK": "0", "CUDA_VISIBLE_DEVICES": "1", "PATH": "/bin"})   # (a given reading)
        self.assertEqual((E.get("MAYA_UPDATE_CHECK"), E.get("CUDA_VISIBLE_DEVICES"), E.get("PATH")), ("0", "1", None))
        from serve.update import Updater
        self.assertFalse(Updater(None).enabled())
        E._frozen = None
        with unittest.mock.patch.dict(os.environ, {"MAYA_UPDATE_CHECK": "1"}):
            self.assertTrue(Updater(None).enabled())


if __name__ == "__main__":
    unittest.main()
