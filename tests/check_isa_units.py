"""Check that each ISA unit's object files contain only that ISA (ctest isa_units).

    check_isa_units.py --objdump OBJDUMP --nm NM --unit=scalar=a.o|b.o --unit=avx2=c.o ...

Fails if
  * a scalar-unit object contains a VEX (AVX, AVX2, FMA, F16C, BMI) or EVEX
    (AVX-512) instruction, or LZCNT/TZCNT/POPCNT, none of which baseline
    x86-64 has;
  * an AVX2-unit object contains an EVEX instruction;
  * a function is emitted out of line (weak/COMDAT: inline functions,
    templates) by objects of two different units. The linker keeps one copy, chosen by
    link order, so code built for a narrower ISA could end up calling the
    AVX2/AVX-512 copy and fault on a CPU without it.

Instructions are classified by their first opcode byte, not by mnemonic: in
64-bit mode 0xC4/0xC5 always start a VEX instruction and 0x62 an EVEX one
(after optional segment or address-size prefixes, which VEX/EVEX allow).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict

INSN = re.compile(r"^\s*[0-9a-f]+:\t([0-9a-f]{2}(?: [0-9a-f]{2})*)\s*\t(\S+)")
SKIP_PREFIX = {"26", "2e", "36", "3e", "64", "65", "67"}
BASELINE_FORBIDDEN_MNEMONICS = ("lzcnt", "tzcnt", "popcnt")


def classify(first_bytes: list[str]) -> str:
    i = 0
    while i < len(first_bytes) and first_bytes[i] in SKIP_PREFIX:
        i += 1
    b = first_bytes[i] if i < len(first_bytes) else ""
    return {"c4": "vex", "c5": "vex", "62": "evex"}.get(b, "legacy")


def scan(objdump: str, obj: str) -> dict[str, list[str]]:
    """Return {'vex': [...], 'evex': [...], 'bitcount': [...]} sample lines."""
    out = subprocess.run([objdump, "-d", obj], capture_output=True, text=True, check=True).stdout
    found: dict[str, list[str]] = defaultdict(list)
    func = "?"
    for line in out.splitlines():
        if line.endswith(">:"):
            func = line.split("<", 1)[1][:-2]
            continue
        m = INSN.match(line)
        if not m:
            continue
        kind = classify(m.group(1).split())
        mnemonic = m.group(2)
        if kind != "legacy":
            found[kind].append(f"{func}: {line.strip()}")
        elif mnemonic.startswith(BASELINE_FORBIDDEN_MNEMONICS):
            found["bitcount"].append(f"{func}: {line.strip()}")
    return found


def weak_symbols(nm: str, obj: str) -> set[str]:
    out = subprocess.run([nm, "--defined-only", obj], capture_output=True, text=True,
                         check=True).stdout
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        # W: weak code (inline functions, template instantiations). Weak data
        # (V, u: e.g. DW.ref.__gxx_personality_v0, static locals) cannot run
        # instructions, so it is not a hazard.
        if len(parts) >= 3 and parts[1] == "W":
            syms.add(parts[2])
    return syms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objdump", required=True)
    ap.add_argument("--nm", required=True)
    ap.add_argument("--unit", action="append", default=[], help="NAME=obj1|obj2|...")
    args = ap.parse_args()

    units: dict[str, list[str]] = defaultdict(list)
    for spec in args.unit:
        name, objs = spec.split("=", 1)
        units[name] += [o for o in objs.split("|") if o]

    forbidden = {"scalar": ("vex", "evex", "bitcount"), "avx2": ("evex",), "avx512": ()}
    errors = []
    owner: dict[str, set[str]] = defaultdict(set)
    for unit, objs in units.items():
        for obj in objs:
            found = scan(args.objdump, obj)
            for kind in forbidden[unit]:
                if found.get(kind):
                    errors.append(f"{unit} unit {obj}: {len(found[kind])} {kind} instructions, "
                                  f"e.g. {found[kind][0]}")
            for sym in weak_symbols(args.nm, obj):
                owner[sym].add(unit)
        print(f"{unit}: {len(objs)} objects checked")
    shared = sorted(s for s, u in owner.items() if len(u) > 1)
    for sym in shared:
        errors.append(f"weak symbol in units {sorted(owner[sym])}: {sym}")
    for e in errors:
        print("FAIL", e)
    print("isa_units:", "FAILED" if errors else "passed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
