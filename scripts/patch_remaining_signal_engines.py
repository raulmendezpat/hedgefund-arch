from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path("src/hf/engines/signals")

TARGETS = [
    "sol_bbrsi_signal.py",
    "sol_vol_breakout_signal.py",
    "sol_trend_pullback_signal.py",
    "sol_extreme_mr_signal.py",
    "sol_vol_compression_signal.py",
]

def backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)

def ensure_penalty_fields(txt: str) -> str:
    if "regime_as_metadata" in txt:
        return txt

    anchor = '    only_if_symbol_contains: str = "SOL"\n'
    insert = '''    only_if_symbol_contains: str = "SOL"

    # research: regime/context as metadata, not hard rejection
    strength_penalty_adx: float = 0.70
    strength_penalty_atrp: float = 0.70
    strength_penalty_rsi: float = 0.85
    strength_penalty_bb_width: float = 0.85
    strength_penalty_range_expansion: float = 0.85
    strength_penalty_directional_close: float = 0.90
    strength_penalty_extension: float = 0.85
    strength_penalty_trend_alignment: float = 0.85
    strength_penalty_donchian: float = 0.85
'''
    if anchor in txt:
        return txt.replace(anchor, insert, 1)
    return txt

def soften_common_hard_filters(txt: str) -> str:
    replacements = [
        (
            re.compile(
                r'(\s*)if\s+([A-Za-z_][A-Za-z0-9_]*)\s*<\s*float\(self\.adx_min\):\n'
                r'\1\s+out\[sym\]\s*=\s*self\._flat\(sym,\s*"adx_low"(.*?)\)\n'
                r'\1\s+continue\n',
                re.S,
            ),
            r'\1adx_low = \2 < float(self.adx_min)\n',
        ),
        (
            re.compile(
                r'(\s*)if\s+([A-Za-z_][A-Za-z0-9_]*)\s*<\s*float\(self\.atrp_min\):\n'
                r'\1\s+out\[sym\]\s*=\s*self\._flat\(sym,\s*"atrp_low"(.*?)\)\n'
                r'\1\s+continue\n',
                re.S,
            ),
            r'\1atrp_low = \2 < float(self.atrp_min)\n',
        ),
        (
            re.compile(
                r'(\s*)if\s+float\(self\.min_range_expansion\)\s*>\s*0\.0:\n'
                r'(\1\s+if\s+[A-Za-z_][A-Za-z0-9_]*\s+is\s+None:\n'
                r'\1\s+out\[sym\]\s*=\s*self\._flat\(sym,\s*"missing_range_expansion"(.*?)\)\n'
                r'\1\s+continue\n)?'
                r'(\1\s+if\s+float\([A-Za-z_][A-Za-z0-9_]*\)\s*<\s*float\(self\.min_range_expansion\):\n'
                r'\1\s+out\[sym\]\s*=\s*self\._flat\(sym,\s*"range_expansion_low"(.*?)\)\n'
                r'\1\s+continue\n)?',
                re.S,
            ),
            lambda m: (
                f'{m.group(1)}range_expansion_low = False\n'
                f'{m.group(1)}if float(self.min_range_expansion) > 0.0:\n'
                f'{m.group(1)}    if range_expansion is None:\n'
                f'{m.group(1)}        range_expansion_low = True\n'
                f'{m.group(1)}    elif float(range_expansion) < float(self.min_range_expansion):\n'
                f'{m.group(1)}        range_expansion_low = True\n'
            ),
        ),
    ]

    out = txt
    for pat, repl in replacements:
        out = pat.sub(repl, out)
    return out

def inject_strength_penalties(txt: str) -> str:
    anchor = '            strength = 1.0 if side != "flat" else 0.0\n'
    if anchor not in txt:
        return txt
    if "regime_as_metadata" in txt and "strength_penalty_adx" in txt and "adx_low" in txt and "strength *= float(self.strength_penalty_adx)" in txt:
        return txt

    block = '''            strength = 1.0 if side != "flat" else 0.0

            if side != "flat" and bool(locals().get("atrp_low", False)):
                strength *= float(self.strength_penalty_atrp)
            if side != "flat" and bool(locals().get("adx_low", False)):
                strength *= float(self.strength_penalty_adx)
            if side != "flat" and bool(locals().get("range_expansion_low", False)):
                strength *= float(self.strength_penalty_range_expansion)
'''
    return txt.replace(anchor, block, 1)

def enrich_meta(txt: str) -> str:
    old = '''                meta={
                    "engine": '''
    if old not in txt:
        return txt

    txt = txt.replace(
'''                    "engine": ''',
'''                    "engine": ''',
1
    )

    txt = re.sub(
        r'(\s+"bb_low":\s*bb_low,\n)(\s+\},\n)',
        r'\1'
        r'\2',
        txt,
        count=1,
    )

    # add metadata flags only once
    if '"regime_as_metadata": True,' in txt:
        return txt

    txt = txt.replace(
'''                    "bb_low": bb_low,
''',
'''                    "bb_low": bb_low,
                    "regime_as_metadata": True,
                    "atrp_low": bool(locals().get("atrp_low", False)),
                    "adx_low": bool(locals().get("adx_low", False)),
                    "range_expansion_low": bool(locals().get("range_expansion_low", False)),
''',
1
    )
    return txt

def patch_file(name: str) -> tuple[str, str]:
    p = ROOT / name
    if not p.exists():
        return name, "missing"

    backup(p)
    txt = p.read_text(encoding="utf-8")

    original = txt
    txt = ensure_penalty_fields(txt)
    txt = soften_common_hard_filters(txt)
    txt = inject_strength_penalties(txt)
    txt = enrich_meta(txt)

    if txt == original:
        return name, "no_change"

    p.write_text(txt, encoding="utf-8")
    return name, "patched"

def main() -> None:
    results = []
    for name in TARGETS:
        results.append(patch_file(name))

    for name, status in results:
        print(f"{status.upper():<10} {name}")

if __name__ == "__main__":
    main()
