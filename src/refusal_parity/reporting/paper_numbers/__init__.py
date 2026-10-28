"""Generate paper/numbers.tex: every number quoted in the paper, computed from result files (no hand-typed values).

    python -m refusal_parity.reporting.paper_numbers

Each macro carries a comment naming its evidence source, which doubles as the claim->evidence registry.

One module per paper part; __main__ runs them in the order that fixes numbers.tex."""
