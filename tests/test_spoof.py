"""spoof.summarise: firing rates, language-ID flip rates and paired deltas vs the original prompt. CPU, instant."""

from refusal_parity.analysis import spoof


def rec(i, cond, fires, detected="ar"):
    return dict(id=i, lang="ar", kind="benign", cond=cond, detected=detected, score=0.0, threshold=0.0, fires=fires)


def test_paired_delta_and_flip_rate():
    recs = [rec(i, "orig", False) for i in range(4)] + [
        rec(0, "en_prefix", True, "en"),
        rec(1, "en_prefix", True, "en"),
        rec(2, "en_prefix", False),
        rec(3, "en_prefix", False),
    ]
    s = {r["cond"]: r for r in spoof.summarise(recs)}
    assert s["orig"]["fire_rate"] == 0.0 and "delta" not in s["orig"]
    assert s["en_prefix"]["fire_rate"] == 0.5
    assert s["en_prefix"]["delta"] == 0.5  # paired: two of four prompts start firing
    assert s["en_prefix"]["lid_flip"] == 0.5  # two of four detected as English
    assert s["en_prefix"]["delta_lo"] <= 0.5 <= s["en_prefix"]["delta_hi"]


if __name__ == "__main__":
    test_paired_delta_and_flip_rate()
    print("ok")
