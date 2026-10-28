"""Write paper/numbers.tex: python -m refusal_parity.reporting.paper_numbers (run from src/)."""

from refusal_parity.reporting.paper_numbers import controls, end_to_end, measurement, mechanism, operations
from refusal_parity.reporting.paper_numbers.core import OUT, lines


def main():
    measurement.judge_validation()
    measurement.closed_model_gemini_judge()
    measurement.baseline_b0()
    measurement.h1_keyword_detectors()
    measurement.gate_on_failures()
    operations.roi()
    mechanism.natural_drift()
    mechanism.tiny_aya_report_values()
    mechanism.drift_power()
    mechanism.circuitkit_parity()
    mechanism.detection_vs_action()
    mechanism.cast_gate_level()
    mechanism.gate_parity_matched_cost()
    end_to_end.arms_end_to_end()
    end_to_end.arms_level_vs_spread()
    mechanism.gate_brittleness()
    operations.hardware_check()
    end_to_end.table_arms()
    end_to_end.eors_vs_gcal()
    end_to_end.followups_alpha_half_and_greeting()
    end_to_end.table_arms_alpha_half()
    controls.controls_primary()
    controls.controls_style_check()
    controls.controls_greeting()
    controls.controls_frontier()
    controls.controls_calibration_shift()
    controls.controls_review_round2()
    mechanism.cast_linear_regime()
    measurement.closed_model_primary_judge()
    controls.table_controls()
    operations.compute_per_arm()
    mechanism.circuitkit_appendix()
    end_to_end.steering_strength()
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({sum(l.startswith(chr(92)) for l in lines)} macros)")


if __name__ == "__main__":
    main()
