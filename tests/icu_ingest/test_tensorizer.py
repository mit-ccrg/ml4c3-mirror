# Imports: standard library
import os
import sys
from typing import Dict, Union

# Imports: third party
import h5py
import numpy as np
import pytest

# Imports: first party
from ml4c3.arguments import parse_args
from ingest.icu.tensorizer import Tensorizer

# pylint: disable=no-member


def test_tensorizer(
    temp_dir,
    monkeypatch,
    test_scale_units: Dict[str, Dict[str, Union[int, float, str]]],
):
    monkeypatch.setattr("definitions.icu.ICU_SCALE_UNITS", test_scale_units)
    monkeypatch.setattr(
        "ingest.icu.tensorizer.ICU_SCALE_UNITS",
        test_scale_units,
    )
    monkeypatch.setattr(
        "ingest.icu.readers.ICU_SCALE_UNITS",
        test_scale_units,
    )

    test_dir = os.path.dirname(__file__)
    sys.argv = f"""
    .
    tensorize_icu_no_edw_pull
    --xref {test_dir}/data/xref_file_tensorize.csv
    --adt {test_dir}/data/edw/adt
    --bedmaster {test_dir}/data/bedmaster
    --edw {test_dir}/data/edw
    --alarms {test_dir}/data/bedmaster_alarms
    --output_folder {temp_dir}/{pytest.run_id}
    --tensors {os.path.join(temp_dir, pytest.run_id)}
    """.split()

    args = parse_args()
    output_file = os.path.join(args.tensors, f"{pytest.example_mrn}.hd5")

    # Make sure the file doesn't exist
    with pytest.raises(OSError):
        with h5py.File(output_file, "r") as tens_file:
            pass

    # Tensorize and check hd5 structure
    tensorizer = Tensorizer(
        args.bedmaster,
        args.alarms,
        args.edw,
        args.xref,
        args.adt,
    )
    tensorizer.tensorize(
        tensors=args.tensors,
        overwrite_hd5=args.overwrite,
        num_workers=args.num_workers,
        allow_one_source=args.allow_one_source,
    )

    with h5py.File(output_file, "r") as tens_file:
        assert sorted(list(tens_file.keys())) == ["bedmaster", "edw"]

        bedmaster_signals = {
            "waveform": ["i", "ii", "iii", "resp", "spo2", "v"],
            "vitals": ["co", "cuff", "hr", "spo2r", "spo2%"],
            "alarms": [
                "arrhy_suspend",
                "cpp_low",
                "hr_hi_156",
                "hr_hi_160",
                "hr_hi_165",
                "ppeak_high",
                "spo2_probe",
                "tachy",
                "tvexp_low",
            ],
        }
        edw_signals = {
            "med": [
                "aspirin_325_mg_tablet",
                (
                    "cefazolin_2_gram|50_ml_in_dextrose_(iso-osmotic)_"
                    "intravenous_piggyback"
                ),
                "lactated_ringers_iv_bolus",
                "norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh",
                "sodium_chloride_0.9_%_intravenous_solution",
            ],
            "flowsheet": [
                "blood_pressure",
                "pulse",
                "r_phs_ob_bp_diastolic_outgoing",
                "r_phs_ob_bp_systolic_outgoing",
                "r_phs_ob_pulse_oximetry_outgoing",
            ],
            "labs": ["creatinine", "lactate_blood", "magnesium", "ph_arterial"],
            "surgery": ["colonoscopy", "coronary_artery_bypass_graft"],
            "procedures": ["hemodialysis", "hemodialysis_|_ultrafiltration"],
            "transfusions": [
                "transfuse_cryoprecipitate",
                "transfuse_platelets",
                "transfuse_red_blood_cells",
            ],
            "events": ["rapid_response_start", "code_start"],
        }

        assert tens_file["bedmaster"].attrs["completed"]
        assert tens_file["edw"].attrs["completed"]

        assert sorted(tens_file["bedmaster/345"].keys()) == sorted(
            list(bedmaster_signals.keys()),
        )
        assert sorted(tens_file["edw/345"].keys()) == sorted(list(edw_signals.keys()))

        bedmaster_attrs = [
            "channel",
            "name",
            "scale_factor",
            "source",
            "units",
        ]
        bedmaster_sig_keys = [
            "time",
            "time_corr_arr",
            "value",
            "samples_per_ts",
            "sample_freq",
        ]
        bedmaster_alarms_keys = ["duration", "start_date"]

        # Test units, scaling factor and sample_freq
        hr_dir = tens_file["bedmaster/345/vitals/hr"]
        assert hr_dir.attrs["units"] == "Bpm"
        assert hr_dir.attrs["scale_factor"] == 0.5
        expected_sf = np.array([(0.5, 0)], dtype="float, int")
        assert np.array_equal(hr_dir["sample_freq"], expected_sf)

        ecg_ii = tens_file["bedmaster/345/waveform/ii"]
        assert ecg_ii.attrs["units"] == "mV"
        assert ecg_ii.attrs["scale_factor"] == 0.0243
        expected_sf = np.array(
            [(240.0, 0), (120.0, 80), (240.0, 5760), (120.0, 5808)],
            dtype="float, int",
        )
        assert np.array_equal(ecg_ii["sample_freq"], expected_sf)

        for sig_type, signals in bedmaster_signals.items():
            sig_type_dir = tens_file["bedmaster/345"][sig_type]
            assert sorted(sig_type_dir.keys()) == sorted(bedmaster_signals[sig_type])
            for signal in signals:
                # Test interbundle correction
                if sig_type == "vitals":
                    assert sorted(sig_type_dir[signal].keys()) == sorted(
                        bedmaster_sig_keys,
                    )
                    assert sorted(sig_type_dir[signal].attrs.keys()) == sorted(
                        bedmaster_attrs,
                    )
                    diff = np.diff(sig_type_dir[signal]["time"][()])
                    if signal == "hr":  # Signal has a data event 1
                        assert np.array_equal(np.where(diff != 2)[0], np.array([18]))
                    else:
                        # Test signals concatenated
                        assert len(sig_type_dir[signal]["time"]) == 20
                        assert all(diff == 2)

                elif sig_type == "waveform":
                    assert sorted(sig_type_dir[signal].keys()) == sorted(
                        bedmaster_sig_keys,
                    )
                    assert sorted(sig_type_dir[signal].attrs.keys()) == sorted(
                        bedmaster_attrs,
                    )
                    diff = np.diff(sig_type_dir[signal]["time"][:3349])
                    if signal == "v":
                        assert np.array_equal(
                            np.where(diff != 0.25)[0],
                            np.array([113]),
                        )
                    else:
                        assert all(diff == 0.25)
                        length = 32 if signal == "resp" else 160
                        assert len(sig_type_dir[signal]["time"]) == length
                else:
                    assert sorted(sig_type_dir[signal].keys()) == sorted(
                        bedmaster_alarms_keys,
                    )

        for key in edw_signals:
            assert sorted(tens_file[f"edw/345/{key}"].keys()) == sorted(
                edw_signals[key],
            )
