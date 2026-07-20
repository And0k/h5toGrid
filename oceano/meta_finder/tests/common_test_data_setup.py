import pytest
import json


@pytest.fixture(scope="session")
def common_test_data_setup():
    """Session-level fixture to create common test data structures in test_data directory."""
    import json
    from pathlib import Path

    base_path = Path("test_data/test_common_setup")
    base_path.mkdir(parents=True, exist_ok=True)

    # Define expected files with their content as a constant dict
    EXPECTED_FILES = {
        # Basic test case 1: cruise_with_device_name_and_subdirs
        base_path / "230507_ABP53_inclinometer": None,  # Directory
        base_path / "230507_ABP53_inclinometer" / "230508_inclinometer@i03": None,  # Directory
        base_path / "230507_ABP53_inclinometer" / "230509_wavegauge@w01": None,  # Directory
        base_path / "230507_ABP53_inclinometer" / "230508_inclinometer@i03" / "info_devices.json": json.dumps({
            "i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),
        base_path / "230507_ABP53_inclinometer" / "230509_wavegauge@w01" / "info_devices.json": json.dumps({
            "w1": ["p2", 51.5, 1.3, "B", 55.5, 21.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),

        # Basic test case 2: cruise_with_device_name_no_device_subdirs
        base_path / "230507_cruise_inclinometer": None,  # Directory
        base_path / "230507_cruise_inclinometer" / "230508_other": None,  # Directory
        base_path / "230507_cruise_inclinometer" / "230508_other" / "info_devices.json": json.dumps({
            "i7": ["p3", 52.5, 1.4, "C", 56.5, 22.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),

        # Basic test case 3: cruise_with_ati_and_subdirs
        base_path / "230507_ABP53@i": None,  # Directory
        base_path / "230507_ABP53@i" / "230508_inclinometer@i03": None,  # Directory
        base_path / "230507_ABP53@i" / "230509@i04": None,  # Directory
        base_path / "230507_ABP53@i" / "230508_inclinometer@i03" / "info_devices.json": json.dumps({
            "i3": ["p4", 53.5, 1.5, "D", 57.5, 23.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),
        base_path / "230507_ABP53@i" / "230509@i04" / "info_devices.json": json.dumps({
            "i4": ["p5", 54.5, 1.6, "E", 58.5, 24.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),

        # Basic test case 4: cruise_with_device_name_dated_subdirs_no_device_files
        base_path / "230507_test_inclinometer": None,  # Directory
        base_path / "230507_test_inclinometer" / "230508_test": None,  # Directory
        base_path / "230507_test_inclinometer" / "230508_test" / "info_devices.json": json.dumps({
            "i8": ["p6", 55.5, 1.7, "F", 59.5, 25.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
        }),

        # Combined device test directory
        base_path / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6": None,  # Directory
        base_path / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "info_devices.json": json.dumps({
            "i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600],
            "i4": ["p2", 51.5, 1.3, "B", 55.5, 21.1, "2019-11-08T12:00:0", "2019-11-08T13:00:00", 300, 600]
        }),
        base_path / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "text_output": None,  # Directory
        base_path / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "text_output" / "230508_1551bin2s@i03.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2019-11-08 12:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Additional edge case directories for comprehensive testing
        # Combined devices test
        base_path / "240101_combined_devices_test": None,  # Directory
        base_path / "240101_combined_devices_test" / "240102_multi@i01,i02": None,  # Directory
        base_path / "240101_combined_devices_test" / "240102_multi@i01,i02" / "info_devices.json": json.dumps({
            "i01": ["p7", 53.0, 1.8, "C", 57.0, 23.5, "2020-01-02T10:00:00", "2020-01-02T11:00:00", 300, 600],
            "i02": ["p8", 53.1, 1.9, "D", 57.1, 23.6, "2020-01-02T10:00:00", "2020-01-02T11:00:00", 300, 600]
        }),
        base_path / "240101_combined_devices_test" / "240102_multi@i01,i02" / "text_output": None,  # Directory
        base_path / "240101_combined_devices_test" / "240102_multi@i01,i02" / "text_output" / "240102_1200bin10s@i01,i02.tsv": "Time\tVabs_i01\tVdir_i01\tv_i01\tu_i01\tTemp_i01\tVabs_i02\tVdir_i02\tv_i02\tu_i02\tTemp_i02\n2020-01-02 10:00:00\t0.1\t180\t0.05\t0.08\t20.1\t0.2\t185\t0.06\t0.09\t20.2\n2020-01-02 10:00:10\t0.15\t185\t0.06\t0.09\t20.2\t0.25\t190\t0.07\t0.10\t20.3",

        # Edge case with cruise-level info_devices.json
        base_path / "240202_edge_case_cruise": None,  # Directory
        base_path / "240202_edge_case_cruise" / "info_devices.json": json.dumps({
            "i5": ["p9", 54.0, 2.0, "E", 58.0, 24.0, "2020-02-02T12:00:00", "2020-02-02T13:00:00", 300, 600]
        }),

        # HDF5 fallback test case
        base_path / "240303_hdf5_fallback_test": None,  # Directory
        base_path / "240303_hdf5_fallback_test" / "240304_device@i05": None,  # Directory
        base_path / "240303_hdf5_fallback_test" / "240304_device@i05" / "info_devices.json": json.dumps({
            "i05": ["p10", 55.0, 2.1, "F", 59.0, 25.0, "?", "?", "-", "-"]
        }),
        base_path / "240303_hdf5_fallback_test" / "240304_device@i05" / "text_output": None,  # Directory
        base_path / "240303_hdf5_fallback_test" / "240304_device@i05" / "text_output" / "240304_1400bin30s@i05.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2020-03-04 14:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2020-03-04 14:00:30\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Archived data test case
        base_path / "240404_archived_data_test": None,  # Directory
        base_path / "240404_archived_data_test" / "240405_archive@w03": None,  # Directory
        base_path / "240404_archived_data_test" / "240405_archive@w03" / "info_devices.json": json.dumps({
            "w03": ["p11", 56.0, 2.2, "G", 60.0, 26.0, "2020-04-05T15:00", "2020-04-05T16:00", 300, 600]
        }),
        base_path / "240404_archived_data_test" / "240405_archive@w03" / "text_output.zip": "PLACEHOLDER FOR ARCHIVE FILE",

        # Raw data test case
        base_path / "240505_raw_data_test": None,  # Directory
        base_path / "240505_raw_data_test" / "240506_raw@i07": None,  # Directory
        base_path / "240505_raw_data_test" / "240506_raw@i07" / "info_devices.json": json.dumps({
            "i07": ["p12", 57.0, 2.3, "H", 61.0, 27.0, "2020-05-06T16:00:00", "2020-05-06T17:00", 300, 600]
        }),
        base_path / "240505_raw_data_test" / "240506_raw@i07" / "_raw": None,  # Directory
        base_path / "240505_raw_data_test" / "240506_raw@i07" / "_raw" / "incl123.h5": "PLACEHOLDER FOR H5 FILE",

        # Complex filename patterns test
        base_path / "240606_complex_patterns": None,  # Directory
        base_path / "240606_complex_patterns" / "240607_pattern@i08": None,  # Directory
        base_path / "240606_complex_patterns" / "240607_pattern@i08" / "info_devices.json": json.dumps({
            "i08": ["p13", 58.0, 2.4, "I", 62.0, 28.0, "2020-06-07T17:00:00", "2020-06-07T18:00:0", 300, 600]
        }),
        base_path / "240606_complex_patterns" / "240607_pattern@i08" / "text_output": None,  # Directory
        base_path / "240606_complex_patterns" / "240607_pattern@i08" / "text_output" / "200113_000_i08.csv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2020-01-13 00:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2020-01-13 00:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Hash pattern test
        base_path / "240707_hash_patterns": None,  # Directory
        base_path / "240707_hash_patterns" / "240708_hash@w04": None,  # Directory
        base_path / "240707_hash_patterns" / "240708_hash@w04" / "info_devices.json": json.dumps({
            "w04": ["p14", 59.0, 2.5, "J", 63.0, 29.0, "2020-07-08T18:00:00", "2020-07-08T19:00:00", 300, 600]
        }),
        base_path / "240707_hash_patterns" / "240708_hash@w04" / "text_output": None,  # Directory
        base_path / "240707_hash_patterns" / "240708_hash@w04" / "text_output" / "191210#07,23,30,32-bin300s.tsv": "Time\tVabs_w07\tVdir_w07\tVabs_w23\tVdir_w23\tVabs_w30\tVdir_w30\tVabs_w32\tVdir_w32\n2019-12-10 18:00:00\t0.1\t180\t0.2\t185\t0.3\t190\t0.4\t195",

        # Mixed patterns test
        base_path / "240808_mixed_patterns": None,  # Directory
        base_path / "240808_mixed_patterns" / "240809_mixed@i09": None,  # Directory
        base_path / "240808_mixed_patterns" / "240809_mixed@i09" / "info_devices.json": json.dumps({
            "i09": ["p15", 60.0, 2.6, "K", 64.0, 30.0, "2020-08-09T19:00", "20-08-09T20:00", 300, 600]
        }),
        base_path / "240808_mixed_patterns" / "240809_mixed@i09" / "text_output": None,  # Directory
        base_path / "240808_mixed_patterns" / "240809_mixed@i09" / "text_output" / "210618_180bin10s.tsv": "Time\tVabs_i09_14\tVdir_i09_14\tv_i09_14\tTemp_i09_14\n2021-06-18 18:00:00.0000\t1.0\t45.0\t0.5\t25.0\n2021-06-18 18:10:00.000\t1.1\t46.0\t0.6\t25.5",

        # GPX test
        base_path / "240909_gpx_test": None,  # Directory
        base_path / "240909_gpx_test" / "240910_gpx@i10": None,  # Directory
        base_path / "240909_gpx_test" / "240910_gpx@i10" / "info_devices.json": json.dumps({
            "i10": ["p16", 61.0, 2.7, "L", "?", "?", "2020-09-10T20:00:00", "2020-09-10T21:00:00", 300, 600]
        }),
        base_path / "240909_gpx_test" / "240910_gpx@i10" / "navigation": None,  # Directory
        base_path / "240909_gpx_test" / "240910_gpx@i10" / "navigation" / "track.gpx": '<?xml version="1.0"?><gpx><wpt lat="61.0" lon="2.7"><name>i10</name></wpt></gpx>',

        # Underscore device test
        base_path / "241010_underscore_devices": None,  # Directory
        base_path / "241010_underscore_devices" / "241011_underscore@i11": None,  # Directory
        base_path / "241010_underscore_devices" / "241011_underscore@i11" / "info_devices.json": json.dumps({
            "i11": ["p17", 62.0, 2.8, "M", 65.0, 31.0, "2020-10-11T21:00:00", "2020-10-11T22:00:00", 300, 600],
            "i11_": ["p17_1", 62.0, 2.8, "M", 65.0, 31.0, "2020-10-11T21:00:00", "2020-10-11T22:00:00", 300, 600]
        }),
        base_path / "241010_underscore_devices" / "241011_underscore@i11" / "text_output": None,  # Directory
        base_path / "241010_underscore_devices" / "241011_underscore@i11" / "text_output" / "241011_1600bin60s@i11_.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2020-10-11 16:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2020-10-11 17:00:00\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # No device subdirectories test
        base_path / "250101_no_device_subdirs": None,  # Directory
        base_path / "250101_no_device_subdirs" / "250102_data": None,  # Directory
        base_path / "2501_no_device_subdirs" / "250102_data" / "info_devices.json": json.dumps({
            "i12": ["p18", 63.0, 2.9, "N", 66.0, 32.0, "2021-01-02T22:00:00", "2021-01-02T23:00:00", 300, 600]
        }),

        # Multiple text outputs test
        base_path / "250202_multiple_text_outputs": None,  # Directory
        base_path / "250202_multiple_text_outputs" / "250203_multi@i12": None,  # Directory
        base_path / "250202_multiple_text_outputs" / "250203_multi@i12" / "info_devices.json": json.dumps({
            "i12": ["p19", 64.0, 3.0, "O", 67.0, 33.0, "2021-02-03T23:00:00", "2021-02-04T00:00:00", 300, 600]
        }),
        base_path / "250202_multiple_text_outputs" / "250203_multi@i12" / "text_output": None,  # Directory
        base_path / "250202_multiple_text_outputs" / "250203_multi@i12" / "text_output" / "250203_1000bin2s@i12.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2021-02-03 10:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2021-02-03 10:02\t0.15\t185\t0.06\t0.09\t5.3\t20.2",
        base_path / "250202_multiple_text_outputs" / "250203_multi@i12" / "text_output" / "250203_1000bin60s@i12.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2021-02-03 10:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2021-02-03 11:00:00\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Burst detection test
        base_path / "250303_burst_detection": None,  # Directory
        base_path / "250303_burst_detection" / "250304_burst@i13": None,  # Directory
        base_path / "250303_burst_detection" / "250304_burst@i13" / "info_devices.json": json.dumps({
            "i13": ["p20", 65.0, 3.1, "P", 68.0, 34.0, "2021-03-04T00:00:00", "2021-03-04T01:00:00", 300, 600]
        }),
        base_path / "250303_burst_detection" / "250304_burst@i13" / "text_output": None,  # Directory
        base_path / "250303_burst_detection" / "250304_burst@i13" / "text_output" / "250304_1200bin2s@i13.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2021-03-04 00:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2021-03-04 00:00:02\t0.15\t185\t0.06\t0.09\t5.3\t20.2\n2021-03-04 00:10:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2021-03-04 00:10:02\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Special characters test
        base_path / "250404_special_chars": None,  # Directory
        base_path / "250404_special_chars" / "250405_special@i14": None,  # Directory
        base_path / "250404_special_chars" / "250405_special@i14" / "info_devices.json": json.dumps({
            "i14": ["p21", 66.0, 3.2, "Q", 69.0, 35.0, "2021-04-05T01:00:00", "2021-04-05T02:00:00", 300, 600]
        }),
        base_path / "250404_special_chars" / "250405_special@i14" / "text_output": None,  # Directory
        base_path / "250404_special_chars" / "250405_special@i14" / "text_output" / "250405_1400@i14.tsv": "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2021-04-05 14:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2021-04-05 14:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2",

        # Parentheses patterns test
        base_path / "250505_parentheses_patterns": None,  # Directory
        base_path / "250505_parentheses_patterns" / "250506_paren@i15": None,  # Directory
        base_path / "250505_parentheses_patterns" / "250506_paren@i15" / "info_devices.json": json.dumps({
            "i15": ["p22", 67.0, 3.3, "R", 70.0, 36.0, "2021-05-06T02:00:00", "2021-05-06T03:00:00", 300, 600],
            "i16": ["p23", 67.1, 3.4, "S", 70.1, 36.1, "2021-05-06T02:00:00", "2021-05-06T03:00:00", 300, 600]
        }),
        base_path / "250505_parentheses_patterns" / "250506_paren@i15" / "text_output": None,  # Directory
        base_path / "250505_parentheses_patterns" / "250506_paren@i15" / "text_output" / "250506_1500bin10s@i(15,16).tsv": "Time\tVabs_i15\tVdir_i15\tv_i15\tu_i15\tTemp_i15\tVabs_i16\tVdir_i16\tv_i16\tu_i16\tTemp_i16\n2021-05-06 15:00:00\t0.1\t180\t0.05\t0.08\t20.1\t0.2\t185\t0.06\t0.09\t20.2\n2021-05-06 15:00:10\t0.15\t185\t0.06\t0.09\t20.2\t0.25\t190\t0.07\t0.10\t20.3",

        # Range patterns test
        base_path / "250606_range_patterns": None,  # Directory
        base_path / "250606_range_patterns" / "250607_range@i17": None,  # Directory
        base_path / "250606_range_patterns" / "250607_range@i17" / "info_devices.json": json.dumps({
            "i17": ["p24", 68.0, 3.5, "T", 71.0, 37.0, "2021-06-07T03:00:00", "2021-06-07T04:00:00", 300, 600],
            "i18": ["p25", 68.1, 3.6, "U", 71.1, 37.1, "2021-06-07T03:00:00", "2021-06-07T04:00:00", 300, 600],
            "i19": ["p26", 68.2, 3.7, "V", 71.2, 37.2, "2021-06-07T03:00:00", "2021-06-07T04:00:00", 300, 600],
            "i20": ["p27", 68.3, 3.8, "W", 71.3, 37.3, "2021-06-07T03:00:00", "2021-06-07T04:00:00", 300, 600]
        }),
        base_path / "250606_range_patterns" / "250607_range@i17" / "text_output": None,  # Directory
        base_path / "250606_range_patterns" / "250607_range@i17" / "text_output" / "250607_1600bin5s@i17-20.tsv": "Time\tVabs_i17\tVdir_i17\tv_i17\tu_i17\tTemp_i17\tVabs_i18\tVdir_i18\tv_i18\tu_i18\tTemp_i18\tVabs_i19\tVdir_i19\tv_i19\tu_i19\tTemp_i19\tVabs_i20\tVdir_i20\tv_i20\tu_i20\tTemp_i20\n2021-06-07 16:00:00\t0.1\t180\t0.05\t0.08\t20.1\t0.1\t181\t0.051\t0.081\t20.11\t0.12\t182\t0.052\t0.082\t20.12\t0.13\t183\t0.053\t0.083\t20.13\n2021-06-07 16:00:05\t0.15\t185\t0.06\t0.09\t20.2\t0.16\t186\t0.061\t0.091\t20.21\t0.17\t187\t0.062\t0.092\t20.22\t0.18\t188\t0.063\t0.093\t20.23",

        # Semicolon patterns test
        base_path / "250707_semicolon_patterns": None,  # Directory
        base_path / "250707_semicolon_patterns" / "250708_semi@i21": None,  # Directory
        base_path / "250707_semicolon_patterns" / "250708_semi@i21" / "info_devices.json": json.dumps({
            "i21": ["p28", 69.0, 3.9, "X", 72.0, 38.0, "2021-07-08T04:00:00", "2021-07-08T05:00:00", 300, 600],
            "i22": ["p29", 69.1, 4.0, "Y", 72.1, 38.1, "2021-07-08T04:00:00", "2021-07-08T05:00:00", 300, 600],
            "ib27": ["p30", 69.2, 4.1, "Z", 72.2, 38.2, "2021-07-08T04:00:00", "2021-07-08T05:00", 300, 600],
            "ib28": ["p31", 69.3, 4.2, "AA", 72.3, 38.3, "2021-07-08T04:00:00", "2021-07-08T05:00", 300, 600],
            "ib29": ["p32", 69.4, 4.3, "AB", 72.4, 38.4, "2021-07-08T04:00:00", "2021-07-08T05:00:00", 300, 600],
            "ib30": ["p33", 69.5, 4.4, "AC", 72.5, 38.5, "2021-07-08T04:00:00", "2021-07-08T05:00:00", 300, 600]
        }),
        base_path / "250707_semicolon_patterns" / "250708_semi@i21" / "text_output": None,  # Directory
        base_path / "250707_semicolon_patterns" / "250708_semi@i21" / "text_output" / "250708_170bin15s@i21,22;ib27-30.tsv": "Time\tVabs_i21\tVdir_i21\tv_i21\tu_i21\tTemp_i21\tVabs_i22\tVdir_i22\tv_i22\tu_i22\tTemp_i22\tVabs_ib27\tVdir_ib27\tv_ib27\tu_ib27\tTemp_ib27\tVabs_ib28\tVdir_ib28\tv_ib28\tu_ib28\tTemp_ib28\tVabs_ib29\tVdir_ib29\tv_ib29\tu_ib29\tTemp_ib29\tVabs_ib30\tVdir_ib30\tv_ib30\tu_ib30\tTemp_ib30\n2021-07-08 17:00:00\t0.1\t180\t0.05\t0.08\t20.1\t0.11\t181\t0.051\t0.081\t20.11\t0.12\t182\t0.052\t0.082\t20.12\t0.13\t183\t0.053\t0.083\t20.13\t0.14\t184\t0.054\t0.084\t20.14\t0.15\t185\t0.055\t0.085\t20.15\n2021-07-08 17:00:15\t0.15\t185\t0.06\t0.09\t20.2\t0.16\t186\t0.061\t0.091\t20.21\t0.17\t187\t0.062\t0.092\t20.2\t0.18\t188\t0.063\t0.093\t20.23\t0.19\t189\t0.064\t0.094\t20.24\t0.20\t190\t0.065\t0.095\t20.25",
    }

    # Check if all files exist by checking a marker file
    marker_file = base_path / ".setup_complete"
    all_exist = marker_file.exists()

    if not all_exist:
        # Process all directories and files in one cycle
        for path, content in EXPECTED_FILES.items():
            if content is None:  # It's a directory
                path.mkdir(parents=True, exist_ok=True)
            else:  # It's a file
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write content only if file doesn't exist
                if not path.exists():
                    path.write_text(content)

        # Create marker file to indicate setup is complete
        marker_file.write_text("setup complete")

    return base_path
