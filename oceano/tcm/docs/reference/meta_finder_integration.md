# meta_finder Integration

## Contracts reused verbatim

- Archive extensions `.zip`/`.7z` (`config.extensions_archive` ↔ `tcm._constants.ARCHIVE_EXTS`).
- Directory exclusion `.*-(?:\.|$)`, `^bad$`, `^test[^.]*$`.
- Device keywords `inclinometers?|incl|tcm|wave_?gau?ges?|pressure|pres|@i[0-9]?`
  plus device-id suffix grammars (complex `i3,5,9,w1-6`, ranges, parentheses).
- Metadata array layout `[point, sea_depth, h_above, symbol, lat, lon,
  time_st, time_en, burst_dt, bursts_t, comment, …]`; file priority
  `info_devices@meta_finder.yaml` > `info_devices.yaml` > `.json`.
- Full specs: [meta_finder I/O formats](../../../meta_finder/docs/reference/io_formats.md),
  [meta_finder config reference](../../../meta_finder/docs/reference/config_reference.md).



`tcm`-side rows: [Console Messages](../user_guide/console_messages.md).
