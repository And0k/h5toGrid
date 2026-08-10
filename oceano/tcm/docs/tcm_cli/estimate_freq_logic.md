Description of the _estimate_freq_np regime B algorithm

For 1 s-floored N-Hz data every second is a "run" of equal timestamps. Run-length = samples recorded that second. A device at fractional Hz (e.g. 5.3 Hz) produces exactly two valid run-lengths — floor(freq) and ceil(freq) — mixed in proportion frac(freq):
5.3 Hz  →  70 % × {5}  +  30 % × {6}  →  mean = 5.30
5.8 Hz  →  20 % × {5}  +  80 % × {6}  →  mean = 5.80
N_base = floor(freq) is always the smaller of the two. The problem: for 5.8 Hz mode(runs) = 6 (majority), yet N_base = 5 (the floor). So the algorithm checks whether mode is the ceiling rather than the floor.
The 80 % gate: if {mode−1, mode} together cover ≥ 80 % of all runs, those two values dominate the distribution the way a genuine fractional rate would — so mode−1 is the true N_base. If they don't reach 80 %, the runs below mode are burst-boundary scatter (partial seconds at burst edges create runs of length 1, 2, 3 … that look like N_base candidates but aren't): N_base stays at mode.
5.8 Hz cont.:  {5, 6} = 100 % ≥ 80 %  →  N_base = mode−1 = 5  ✓
burst 5 Hz:    {4, 5} = 55 %  < 80 %  →  N_base = mode = 5    ✓  (4 s are artifacts)