# Algorithm of frequency estimation `utils_time_corr._estimate_freq_np`

## Regime B

For 1 s-floored N-Hz data every second is a "run" of equal timestamps. Run-length = samples recorded
that second. A device at fractional Hz (e.g. 5.3 Hz) produces exactly two valid run-lengths —
\(\lfloor f \rfloor\) and \(\lceil f \rceil\) — mixed in proportion \(\text{frac}(f)\):

```text
5.3 Hz  →  70 % × {5}  +  30 % × {6}  →  mean = 5.30
5.8 Hz  →  20 % × {5}  +  80 % × {6}  →  mean = 5.80
```

\(N_{\text{base}} = \lfloor f \rfloor\) is always the smaller of the two. The problem: for 5.8 Hz
\(\text{mode}(\text{runs}) = 6\) (majority), yet \(N_{\text{base}} = 5\) (the floor). So the algorithm
checks whether mode is the ceiling rather than the floor.

The 80 % gate: if \(\{\text{mode}-1,\ \text{mode}\}\) together cover \(\ge 80\,\%\) of all runs, those
two values dominate the distribution the way a genuine fractional rate would — so \(\text{mode}-1\) is
the true \(N_{\text{base}}\). If they don't reach 80 %, the runs below mode are burst-boundary scatter
(partial seconds at burst edges create runs of length 1, 2, 3 … that look like \(N_{\text{base}}\)
candidates but aren't): \(N_{\text{base}}\) stays at mode.

```text
5.8 Hz cont.:  {5, 6} = 100 % ≥ 80 %  →  N_base = mode−1 = 5  ✓
burst 5 Hz:    {4, 5} = 55 %  < 80 %  →  N_base = mode = 5    ✓  (4 s are artifacts)
```
