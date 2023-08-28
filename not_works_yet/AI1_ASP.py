#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: 
  Created: 11.11.2014
"""


import unittest

from pyzcasp import potassco

clingo = potassco.Clingo(r"c:\Programs\_coding\_AI\clingo\clingo.exe")  # path to clingo binary

prog = r"""bird(tux). penguin(tux).
bird(tweety). chicken(tweety).
-flies(X) :- bird(X), not flies(X).
-flies(X) :- penguin(X).
:- flies(tux), -flies(tux).
:- flies(tweety), -flies(tweety).
"""

answers = clingo.run(prog)  # , grounder_args=["-c k=2"], solver_args=["-n0"]
for ans in answers:
    print([(str(t)) for t in ans])  # , t.pred, t.arg(0)

if __name__ == '__main__':
    unittest.main()
