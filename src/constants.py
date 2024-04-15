#!/usr/bin/python

import gmpy2
from gmpy2 import mpfr

# numerical precision of mpfr numbers
mpfr_precision = 1000
gmpy2.get_context().precision=1000

constants = {
    "precision": mpfr_precision,
    "N_A": mpfr('6.022') * gmpy2.exp10(mpfr('23')),  # Avogadro's constant
    "e": mpfr('1.602') * gmpy2.exp10(mpfr('-19')),  # elementary charge
    "k_B": mpfr('1.38') * gmpy2.exp10(mpfr('-23')),  # Boltzmann constant
    "T": mpfr('310'),  # Temperature
    "eps_0": mpfr('8.854') * gmpy2.exp10(mpfr('-12')),  # vacuum permittivity
    "pi": mpfr('3.141592653589793238462643383279502884197169399375105820974944592307816406286'),  # pi 
}
