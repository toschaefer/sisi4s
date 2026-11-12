#!/usr/bin/env python3
# Compare Sisi4s' energies with the cc4s reference (cc4s test r-bn_gamma (correct.out.yaml, test-resources 8c7a8d34); VASP data, energies in eV.)
# The cc4s run used energy and amplitudes convergence 1e-8, as in.yaml here.

from testis import read_yaml

ACCURACY = 1e-7
REFERENCE = {
    "BasisSetCorrection": {
        "correction": -5.299802224716957,
        "pplCorrection": 1.2013854043019938,
        "secondOrderCorrection": -6.5011876290189505,
        "uncorrectedCorrelation": -25.445627726271653
    },
    "CcsdPerturbativeTriples": {
        "ccsd": -25.44562772627135,
        "triples": -0.8225306501106308,
        "starCorrection": -0.2195121258171968
    }
}

steps = read_yaml("sisi4s.out.yaml")["steps"]
for name, ref in REFERENCE.items():
    found = [s for s in steps if s["name"] == name]
    if len(found) != 1:
        raise Exception("expected one step %s, found %d" % (name, len(found)))
    energy = found[0]["energy"]
    for key, value in ref.items():
        diff = abs(float(energy[key]) - value)
        print("%s.%s: %.16f (cc4s %.16f, diff %.2e)" % (name, key, energy[key], value, diff))
        if diff > ACCURACY:
            raise Exception("%s.%s differs from cc4s by %e > %e" % (name, key, diff, ACCURACY))
