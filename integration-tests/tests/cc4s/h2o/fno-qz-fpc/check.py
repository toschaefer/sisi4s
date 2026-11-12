#!/usr/bin/env python3
# Compare Sisi4s' energies with the cc4s reference (cc4s test h2o_fno-qz (correct.out.yaml, test-resources 8c7a8d34); unit 1 (Hartree).)
# The cc4s run used energy and amplitudes convergence 1e-8, as in.yaml here.

from testis import read_yaml

ACCURACY = 1e-7
REFERENCE = {
    "BasisSetCorrection": {
        "correction": -0.016542395813125374,
        "pplCorrection": 0.006390805404366886,
        "secondOrderCorrection": -0.02293320121749226,
        "uncorrectedCorrelation": -0.2813034283659386
    },
    "CcsdPerturbativeTriples": {
        "ccsd": -0.28130342836593875,
        "triples": -0.00872706734651425
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
