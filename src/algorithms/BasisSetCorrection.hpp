#ifndef BASIS_SET_CORRECTION_DEFINED
#define BASIS_SET_CORRECTION_DEFINED

#include <algorithms/Algorithm.hpp>

namespace sisi4s {
/**
 * \brief Focal-point basis-set correction of the CCSD correlation energy
 * (MP2 pair-energy increment + ps-ppl correction), as cc4s'
 * BasisSetCorrection (Irmler, Gallo, Grueneis, JCP 154, 234103 (2021),
 * Eq. (31)).
 *
 * Inputs (restricted, spatial orbitals; real or complex amplitudes):
 *  - CcsdSinglesAmplitudes [Nv,No], CcsdDoublesAmplitudes [Nv,Nv,No,No]
 *  - PPHHCoulombIntegrals V_abij [Nv,Nv,No,No]
 *  - HoleEigenEnergies [No], ParticleEigenEnergies [Nv]
 *  - Mp2PairEnergies [No,No] (real): CBS estimates of the MP2 pair energies
 *  - DeltaIntegralsPPHH D_abij = <ab|delta(r12)|ij> [Nv,Nv,No,No]
 *  - DeltaIntegralsHH n_ij = <ij|delta(r12)|ij> [No,No]
 *  - unrestricted (default 0), scheme (default "cc4s"): only the restricted
 *    cc4s scheme is implemented; anything else aborts.
 * Output (yaml "energy" and log): correction, pplCorrection,
 *  secondOrderCorrection, uncorrectedCorrelation; optional real outputs
 *  BasisSetCorrection (= correction) and CorrectedCcsdEnergy.
 */
DEFINE_ALGORITHM_HEADER(BasisSetCorrection,
                        template <typename F>
                        void run();
                        template <typename F>
                        Tensor<F> *getConvertedTensor(std::string const &name););

} // namespace sisi4s

#endif
