#include <algorithms/BasisSetCorrection.hpp>
#include <math/Complex.hpp>
#include <math/ComplexTensor.hpp>
#include <math/MathFunctions.hpp>
#include <util/Log.hpp>
#include <util/Emitter.hpp>
#include <util/Exception.hpp>
#include <util/Tensor.hpp>
#include <Sisi4s.hpp>

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace sisi4s;

ALGORITHM_REGISTRAR_DEFINITION(BasisSetCorrection);

namespace {

// abort with a message if the lengths of tensor `name` differ from `expected`
template <typename F>
void checkLens(Tensor<F> const &t,
               std::string const &name,
               std::vector<int64_t> const &expected,
               std::string const &expectedText) {
  bool ok(t.order == static_cast<int>(expected.size()));
  for (int d(0); ok && d < t.order; ++d) ok = t.lens[d] == expected[d];
  if (!ok) {
    std::stringstream s;
    s << "BasisSetCorrection: " << name << " has dimensions [";
    for (int d(0); d < t.order; ++d) s << (d ? "," : "") << t.lens[d];
    s << "], expected " << expectedText << " = [";
    for (size_t d(0); d < expected.size(); ++d)
      s << (d ? "," : "") << expected[d];
    s << "] (No, Nv from HoleEigenEnergies, ParticleEigenEnergies)";
    throw new EXCEPTION(s.str());
  }
}

template <typename F>
void conjugateInPlace(Tensor<F> &t, const char *indices) {
  CTF::Transform<F>(std::function<void(F &)>([](F &x) { x = sisi4s::conj(x); }))(
      t[indices]);
}

template <typename F>
void realPart(Tensor<F> &c, Tensor<double> &r, const char *indices) {
  CTF::Transform<F, double>(std::function<void(F, double &)>(
      [](F x, double &y) { y = std::real(x); }))(c[indices], r[indices]);
}

// g <- g * (1/n), elementwise, as cc4s
template <typename F>
void divideBy(Tensor<F> &n, Tensor<F> &g, const char *indices) {
  CTF::Transform<F, F>(std::function<void(F, F &)>(
      [](F x, F &y) { y = y * (F(1.0) / x); }))(n[indices], g[indices]);
}

template <typename F>
std::vector<F> readAll(Tensor<F> &t) {
  int64_t n(1);
  for (int d(0); d < t.order; ++d) n *= t.lens[d];
  std::vector<F> v(n);
  t.read_all(v.data());
  return v;
}

} // namespace

template <typename F>
Tensor<F> *BasisSetCorrection::getConvertedTensor(std::string const &name) {
  Data *data(getArgumentData(name));
  if (auto td = dynamic_cast<TensorData<F> *>(data)) {
    return new Tensor<F>(*td->value);
  }
  auto rd(dynamic_cast<TensorData<double> *>(data));
  if (rd && !std::is_same<F, double>::value) {
    // real data (e.g. vasp_gam Delta integrals) with complex amplitudes
    auto t(new Tensor<F>(rd->value->order,
                         rd->value->lens,
                         rd->value->sym,
                         *rd->value->wrld,
                         name.c_str()));
    toComplexTensor(*rd->value, *t);
    return t;
  }
  throw new EXCEPTION("BasisSetCorrection: " + name
                      + " must be a tensor of " + TypeTraits<F>::getName()
                      + " (the type of PPHHCoulombIntegrals), found "
                      + data->getTypeName());
}

void BasisSetCorrection::run() {
  const int unrestricted(getIntegerArgument("unrestricted", 0));
  const std::string scheme(getTextArgument("scheme", "cc4s"));
  if (unrestricted != 0) {
    throw new EXCEPTION("BasisSetCorrection: unrestricted (spin-orbital) "
                        "basis-set correction not implemented yet; use "
                        "unrestricted: 0");
  }
  if (scheme != "cc4s") {
    throw new EXCEPTION("BasisSetCorrection: unknown or unimplemented scheme '"
                        + scheme + "'; only 'cc4s' (restricted) is available");
  }
  EMIT() << YAML::Key << "unrestricted" << YAML::Value << unrestricted
         << YAML::Key << "scheme" << YAML::Value << scheme;

  Data *Vabij(getArgumentData("PPHHCoulombIntegrals"));
  if (dynamic_cast<TensorData<double> *>(Vabij)) {
    run<double>();
  } else {
    run<sisi4s::complex>();
  }
}

template <typename F>
void BasisSetCorrection::run() {
  auto epsi(getTensorArgument<double>("HoleEigenEnergies"));
  auto epsa(getTensorArgument<double>("ParticleEigenEnergies"));
  const int64_t No(epsi->lens[0]), Nv(epsa->lens[0]);

  auto Tai(getTensorArgument<F>("CcsdSinglesAmplitudes"));
  auto Tabij(getTensorArgument<F>("CcsdDoublesAmplitudes"));
  auto Vabij(getTensorArgument<F>("PPHHCoulombIntegrals"));
  auto epsCbs(getTensorArgument<double>("Mp2PairEnergies"));
  std::unique_ptr<Tensor<F>> Dabij(getConvertedTensor<F>("DeltaIntegralsPPHH"));
  std::unique_ptr<Tensor<F>> nij(getConvertedTensor<F>("DeltaIntegralsHH"));

  checkLens(*Tai, "CcsdSinglesAmplitudes", {Nv, No}, "[Nv,No]");
  checkLens(*Tabij, "CcsdDoublesAmplitudes", {Nv, Nv, No, No}, "[Nv,Nv,No,No]");
  checkLens(*Vabij, "PPHHCoulombIntegrals", {Nv, Nv, No, No}, "[Nv,Nv,No,No]");
  checkLens(*epsCbs, "Mp2PairEnergies", {No, No}, "[No,No]");
  checkLens(*Dabij, "DeltaIntegralsPPHH", {Nv, Nv, No, No}, "[Nv,Nv,No,No]");
  checkLens(*nij, "DeltaIntegralsHH", {No, No}, "[No,No]");

  // n_ij = <ij|delta|ij> must not vanish (it is a denominator)
  {
    auto n(readAll(*nij));
    for (int64_t k(0); k < No * No; ++k) {
      if (n[k] == F(0) || !std::isfinite(std::abs(n[k]))) {
        std::stringstream s;
        s << "BasisSetCorrection: DeltaIntegralsHH n_ij = " << n[k]
          << " for i=" << k % No << ", j=" << k / No
          << " (zero or not finite); g_ij = .../n_ij is undefined";
        throw new EXCEPTION(s.str());
      }
    }
  }

  LOG(1, "BasisSetCorrection") << "No=" << No << " Nv=" << Nv
                               << " scalar type: " << TypeTraits<F>::getName()
                               << std::endl;

  const int64_t oo[] = {No, No};
  const int syms2[] = {NS, NS};
  auto &world(*Sisi4s::world);

  Tensor<F> Fepsi(1, &epsi->lens[0], epsi->sym, world, "Fepsi");
  toComplexTensor(*epsi, Fepsi);
  Tensor<F> Fepsa(1, &epsa->lens[0], epsa->sym, world, "Fepsa");
  toComplexTensor(*epsa, Fepsa);

  // MP2 amplitudes M_abij = conj(V_abij) / (e_i + e_j - e_a - e_b)
  Tensor<F> Mabij(false, *Vabij);
  Mabij.set_name("Mabij");
  Mabij["abij"] = Fepsi["i"];
  Mabij["abij"] += Fepsi["j"];
  Mabij["abij"] -= Fepsa["a"];
  Mabij["abij"] -= Fepsa["b"];
  {
    Tensor<F> cV(*Vabij);
    conjugateInPlace(cV, "abij");
    CTF::Transform<F, F>(std::function<void(F, F &)>(
        [](F v, F &d) { d = v * (F(1.0) / d); }))(cV["abij"], Mabij["abij"]);
  }

  // MP2 pair energies in the finite (FNO) basis
  Tensor<F> epsFno(2, oo, syms2, world, "epsFno");
  epsFno["ij"] = 2.0 * Mabij["abij"] * (*Vabij)["abij"];
  epsFno["ij"] += (-1.0) * Mabij["abij"] * (*Vabij)["abji"];
  CTF::Scalar<F> eMp2Scalar(world);
  eMp2Scalar[""] = epsFno["ij"];
  CTF::Scalar<double> eMp2CbsScalar(world);
  eMp2CbsScalar[""] = (*epsCbs)["ij"];

  // T_abij = conj(t_abij + t_ai t_bj); CCSD energy recomputed from T
  Tensor<F> T(*Tabij);
  T.set_name("T");
  T["abij"] += (*Tai)["ai"] * (*Tai)["bj"];
  conjugateInPlace(T, "abij");
  CTF::Scalar<F> eCcsdScalar(world);
  eCcsdScalar[""] = 2.0 * T["abij"] * (*Vabij)["abij"];
  eCcsdScalar[""] += (-1.0) * T["abij"] * (*Vabij)["abji"];

  // g^cc_ij = sum_ab D*_abij T_abij / n_ij, g^(1)_ij = sum_ab D*_abij M_abij / n_ij
  conjugateInPlace(*Dabij, "abij");
  Tensor<F> gcc(2, oo, syms2, world, "gcc");
  Tensor<F> g1(2, oo, syms2, world, "g1");
  gcc["ij"] = (*Dabij)["abij"] * T["abij"];
  g1["ij"] = (*Dabij)["abij"] * Mabij["abij"];
  divideBy(*nij, gcc, "ij");
  divideBy(*nij, g1, "ij");

  // g^eff = Re g^cc + Re g^(1) + Re g^(1) Re g^cc
  Tensor<double> gccR(2, oo, syms2, world, "gccR");
  Tensor<double> g1R(2, oo, syms2, world, "g1R");
  realPart(gcc, gccR, "ij");
  realPart(g1, g1R, "ij");
  Tensor<double> geff(2, oo, syms2, world, "geff");
  geff["ij"] = gccR["ij"];
  geff["ij"] += g1R["ij"];
  geff["ij"] += g1R["ij"] * gccR["ij"];

  // Delta E_ppl = sum_ij g^eff_ij (eps^CBS_ij - Re eps^FNO_ij)
  Tensor<double> epsFnoR(2, oo, syms2, world, "epsFnoR");
  realPart(epsFno, epsFnoR, "ij");
  Tensor<double> epsDel(2, oo, syms2, world, "epsDel");
  epsDel["ij"] = (*epsCbs)["ij"];
  epsDel["ij"] += (-1.0) * epsFnoR["ij"];
  CTF::Scalar<double> eDeltaPplScalar(world);
  eDeltaPplScalar[""] = geff["ij"] * epsDel["ij"];

  const double eMp2(std::real(eMp2Scalar.get_val()));
  const double eMp2Cbs(eMp2CbsScalar.get_val());
  const double eCcsd(std::real(eCcsdScalar.get_val()));
  const double pplCorrection(eDeltaPplScalar.get_val());
  const double secondOrderCorrection(-eMp2 + eMp2Cbs);
  const double correction(-eMp2 + eMp2Cbs + pplCorrection);

  // per-pair table (log file only, level 2)
  {
    auto vCbs(readAll(*epsCbs)), vFno(readAll(epsFnoR)), vGcc(readAll(gccR)),
        vG1(readAll(g1R)), vGeff(readAll(geff));
    auto vN(readAll(*nij));
    LOG(2, "BasisSetCorrection")
        << "pairs: i j epsCbs epsFno gcc g1 geff n_ij (column-major ij)"
        << std::endl;
    for (int64_t k(0); k < No * No; ++k) {
      LOG(2, "BasisSetCorrection")
          << std::setprecision(16) << "pair " << k % No << " " << k / No
          << " " << vCbs[k] << " " << vFno[k] << " " << vGcc[k] << " "
          << vG1[k] << " " << vGeff[k] << " " << std::real(vN[k]) << std::endl;
    }
  }

  std::streamsize ss(std::cout.precision());
  LOG(0, "BasisSetCorrection")
      << std::setprecision(16) << "correction=" << correction << std::endl;
  LOG(1, "BasisSetCorrection")
      << "pplCorrection=" << pplCorrection << std::endl;
  LOG(1, "BasisSetCorrection")
      << "secondOrderCorrection=" << secondOrderCorrection << std::endl;
  LOG(1, "BasisSetCorrection")
      << "uncorrectedCorrelation=" << eCcsd << std::endl;
  LOG(1, "BasisSetCorrection") << "mp2=" << eMp2 << std::endl;
  LOG(1, "BasisSetCorrection") << "mp2Cbs=" << eMp2Cbs << std::endl;
  LOG(1, "BasisSetCorrection")
      << "correctedCorrelation=" << eCcsd + correction << std::setprecision(ss)
      << std::endl;

  EMIT() << YAML::Key << "energy" << YAML::Value << YAML::BeginMap
         << YAML::Key << "correction" << YAML::Value << correction
         << YAML::Key << "pplCorrection" << YAML::Value << pplCorrection
         << YAML::Key << "secondOrderCorrection" << YAML::Value
         << secondOrderCorrection << YAML::Key << "uncorrectedCorrelation"
         << YAML::Value << eCcsd << YAML::Key << "mp2" << YAML::Value << eMp2
         << YAML::Key << "mp2Cbs" << YAML::Value << eMp2Cbs << YAML::EndMap;

  if (isArgumentGiven("BasisSetCorrection")) {
    setRealArgument("BasisSetCorrection", correction);
  }
  if (isArgumentGiven("CorrectedCcsdEnergy")) {
    setRealArgument("CorrectedCcsdEnergy", eCcsd + correction);
  }
}

template void BasisSetCorrection::run<double>();
template void BasisSetCorrection::run<sisi4s::complex>();
