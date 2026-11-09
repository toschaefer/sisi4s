#include <Sisi4s.hpp>
#include <util/Yaml.hpp>
#include <util/Tensor.hpp>
#include <util/Exception.hpp>
#include <algorithm>
#include <algorithms/cc4s/DefineHolesAndParticles.hpp>

namespace sisi4s {

IMPLEMENT_ALGORITHM(DefineHolesAndParticles) {

  const std::string fileName = getTextArgument("fileName");

  cc4s::HPHeader h = YAML::LoadFile(fileName).as<cc4s::HPHeader>();

  LOG(0, "DefineHolesAndParticles") << "fermiEnergy: " << h.fermiEnergy << "\n";

  LOG(0, "DefineHolesAndParticles")
      << "read elements: " << h.energies.size() << "\n";

  // The states are not reordered: the Coulomb vertex read separately uses
  // the same state ordering as the energies, so all occupied states
  // (e <= fermiEnergy) must come first.
  const auto isHole = [&h](double e) { return e <= h.fermiEnergy; };
  if (!std::is_partitioned(h.energies.begin(), h.energies.end(), isHole)) {
    throw new EXCEPTION("DefineHolesAndParticles: energies must list all "
                        "states with e <= fermiEnergy before all others");
  }
  const auto it =
      std::partition_point(h.energies.begin(), h.energies.end(), isHole);

  const TensorIndex No = it - h.energies.begin(), Nv = h.energies.size() - No;

  const int dense = NS;
  auto *epsi = new Tensor<double>(1, &No, &dense, *Sisi4s::world),
       *epsa = new Tensor<double>(1, &Nv, &dense, *Sisi4s::world);

  LOG(0, "DefineHolesAndParticles") << "No: " << No << "\n";
  LOG(0, "DefineHolesAndParticles") << "Nv: " << Nv << "\n";

  {
    // write epsi
    auto &t = epsi;
    const auto count = Sisi4s::world->rank == 0 ? No : 0;
    std::vector<TensorIndex> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    t->write(count, indices.data(), h.energies.data());
    allocatedTensorArgument<double>("HoleEigenEnergies", t);
  }

  {
    // write epsa
    auto &t = epsa;
    const auto count = Sisi4s::world->rank == 0 ? Nv : 0;
    std::vector<TensorIndex> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    t->write(count, indices.data(), h.energies.data() + No);
    allocatedTensorArgument<double>("ParticleEigenEnergies", t);
  }
}

} // namespace sisi4s

namespace YAML {

using namespace sisi4s::cc4s;
template <>
struct convert<HPHeader> {
  using Native = HPHeader;
  static bool decode(Node const &n, Native &r) {
    if (!n.IsMap()) return false;
    YAML_ASSERT_KEY(n, "metaData");
    YAML_ASSERT_KEY(n["metaData"], "fermiEnergy");
    YAML_ASSERT_KEY(n["metaData"], "energies");
    r.fermiEnergy = n["metaData"]["fermiEnergy"].as<double>();
    r.energies = n["metaData"]["energies"].as<std::vector<double>>();
    return true;
  }
};

} // namespace YAML
