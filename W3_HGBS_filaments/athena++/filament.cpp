// ============================================================
// Athena++ Problem Generator: 3D Filament Fragmentation
//
// Purpose: Set up initial conditions for self-gravitating
//          isothermal filament with perturbations
//
// Initial Conditions:
//   - Cylindrical filament with density profile
//   - Sinusoidal perturbations along axis
//   - Optional magnetic field along filament
//
// Reference: Inutsuka & Miyama 1992, ApJ 388, 392
// ============================================================

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "hydro/eos/eos.hpp"
#include "field/field.hpp"
#include "utils/utils.hpp"

// ============================================================================
// Problem Generator: User-defined parameters from athinput
// ============================================================================

namespace {
// Physical constants
const double pc_to_cm = 3.086e18;         // Parsec to cm
const double km_s_to_cm_s = 1.0e5;        // km/s to cm/s
const double mu_H2 = 2.33;                // Mean molecular weight
const double m_H = 1.67e-24;               // Proton mass (g)
const double kB = 1.38e-16;               // Boltzmann constant (erg/K)

// Problem parameters (set in athinput <problem> block)
Real rho0;           // Initial density (g/cm³)
Real pert_amp;       // Perturbation amplitude
Real n_waves;        // Number of wavelengths
Real wavelength;     // Perturbation wavelength (pc)
Real r_filament;     // Filament radius (pc)
Real rho_center;     // Central density enhancement
Real rho_edge;       // Edge density (normalized)
Real temp;           // Temperature (K)
Real mu;             // Mean molecular weight
Real P_ext;          // External pressure (optional)

// Derived quantities
Real cs_isothermal;  // Isothermal sound speed
Real G_phys;         // Gravitational constant (physical units)
Real L_unit;         // Length unit (pc in cm)
Real V_unit;         // Velocity unit (km/s in cm/s)
Real rho_unit;       // Density unit (g/cm³)
Real P_unit;         // Pressure unit
Real t_unit;         // Time unit

// Filament parameters
Real aspect_ratio;   // L/D ratio
Real k_pert;         // Perturbation wavenumber

} // namespace

// ============================================================================
// Mesh initialization function
// ============================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters from athinput file
  rho0 = pin->GetReal("problem", "rho0");
  pert_amp = pin->GetOrAddReal("problem", "pert", 0.05);
  n_waves = pin->GetOrAddReal("problem", "n_waves", 5.0);
  wavelength = pin->GetOrAddReal("problem", "lambda", 0.16);
  r_filament = pin->GetOrAddReal("problem", "r_filament", 0.04);
  rho_center = pin->GetOrAddReal("problem", "rho_center", 2.0);
  rho_edge = pin->GetOrAddReal("problem", "rho_edge", 1.0);
  temp = pin->GetOrAddReal("problem", "temp", 10.0);
  mu = pin->GetOrAddReal("problem", "mu", 2.33);
  P_ext = pin->GetOrAddReal("problem", "P_ext", 0.0);

  // Calculate derived quantities
  cs_isothermal = std::sqrt(kB * temp / (mu * m_H)) / km_s_to_cm_s;  // km/s

  // Units
  L_unit = 1.0;  // Code length unit = pc
  V_unit = 1.0;  // Code velocity unit = km/s
  rho_unit = rho0;  // Code density unit = initial density
  P_unit = rho_unit * SQR(cs_isothermal);
  t_unit = L_unit * pc_to_cm / V_unit / km_s_to_cm_s;  // s

  // Gravitational constant in code units
  // G_phys = 6.674e-8 cm³/g/s²
  // G_code = G_phys * rho_unit * t_unit² / L_unit³
  G_phys = 6.674e-8;
  Real G_code = G_phys * rho_unit * SQR(t_unit) / CUBE(L_unit * pc_to_cm);

  // Set gravitational constant
  if (SELF_GRAVITY_ENABLED) {
    phydro->pf_grav->SetGConstant(G_code);
  }

  // Calculate perturbation wavenumber
  Real L_domain = mesh_size.x3max - mesh_size.x3min;
  k_pert = 2.0 * PI * n_waves / L_domain;

  // Enroll history outputs for monitoring
  // Allocate memory for user-defined history output
  // ...
}

// ============================================================================
// Problem Generator: Set initial conditions
// ============================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get coordinates
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Get position (in pc)
        Real x = pcoord->x1v(i);
        Real y = pcoord->x2v(j);
        Real z = pcoord->x3v(k);

        // Calculate cylindrical radius from filament axis
        Real r_cyl = std::sqrt(SQR(x) + SQR(y));

        // Density profile: Ostriker (1964) isothermal cylinder
        // Modified with edge tapering
        Real rho_norm;
        if (r_cyl < r_filament) {
          // Inside filament: smooth cylinder profile
          rho_norm = rho_edge + (rho_center - rho_edge) *
                     std::exp(-SQR(r_cyl / r_filament));
        } else {
          // Outside filament: exponential falloff
          rho_norm = rho_edge * std::exp(-(r_cyl - r_filament) / r_filament);
        }

        // Apply sinusoidal perturbation along filament axis (z)
        Real perturbation = 1.0 + pert_amp * std::cos(k_pert * z);
        Real rho = rho_norm * perturbation;

        // Set conserved variables
        phydro->u(IDN, k, j, i) = rho;

        // Zero velocity initially
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;

        // Energy (isothermal)
        phydro->u(IEN, k, j, i) = P_unit / (GAMMA_MINUS_1.0);

        // Magnetic field (if enabled)
        if (MAGNETIC_FIELDS_ENABLED) {
          // Uniform field along filament axis
          // B0 = sqrt(2 * P_ext / beta) for specified plasma beta
          Real B0 = 0.0;  // microG (convert to code units)
          Real B_code = B0 * 1.0e-6;  // Gauss to code units (needs proper scaling)

          pfield->b.x1f(k, j, i) = B_code;
          pfield->b.x2f(k, j, i) = B_code;
          pfield->b.x3f(k, j, i) = B_code;

          // Calculate magnetic pressure
          Real pb = SQR(B_code);

          // Update energy with magnetic pressure
          phydro->u(IEN, k, j, i) += 0.5 * pb;
        }
      }
    }
  }
}

// ============================================================================
// Analysis functions for core detection
// ============================================================================

// Called at each output to analyze filament structure
void MeshBlock::UserWorkInLoop(void) {
  // Nothing to do during main loop
  // Analysis can be done in Python during post-processing
}

// ============================================================================
// History output functions (optional)
// ============================================================================

// User-defined history output
// Real Mesh::UserHistoryOutput(int i, ...) {
//   // Return various diagnostics:
//   // - Maximum density
//   // - Minimum density
//   // - Total mass
//   // - Center of mass
//   // - Velocity dispersion
// }

// ============================================================================
// Notes:
// - This problem generator sets up a perturbed isothermal filament
// - The filament will fragment due to the Jeans instability
// - Core spacing can be measured from the final density field
// - Compare results to observed spacing: 0.213 ± 0.007 pc
// ============================================================================
