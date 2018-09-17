/*
 * @BEGIN LICENSE
 *
 * ownmp2 by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2018 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Psi4; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */
#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/vector.h" // <- needed to access SharedVector
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "psi4/libmints/factory.h"
#include "psi4/libmints/mintshelper.h"

// This allows us to be lazy in getting the spaces in DPD calls
#define ID(x) ints.DPD_ID(x)

namespace psi {
namespace ownmp2 {

extern "C" PSI_API int read_options(std::string name, Options &options) {
  if (name == "OWNMP2" || options.read_globals()) {
    /*- The amount of information printed
        to the output file -*/
    options.add_int("PRINT", 1);
    options.add_str("REFERENCE", "HF_DEFAULT");
    options.add_str("ROTATE_INT", "READ");
  }

  return true;
}

double ao_int_index(SharedMatrix TEI, int u, int v, int l, int s,
                    int nmo) {
  return TEI->get(u * nmo + v,
                  l * nmo + s); // Currently works only for C1 !!!
}

extern "C" PSI_API SharedWavefunction ownmp2(SharedWavefunction ref_wfn,
                                             Options &options) {
  /*
   * This plugin shows a simple way of obtaining MO basis integrals, directly
   * from a DPD buffer.  It is also possible to generate integrals with labels
   * (IWL) formatted files, but that's not shown here.
   */
  int print = options.get_int("PRINT");

  // Grab the global (default) PSIO object, for file I/O
  std::shared_ptr<PSIO> psio(_default_psio_lib_);

  // Have the reference (SCF) wavefunction, ref_wfn
  if (!ref_wfn)
    throw PSIEXCEPTION("SCF has not been run yet!");

  // Quickly check that there are no open shell orbitals here...
  int nirrep = ref_wfn->nirrep();

  std::vector<std::shared_ptr<MOSpace>> spaces;
  spaces.push_back(MOSpace::all);
  IntegralTransform ints(ref_wfn, spaces,
                         IntegralTransform::TransformationType::Restricted);
  ints.transform_tei(MOSpace::all, MOSpace::all, MOSpace::all, MOSpace::all);
  // Use the IntegralTransform object's DPD instance, for convenience
  dpd_set_default(ints.get_dpd_id());

  /*
   * Now, loop over the DPD buffer, printing the integrals
   */
  dpdbuf4 K;
  psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
  // To only process the permutationally unique integrals, change the
  // ID("[A,A]") to ID("[A>=A]+")
  global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[A,A]"), ID("[A,A]"),
                         ID("[A>=A]+"), ID("[A>=A]+"), 0, "MO Ints (AA|AA)");

  // 1. Read and store the one and two-electron integrals in chemist notation
  // (pq|rs) allocate a vector of size nmo^4
  size_t nmo = ref_wfn->nmo();
  size_t nmo4 = nmo * nmo * nmo * nmo;
  std::vector<double> mo_ints(nmo4, 0.0);

  // function to address a 2d tensor
  auto two_idx = [&](size_t p, size_t q, size_t dim) -> size_t {
    return (p * dim + q);
  };

  // Read and store one electron integrals

  // grab T and one electron V in AO basis
  SharedMatrix T = SharedMatrix(ref_wfn->matrix_factory()->create_matrix(
      PSIF_SO_T)); // initialize matrix of wfn's SO size
  SharedMatrix V_oe =
      SharedMatrix(ref_wfn->matrix_factory()->create_matrix(PSIF_SO_V));

  MintsHelper mints(ref_wfn); // Mintshelper read many ints (T, V, Prop, ...)
  T = mints.so_kinetic();     // read from mints
  V_oe = mints.so_potential();

  // Now build alpha/beta oei in AO basis
  SharedMatrix Ha = T->clone();
  SharedMatrix Hb = T->clone();

  Ha->add(V_oe);
  Hb->add(V_oe);

  // One can rotate orbs here in any potential/sets

  // Now transform Ha/Hb to MO basis
  // First grab C matrices
  SharedMatrix Ca = ref_wfn->Ca();
  SharedMatrix Cb = ref_wfn->Cb();

  // CaHaCa^T & CbHbCb^T
  Ha->transform(Ca);
  Hb->transform(Cb);

  // Move MO oeis to SO basis

  // function to address a four-dimensional tensor of dimension dim * dim * dim
  // * dim
  auto four_idx = [&](size_t p, size_t q, size_t r, size_t s,
                      size_t dim) -> size_t {
    size_t dim2 = dim * dim;
    size_t dim3 = dim2 * dim;
    return (p * dim3 + q * dim2 + r * dim + s);
  };

  // read two electron integrals
  if (options.get_str("ROTATE_INT") == "READ") {
	  outfile->Printf(
		  "\n Read integrals from DPD buffer. Note that integrals will NOT be rotated here! \n");
    for (int h = 0; h < nirrep; ++h) {
      global_dpd_->buf4_mat_irrep_init(&K, h);
      global_dpd_->buf4_mat_irrep_rd(&K, h);
      for (int pq = 0; pq < K.params->rowtot[h]; ++pq) {
        int p = K.params->roworb[h][pq][0];
        int q = K.params->roworb[h][pq][1];
        int psym = K.params->psym[p];
        int qsym = K.params->qsym[q];
        int prel = p - K.params->poff[psym];
        int qrel = q - K.params->qoff[qsym];
        for (int rs = 0; rs < K.params->coltot[h]; ++rs) {
          int r = K.params->colorb[h][rs][0];
          int s = K.params->colorb[h][rs][1];
          int rsym = K.params->rsym[r];
          int ssym = K.params->ssym[s];
          int rrel = r - K.params->roff[rsym];
          int srel = s - K.params->soff[ssym];
          // store the integrals
          mo_ints[four_idx(p, q, r, s, nmo)] = K.matrix[h][pq][rs];
        }
      }
      global_dpd_->buf4_mat_irrep_close(&K, h);
    }
  }
  global_dpd_->buf4_close(&K);
  psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);

  // Rotate integral according to wfn->S() if using IBO reference
  if (options.get_str("ROTATE_INT") == "ROTATE") {
	  outfile->Printf(
		  "\n Rotate integrals according to Utrans. \n");
    // Take a look at Utrans
    outfile->Printf(
        "\n Take a look at Utrans (reordered MO->IBO rotation matrix): \n");
    SharedMatrix Utrans = ref_wfn->S();
    Utrans->print();
    // Rotate mo ints with Utran
    // Rotate ao ints with C, whey should equal
  }

  SharedMatrix TEI = mints.ao_eri();
  if (options.get_str("ROTATE_INT") ==
      "CALCULATE") { // This part doesn't work for symmetry higher than C1!!
	  // <aa|aa> only, add another 2 Matrix for open shell in the future
	  outfile->Printf(
		  "\n Calculate integrals based on AOint and Ca. \n");
	  Ca->print();
    for (int p = 0; p < nmo; ++p) {
      for (int q = 0; q < nmo; ++q) {
        for (int r = 0; r < nmo; ++r) {
          for (int s = 0; s < nmo; ++s) {
            for (int u = 0; u < nmo; ++u) {
              for (int v = 0; v < nmo; ++v) {
                for (int l = 0; l < nmo; ++l) {
                  for (int sig = 0; sig < nmo; ++sig) {
                    mo_ints[four_idx(p, q, r, s, nmo)] +=
                        Ca->get(0, u, p) * Ca->get(0, v, q) * Ca->get(0, l, r) *
                        Ca->get(0, sig, s) *
                        ao_int_index(TEI, u, v, l, sig, nmo);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 2. Build the antisymmetrized two-electron integrals in a spin orbital basis
  size_t nso = 2 * nmo;

  // define the order of spin orbitals and store it as a vector of pairs
  // (orbital index,spin)
  Dimension noccpi = ref_wfn->doccpi();
  Dimension nmopi = ref_wfn->nmopi();

  size_t offset_here = 0;
  std::vector<std::tuple<size_t, int, int>> so_labels(nso);
  for (int h = 0; h < nirrep; ++h) {
    for (size_t n = 0; n < nmopi[h]; n++) {
      so_labels[2 * n + offset_here] = std::make_tuple(n, h, 0); // 0 = alpha
      so_labels[2 * n + 1 + offset_here] = std::make_tuple(n, h, 1); // 1 = beta
    }
    offset_here += nmopi[h];
  }

  // build oeis in SO basis
  // Assume C1 !!!
  size_t nso2 = nso * nso;
  std::vector<double> so_oeis(nso2, 0.0);
  for (size_t p = 0; p < nso; ++p) {
    size_t p_orb = std::get<0>(so_labels[p]);
    size_t p_spin = std::get<2>(so_labels[p]);
    for (size_t q = 0; q < nso; ++q) {
      size_t q_orb = std::get<0>(so_labels[q]);
      size_t q_spin = std::get<2>(so_labels[q]);
      if (p_spin == q_spin) {
        double value =
            p_spin ? Hb->get(0, p_orb, q_orb) : Ha->get(0, p_orb, q_orb);
        so_oeis[two_idx(p, q, nso)] = value;
      }
    }
  }

  // allocate the vector that will store the spin orbital integrals
  size_t nso4 = nso * nso * nso * nso;
  std::vector<double> so_ints(nso4, 0.0);

  // form the integrals <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
  for (size_t p = 0; p < nso; p++) {
    size_t p_orb = std::get<0>(so_labels[p]);
    int p_spin = std::get<2>(so_labels[p]);
    for (size_t q = 0; q < nso; q++) {
      size_t q_orb = std::get<0>(so_labels[q]);
      int q_spin = std::get<2>(so_labels[q]);
      for (size_t r = 0; r < nso; r++) {
        size_t r_orb = std::get<0>(so_labels[r]);
        int r_spin = std::get<2>(so_labels[r]);
        for (size_t s = 0; s < nso; s++) {
          size_t s_orb = std::get<0>(so_labels[s]);
          int s_spin = std::get<2>(so_labels[s]);

          double integral = 0.0;
          if ((p_spin == r_spin) and (q_spin == s_spin)) {
            integral += mo_ints[four_idx(p_orb, r_orb, q_orb, s_orb, nmo)];
          }
          if ((p_spin == s_spin) and (q_spin == r_spin)) {
            integral -= mo_ints[four_idx(p_orb, s_orb, q_orb, r_orb, nmo)];
          }
          so_ints[four_idx(p, q, r, s, nso)] = integral;
        }
      }
    }
  }

  // 3. Get the orbital energies from the reference wave function

  // sym awareness
  // SharedVector epsilon_a = ref_wfn->epsilon_a();
  // SharedVector epsilon_b = ref_wfn->epsilon_b();
  std::vector<double> epsilon(nso, 0.0);

  // Build Fock Matrix
  SharedMatrix F(std::make_shared<Matrix>("Fock", nso, nso));

  /*
  Dimension nzeropi = ref_wfn->nmopi();
  Dimension nsopi = ref_wfn->nmopi();
  for (int h = 0; h < nirrep; ++h) {
          nzeropi[h] = 0;
          nsopi[h] = nmopi[h]*2;
  }

  Slice alpha(nzeropi, nmopi);
  Slice beta(nmopi, nsopi);

  F->zero();
  F->set_block(alpha, alpha, ref_wfn->Fa());
  F->set_block(beta, beta, ref_wfn->Fb());
  */
  SharedMatrix Fa = ref_wfn->Fa();
  SharedMatrix Fb = ref_wfn->Fb();

  if (options.get_str("REFERENCE") == "HF_READF") {
    Fa->transform(Ca);
    Fb->Matrix::triplet(Cb, Fb, Cb, true, false, false);

    Fa->print();
    Fb->print();

    for (size_t p = 0; p < nso; ++p) {
      size_t p_orb = std::get<0>(so_labels[p]);
      size_t p_spin = std::get<2>(so_labels[p]);
      for (size_t q = 0; q < nso; ++q) {
        size_t q_orb = std::get<0>(so_labels[q]);
        size_t q_spin = std::get<2>(so_labels[q]);
        if (p_spin == q_spin) {
          if (p_spin == 0) {
            outfile->Printf("\n %d, %d", p, q);
            F->set(p, q, Fa->get(p / 2, q / 2));
          } else {
            outfile->Printf("\n %d, %d", p, q);
            F->set(p, q, Fb->get((p - 1) / 2, (q - 1) / 2));
          }
        } else {
          F->set(p, q, 0.0);
        }
      }
    }
  }

  if (options.get_str("REFERENCE") == "OTHER_FOCK") {
    // Fa->Matrix::triplet(Ca, Fa, Ca, true, false, false);
    // Fb->Matrix::triplet(Cb, Fb, Cb, true, false, false);
    Fa->print();

    for (size_t p = 0; p < nso; ++p) {
      size_t p_orb = std::get<0>(so_labels[p]);
      size_t p_spin = std::get<2>(so_labels[p]);
      for (size_t q = 0; q < nso; ++q) {
        size_t q_orb = std::get<0>(so_labels[q]);
        size_t q_spin = std::get<2>(so_labels[q]);
        if (p_spin == q_spin) {
          if (p_spin == 0) {
            outfile->Printf("\n %d, %d", p, q);
            F->set(p, q, Fa->get(p / 2, q / 2));
          } else {
            outfile->Printf("\n %d, %d", p, q);
            F->set(p, q, Fb->get((p - 1) / 2, (q - 1) / 2));
          }
        } else {
          F->set(p, q, 0.0);
        }
      }
    }
  }

  std::vector<size_t> O; // occupied mos
  std::vector<size_t> V;

  size_t offset = 0;
  for (int h = 0; h < nirrep; ++h) {
    size_t nocc_h = 2 * noccpi[h];
    size_t nmo_h = 2 * nmopi[h];
    for (size_t i = 0; i < nocc_h; ++i) {
      //	    epsilon[offset + i] = epsilon_a->get(h,i/2);
      O.push_back(offset + i);
    }
    for (size_t a = nocc_h; a < nmo_h; ++a) {
      //      epsilon[offset + a] = epsilon_a->get(h,a/2);
      V.push_back(offset + a);
    }
    offset += nmo_h;
  }

  // for (auto& o : O) {
  //	outfile->Printf("\n %d", o);
  //}

  // for (auto& v : V) {
  //	outfile->Printf("\n %d", v);
  //}
  //
  //    for(size_t p = 0; p < nso; p++) {
  //	outfile->Printf("\n %d  %8.4f", p, epsilon[p]);
  //    }

  //   for (size_t p = 0; p < nso; p++) {
  //       size_t p_orb = so_labels[p].first;
  //       size_t p_spin = so_labels[p].first;
  //       if (p_spin == 0){
  //           epsilon[p] = epsilon_a->get(p_orb);
  //       }else{
  //           epsilon[p] = epsilon_b->get(p_orb);
  //       }
  //   }
  //
  // 4. Form list of occupied and virtual orbitals
  int na = ref_wfn->nalpha();
  int nb = ref_wfn->nbeta();
  int nocc = na + nb;
  // ASSUMES RESTRICTED ORBITALS

  //   std::vector<size_t> O; //occupied mos
  //   std::vector<size_t> V;
  //
  //   for (int i = 0; i < nocc; i++) {
  //       O.push_back(i);
  //   }
  //   for (int a = nocc; a < nso; a++) {
  //       V.push_back(a);
  //   }
  //

  if (options.get_str("REFERENCE") == "HF_DEFAULT") {
    for (size_t p = 0; p < nso; ++p) {
      size_t p_orb = std::get<0>(so_labels[p]);
      size_t p_spin = std::get<2>(so_labels[p]);
      for (size_t q = 0; q < nso; ++q) {
        size_t q_orb = std::get<0>(so_labels[q]);
        size_t q_spin = std::get<2>(so_labels[q]);
        double value = 0.0;
        value += so_oeis[two_idx(p, q, nso)];
        for (size_t r : O) {
          value += so_ints[four_idx(p, r, q, r, nso)];
        }
        F->set(p, q, value);
      }
    }
  }

  F->print();
  for (size_t p = 0; p < nso; ++p) {
    epsilon[p] = F->get(p, p);
  }

  double mp2_energy = 0.0;
  for (int i : O) {
    for (int j : O) {
      for (int a : V) {
        for (int b : V) {
          double Vijab = so_ints[four_idx(i, j, a, b, nso)];
          double Dijab = epsilon[i] + epsilon[j] - epsilon[a] - epsilon[b];
          mp2_energy += 0.25 * Vijab * Vijab / Dijab;
        }
      }
    }
  }

  double rhf_energy = ref_wfn->reference_energy();

  outfile->Printf("\n\n    ==> Spin orbital MP2 energy <==\n");
  outfile->Printf("    RHF total energy         %20.12f\n", rhf_energy);
  outfile->Printf("    MP2 correlation energy   %20.12f\n", mp2_energy);
  outfile->Printf("    MP2 Total Energy         %20.12f\n",
                  rhf_energy + mp2_energy);

  Process::environment.globals["CURRENT ENERGY"] = rhf_energy + mp2_energy;

  return ref_wfn;
}

} // namespace ownmp2
} // namespace psi
