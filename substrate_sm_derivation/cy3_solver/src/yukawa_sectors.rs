// LEGACY-SUPERSEDED-BY-ROUTE34: this module hardcodes the SU(5)
// 10/5bar/5bar split as bundle-moduli slices [0,1,2]/[3,4,5]/[6,7,8].
// Real heterotic compactification has no such hardcoded slice; the
// sectors are determined dynamically by the E_8 -> E_6 x SU(3) Wilson
// line and the cohomology indices of the chosen monad bundle (Anderson-
// Constantin-Lukas-Palti 2017). Superseded by:
//   * route34::yukawa_sectors_real (E_8 -> E_6 x SU(3) decomposition with
//                                   sector-to-cohomology assignment
//                                   computed from the Wilson-line element
//                                   and the bundle Chern data, not
//                                   hardcoded slot indices)
// Do not modify in place; add new sector logic to the route34 module.
//
//! Separated up-, down-, and lepton-sector Yukawa tensors with CKM
//! rotation extraction.
//!
//! In SU(5) GUT framing the matter lives in three reps:
//!   - 10̄ (Q_L, u_R^c, e_R^c) -- generates Y_u via 10·10·5_H
//!   - 5  (d_R^c, L_L)        -- generates Y_d via 10·5̄·5̄_H
//!   - 5̄  (anti-Q_L, anti-d_R, ν_L, e_L) -- generates Y_e via 10·5̄·5̄_H
//!
//! The down-quark and lepton sectors share a Yukawa structure at the
//! GUT scale (Y_d = Y_e^T in minimal SU(5)); RG running breaks the
//! degeneracy at low energy. We model the GUT-scale Yukawas as three
//! distinct tensors derived from disjoint slices of the bundle moduli
//! and zero-mode centers.
//!
//! The 9 charged-fermion masses come from three 3x3 sub-matrices:
//!   M_u = (v_u/sqrt(2)) Y_u (3 generations of u, c, t)
//!   M_d = (v_d/sqrt(2)) Y_d (d, s, b)
//!   M_e = (v/sqrt(2)) Y_e   (e, mu, tau)
//!
//! Diagonalisation via the singular-value decomposition gives the
//! mass eigenvalues plus the left-handed rotations V_uL, V_dL.
//! The CKM matrix is V_CKM = V_uL · V_dL†.


/// Build a 3x3 Yukawa sub-matrix from a Yukawa tensor's restriction
/// onto a 3-mode sector. The sector indexing maps the 3 generations
/// onto specific zero-mode indices in the n_modes mode space.
///
/// Convention: with n_modes >= 9, sectors are:
///   up-type:    indices [0, 1, 2]
///   down-type:  indices [3, 4, 5]
///   lepton:     indices [6, 7, 8]
///
/// The Yukawa contraction is M_ij = sum_k Y_ijk * h_k where h_k is
/// uniform 1/sqrt(n_modes), restricted to the 3 sector indices.
pub fn extract_sector_3x3(
    yukawa: &[f64],
    n_modes: usize,
    sector: [usize; 3],
) -> [f64; 9] {
    let mut m = [0.0f64; 9];
    let h_val = 1.0 / (n_modes as f64).sqrt();
    for (i_local, &i) in sector.iter().enumerate() {
        for (j_local, &j) in sector.iter().enumerate() {
            let mut s = 0.0;
            for k in 0..n_modes {
                s += yukawa[i * n_modes * n_modes + j * n_modes + k] * h_val;
            }
            m[i_local * 3 + j_local] = s;
        }
    }
    m
}

/// Singular-value decomposition for a 3x3 real matrix.
///
/// Returns (U, sigma, V^T) where M = U diag(sigma) V^T.
/// sigma is sorted descending. Used to extract:
///   sigma[0..3]  -- the three mass eigenvalues
///   U            -- left rotation V_{u/d/e L}
///   V^T          -- right rotation V_{u/d/e R}
///
/// Algorithm: form M^T M (3x3 symmetric PSD); diagonalise via Jacobi
/// rotations to get eigenvalues and V; then sigma = sqrt(eigenvalues)
/// and U = M V / sigma (column-by-column).
pub fn svd_3x3(m: &[f64; 9]) -> ([f64; 9], [f64; 3], [f64; 9]) {
    // Step 1: Form M^T M.
    let mut mtm = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += m[k * 3 + i] * m[k * 3 + j];
            }
            mtm[i * 3 + j] = s;
        }
    }
    // Step 2: Jacobi diagonalisation of mtm.
    let (eigvals, v) = jacobi_eigen_3x3(&mtm);
    // eigvals[0..3] are the eigenvalues of M^T M; sigma = sqrt(|·|).
    let mut sigma_unsorted = [0.0f64; 3];
    for k in 0..3 {
        sigma_unsorted[k] = eigvals[k].abs().sqrt();
    }
    // **Sort ASCENDING** (lightest generation first) so V_uL[:, 0]
    // corresponds to the up-quark / electron / down-quark direction
    // (lightest), and CKM[0, j] = ⟨u|d_j⟩ matches PDG row = up
    // generation indexing. Previous version sorted descending, which
    // gave V_uL[:, 0] = top direction and made the CKM labels wrong
    // (|V_us| was actually |V_ts| at the entry level — the bug only
    // numerically vanished because both V_uL and V_dL got the same
    // permutation that cancelled in V_uL · V_dL†, but the convention
    // was inconsistent with PDG).
    let mut idx = [0_usize, 1, 2];
    idx.sort_by(|a, b| {
        sigma_unsorted[*a]
            .partial_cmp(&sigma_unsorted[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sigma = [0.0f64; 3];
    let mut v_sorted = [0.0f64; 9];
    for (new_k, &old_k) in idx.iter().enumerate() {
        sigma[new_k] = sigma_unsorted[old_k];
        for i in 0..3 {
            v_sorted[i * 3 + new_k] = v[i * 3 + old_k];
        }
    }
    // Step 3: U = M V / sigma (column-by-column).
    let mut u = [0.0f64; 9];
    for j in 0..3 {
        if sigma[j] < 1e-14 {
            // Degenerate column: set to standard basis vector to maintain
            // unitarity. Subsequent Gram-Schmidt step rescues us.
            for i in 0..3 {
                u[i * 3 + j] = if i == j { 1.0 } else { 0.0 };
            }
            continue;
        }
        for i in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += m[i * 3 + k] * v_sorted[k * 3 + j];
            }
            u[i * 3 + j] = s / sigma[j];
        }
    }
    // Re-orthogonalise U via modified Gram-Schmidt for numerical safety.
    for j in 0..3 {
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..3 {
                dot += u[i * 3 + k] * u[i * 3 + j];
            }
            for i in 0..3 {
                u[i * 3 + j] -= dot * u[i * 3 + k];
            }
        }
        let mut nrm_sq = 0.0;
        for i in 0..3 {
            nrm_sq += u[i * 3 + j] * u[i * 3 + j];
        }
        let nrm = nrm_sq.sqrt();
        if nrm > 1e-14 {
            for i in 0..3 {
                u[i * 3 + j] /= nrm;
            }
        }
    }
    // Build V^T from V_sorted.
    let mut vt = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            vt[i * 3 + j] = v_sorted[j * 3 + i];
        }
    }
    (u, sigma, vt)
}

/// Jacobi eigendecomposition for a 3x3 symmetric matrix. Returns
/// (eigenvalues, eigenvectors-as-columns-of-V).
fn jacobi_eigen_3x3(a: &[f64; 9]) -> ([f64; 3], [f64; 9]) {
    let mut a = *a;
    let mut v = [
        1.0_f64, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    for _ in 0..50 {
        // Find largest off-diagonal entry.
        let mut max_off = 0.0;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let aij = a[i * 3 + j].abs();
                if aij > max_off {
                    max_off = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p * 3 + p];
        let aqq = a[q * 3 + q];
        let apq = a[p * 3 + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Apply rotation to a.
        let app_new = app - t * apq;
        let aqq_new = aqq + t * apq;
        a[p * 3 + p] = app_new;
        a[q * 3 + q] = aqq_new;
        a[p * 3 + q] = 0.0;
        a[q * 3 + p] = 0.0;
        for i in 0..3 {
            if i != p && i != q {
                let aip = a[i * 3 + p];
                let aiq = a[i * 3 + q];
                a[i * 3 + p] = c * aip - s * aiq;
                a[p * 3 + i] = a[i * 3 + p];
                a[i * 3 + q] = s * aip + c * aiq;
                a[q * 3 + i] = a[i * 3 + q];
            }
        }
        // Update V.
        for i in 0..3 {
            let vip = v[i * 3 + p];
            let viq = v[i * 3 + q];
            v[i * 3 + p] = c * vip - s * viq;
            v[i * 3 + q] = s * vip + c * viq;
        }
    }
    let eigvals = [a[0], a[4], a[8]];
    (eigvals, v)
}

/// Compute V_CKM = V_uL · V_dL^† from up- and down-sector left-rotations.
pub fn compute_ckm(v_ul: &[f64; 9], v_dl: &[f64; 9]) -> [f64; 9] {
    // V_dL^† for real matrices is V_dL^T (transpose).
    let mut ckm = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += v_ul[i * 3 + k] * v_dl[j * 3 + k];
            }
            ckm[i * 3 + j] = s;
        }
    }
    ckm
}

/// Extract the three Wolfenstein-style CKM angles (|V_us|, |V_cb|,
/// |V_ub|) from the CKM matrix. PDG values:
///   |V_us| ~ 0.225  (Cabibbo angle)
///   |V_cb| ~ 0.041
///   |V_ub| ~ 0.00357
pub fn ckm_magnitudes(ckm: &[f64; 9]) -> [f64; 3] {
    [
        ckm[0 * 3 + 1].abs(), // V_us
        ckm[1 * 3 + 2].abs(), // V_cb
        ckm[0 * 3 + 2].abs(), // V_ub
    ]
}

/// CKM mixing-angle loss: log-ratio mismatch against PDG values.
pub fn ckm_mixing_loss(ckm: &[f64; 9]) -> f64 {
    let pred = ckm_magnitudes(ckm);
    let measured = [0.22500_f64, 0.04200, 0.00357];
    let mut total = 0.0;
    for k in 0..3 {
        let p = pred[k].max(1e-12);
        let m = measured[k];
        let log_pred = p.ln();
        let log_meas = m.ln();
        let rel = (log_pred - log_meas) / log_meas.abs().max(1.0);
        total += rel * rel;
    }
    total / 3.0
}

/// Aggregate Yukawa-sector loss: sum of mass-spectrum loss for each
/// of the three sectors plus the CKM mixing loss.
#[derive(Debug, Clone, Copy, Default)]
pub struct YukawaSectorLoss {
    pub up_quark_masses: f64,    // (m_u, m_c, m_t) vs PDG
    pub down_quark_masses: f64,  // (m_d, m_s, m_b)
    pub lepton_masses: f64,      // (m_e, m_mu, m_tau)
    pub ckm_mixing: f64,         // |V_us|, |V_cb|, |V_ub|
}

impl YukawaSectorLoss {
    pub fn total(&self) -> f64 {
        self.up_quark_masses + self.down_quark_masses + self.lepton_masses + self.ckm_mixing
    }
}

/// Compute the full Yukawa-sector loss given a Yukawa tensor with
/// n_modes >= 9.
pub fn yukawa_sector_loss(yukawa: &[f64], n_modes: usize) -> YukawaSectorLoss {
    if n_modes < 9 || yukawa.len() < n_modes * n_modes * n_modes {
        return YukawaSectorLoss::default();
    }
    let m_u = extract_sector_3x3(yukawa, n_modes, [0, 1, 2]);
    let m_d = extract_sector_3x3(yukawa, n_modes, [3, 4, 5]);
    let m_e = extract_sector_3x3(yukawa, n_modes, [6, 7, 8]);
    let (v_ul, sigma_u, _) = svd_3x3(&m_u);
    let (v_dl, sigma_d, _) = svd_3x3(&m_d);
    let (_, sigma_e, _) = svd_3x3(&m_e);

    // PDG masses (GeV). Up-type: m_u, m_c, m_t. Down-type: m_d, m_s, m_b.
    // Lepton: m_e, m_mu, m_tau.
    let pdg_u = [0.00216_f64, 1.273, 172.76];
    let pdg_d = [0.00467_f64, 0.0934, 4.18];
    let pdg_e = [0.000511_f64, 0.10566, 1.7768];

    // Yukawa scale: y_i * v / sqrt(2) where v ~ 246 GeV.
    let v_eff = 246.0_f64 / std::f64::consts::SQRT_2;
    let log_loss = |sigma: [f64; 3], pdg: [f64; 3]| -> f64 {
        // SVD now returns σ ascending (PDG-aligned), so no resort needed.
        let mut total = 0.0;
        for k in 0..3 {
            let predicted = (sigma[k] * v_eff).max(1e-12);
            let log_pred = predicted.ln();
            let log_meas = pdg[k].ln();
            let rel = (log_pred - log_meas) / log_meas.abs().max(1.0);
            total += rel * rel;
        }
        total / 3.0
    };

    let ckm = compute_ckm(&v_ul, &v_dl);
    YukawaSectorLoss {
        up_quark_masses: log_loss(sigma_u, pdg_u),
        down_quark_masses: log_loss(sigma_d, pdg_d),
        lepton_masses: log_loss(sigma_e, pdg_e),
        ckm_mixing: ckm_mixing_loss(&ckm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_recovers_diagonal_3x3() {
        // M = diag(3, 2, 1); SVD with ASCENDING sort recovers
        // sigma = (1, 2, 3) (lightest first; PDG-aligned convention).
        let m = [
            3.0_f64, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let (_u, sigma, _vt) = svd_3x3(&m);
        assert!(
            (sigma[0] - 1.0).abs() < 1e-10
                && (sigma[1] - 2.0).abs() < 1e-10
                && (sigma[2] - 3.0).abs() < 1e-10,
            "sigma = {:?} (expected ascending [1, 2, 3])",
            sigma
        );
    }

    #[test]
    fn svd_orthogonal_u_v() {
        // For a generic 3x3 matrix, U and V^T must be orthogonal.
        let m = [
            1.0_f64, 0.5, 0.2,
            0.3, 1.5, 0.1,
            0.4, 0.2, 2.0,
        ];
        let (u, _sigma, vt) = svd_3x3(&m);
        // Check U^T U = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += u[k * 3 + i] * u[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-9,
                    "U^T U[{i},{j}] = {s} (expected {expected})"
                );
            }
        }
        // Check V^T (V^T)^T = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += vt[i * 3 + k] * vt[j * 3 + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-9,
                    "V V^T[{i},{j}] = {s} (expected {expected})"
                );
            }
        }
    }

    #[test]
    fn svd_reconstructs_matrix() {
        let m = [
            1.0_f64, 0.5, 0.2,
            0.3, 1.5, 0.1,
            0.4, 0.2, 2.0,
        ];
        let (u, sigma, vt) = svd_3x3(&m);
        // Reconstruct M = U diag(sigma) V^T.
        let mut m_rec = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += u[i * 3 + k] * sigma[k] * vt[k * 3 + j];
                }
                m_rec[i * 3 + j] = s;
            }
        }
        for k in 0..9 {
            assert!(
                (m_rec[k] - m[k]).abs() < 1e-8,
                "reconstruction failed at {k}: got {} expected {}",
                m_rec[k],
                m[k]
            );
        }
    }

    #[test]
    fn ckm_is_unitary_for_orthogonal_inputs() {
        // V_uL = V_dL = I gives CKM = I.
        let v_ul = [
            1.0_f64, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let v_dl = v_ul;
        let ckm = compute_ckm(&v_ul, &v_dl);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ckm[i * 3 + j] - expected).abs() < 1e-12);
            }
        }
    }

    /// REGRESSION (Bug 7): CKM matrix column ordering.
    ///
    /// SVD returns σ in descending order (heaviest first), so V_uL
    /// columns are in (t, c, u) order. But CKM index convention is
    /// row=up generation ascending, col=down generation ascending:
    /// |V_us| = ckm[0, 1] should map (u → s), not (t → s) or (t → b).
    ///
    /// Test: build hierarchical M_u with σ = [173, 1.27, 0.002] and
    /// M_d = identity. Then V_uL after SVD is in heaviest-first order.
    /// V_dL = identity. CKM = V_uL · V_dL† = V_uL.
    /// The (0, 1) entry of V_uL is the (top, charm-direction) coupling
    /// — interpreting it as |V_us| would be WRONG by ~5 orders of
    /// magnitude (since the actual |V_us| of identity is 0).
    ///
    /// Correct behaviour: permute U columns to ASCENDING σ order, so
    /// V_uL[:, 0] is the up direction (lightest), [:, 1] is charm,
    /// [:, 2] is top.
    #[test]
    fn ckm_uses_ascending_generation_order() {
        // Hierarchical M_u in standard (u, c, t) flavor order with
        // ascending masses. Descending-σ SVD permutes V_uL columns to
        // (t, c, u) order, breaking the row-0 = up convention.
        //
        // Build M_d with a Cabibbo rotation in the (d, s) block:
        // M_d = R_C(θ_C) · diag([0.005, 0.093, 4.18]) · R_C^T.
        // No rotation in M_u (M_u_diag).
        //
        // Expected: |V_us| = sin(θ_C) ≈ 0.225, |V_cb| ≈ 0, |V_ub| ≈ 0.
        // With BUG: row 0 of CKM corresponds to the *top* direction,
        //   so ckm[0, 1] = |V_t,?| ≠ |V_us|. The numerical value can
        //   be anywhere from 0 to ≈ 1 depending on the permutation.
        let theta_c = 0.225_f64;
        let c = theta_c.cos();
        let s = theta_c.sin();
        // M_u = diag(m_u, m_c, m_t) in flavor (u, c, t) basis.
        let m_u = [
            0.00216_f64, 0.0, 0.0,
            0.0, 1.273, 0.0,
            0.0, 0.0, 173.0,
        ];
        // R_C in (d, s) block:
        // R_C = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        // M_d = R_C · diag(m_d, m_s, m_b) · R_C^T
        // Compute manually:
        let dm = [0.00467_f64, 0.0934, 4.18];
        let m_d = [
            c * c * dm[0] + s * s * dm[1],
            c * s * (dm[0] - dm[1]),
            0.0,
            c * s * (dm[0] - dm[1]),
            s * s * dm[0] + c * c * dm[1],
            0.0,
            0.0,
            0.0,
            dm[2],
        ];
        let (v_ul, _sigma_u, _) = svd_3x3(&m_u);
        let (v_dl, _sigma_d, _) = svd_3x3(&m_d);
        let ckm = compute_ckm(&v_ul, &v_dl);
        let mag = ckm_magnitudes(&ckm);
        eprintln!("|V_us| = {:.4}, expected sin(θ_C) ≈ {:.4}", mag[0], s);
        eprintln!("|V_cb| = {:.4}, expected ≈ 0", mag[1]);
        eprintln!("|V_ub| = {:.4}, expected ≈ 0", mag[2]);
        assert!(
            (mag[0] - s).abs() < 0.01,
            "|V_us| = {} should equal sin(θ_C) = {}; ordering bug?",
            mag[0],
            s
        );
        assert!(
            mag[1] < 0.01,
            "|V_cb| = {} should be ~0; ordering bug?",
            mag[1]
        );
        assert!(
            mag[2] < 0.01,
            "|V_ub| = {} should be ~0; ordering bug?",
            mag[2]
        );
    }

    #[test]
    fn ckm_off_diagonal_when_rotations_differ() {
        // Up-rotation = identity; down-rotation = small mixing.
        // CKM = V_uL V_dL^T = V_dL^T (transpose of V_dL).
        let v_ul = [
            1.0_f64, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        // 2D rotation by Cabibbo-like angle in (1,2) block.
        let theta = 0.225_f64;
        let c = theta.cos();
        let s = theta.sin();
        let v_dl = [
            c, s, 0.0,
            -s, c, 0.0,
            0.0, 0.0, 1.0,
        ];
        let ckm = compute_ckm(&v_ul, &v_dl);
        // CKM = V_uL V_dL^T = V_dL^T. So |V_us| = |CKM[0][1]| = |s| = 0.223.
        let mag = ckm_magnitudes(&ckm);
        assert!(
            (mag[0] - s.abs()).abs() < 1e-10,
            "|V_us| = {}, expected {}",
            mag[0],
            s.abs()
        );
    }
}
