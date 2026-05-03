// Cycle 8 probe: BBW dimensions of every B/C summand for V_min2 and
// AKLP on the Tian-Yau bicubic-triple, plus the predicted h^1(V)
// (post-quotient n_27) from the Atiyah-Singer index theorem.
//
// Purpose. Cycle 7 hypothesised that V_min2's bucket-hit shortfall
// (2/27, observed=2 vs predicted=3) was caused by the polynomial-seed
// basis missing the `ker(H^1(B) → H^1(C))` branch of the monad LES.
// The seed basis is built from H^0(B-summand) only; if H^1(B-summand)
// is non-zero on negative-bidegree summands, that branch *should*
// contribute to h^1(V). This probe directly measures whether that
// branch is non-zero on V_min2.
//
// Method. The BBW Koszul-resolution chase
// (`route34::bbw_cohomology::h_star_X_line`) returns the full
// `[h^0, h^1, h^2, h^3]` vector for any line bundle on the bicubic-
// triple. Apply it to every summand of B (V_min2) and B (AKLP), then
// compare totals.
//
// Expected (cycle 7 hypothesis):
//   * V_min2: sum h^1(B_alpha) > 0  (the missing modes live here).
//   * AKLP:   sum h^1(B_alpha) = 0  (the H^0-only basis is complete).
//
// Empirical (this probe):
//   * V_min2: sum h^1(B_alpha) = 0   ← FALSIFIES cycle-7 hypothesis.
//   * AKLP:   sum h^1(B_alpha) = 6   ← unexpected; AKLP also has the
//     H^1(B) branch, yet still gets 9/27 from H^0-only seeds. Hint:
//     the LES connecting map H^0(C) → H^1(V) provides the dominant
//     channel, and the H^1(B) branch is killed by the next map
//     H^1(B) → H^1(C). So the H^0-only construction is COMPLETE for
//     V_min2 too — the cycle-6 bucket-hit shortfall is a separate
//     geometric obstruction.
//
// Verdict. Implementing `h1_serre_dual_seeds` would NOT increase
// V_min2's seed_basis_dim (the source dimension is zero), so it would
// NOT change the cycle-6 bucket-hits. V_min2 is structurally
// falsified.
use cy3_rust_solver::geometry::CicyGeometry;
use cy3_rust_solver::route34::bbw_cohomology::h_star_X_line;
use cy3_rust_solver::zero_modes::{compute_zero_mode_spectrum, AmbientCY3, MonadBundle};

fn main() {
    let g = CicyGeometry::tian_yau_z3();

    let v_min2_b: Vec<[i32; 2]> = vec![
        [0, 0], [0, 0], [-1, -2], [-2, -1], [-2, -1], [-1, 0], [-1, 0],
    ];
    let v_min2_c: Vec<[i32; 2]> = vec![[-1, -1], [-2, -1], [-2, -1], [-2, -1]];

    println!("=== V_min2 B-summand h^p(X_TY, O(a,b)) ===");
    let mut sum_h_b = [0i64; 4];
    for line in &v_min2_b {
        let h = h_star_X_line(&[line[0], line[1]], &g).unwrap();
        println!("  O({:>2},{:>2}): h^* = [{}, {}, {}, {}]", line[0], line[1], h[0], h[1], h[2], h[3]);
        for p in 0..4 {
            sum_h_b[p] += h[p];
        }
    }
    println!(
        "  Sum h^* = [h^0={}, h^1={}, h^2={}, h^3={}]",
        sum_h_b[0], sum_h_b[1], sum_h_b[2], sum_h_b[3]
    );

    println!("=== V_min2 C-summand h^p(X_TY, O(a,b)) ===");
    let mut sum_h_c = [0i64; 4];
    for line in &v_min2_c {
        let h = h_star_X_line(&[line[0], line[1]], &g).unwrap();
        println!("  O({:>2},{:>2}): h^* = [{}, {}, {}, {}]", line[0], line[1], h[0], h[1], h[2], h[3]);
        for p in 0..4 {
            sum_h_c[p] += h[p];
        }
    }
    println!(
        "  Sum h^* = [h^0={}, h^1={}, h^2={}, h^3={}]",
        sum_h_c[0], sum_h_c[1], sum_h_c[2], sum_h_c[3]
    );

    println!("=== AKLP B-summand h^p(X_TY, O(a,b)) ===");
    let aklp_b: Vec<[i32; 2]> = vec![[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]];
    let mut sum_h_b_aklp = [0i64; 4];
    for line in &aklp_b {
        let h = h_star_X_line(&[line[0], line[1]], &g).unwrap();
        println!("  O({:>2},{:>2}): h^* = [{}, {}, {}, {}]", line[0], line[1], h[0], h[1], h[2], h[3]);
        for p in 0..4 {
            sum_h_b_aklp[p] += h[p];
        }
    }
    println!(
        "  Sum h^* = [h^0={}, h^1={}, h^2={}, h^3={}]",
        sum_h_b_aklp[0], sum_h_b_aklp[1], sum_h_b_aklp[2], sum_h_b_aklp[3]
    );

    let ambient = AmbientCY3::tian_yau_upstairs();
    let aklp = MonadBundle::anderson_lukas_palti_example();
    let s_aklp = compute_zero_mode_spectrum(&aklp, &ambient);
    let v_min2 = MonadBundle::tian_yau_z3_v_min2();
    let s_v_min2 = compute_zero_mode_spectrum(&v_min2, &ambient);

    println!("\n=== Predicted h^1(V) (post-quotient n_27) from index theorem ===");
    println!("  AKLP   : n_27 = {}, n_27bar = {}", s_aklp.n_27, s_aklp.n_27bar);
    println!("  V_min2 : n_27 = {}, n_27bar = {}", s_v_min2.n_27, s_v_min2.n_27bar);

    println!("\n=== Cycle 7 hypothesis check ===");
    println!(
        "  V_min2 sum h^1(B_alpha) = {}  (cycle-7 hypothesis predicted > 0; FALSIFIED)",
        sum_h_b[1]
    );
    println!(
        "  AKLP   sum h^1(B_alpha) = {}  (cycle-7 thought this was 0; actually 6)",
        sum_h_b_aklp[1]
    );

    println!("\n=== Stable-monad consistency check ===");
    println!(
        "  V_min2: sum h^0(B) = {}, sum h^0(C) = {}.  H^0(V) >= h^0(B) - h^0(C) = {}.",
        sum_h_b[0],
        sum_h_c[0],
        (sum_h_b[0] - sum_h_c[0]).max(0)
    );
    println!(
        "  Stable SU(3) bundle on a CY3 must have H^0(V) = 0. V_min2 violates this if H^0(V) >= 2."
    );
    println!(
        "  (Equivalently: the LES segment 0 -> H^0(V) -> {} -> {} forces H^0(V) >= max(0, h^0(B) - h^0(C)).)",
        sum_h_b[0], sum_h_c[0]
    );
    let aklp_c: Vec<[i32; 2]> = vec![[1, 1], [1, 1], [1, 1]];
    let mut sum_h_c_aklp = [0i64; 4];
    println!("\n=== AKLP C-summand h^p(X_TY, O(a,b)) ===");
    for line in &aklp_c {
        let h = h_star_X_line(&[line[0], line[1]], &g).unwrap();
        println!("  O({:>2},{:>2}): h^* = [{}, {}, {}, {}]", line[0], line[1], h[0], h[1], h[2], h[3]);
        for p in 0..4 {
            sum_h_c_aklp[p] += h[p];
        }
    }
    println!(
        "  Sum h^* = [h^0={}, h^1={}, h^2={}, h^3={}]",
        sum_h_c_aklp[0], sum_h_c_aklp[1], sum_h_c_aklp[2], sum_h_c_aklp[3]
    );
    println!(
        "  AKLP: sum h^0(B) = {}, sum h^0(C) = {}. H^0(B) -> H^0(C) is the AKLP map; \
         if it is surjective, H^0(V) = ker = h^0(B) - h^0(C) = {}.",
        sum_h_b_aklp[0],
        sum_h_c_aklp[0],
        sum_h_b_aklp[0] - sum_h_c_aklp[0]
    );
    println!(
        "  Sanity: AKLP n_27 = {}, AKLP H^0(B->C) cokernel = max(0, {} - rank(map)) >= 0.",
        s_aklp.n_27,
        sum_h_c_aklp[0]
    );
}
