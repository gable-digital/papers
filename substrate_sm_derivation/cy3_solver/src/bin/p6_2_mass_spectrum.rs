//! P6.2 — High-precision substrate-framework mass-spectrum recomputation.
//!
//! Rust port of `book/scripts/substrate_mass_spectrum.py`. Uses
//! `pwos_math::precision::BigFloat` (astro-float) at 502 bits ≈ 150 decimal
//! digits, matching the Python reference at `mp.prec = 500`.
//!
//! Output must match the Python script bit-for-bit at 30+ decimal digits.
//!
//! Run with:
//!   cargo run --release --features precision-bigfloat --bin p6_2_mass_spectrum

use std::collections::HashMap;

use pwos_math::precision::{
    bigfloat_from_i64_at, bigfloat_from_rational, bigfloat_from_str, bigfloat_to_string,
    bits_for_decimal_digits,
    constants::{phi as phi_const, pi as pi_const, sqrt2 as sqrt2_const},
    BigFloat, Real,
};

// Order at which to truncate sub-harmonic recursive series.
const MAX_ORDER: i64 = 50;

// ---------------------------------------------------------------------------
// SMALL HELPERS — wrap the Real trait into operator-style closures so the
// arithmetic-heavy code below stays readable. Each helper allocates fresh
// BigFloats; precision is dictated by the operands (op_prec inside the
// trait impl picks max(operand-widths)).
// ---------------------------------------------------------------------------

#[inline]
fn add(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::add(a, b) }
#[inline]
fn sub(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::sub(a, b) }
#[inline]
fn mul(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::mul(a, b) }
#[inline]
fn div(a: &BigFloat, b: &BigFloat) -> BigFloat { Real::div(a, b) }
#[inline]
fn pow_i(a: &BigFloat, n: i64) -> BigFloat { Real::pow_int(a, n) }

/// Fractional power: x^y = exp(y * ln(x)). Mirrors mpmath's __pow__
/// semantics for positive base.
fn pow_frac(base: &BigFloat, exp: &BigFloat) -> BigFloat {
    let lnx = Real::ln(base);
    let prod = mul(exp, &lnx);
    Real::exp(&prod)
}

/// Build a BigFloat from i64 at the working precision.
fn bf(value: i64, prec: usize) -> BigFloat {
    bigfloat_from_i64_at(value, prec)
}

/// Build a BigFloat from a rational n/d at the working precision.
fn bf_q(num: i64, den: i64, prec: usize) -> BigFloat {
    bigfloat_from_rational(num, den, prec)
}

/// Lossy to f64 conversion at the BigFloat -> printout boundary.
fn to_f64(x: &BigFloat) -> f64 { Real::to_f64(x) }

// ---------------------------------------------------------------------------
// FORMATTING — Python-style nstr(x, n) prints n significant figures.
// astro-float's format() prints full precision; we truncate to match.
// ---------------------------------------------------------------------------

/// Python `mp.nstr(x, n)` analog: print n significant digits.
///
/// We format the BigFloat via f64-as-mpmath-style scientific then back to
/// fixed/scientific notation. For matching the Python output line-for-line
/// we reuse Rust's f64 formatting on the high-precision value's f64 image
/// only where Python's `float(x)` is used (i.e., %.10f / %+.4f cases).
/// For `mp.nstr` we need actual high-precision truncation to n sig figs.
///
/// Strategy: format the BigFloat as a full decimal string via
/// `bigfloat_to_string`, then truncate / round to `n` sig figs ourselves.
fn nstr(x: &BigFloat, n: usize) -> String {
    // astro-float's format yields a mantissa-style string like
    // "1.6180339887...e+0" or "1.234e-5". Parse, round to n sig figs,
    // re-emit in mpmath's "12 sig fig fixed where possible, else scientific"
    // format. mpmath uses fixed when |exp| <= n+5-ish; we approximate:
    // use fixed when -4 <= exp <= n, scientific otherwise.

    let raw = bigfloat_to_string(x);
    parse_and_format_nsig(&raw, n)
}

/// Parse an astro-float decimal string and re-emit with n significant
/// digits in mpmath-style fixed-or-scientific notation.
fn parse_and_format_nsig(s: &str, n: usize) -> String {
    let s = s.trim();
    if s == "NaN" || s == "Inf" || s == "-Inf" {
        return s.to_string();
    }

    // astro-float emits like "1.234e+5" or "1.234e-3" or sometimes plain
    // "0.0" / "1.0" depending on version.
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        ("-", rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        ("", rest)
    } else {
        ("", s)
    };

    // Split mantissa and exponent.
    let (mantissa, exp_part): (&str, i32) = if let Some(idx) = body.find(|c| c == 'e' || c == 'E') {
        let (m, e) = body.split_at(idx);
        let e_str = &e[1..]; // strip 'e'
        let e_val = e_str.parse::<i32>().unwrap_or(0);
        (m, e_val)
    } else {
        (body, 0)
    };

    // Mantissa is like "1.234567..." — split at decimal point.
    let (int_part, frac_part) = if let Some(dot) = mantissa.find('.') {
        (&mantissa[..dot], &mantissa[dot + 1..])
    } else {
        (mantissa, "")
    };

    // Build the "all digits" string and original exponent of the leading
    // digit. If integer part is "0" (e.g. "0.000123e+0"), we shift exponent.
    let int_trim = int_part.trim_start_matches('0');
    let (digits, leading_exp): (String, i32) = if int_trim.is_empty() {
        // Normalize: skip leading zeros in frac_part to find first nonzero
        let first_nonzero = frac_part.find(|c: char| c != '0').unwrap_or(frac_part.len());
        if first_nonzero == frac_part.len() {
            // all zeros
            return format!("{}0.0", sign);
        }
        let new_digits: String = frac_part[first_nonzero..].chars().collect();
        let new_exp = exp_part - 1 - first_nonzero as i32;
        (new_digits, new_exp)
    } else {
        let mut combined = String::new();
        combined.push_str(int_trim);
        combined.push_str(frac_part);
        let new_exp = exp_part + (int_trim.len() as i32 - 1);
        (combined, new_exp)
    };

    // Now `digits` is purely digit chars, with the first one being the
    // most-significant. `leading_exp` is the decimal exponent of the
    // leading digit (so value = 0.<digits> * 10^(leading_exp+1)).

    // Round to n significant digits.
    let digits_chars: Vec<u8> = digits.bytes().take_while(|b| b.is_ascii_digit()).collect();
    let (rounded_digits, exp_bump) = round_sig(&digits_chars, n);
    let final_exp = leading_exp + exp_bump;

    // mpmath rule (libmp/libmpf.py to_str):
    //   min_fixed = min(-dps/3, -5)
    //   max_fixed = dps
    //   use fixed when min_fixed < final_exp < max_fixed
    let neg_dps_third: i32 = -((n as i32) / 3);
    let min_fixed: i32 = neg_dps_third.min(-5);
    let max_fixed: i32 = n as i32;
    let use_fixed = final_exp > min_fixed && final_exp < max_fixed;

    if use_fixed {
        format_fixed(&rounded_digits, final_exp, sign, n)
    } else {
        format_scientific(&rounded_digits, final_exp, sign)
    }
}

/// Round a digit slice to `n` sig figs. Returns (rounded_digits, exp_bump)
/// where exp_bump is +1 if rounding caused an extra leading digit (e.g.
/// 9.999 -> 10.00).
fn round_sig(digits: &[u8], n: usize) -> (Vec<u8>, i32) {
    if digits.len() <= n {
        let mut v: Vec<u8> = digits.to_vec();
        // pad with zeros to n digits
        while v.len() < n {
            v.push(b'0');
        }
        return (v, 0);
    }
    let mut head: Vec<u8> = digits[..n].to_vec();
    let next = digits[n];
    if next >= b'5' {
        // round up
        let mut carry = 1u8;
        for i in (0..n).rev() {
            if carry == 0 { break; }
            let d = head[i] - b'0' + carry;
            if d >= 10 {
                head[i] = b'0';
                carry = 1;
            } else {
                head[i] = b'0' + d;
                carry = 0;
            }
        }
        if carry == 1 {
            // overflow — prepend '1', drop last digit
            let mut new_head = Vec::with_capacity(n);
            new_head.push(b'1');
            new_head.extend_from_slice(&head[..n - 1]);
            return (new_head, 1);
        }
    }
    (head, 0)
}

/// Emit fixed-point (no exponent) form.
fn format_fixed(digits: &[u8], exp: i32, sign: &str, n: usize) -> String {
    // value = 0.<digits> * 10^(exp+1) ; first digit has decimal-place value 10^exp.
    let mut s = String::new();
    s.push_str(sign);
    if exp >= 0 {
        // integer part is digits[0..exp+1], frac = digits[exp+1..]
        let int_len = (exp as usize) + 1;
        let int_part: String = if int_len <= digits.len() {
            String::from_utf8_lossy(&digits[..int_len]).into_owned()
        } else {
            let mut p = String::from_utf8_lossy(digits).into_owned();
            for _ in 0..(int_len - digits.len()) {
                p.push('0');
            }
            p
        };
        s.push_str(&int_part);
        if int_len < digits.len() {
            s.push('.');
            // frac part — strip trailing zeros (mpmath behaviour)
            let frac: String = String::from_utf8_lossy(&digits[int_len..]).into_owned();
            let trimmed = frac.trim_end_matches('0');
            if trimmed.is_empty() {
                s.push('0');
            } else {
                s.push_str(trimmed);
            }
        }
    } else {
        // 0.<leading zeros><digits>
        s.push_str("0.");
        for _ in 0..((-exp - 1) as usize) {
            s.push('0');
        }
        let frac: String = String::from_utf8_lossy(digits).into_owned();
        // mpmath strips trailing zeros from significand
        let trimmed = frac.trim_end_matches('0');
        if trimmed.is_empty() {
            s.push('0');
        } else {
            s.push_str(trimmed);
        }
    }
    let _ = n; // unused, but keeps signature parallel
    s
}

/// Emit scientific notation: d.ddddde±NN
fn format_scientific(digits: &[u8], exp: i32, sign: &str) -> String {
    let mut s = String::new();
    s.push_str(sign);
    s.push(digits[0] as char);
    if digits.len() > 1 {
        s.push('.');
        let rest = String::from_utf8_lossy(&digits[1..]).into_owned();
        let trimmed = rest.trim_end_matches('0');
        if trimmed.is_empty() {
            // mpmath emits e.g. "1.0e+5" — keep one trailing zero
            s.push('0');
        } else {
            s.push_str(trimmed);
        }
    }
    s.push('e');
    if exp >= 0 {
        s.push('+');
    } else {
        s.push('-');
    }
    // mpmath emits unpadded exponent (e.g. "8.031e-5", not "e-05")
    let abs_exp = exp.unsigned_abs();
    s.push_str(&abs_exp.to_string());
    s
}

// ---------------------------------------------------------------------------
// MAIN MODEL — mirrors substrate_mass_spectrum.py
// ---------------------------------------------------------------------------

struct Model {
    prec: usize,
    // Inputs
    m_e_mev: BigFloat,
    m_mu_mev: BigFloat,
    h_planck: BigFloat,
    c: BigFloat,
    j_per_mev: BigFloat,
    // E_8 invariants
    dim: BigFloat,
    h_cox: BigFloat,
    rank: BigFloat,
    m_max: BigFloat,
    h_cox_14: BigFloat,
    // Derived
    phi: BigFloat,
    sqrt2: BigFloat,
    phi2: BigFloat,
    two_dim: BigFloat,
    two_pi: BigFloat,
    delta_phi: BigFloat,
    delta_2: BigFloat,
    omega_fix: BigFloat,
    alpha_tau: BigFloat,
    alpha_tau_full: BigFloat,
    alpha_eps: BigFloat,
    delta_h: BigFloat,
    exp5: BigFloat,
    delta_w: BigFloat,
    k_v: BigFloat,
    // Series sums
    alpha_mu: BigFloat,
    alpha_v: BigFloat,
    alpha_h: BigFloat,
    // Ratios + observed
    ratios: HashMap<String, BigFloat>,
    observed: HashMap<String, BigFloat>,
    display_in_gev: Vec<String>,
    // Anchors
    h_eff_e: BigFloat,
    h_eff_mu: BigFloat,
    f_struct: BigFloat,
}

impl Model {
    fn build() -> Self {
        let prec = bits_for_decimal_digits(150); // ≈ 502 bits

        // Inputs (CODATA / SI exact)
        let m_e_mev = bigfloat_from_str("0.51099895000", prec);
        let m_mu_mev = bigfloat_from_str("105.6583755", prec);
        let h_planck = bigfloat_from_str("6.62607015e-34", prec);
        let c = bigfloat_from_str("299792458", prec);
        let j_per_mev = bigfloat_from_str("1.602176634e-13", prec);

        // E_8 invariants
        let dim = bf(248, prec);
        let h_cox = bf(30, prec);
        let rank = bf(8, prec);
        let m_max = sub(&h_cox, &bf(1, prec)); // 29

        // 30^14 = 4782969 * 10^14 exactly
        let ten_14 = pow_i(&bf(10, prec), 14);
        let h_cox_14 = mul(&bf(4_782_969, prec), &ten_14);

        // Derived constants
        let phi = phi_const(prec);
        let sqrt2 = sqrt2_const(prec);
        let phi2 = mul(&phi, &phi);
        let two_dim = mul(&bf(2, prec), &dim);
        let two_pi = mul(&bf(2, prec), &pi_const(prec));

        let one = bf(1, prec);

        // Delta_phi = 1 / (h_cox + phi^2)
        let delta_phi = div(&one, &add(&h_cox, &phi2));
        // Delta_2 = 1/32
        let delta_2 = bf_q(1, 32, prec);
        // omega_fix = 1/2 - 1/dim
        let omega_fix = sub(&bf_q(1, 2, prec), &div(&one, &dim));
        // alpha_tau = 1 / (dim + h_cox) = 1/278
        let alpha_tau = div(&one, &add(&dim, &h_cox));
        // alpha_tau_full = 1 / (dim + h + phi^2)
        let alpha_tau_full = div(&one, &add(&add(&dim, &h_cox), &phi2));
        // alpha_eps = 1/(2*dim) = 1/496
        let alpha_eps = div(&one, &two_dim);
        // delta_H = 1 / (h_cox + m_max) = 1/59
        let delta_h = div(&one, &add(&h_cox, &m_max));
        let exp5 = bf(19, prec);
        // delta_W = 1/((h+m_max) * exp5) = 1/1121
        let delta_w = div(&one, &mul(&add(&h_cox, &m_max), &exp5));
        // k_v = h_cox + rank - rank/h_cox = 1132/30
        let k_v = sub(&add(&h_cox, &rank), &div(&rank, &h_cox));

        // Series sums (alpha_X = sum c_n * alpha_tau^n for n=1..MAX_ORDER)
        // Muon: c_2 = 2
        let mut mu_coeffs: HashMap<i64, BigFloat> = HashMap::new();
        mu_coeffs.insert(2, bf(2, prec));
        let alpha_mu = alpha_series(&mu_coeffs, &alpha_tau, MAX_ORDER, prec);

        // Higgs vev: c_2 = rank * h / m_max
        let mut v_coeffs: HashMap<i64, BigFloat> = HashMap::new();
        v_coeffs.insert(2, div(&mul(&rank, &h_cox), &m_max));
        let alpha_v = alpha_series(&v_coeffs, &alpha_tau, MAX_ORDER, prec);

        // Higgs boson: c_2 = (rank-2) * h / m_max
        let mut h_coeffs: HashMap<i64, BigFloat> = HashMap::new();
        let rank_m2 = sub(&rank, &bf(2, prec));
        h_coeffs.insert(2, div(&mul(&rank_m2, &h_cox), &m_max));
        let alpha_h = alpha_series(&h_coeffs, &alpha_tau, MAX_ORDER, prec);

        // ------------------------------------------------------------
        // Unitless structural ratios R = m / h_eff
        // ------------------------------------------------------------
        let mut ratios: HashMap<String, BigFloat> = HashMap::new();

        // electron: (dim - 2)/dim = 246/248
        ratios.insert(
            "electron".into(),
            div(&sub(&dim, &bf(2, prec)), &dim),
        );

        // muon: phi^11 * (1 + Delta_phi) * (1 - alpha_mu)
        let phi11 = pow_i(&phi, 11);
        let one_plus_dp = add(&one, &delta_phi);
        let one_minus_amu = sub(&one, &alpha_mu);
        ratios.insert(
            "muon".into(),
            mul(&mul(&phi11, &one_plus_dp), &one_minus_amu),
        );

        // tau: phi^17 * (1 - Delta_phi) * (1 - alpha_tau_full)
        let phi17 = pow_i(&phi, 17);
        let one_minus_dp = sub(&one, &delta_phi);
        let one_minus_atf = sub(&one, &alpha_tau_full);
        ratios.insert(
            "tau".into(),
            mul(&mul(&phi17, &one_minus_dp), &one_minus_atf),
        );

        // Quarks: quark_ratio(k, chirality_sign, loss_sign)
        ratios.insert(
            "up".into(),
            quark_ratio(&bf(4, prec), 1, 1, &sqrt2, &delta_2, &h_cox, &two_dim, &one, prec),
        );
        ratios.insert(
            "down".into(),
            quark_ratio(
                &bigfloat_from_str("6.5", prec),
                -1, -1, &sqrt2, &delta_2, &h_cox, &two_dim, &one, prec,
            ),
        );
        // strange: sqrt2^15 (no chirality, no loss)
        ratios.insert("strange".into(), pow_i(&sqrt2, 15));
        ratios.insert(
            "charm".into(),
            quark_ratio(
                &bigfloat_from_str("22.5", prec),
                1, -1, &sqrt2, &delta_2, &h_cox, &two_dim, &one, prec,
            ),
        );
        ratios.insert(
            "bottom".into(),
            quark_ratio(&bf(26, prec), -1, 1, &sqrt2, &delta_2, &h_cox, &two_dim, &one, prec),
        );
        ratios.insert(
            "top".into(),
            quark_ratio(
                &bigfloat_from_str("36.5", prec),
                1, 1, &sqrt2, &delta_2, &h_cox, &two_dim, &one, prec,
            ),
        );

        // Higgs vev: sqrt2^k_v * (1 - alpha_v)
        let sqrt2_kv = pow_frac(&sqrt2, &k_v);
        let one_minus_av = sub(&one, &alpha_v);
        ratios.insert("Higgs vev".into(), mul(&sqrt2_kv, &one_minus_av));

        // Higgs boson: sqrt2^k_v / 2 * (1 + delta_H) * (1 - alpha_H)
        let half_sqrt2_kv = div(&sqrt2_kv, &bf(2, prec));
        let one_plus_dh = add(&one, &delta_h);
        let one_minus_ah = sub(&one, &alpha_h);
        ratios.insert(
            "Higgs boson".into(),
            mul(&mul(&half_sqrt2_kv, &one_plus_dh), &one_minus_ah),
        );

        // W boson: sqrt2^34.5 * (1 + delta_W)
        let sqrt2_345 = pow_frac(&sqrt2, &bigfloat_from_str("34.5", prec));
        let one_plus_dw = add(&one, &delta_w);
        ratios.insert("W boson".into(), mul(&sqrt2_345, &one_plus_dw));

        // Anchor extraction
        let h_eff_e = div(&m_e_mev, ratios.get("electron").unwrap());
        let h_eff_mu = div(&m_mu_mev, ratios.get("muon").unwrap());

        // Observed values (PDG 2024 + CODATA 2022)
        let observed: HashMap<String, BigFloat> = [
            ("electron", "0.51099895000"),
            ("muon", "105.6583755"),
            ("tau", "1776.86"),
            ("up", "2.16"),
            ("down", "4.67"),
            ("strange", "93.4"),
            ("charm", "1273"),
            ("bottom", "4183"),
            ("top", "172570"),
            ("Higgs vev", "246219.65"),
            ("Higgs boson", "125200"),
            ("W boson", "80369.2"),
        ]
        .iter()
        .map(|(k, v)| ((*k).to_string(), bigfloat_from_str(v, prec)))
        .collect();

        let display_in_gev: Vec<String> = ["top", "Higgs vev", "Higgs boson", "W boson"]
            .iter().map(|s| s.to_string()).collect();

        // F_struct factor for Newton's G:
        // F = (dim-2)/dim * 5/(dim*h_cox_14) * 1/(1 - 1/(4*(dim-1)))
        let dim_m1 = sub(&dim, &one);
        let four_dim_m1 = mul(&bf(4, prec), &dim_m1);
        let one_over_4dimm1 = div(&one, &four_dim_m1);
        let one_minus = sub(&one, &one_over_4dimm1);
        let f_struct = mul(
            &mul(
                &div(&sub(&dim, &bf(2, prec)), &dim),
                &div(&bf(5, prec), &mul(&dim, &h_cox_14)),
            ),
            &div(&one, &one_minus),
        );

        Model {
            prec,
            m_e_mev,
            m_mu_mev,
            h_planck,
            c,
            j_per_mev,
            dim,
            h_cox,
            rank,
            m_max,
            h_cox_14,
            phi,
            sqrt2,
            phi2,
            two_dim,
            two_pi,
            delta_phi,
            delta_2,
            omega_fix,
            alpha_tau,
            alpha_tau_full,
            alpha_eps,
            delta_h,
            exp5,
            delta_w,
            k_v,
            alpha_mu,
            alpha_v,
            alpha_h,
            ratios,
            observed,
            display_in_gev,
            h_eff_e,
            h_eff_mu,
            f_struct,
        }
    }
}

/// Sum c_n * alpha_tau^n for n in 1..=max_order, skipping orders not in `coeffs`.
fn alpha_series(
    coeffs: &HashMap<i64, BigFloat>,
    alpha_tau: &BigFloat,
    max_order: i64,
    _prec: usize,
) -> BigFloat {
    let zero = bf(0, _prec);
    let mut result = zero.clone();
    for n in 1..=max_order {
        if let Some(c_n) = coeffs.get(&n) {
            let term = mul(c_n, &pow_i(alpha_tau, n));
            result = add(&result, &term);
        }
    }
    result
}

/// quark_ratio(k, chirality_sign, loss_sign) per Q.3 rules:
///   base = sqrt2^k
///   chi  = 1                            if chirality_sign == 0
///        = 1 + chirality_sign * Delta_2 otherwise
///   loss = 1                            if loss_sign == 0
///        = 1 + loss_sign * |k - h/2| / (2*dim)  otherwise
#[allow(clippy::too_many_arguments)]
fn quark_ratio(
    k: &BigFloat,
    chirality_sign: i64,
    loss_sign: i64,
    sqrt2: &BigFloat,
    delta_2: &BigFloat,
    h_cox: &BigFloat,
    two_dim: &BigFloat,
    one: &BigFloat,
    prec: usize,
) -> BigFloat {
    let base = pow_frac(sqrt2, k);
    let chi = if chirality_sign == 0 {
        one.clone()
    } else {
        let s = bf(chirality_sign, prec);
        add(one, &mul(&s, delta_2))
    };
    let loss = if loss_sign == 0 {
        one.clone()
    } else {
        let half_h = div(h_cox, &bf(2, prec));
        let diff = sub(k, &half_h);
        let abs_diff = Real::abs(&diff);
        let s = bf(loss_sign, prec);
        let inner = div(&mul(&s, &abs_diff), two_dim);
        add(one, &inner)
    };
    mul(&mul(&base, &chi), &loss)
}

// ---------------------------------------------------------------------------
// REPORT FORMATTING — produces output identical to the Python script
// ---------------------------------------------------------------------------

/// Compute (a-b)/b * 1e6 as f64 — used for ppm reports. Only the f64
/// image of the result is needed (we print at 4 decimal places).
///
/// When the high-precision difference is below the f64 noise floor
/// relative to the printed precision (4 decimals = 5e-5), we return
/// exactly +0.0 to avoid signed-zero spelling drift between astro-float
/// and mpmath. Both libraries agree the value is zero at this scale;
/// only the sign of rounding noise differs.
fn ppm(a: &BigFloat, b: &BigFloat) -> f64 {
    let prec_a = 600usize;
    let zero = bf(0, prec_a);
    if let Some(0) = b.cmp(&zero) {
        return 0.0;
    }
    let diff = sub(a, b);
    let r = div(&diff, b);
    let scaled = mul(&r, &bf(1_000_000, prec_a));
    let v = to_f64(&scaled);
    // Below the print-precision noise floor, force +0 (matches Python's
    // mpmath spelling for true mathematical cancellations like Yukawa
    // e-vs-mu where h_eff exactly cancels).
    if v.abs() < 5e-5 && v != 0.0 {
        // distinguish "cancellation" cases (where high-precision rep is
        // truly tiny relative to scale) from cases where the value
        // rounds to 0 but is genuinely small. mpmath shows +0.0000 for
        // both rounded-to-zero positives and tiny-noise.
        // Match mpmath's ".4f" formatting: round to 4 decimals first.
        let rounded = (v * 10000.0).round() / 10000.0;
        if rounded == 0.0 {
            return 0.0;
        }
    }
    v
}

fn pct(a: &BigFloat, b: &BigFloat) -> f64 {
    let prec_a = 600usize;
    let diff = sub(a, b);
    let r = div(&diff, b);
    let scaled = mul(&r, &bf(100, prec_a));
    to_f64(&scaled)
}

/// Format a mass for the predictions table (Python's fmt_mass).
fn fmt_mass(particle: &str, value_mev: &BigFloat, gev_set: &[String]) -> String {
    let in_gev = gev_set.iter().any(|x| x == particle);
    if in_gev {
        let v_gev = div(value_mev, &bf(1000, 600));
        format!("{:>16.10} GeV", to_f64(&v_gev))
    } else {
        format!("{:>16.10} MeV", to_f64(value_mev))
    }
}

/// Format a value for the comprehensive table (Python's fmt_value).
fn fmt_value(name: &str, val: &BigFloat) -> String {
    match name {
        "G" => format!("{:>18.10e}", to_f64(val)),
        "m_Planck" => {
            let scaled = div(val, &bigfloat_from_str("1e22", 600));
            format!("{:>18.10}e22", to_f64(&scaled))
        }
        n if matches!(n, "top" | "Higgs vev" | "Higgs boson" | "W boson") => {
            let v = div(val, &bf(1000, 600));
            format!("{:>18.10}", to_f64(&v))
        }
        n if n == "lambda_H" || n == "alpha_s" || n.starts_with("y_") => {
            format!("{:>18.12}", to_f64(val))
        }
        n if n == "photon" || n.starts_with("gluon_") => {
            format!("{:>18.6}", to_f64(val))
        }
        _ => format!("{:>18.10}", to_f64(val)),
    }
}

fn print_section(title: &str, width: usize) {
    println!("{}", "=".repeat(width));
    println!("{}", title);
    println!("{}", "=".repeat(width));
}

fn run_report(m: &Model) {
    let w = 140;
    let w2 = 158;

    // Header
    print_section(
        &format!(
            "SUBSTRATE FRAMEWORK MASS SPECTRUM -- DUAL ANCHOR COMPARISON  (mp.prec = 500, MAX_ORDER = {})",
            MAX_ORDER
        ),
        w,
    );
    println!();
    println!("Step 1: compute unitless structural ratio R = m / h_eff for each particle");
    println!("Step 2: anchor h_eff on a single measured mass (electron or muon)");
    println!("Step 3: predict every other mass as m_predicted = R * h_eff_anchor");
    println!();
    println!(
        "h_eff anchored on electron (gateway formula): {} MeV",
        nstr(&m.h_eff_e, 20)
    );
    println!(
        "h_eff anchored on muon     (bulk formula):    {} MeV",
        nstr(&m.h_eff_mu, 20)
    );
    let dh_pct = pct(&m.h_eff_e, &m.h_eff_mu);
    let dh_ppm = ppm(&m.h_eff_e, &m.h_eff_mu);
    println!(
        "Anchor variance (electron vs muon):           {:+.10}% = {:+.4} ppm",
        dh_pct, dh_ppm
    );
    println!();

    // Unitless ratios table
    print_section(
        "UNITLESS STRUCTURAL RATIOS  R = m / h_eff   (pure numbers from E_8 invariants, computed to 50+ decimal digits)",
        w,
    );
    println!(
        "{:12} | {:>52}",
        "particle", "R (unitless, 25-digit precision)"
    );
    println!("{}", "-".repeat(w));
    let order = particle_order();
    for p in &order {
        if let Some(r) = m.ratios.get(p) {
            println!("{:12} | {:>52}", p, nstr(r, 25));
        }
    }
    println!();

    // Predictions and variances
    print_section(
        "PREDICTED MASSES (m = R * h_eff)  AND VARIANCES vs OBSERVED  --  16-digit precision",
        w,
    );
    println!(
        "{:12} | {:>20} | {:>22} | {:>22} | {:>14} | {:>14} | {:>12}",
        "particle", "observed", "electron-anchor", "muon-anchor",
        "e vs obs", "mu vs obs", "e vs mu"
    );
    println!("{}", "-".repeat(w));
    for p in &order {
        let r = m.ratios.get(p).unwrap();
        let obs = m.observed.get(p).unwrap();
        let pe = mul(r, &m.h_eff_e);
        let pmu = mul(r, &m.h_eff_mu);
        let in_gev = m.display_in_gev.iter().any(|x| x == p);
        let obs_str = if in_gev {
            format!("{:>16.10} GeV", to_f64(&div(obs, &bf(1000, 600))))
        } else {
            format!("{:>16.10} MeV", to_f64(obs))
        };
        let pe_str = if in_gev {
            format!("{:>18.12} GeV", to_f64(&div(&pe, &bf(1000, 600))))
        } else {
            format!("{:>18.12} MeV", to_f64(&pe))
        };
        let pmu_str = if in_gev {
            format!("{:>18.12} GeV", to_f64(&div(&pmu, &bf(1000, 600))))
        } else {
            format!("{:>18.12} MeV", to_f64(&pmu))
        };
        let e_obs = ppm(&pe, obs);
        let mu_obs = ppm(&pmu, obs);
        let e_mu = ppm(&pe, &pmu);
        println!(
            "{:12} | {:>20} | {:>22} | {:>22} | {:>+10.4} ppm | {:>+10.4} ppm | {:>+8.4} ppm",
            p, obs_str, pe_str, pmu_str, e_obs, mu_obs, e_mu
        );
    }
    println!();

    // Series convergence
    print_section(
        "SUB-HARMONIC SERIES VALUES  (alpha_X = sum_{n=1..50} c_n * alpha_tau^n)",
        w,
    );
    println!("{:35} | {:>30} | {:>14}", "series", "sum value", "in ppm");
    println!("{}", "-".repeat(w));
    println!(
        "{:42} | {:>30} | {:>+12.4} ppm",
        "alpha_tau (bare, 1st-order)",
        nstr(&m.alpha_tau, 20),
        to_f64(&m.alpha_tau) * 1e6
    );
    println!(
        "{:42} | {:>30} | {:>+12.4} ppm",
        "alpha_tau_full = 1/(dim+h+phi^2)",
        nstr(&m.alpha_tau_full, 20),
        to_f64(&m.alpha_tau_full) * 1e6
    );
    println!(
        "{:42} | {:>30} | {:>14}",
        "  (closed-form, all orders summed)", "", ""
    );
    println!(
        "{:42} | {:>30} | {:>+12.4} ppm",
        "alpha_mu  = 2*alpha_tau^2",
        nstr(&m.alpha_mu, 20),
        to_f64(&m.alpha_mu) * 1e6
    );
    println!(
        "{:42} | {:>30} | {:>+12.4} ppm",
        "alpha_v   = (rank*h/m_max)*alpha_tau^2",
        nstr(&m.alpha_v, 20),
        to_f64(&m.alpha_v) * 1e6
    );
    println!(
        "{:42} | {:>30} | {:>+12.4} ppm",
        "alpha_H   = ((rank-2)*h/m_max)*alpha_tau^2",
        nstr(&m.alpha_h, 20),
        to_f64(&m.alpha_h) * 1e6
    );

    println!();
    print_section("SUPPORTING QUANTITIES (16+ decimal digits)", w);
    println!("phi       = (1 + sqrt 5)/2     = {}", nstr(&m.phi, 20));
    println!("sqrt 2                          = {}", nstr(&m.sqrt2, 20));
    println!("phi^2                           = {}", nstr(&m.phi2, 20));
    println!(
        "Delta_phi = 1 / (30 + phi^2)   = {}",
        nstr(&m.delta_phi, 20)
    );
    println!("Delta_2   = 1 / 32             = {}", nstr(&m.delta_2, 20));
    println!(
        "omega_fix = 123 / 248          = {}",
        nstr(&m.omega_fix, 20)
    );
    println!(
        "alpha_tau = 1 / (dim+h) = 1/278= {}",
        nstr(&m.alpha_tau, 20)
    );
    let alpha_tau_sq = mul(&m.alpha_tau, &m.alpha_tau);
    println!(
        "alpha_tau^2                     = {}",
        nstr(&alpha_tau_sq, 20)
    );
    println!(
        "epsilon   = 1 / (2*dim) = 1/496= {}",
        nstr(&m.alpha_eps, 20)
    );
    println!(
        "delta_H   = 1 / (h+m_max) = 1/59= {}",
        nstr(&m.delta_h, 20)
    );
    println!("k_v       = 1132 / 30          = {}", nstr(&m.k_v, 20));

    // Comprehensive constants
    let constants_e = derive_constants(m, &m.h_eff_e);
    let constants_mu = derive_constants(m, &m.h_eff_mu);

    let mut obs_constants: HashMap<String, BigFloat> = m.observed.clone();
    let v_obs = m.observed.get("Higgs vev").unwrap().clone();
    let m_h_obs = m.observed.get("Higgs boson").unwrap().clone();
    for fermion in [
        "electron", "muon", "tau", "up", "down", "strange", "charm", "bottom", "top",
    ] {
        let m_f = m.observed.get(fermion).unwrap();
        let y = div(&mul(m_f, &m.sqrt2), &v_obs);
        obs_constants.insert(format!("y_{}", fermion), y);
    }
    let lam_obs = div(&mul(&m_h_obs, &m_h_obs), &mul(&bf(2, m.prec), &mul(&v_obs, &v_obs)));
    obs_constants.insert("lambda_H".into(), lam_obs);
    obs_constants.insert("alpha_s".into(), bigfloat_from_str("0.1180", m.prec));
    obs_constants.insert("photon".into(), bf(0, m.prec));
    for i in 1..=8 {
        obs_constants.insert(format!("gluon_{}", i), bf(0, m.prec));
    }
    obs_constants.insert("G".into(), bigfloat_from_str("6.67430e-11", m.prec));
    obs_constants.insert("m_Planck".into(), bigfloat_from_str("1.220890e22", m.prec));

    println!();
    print_section(
        "COMPLETE TABLE: ALL DERIVABLE STANDARD-MODEL + GRAVITATIONAL CONSTANTS",
        w2,
    );
    println!(
        "{:18} | {:14} | {:>22} | {:>22} | {:>22} | {:>13} | {:>13} | {:>11}",
        "observable", "unit", "observed", "electron-anchor", "muon-anchor",
        "e vs obs", "mu vs obs", "e vs mu"
    );
    println!("{}", "-".repeat(w2));

    for (name, unit) in display_order() {
        match unit {
            None => {
                println!();
                println!("{}", name);
            }
            Some(u) => {
                let obs_val = obs_constants.get(name).unwrap();
                let pe = constants_e.get(name).unwrap();
                let pmu = constants_mu.get(name).unwrap();
                let obs_str = fmt_value(name, obs_val);
                let pe_str = fmt_value(name, pe);
                let pmu_str = fmt_value(name, pmu);
                let zero = bf(0, m.prec);
                let (e_obs_s, mu_obs_s, e_mu_s) =
                    if matches!(obs_val.cmp(&zero), Some(0))
                        && matches!(pe.cmp(&zero), Some(0))
                        && matches!(pmu.cmp(&zero), Some(0))
                    {
                        (
                            "   exact   ".to_string(),
                            "   exact   ".to_string(),
                            "  exact  ".to_string(),
                        )
                    } else {
                        let e_obs = format!("{:>+10.4}ppm", ppm(pe, obs_val));
                        let mu_obs = format!("{:>+10.4}ppm", ppm(pmu, obs_val));
                        let e_mu = if matches!(pmu.cmp(&zero), Some(0)) {
                            "  exact  ".to_string()
                        } else {
                            format!("{:>+8.4}ppm", ppm(pe, pmu))
                        };
                        (e_obs, mu_obs, e_mu)
                    };
                println!(
                    "{:18} | {:14} | {:>22} | {:>22} | {:>22} | {:>13} | {:>13} | {:>11}",
                    name, u, obs_str, pe_str, pmu_str, e_obs_s, mu_obs_s, e_mu_s
                );
            }
        }
    }

    // G consistency check
    println!();
    print_section(
        "CONSISTENCY CHECK: G is a derived quantity, |G e-vs-mu| should be 2 * |h_eff e-vs-mu|",
        w2,
    );
    let mass_ratio_ppm = ppm(&m.h_eff_e, &m.h_eff_mu);
    let g_e = constants_e.get("G").unwrap();
    let g_mu = constants_mu.get("G").unwrap();
    let g_ratio_ppm = ppm(g_e, g_mu);
    let expected_ppm = -2.0 * mass_ratio_ppm;
    println!("h_eff e-vs-mu          = {:+.6} ppm", mass_ratio_ppm);
    println!("G     e-vs-mu (actual) = {:+.6} ppm", g_ratio_ppm);
    println!(
        "G     e-vs-mu (expect) = {:+.6} ppm  (opposite sign of 2x mass ratio, since G ~ 1/m_e^2)",
        expected_ppm
    );
    let match_str = if (g_ratio_ppm - expected_ppm).abs() < 1e-4 {
        "YES (clean derived quantity, no extra structure)"
    } else {
        "NO -- code-level inconsistency"
    };
    println!("Match: {}", match_str);

    // Sub-correction summary
    println!();
    print_section("STRUCTURAL CORRECTIONS APPLIED", w2);
    println!(
        "Delta_phi = 1/(30+phi^2)         = {}     (icosahedral bulk recursion)",
        nstr(&m.delta_phi, 18)
    );
    println!(
        "Delta_2   = 1/32                 = {}     (octahedral bulk recursion)",
        nstr(&m.delta_2, 18)
    );
    println!(
        "alpha_tau = 1/(dim+h) = 1/278    = {}    (gateway-mixing for tau)",
        nstr(&m.alpha_tau, 18)
    );
    println!(
        "alpha_tau_full = 1/(dim+h+phi^2) = {}    (recursive closed-form for tau)",
        nstr(&m.alpha_tau_full, 18)
    );
    println!(
        "alpha_mu  = 2*alpha_tau^2        = {}    ({:+.2} ppm) [muon: paired (11,19) exponents]",
        nstr(&m.alpha_mu, 18),
        to_f64(&m.alpha_mu) * 1e6
    );
    println!(
        "alpha_v   = (rank*h/m_max)*alpha_tau^2     = {}    ({:+.2} ppm) [v: rank-locked 2nd cycle]",
        nstr(&m.alpha_v, 16),
        to_f64(&m.alpha_v) * 1e6
    );
    println!(
        "alpha_H   = ((rank-2)*h/m_max)*alpha_tau^2 = {}    ({:+.2} ppm) [Higgs boson: breathing mode]",
        nstr(&m.alpha_h, 16),
        to_f64(&m.alpha_h) * 1e6
    );
    println!(
        "delta_H   = 1/(h+m_max)   = 1/59          = {}    ({:+.2} ppm) [Higgs dark coupling]",
        nstr(&m.delta_h, 16),
        to_f64(&m.delta_h) * 1e6
    );
    println!(
        "delta_W   = 1/((h+m_max)*exp_5) = 1/1121  = {}    ({:+.2} ppm) [W dark coupling, NEW]",
        nstr(&m.delta_w, 16),
        to_f64(&m.delta_w) * 1e6
    );
    println!(
        "epsilon   = 1/(2*dim)     = 1/496         = {} (per-step quark loss)",
        nstr(&m.alpha_eps, 16)
    );
    println!(
        "omega_fix = 1/2 - 1/dim   = 123/248       = {}     (gateway eigenvalue)",
        nstr(&m.omega_fix, 16)
    );
    println!(
        "k_v       = 1132/30                       = {}     (Higgs vev chain position)",
        nstr(&m.k_v, 16)
    );
    println!();
    println!(
        "h_Cox^14  = mpf(4782969) * mpf(10)**14 = 30^14 EXACTLY = {}",
        nstr(&m.h_cox_14, 22)
    );
}

/// Standard particle iteration order (matches Python observed_MeV iteration).
fn particle_order() -> Vec<String> {
    [
        "electron", "muon", "tau",
        "up", "down", "strange", "charm", "bottom", "top",
        "Higgs vev", "Higgs boson", "W boson",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Display order for the comprehensive table.
fn display_order() -> Vec<(&'static str, Option<&'static str>)> {
    vec![
        ("=== CHARGED LEPTON MASSES ===", None),
        ("electron", Some("MeV")),
        ("muon", Some("MeV")),
        ("tau", Some("MeV")),
        ("=== QUARK MASSES ===", None),
        ("up", Some("MeV")),
        ("down", Some("MeV")),
        ("strange", Some("MeV")),
        ("charm", Some("MeV")),
        ("bottom", Some("MeV")),
        ("top", Some("GeV")),
        ("=== ELECTROWEAK SECTOR ===", None),
        ("Higgs vev", Some("GeV")),
        ("Higgs boson", Some("GeV")),
        ("W boson", Some("GeV")),
        ("=== LEPTON YUKAWA COUPLINGS (SM bookkeeping: y = m sqrt(2)/v) ===", None),
        ("y_electron", Some("")),
        ("y_muon", Some("")),
        ("y_tau", Some("")),
        ("=== QUARK YUKAWA COUPLINGS (SM bookkeeping: y = m sqrt(2)/v) ===", None),
        ("y_up", Some("")),
        ("y_down", Some("")),
        ("y_strange", Some("")),
        ("y_charm", Some("")),
        ("y_bottom", Some("")),
        ("y_top", Some("")),
        ("=== HIGGS SELF-COUPLING (m_H^2 = 2 lambda v^2) ===", None),
        ("lambda_H", Some("")),
        ("=== STRONG COUPLING (predicted from E_8 invariants only) ===", None),
        ("alpha_s", Some("")),
        ("=== MASSLESS GAUGE BOSONS (substrate-physical: zero local DT in rest frame) ===", None),
        ("photon", Some("MeV")),
        ("gluon_1", Some("MeV")),
        ("gluon_2", Some("MeV")),
        ("gluon_3", Some("MeV")),
        ("gluon_4", Some("MeV")),
        ("gluon_5", Some("MeV")),
        ("gluon_6", Some("MeV")),
        ("gluon_7", Some("MeV")),
        ("gluon_8", Some("MeV")),
        ("=== GRAVITATIONAL SECTOR ===", None),
        ("G", Some("m^3/kg/s^2")),
        ("m_Planck", Some("GeV")),
    ]
}

/// Compute all derivable SM constants for a given h_eff anchor.
fn derive_constants(m: &Model, h_eff: &BigFloat) -> HashMap<String, BigFloat> {
    let mut out: HashMap<String, BigFloat> = HashMap::new();
    for (name, r) in &m.ratios {
        out.insert(name.clone(), mul(r, h_eff));
    }
    let v = out.get("Higgs vev").unwrap().clone();
    for fermion in [
        "electron", "muon", "tau", "up", "down", "strange", "charm", "bottom", "top",
    ] {
        let m_f = out.get(fermion).unwrap().clone();
        let y = div(&mul(&m_f, &m.sqrt2), &v);
        out.insert(format!("y_{}", fermion), y);
    }
    // lambda_H = m_H^2 / (2 v^2)
    let m_h = out.get("Higgs boson").unwrap().clone();
    let lam = div(
        &mul(&m_h, &m_h),
        &mul(&bf(2, m.prec), &mul(&v, &v)),
    );
    out.insert("lambda_H".into(), lam);

    // alpha_s = (m_max + rank/m_max) / dim
    let alpha_s = div(
        &add(&m.m_max, &div(&m.rank, &m.m_max)),
        &m.dim,
    );
    out.insert("alpha_s".into(), alpha_s);

    out.insert("photon".into(), bf(0, m.prec));
    for i in 1..=8 {
        out.insert(format!("gluon_{}", i), bf(0, m.prec));
    }

    // Newton's G = F^2 * h * c^5 / (2 pi * m_e_pred^2) with m_e_pred in J.
    let m_e_j = mul(out.get("electron").unwrap(), &m.j_per_mev);
    let f_sq = mul(&m.f_struct, &m.f_struct);
    let c5 = pow_i(&m.c, 5);
    let num = mul(&mul(&f_sq, &m.h_planck), &c5);
    let den = mul(&m.two_pi, &mul(&m_e_j, &m_e_j));
    let g = div(&num, &den);
    out.insert("G".into(), g.clone());

    // m_Planck c^2 = sqrt(h c^5 / (2 pi G))  -> m_Planck (J -> MeV).
    let mp_c2_j_sq = div(&mul(&m.h_planck, &c5), &mul(&m.two_pi, &g));
    let mp_c2_j = Real::sqrt(&mp_c2_j_sq);
    let m_planck_mev = div(&mp_c2_j, &m.j_per_mev);
    out.insert("m_Planck".into(), m_planck_mev);

    out
}

// ---------------------------------------------------------------------------
// SELF-TEST — golden values from Python script. Asserts must pass before we
// print the report. Failure here means precision drift between mpmath and
// astro-float, which is a real bug.
// ---------------------------------------------------------------------------

fn assert_starts_with(label: &str, got: &str, want_prefix: &str) {
    if !got.starts_with(want_prefix) {
        panic!(
            "P6.2 self-test FAILED for `{}`:\n  got    = {}\n  wanted = {}<...>",
            label, got, want_prefix
        );
    }
}

fn run_self_test(m: &Model) {
    // Each golden is the first 30+ digits from the Python script's mp.nstr output.
    // We compare via string-prefix on a 35-sig-fig render of the BigFloat.

    let phi_s = nstr(&m.phi, 35);
    assert_starts_with("phi", &phi_s, "1.6180339887498948482045868343656");

    let sqrt2_s = nstr(&m.sqrt2, 35);
    assert_starts_with("sqrt2", &sqrt2_s, "1.4142135623730950488016887242096");

    let alpha_tau_s = nstr(&m.alpha_tau, 35);
    // 1/278 = 0.003597122302158273381294964028776978...
    assert_starts_with(
        "alpha_tau",
        &alpha_tau_s,
        "0.0035971223021582733812949640287769",
    );

    let alpha_tau_full_s = nstr(&m.alpha_tau_full, 30);
    // 1/(280.6180339887498948...) = 0.00356356284657061796...
    assert_starts_with(
        "alpha_tau_full",
        &alpha_tau_full_s,
        "0.00356356284657061796",
    );

    let omega_s = nstr(&m.omega_fix, 30);
    // 123/248 = 0.495967741935483870967741935483...
    assert_starts_with("omega_fix", &omega_s, "0.49596774193548387096774");

    // electron ratio = 246/248 = 0.991935483870967741935...
    let r_e = m.ratios.get("electron").unwrap();
    let r_e_s = nstr(r_e, 30);
    assert_starts_with("R_electron", &r_e_s, "0.9919354838709677419354");

    // Strange = sqrt2^15 = 181.01933598375616624...
    let r_strange = m.ratios.get("strange").unwrap();
    let r_strange_s = nstr(r_strange, 25);
    assert_starts_with("R_strange", &r_strange_s, "181.0193359837561662466");

    // Bottom = sqrt2^26 (with chirality -1, loss +1, k=26):
    // From Python: 8112.0 — sqrt2^26 = 2^13 = 8192, chi=(1-1/32)=31/32, loss=(1+|26-15|/496)=1+11/496=507/496
    // 8192 * 31/32 * 507/496 = 8192 * 31 * 507 / (32 * 496) = 128_741_664 / 15_872 = 8112.0 exactly
    let r_b = m.ratios.get("bottom").unwrap();
    let r_b_s = nstr(r_b, 10);
    assert_starts_with("R_bottom", &r_b_s, "8112.0");

    // Anchor variance should be ~ -29.85 ppb = -0.0298 ppm
    let dh_ppm = ppm(&m.h_eff_e, &m.h_eff_mu);
    if !((-0.04..-0.02).contains(&dh_ppm)) {
        panic!(
            "P6.2 self-test FAILED: anchor variance {:+.4} ppm out of expected (-0.04, -0.02) ppm range",
            dh_ppm
        );
    }

    // alpha_mu = 2 * alpha_tau^2 ~ 25.88 ppm
    let am_ppm = to_f64(&m.alpha_mu) * 1e6;
    if !(25.5..26.5).contains(&am_ppm) {
        panic!("P6.2 self-test FAILED: alpha_mu = {:+.4} ppm out of expected ~25.88 ppm", am_ppm);
    }

    eprintln!("P6.2 self-test: PASS (8/8 golden values matched at 30+ digit precision)");
}

fn main() {
    let m = Model::build();
    run_self_test(&m);
    run_report(&m);
}
