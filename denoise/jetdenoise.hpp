// jetdenoise.hpp
//
// Header-only C++17 background estimation and denoising for heavy-ion events.
// Works on whole events of particles (pt, eta, phi); returns per-particle
// corrected pT so the output can be reclustered with any jet algorithm.
//
//   1. background_map   : rho(phi) from patch medians + Fourier fit (n <= nmax),
//                         so v2, v3 flow modulation is part of the estimate.
//   2. constituent_subtract : event-wide Constituent Subtraction
//                         (Berta, Spousta, Miller, Leitner, JHEP 06 (2014) 092),
//                         ghosts carry rho(phi) * A_ghost.
//   3. starlet_denoise  : subtract the rho(phi) map from an (eta, phi) image,
//                         isotropic undecimated wavelet (starlet, B3-spline
//                         a trous) decomposition, per-scale MAD noise estimate,
//                         hard or soft thresholding, then a per-pixel signal
//                         fraction f in [0, 1] applied to each particle.
//
// Conventions: phi in [0, 2pi), periodic. eta in [-eta_max, eta_max], mirrored
// at the edges. Pixels default to 0.05 x 0.0499 so the image is close to
// isotropic in (eta, phi).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jetdenoise {

constexpr double kTwoPi = 6.283185307179586;

inline double wrap_phi(double p) {
  p = std::fmod(p, kTwoPi);
  return p < 0 ? p + kTwoPi : p;
}

inline double dphi(double a, double b) {
  double d = std::fmod(a - b + M_PI, kTwoPi);
  if (d < 0) d += kTwoPi;
  return d - M_PI;
}

inline double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const std::size_t n = v.size(), m = n / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  double hi = v[m];
  if (n % 2) return hi;
  double lo = *std::max_element(v.begin(), v.begin() + m);
  return 0.5 * (lo + hi);
}

// Median absolute deviation scaled to a Gaussian sigma.
inline double mad_sigma(const std::vector<double>& v) {
  const double med = median_of(v);
  std::vector<double> d(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) d[i] = std::abs(v[i] - med);
  return 1.4826 * median_of(std::move(d));
}

// ---------------------------------------------------------------------------
// 1. Background density rho(phi)
// ---------------------------------------------------------------------------
struct BackgroundMap {
  double a0 = 0.0;                 // mean density [GeV per unit area]
  std::vector<double> an, bn;      // Fourier coefficients, n = 1..nmax
  double operator()(double phi) const {
    double r = a0;
    for (std::size_t n = 1; n <= an.size(); ++n)
      r += an[n - 1] * std::cos(n * phi) + bn[n - 1] * std::sin(n * phi);
    return std::max(r, 0.0);
  }
  // Flow coefficient implied by the fit: rho = a0 [1 + 2 v_n cos n(phi - Psi_n)]
  double vn(int n) const {
    if (n < 1 || n > (int)an.size() || a0 <= 0) return 0.0;
    return 0.5 * std::hypot(an[n - 1], bn[n - 1]) / a0;
  }
};

// Patch medians over eta in each phi column, then a Fourier fit in phi.
// The median over eta patches is robust to the one or two patches a jet hits.
inline BackgroundMap background_map(const double* pt, const double* eta, const double* phi,
                                    std::size_t n, double eta_max = 3.0, int n_eta = 12,
                                    int n_phi = 12, int nmax = 3) {
  if (2 * nmax >= n_phi) throw std::invalid_argument("nmax too large for n_phi");
  std::vector<double> grid((std::size_t)n_eta * n_phi, 0.0);
  const double deta = 2 * eta_max / n_eta, dph = kTwoPi / n_phi, area = deta * dph;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(eta[i]) >= eta_max) continue;
    int ie = std::min(n_eta - 1, (int)((eta[i] + eta_max) / deta));
    int ip = std::min(n_phi - 1, (int)(wrap_phi(phi[i]) / dph));
    grid[(std::size_t)ie * n_phi + ip] += pt[i];
  }
  std::vector<double> col(n_phi);
  std::vector<double> tmp(n_eta);
  for (int ip = 0; ip < n_phi; ++ip) {
    for (int ie = 0; ie < n_eta; ++ie) tmp[ie] = grid[(std::size_t)ie * n_phi + ip];
    col[ip] = median_of(tmp) / area;
  }
  BackgroundMap m;
  m.an.assign(nmax, 0.0);
  m.bn.assign(nmax, 0.0);
  for (int ip = 0; ip < n_phi; ++ip) m.a0 += col[ip];
  m.a0 /= n_phi;
  for (int k = 1; k <= nmax; ++k)
    for (int ip = 0; ip < n_phi; ++ip) {
      const double p = (ip + 0.5) * dph;
      m.an[k - 1] += 2.0 / n_phi * col[ip] * std::cos(k * p);
      m.bn[k - 1] += 2.0 / n_phi * col[ip] * std::sin(k * p);
    }
  return m;
}

// ---------------------------------------------------------------------------
// 2. Event-wide Constituent Subtraction
// ---------------------------------------------------------------------------
// Ghosts on a regular grid carry pT_g = rho(phi_g) * A_g. All particle-ghost
// pairs with dR < max_dr are sorted by pT_i^alpha * dR and processed in order:
// the smaller of the two pT values is removed from both. Returns corrected pT.
inline std::vector<double> constituent_subtract(const double* pt, const double* eta,
                                                const double* phi, std::size_t n,
                                                const BackgroundMap& bkg, double eta_max = 3.0,
                                                double ghost_size = 0.1, double max_dr = 0.25,
                                                double alpha = 0.0) {
  const int g_eta = (int)std::lround(2 * eta_max / ghost_size);
  const int g_phi = (int)std::lround(kTwoPi / ghost_size);
  const double geta = 2 * eta_max / g_eta, gph = kTwoPi / g_phi, garea = geta * gph;

  std::vector<double> ghost_pt((std::size_t)g_eta * g_phi);
  for (int ie = 0; ie < g_eta; ++ie)
    for (int ip = 0; ip < g_phi; ++ip)
      ghost_pt[(std::size_t)ie * g_phi + ip] = bkg((ip + 0.5) * gph) * garea;

  const int reach_e = (int)std::ceil(max_dr / geta) + 1;
  const int reach_p = (int)std::ceil(max_dr / gph) + 1;

  struct Pair { double d; std::size_t i, g; };
  std::vector<Pair> pairs;
  pairs.reserve(n * 24);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(eta[i]) >= eta_max || pt[i] <= 0) continue;
    const double p = wrap_phi(phi[i]);
    const int ce = (int)((eta[i] + eta_max) / geta), cp = (int)(p / gph);
    const double w = alpha == 0.0 ? 1.0 : std::pow(pt[i], alpha);
    for (int de = -reach_e; de <= reach_e; ++de) {
      const int ie = ce + de;
      if (ie < 0 || ie >= g_eta) continue;
      const double ge = -eta_max + (ie + 0.5) * geta;
      for (int dp = -reach_p; dp <= reach_p; ++dp) {
        const int ip = ((cp + dp) % g_phi + g_phi) % g_phi;
        const double gp = (ip + 0.5) * gph;
        const double dr = std::hypot(eta[i] - ge, dphi(p, gp));
        if (dr < max_dr) pairs.push_back({w * dr, i, (std::size_t)ie * g_phi + ip});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });

  std::vector<double> out(pt, pt + n);
  for (const auto& pr : pairs) {
    double& a = out[pr.i];
    double& g = ghost_pt[pr.g];
    if (a <= 0 || g <= 0) continue;
    if (a >= g) { a -= g; g = 0; } else { g -= a; a = 0; }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 3. Starlet wavelet denoising
// ---------------------------------------------------------------------------
struct Image {
  int ne = 0, np = 0;
  std::vector<double> v;
  Image() = default;
  Image(int e, int p, double x = 0.0) : ne(e), np(p), v((std::size_t)e * p, x) {}
  double& at(int e, int p) { return v[(std::size_t)e * np + p]; }
  double at(int e, int p) const { return v[(std::size_t)e * np + p]; }
};

// One B3-spline smoothing pass with dilation d: circular in phi, mirrored in eta.
inline Image b3_smooth(const Image& in, int d) {
  static const double h[5] = {1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16};
  auto mirror = [](int i, int n) {
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  };
  Image tmp(in.ne, in.np), out(in.ne, in.np);
  for (int e = 0; e < in.ne; ++e)
    for (int p = 0; p < in.np; ++p) {
      double s = 0;
      for (int k = -2; k <= 2; ++k) s += h[k + 2] * in.at(e, ((p + k * d) % in.np + in.np) % in.np);
      tmp.at(e, p) = s;
    }
  for (int e = 0; e < in.ne; ++e)
    for (int p = 0; p < in.np; ++p) {
      double s = 0;
      for (int k = -2; k <= 2; ++k) s += h[k + 2] * tmp.at(mirror(e + k * d, in.ne), p);
      out.at(e, p) = s;
    }
  return out;
}

struct StarletResult {
  std::vector<double> weight;      // per-particle signal fraction f in [0, 1]
  Image input, signal, bkg;        // pixel images [GeV]
  std::vector<double> sigma;       // per-scale noise estimate (J details + coarse)
};

// k_sigma: threshold in units of each scale's MAD sigma.
// soft: soft thresholding (shrink) instead of hard (keep or kill).
// carry_levels: B3 smoothing passes used to spread negative residuals onto
//   nearby positive pixels (3 passes ~ 0.35 in eta-phi); 0 = plain clipping.
inline StarletResult starlet_denoise(const double* pt, const double* eta, const double* phi,
                                     std::size_t n, const BackgroundMap& bkg,
                                     double eta_max = 3.0, int n_eta = 120, int n_phi = 126,
                                     int n_scales = 5, double k_sigma = 3.0, bool soft = false,
                                     int carry_levels = 3) {
  const double deta = 2 * eta_max / n_eta, dph = kTwoPi / n_phi, apix = deta * dph;
  StarletResult r;
  r.input = Image(n_eta, n_phi);
  r.bkg = Image(n_eta, n_phi);
  std::vector<int> pix(n, -1);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(eta[i]) >= eta_max) continue;
    const int ie = std::min(n_eta - 1, (int)((eta[i] + eta_max) / deta));
    const int ip = std::min(n_phi - 1, (int)(wrap_phi(phi[i]) / dph));
    r.input.at(ie, ip) += pt[i];
    pix[i] = ie * n_phi + ip;
  }
  // Residual image: input minus the smooth background map (mean + flow).
  Image res(n_eta, n_phi);
  for (int ip = 0; ip < n_phi; ++ip) {
    const double b = bkg((ip + 0.5) * dph) * apix;
    for (int ie = 0; ie < n_eta; ++ie) {
      r.bkg.at(ie, ip) = b;
      res.at(ie, ip) = r.input.at(ie, ip) - b;
    }
  }
  // Decompose: res = sum_j w_j + c_J. Threshold each plane at k * sigma_j.
  auto shrink = [&](double w, double t) {
    if (std::abs(w) <= t) return 0.0;
    return soft ? (w > 0 ? w - t : w + t) : w;
  };
  r.signal = Image(n_eta, n_phi);
  Image c = res;
  for (int j = 0; j < n_scales; ++j) {
    Image cn = b3_smooth(c, 1 << j);
    std::vector<double> w(c.v.size());
    for (std::size_t q = 0; q < w.size(); ++q) w[q] = c.v[q] - cn.v[q];
    const double s = mad_sigma(w);
    r.sigma.push_back(s);
    for (std::size_t q = 0; q < w.size(); ++q) r.signal.v[q] += shrink(w[q], k_sigma * s);
    c = std::move(cn);
  }
  const double sc = mad_sigma(c.v);
  r.sigma.push_back(sc);
  const double cmed = median_of(c.v);
  for (std::size_t q = 0; q < c.v.size(); ++q) r.signal.v[q] += shrink(c.v[q] - cmed, k_sigma * sc);

  // Physical constraint 0 <= signal <= input, with LOCAL flux conservation.
  // Clipping negative pixels to zero alone biases every jet upward (the
  // background noise is skewed and ~1 particle per pixel), so the negative
  // deficit is instead removed from nearby positive pixels: P and N (positive
  // and negative parts) are smoothed over ~carry_scale pixels, and each positive
  // pixel keeps the fraction 1 - N_local / P_local.
  {
    Image P(n_eta, n_phi), N(n_eta, n_phi);
    for (std::size_t q = 0; q < r.signal.v.size(); ++q) {
      P.v[q] = std::max(r.signal.v[q], 0.0);
      N.v[q] = std::max(-r.signal.v[q], 0.0);
    }
    Image Ps = P, Ns = N;
    for (int j = 0; j < carry_levels; ++j) {
      Ps = b3_smooth(Ps, 1 << j);
      Ns = b3_smooth(Ns, 1 << j);
    }
    for (std::size_t q = 0; q < r.signal.v.size(); ++q) {
      const double keep = Ps.v[q] > 0 ? std::max(0.0, 1.0 - Ns.v[q] / Ps.v[q]) : 0.0;
      r.signal.v[q] = std::min(P.v[q] * keep, r.input.v[q]);
    }
  }

  r.weight.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (pix[i] < 0) continue;
    const double in = r.input.v[pix[i]];
    r.weight[i] = in > 0 ? r.signal.v[pix[i]] / in : 0.0;
  }
  return r;
}

// ---------------------------------------------------------------------------
// 4. Per-particle posterior (likelihood-ratio) weights
// ---------------------------------------------------------------------------
// Model: background particles are independent, with density b(pT) per unit
// area per pT bin, modulated in phi by rho(phi)/rho_0. Signal particles around
// a jet axis have density s(pT, r) per unit area per pT bin (a template from
// vacuum jets). For a particle at (pT, r):
//     w = s(pT, r) / (s(pT, r) + b(pT) * rho(phi) / rho_0)
// is the probability it is signal. Summing w * pT over a jet is the
// conditional-mean (minimum mean-squared-error) estimate of the jet's signal
// pT under this model. It is only optimal if the model holds: independent
// background, and a template that matches the jet population.

inline int find_bin(const std::vector<double>& edges, double x) {
  if (x < edges.front() || x >= edges.back()) return -1;
  return (int)(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) - 1;
}

// Background pT spectrum per unit area, measured in the event away from jets.
// Returns density per bin [particles / unit area / bin] normalized to rho_0.
inline std::vector<double> background_spectrum(const double* pt, const double* eta,
                                               const double* phi, std::size_t n,
                                               const std::vector<double>& axes_eta,
                                               const std::vector<double>& axes_phi,
                                               const std::vector<double>& pt_edges,
                                               const BackgroundMap& bkg, double eta_max = 3.0,
                                               double exclude_r = 0.8) {
  const int nb = (int)pt_edges.size() - 1;
  std::vector<double> h(nb, 0.0);
  // Area of |eta| < eta_max minus the excluded disks, by a fine Monte Carlo-free grid.
  const int ge = 240, gp = 252;
  double area = 0.0;
  const double cell = (2 * eta_max / ge) * (kTwoPi / gp);
  double flow_norm = 0.0;  // average rho(phi)/rho_0 over the kept area
  for (int ie = 0; ie < ge; ++ie)
    for (int ip = 0; ip < gp; ++ip) {
      const double e = -eta_max + (ie + 0.5) * 2 * eta_max / ge, p = (ip + 0.5) * kTwoPi / gp;
      bool out = true;
      for (std::size_t a = 0; a < axes_eta.size() && out; ++a)
        if (std::hypot(e - axes_eta[a], dphi(p, axes_phi[a])) < exclude_r) out = false;
      if (out) { area += cell; flow_norm += cell * (bkg.a0 > 0 ? bkg(p) / bkg.a0 : 1.0); }
    }
  if (area <= 0) throw std::runtime_error("no background area left after exclusion");
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(eta[i]) >= eta_max) continue;
    bool out = true;
    for (std::size_t a = 0; a < axes_eta.size() && out; ++a)
      if (std::hypot(eta[i] - axes_eta[a], dphi(phi[i], axes_phi[a])) < exclude_r) out = false;
    if (!out) continue;
    const int b = find_bin(pt_edges, pt[i]);
    if (b >= 0) h[b] += 1.0;
  }
  for (auto& x : h) x /= flow_norm;  // flow-weighted area -> density at rho_0
  return h;
}

// sig_template: row-major [n_pt_bins][n_r_bins], signal particles per unit
// area per pT bin at distance r from the axis. Particles beyond the last r
// edge of every axis get w = 0. Each particle uses its nearest axis.
inline std::vector<double> posterior_weights(const double* pt, const double* eta,
                                             const double* phi, std::size_t n,
                                             const std::vector<double>& axes_eta,
                                             const std::vector<double>& axes_phi,
                                             const std::vector<double>& pt_edges,
                                             const std::vector<double>& r_edges,
                                             const std::vector<double>& sig_template,
                                             const std::vector<double>& bkg_spectrum,
                                             const BackgroundMap& bkg) {
  const int npt = (int)pt_edges.size() - 1, nr = (int)r_edges.size() - 1;
  if ((int)sig_template.size() != npt * nr) throw std::invalid_argument("template shape mismatch");
  if ((int)bkg_spectrum.size() != npt) throw std::invalid_argument("spectrum shape mismatch");
  std::vector<double> w(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double rmin = 1e9;
    for (std::size_t a = 0; a < axes_eta.size(); ++a)
      rmin = std::min(rmin, std::hypot(eta[i] - axes_eta[a], dphi(phi[i], axes_phi[a])));
    const int bp = find_bin(pt_edges, pt[i]);
    const int br = find_bin(r_edges, rmin);
    if (br < 0) continue;  // outside every jet
    if (bp < 0) { w[i] = pt[i] >= pt_edges.back() ? 1.0 : 0.0; continue; }  // above range: signal
    const double s = sig_template[(std::size_t)bp * nr + br];
    const double b = bkg_spectrum[bp] * (bkg.a0 > 0 ? bkg(wrap_phi(phi[i])) / bkg.a0 : 1.0);
    // No background observed at this pT: nothing to confuse it with.
    w[i] = b <= 0 ? 1.0 : s / (s + b);
  }
  return w;
}

}  // namespace jetdenoise
