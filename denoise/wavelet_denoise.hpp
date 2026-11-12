// wavelet_denoise.hpp
//
// Wavelet analysis of jet images: separates a jet from a fluctuating
// background and puts numbers on both, one angular scale at a time.
//
// Transform: the starlet (isotropic, undecimated "a trous" wavelet transform
// with the B3-spline kernel [1 4 6 4 1]/16). It splits an image exactly into
//
//     image = w_1 + w_2 + ... + w_J + c_J
//
// where the detail plane w_j holds structure about 2^j pixels across and the
// smooth plane c_J holds everything broader. Every plane has the size of the
// image, so a coefficient belongs to a definite place AND a definite scale.
//
// Why it suits jets in a heavy-ion background:
//   - a flat background pedestal lands entirely in c_J; the w_j never see it
//   - uncorrelated background fluctuations shrink quickly with scale, while a
//     jet is a coherent deposit that stays large on the scales of its core
//
// Method (multiresolution support, Starck & Murtagh):
//   1. transform the image
//   2. measure the background fluctuation sigma_j of every detail plane
//      (median absolute deviation, which ignores the few pixels under the jet)
//   3. mark coefficients with w_j > k sigma_j as significant: the "support"
//   4. rebuild the jet from the significant coefficients only, iterating so
//      the result reproduces the data on the support, and keeping it >= 0
//   5. background map = smooth plane of (image - jet)
//
// Choosing J: anything broader than 2^J pixels is called background, so set
// 2^J pixels to about the jet radius.
//
// Choosing k: the thresholds are well above the Gaussian 3 sigma because a
// particle background is not Gaussian. Its pT spectrum has an exponential
// tail, strongest on the finest scale, and lower thresholds let dozens of
// single background particles through per image.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace dn {

struct Starlet {
    std::vector<cv::Mat1f> detail;  // w_1 (finest) ... w_J (coarsest)
    cv::Mat1f smooth;               // c_J
};

inline Starlet starletTransform(const cv::Mat1f& img, int nScales) {
    static const float b3[5] = {1 / 16.f, 4 / 16.f, 6 / 16.f, 4 / 16.f, 1 / 16.f};
    Starlet s;
    cv::Mat1f c = img;
    for (int j = 0; j < nScales; ++j) {
        const int step = 1 << j;  // kernel taps sit 2^j pixels apart (the "holes")
        cv::Mat1f kernel = cv::Mat1f::zeros(1, 4 * step + 1);
        for (int i = 0; i < 5; ++i) kernel(0, i * step) = b3[i];
        cv::Mat1f next;
        cv::sepFilter2D(c, next, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
        s.detail.push_back(cv::Mat1f(c - next));
        c = next;
    }
    s.smooth = c;
    return s;
}

// Standard deviation of each detail plane when the input is white noise of
// unit standard deviation (the L2 norm of each wavelet's impulse response).
// sigma_j / gain_j turns a measured plane fluctuation back into the per-pixel
// white noise that would produce it: equal across scales means the background
// is uncorrelated, rising with scale means it has large-scale structure.
inline std::vector<double> starletNoiseGain(int nScales) {
    const int n = (4 << nScales) + 1;  // large enough to hold the coarsest wavelet
    cv::Mat1f delta = cv::Mat1f::zeros(n, n);
    delta(n / 2, n / 2) = 1.0f;
    std::vector<double> gain;
    for (const cv::Mat1f& w : starletTransform(delta, nScales).detail) gain.push_back(cv::norm(w));
    return gain;
}

inline float medianInPlace(std::vector<float>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Robust standard deviation of a zero-centred plane: 1.4826 x median(|w|).
inline double madSigma(const cv::Mat1f& plane) {
    std::vector<float> v;
    v.reserve(plane.total());
    for (float x : plane) v.push_back(std::fabs(x));
    return 1.4826 * medianInPlace(v);
}

struct WaveletOptions {
    int nScales = 3;        // J: detail planes of 2, 4, ..., 2^J pixels
    double k = 4.0;         // significance threshold in units of sigma_j
    double kFinest = 6.0;   // threshold for w_1, where the noise is least Gaussian
    int iterations = 10;    // reconstruction passes
};

struct WaveletScale {
    double sigma = 0.0;         // background fluctuation of this plane, jet removed
    double peak = 0.0;          // largest coefficient (the jet, when there is one)
    double peakSignificance = 0.0;  // peak / sigma
    int nSignificant = 0;       // pixels in the support
    double jetEnergy = 0.0;     // sum of squared coefficients of the rebuilt jet
};

struct WaveletResult {
    cv::Mat1f jet;         // denoised, background-free image (>= 0)
    cv::Mat1f background;  // smooth background map, same units per pixel
    double rho = 0.0;      // median of the background map (level per pixel)
    std::vector<WaveletScale> scales;  // finest first
};

inline WaveletResult waveletDenoise(const cv::Mat1f& img, const WaveletOptions& opt = WaveletOptions()) {
    WaveletResult out;
    out.scales.resize(opt.nScales);

    // Steps 1-3: transform, noise per scale, support.
    Starlet data = starletTransform(img, opt.nScales);
    std::vector<cv::Mat> support(opt.nScales);
    for (int j = 0; j < opt.nScales; ++j) {
        double sigma = madSigma(data.detail[j]);
        support[j] = data.detail[j] > (j == 0 ? opt.kFinest : opt.k) * sigma;
        out.scales[j].nSignificant = cv::countNonZero(support[j]);
        cv::minMaxLoc(data.detail[j], nullptr, &out.scales[j].peak);
    }

    // Step 4: add the significant coefficients of whatever the current jet
    // estimate still fails to explain. One pass alone over-counts: it keeps
    // each peak but drops the many small negative coefficients that surround
    // it. Repeating makes the estimate's own transform match the data on the
    // support, and positivity recovers the part of the jet the wavelets
    // cannot see (its share of c_J).
    out.jet = cv::Mat1f::zeros(img.size());
    for (int it = 0; it < opt.iterations; ++it) {
        Starlet r = it == 0 ? data : starletTransform(cv::Mat1f(img - out.jet), opt.nScales);
        for (int j = 0; j < opt.nScales; ++j) cv::add(out.jet, r.detail[j], out.jet, support[j]);
        cv::max(out.jet, 0.0, out.jet);
    }

    // Step 5: what is left is the background. Its smooth plane is the level
    // map; its detail planes, now free of the jet, give the fluctuations.
    Starlet bkg = starletTransform(cv::Mat1f(img - out.jet), opt.nScales);
    out.background = bkg.smooth;
    std::vector<float> levels(out.background.begin(), out.background.end());
    out.rho = medianInPlace(levels);

    Starlet jet = starletTransform(out.jet, opt.nScales);
    for (int j = 0; j < opt.nScales; ++j) {
        WaveletScale& s = out.scales[j];
        s.sigma = madSigma(bkg.detail[j]);
        s.peakSignificance = s.sigma > 0.0 ? s.peak / s.sigma : 0.0;
        double n = cv::norm(jet.detail[j]);
        s.jetEnergy = n * n;
    }
    return out;
}

}  // namespace dn
