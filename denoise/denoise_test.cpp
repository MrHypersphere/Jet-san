// denoise_test.cpp
//
// Self-contained test of image denoisers on toy jet images (toy_jets.hpp),
// using OpenCV.
//
// Denoisers compared (input: noisy, target: clean):
//   pedestal   subtract the median background level per pixel (area subtraction;
//              pixels may go negative, which keeps the jet pT unbiased)
//   gaussian   cv::GaussianBlur
//   median     cv::medianBlur, 3x3 (the classic salt-and-pepper filter)
//   bilateral  cv::bilateralFilter
//   dct        pedestal subtraction, then sliding-patch DCT thresholding
//              (dct_denoise.hpp, the same routine gw_test.cpp uses)
//   wavelet    starlet wavelet thresholding (wavelet_denoise.hpp); see
//              wavelet_test.cpp for the scale-by-scale numbers
//
// Scores, averaged over events:
//   RMSE       per-pixel root-mean-square error against the clean image
//   pT bias    mean of (pT in cone R after denoising) - (true pT in cone R)
//   pT res     standard deviation of that difference
//
// Build:  make            (needs OpenCV 4: sudo apt install libopencv-dev pkg-config)
// Run:    ./denoise_test [--events N] [--seed S] [--jet-pt GEV] [--out panel.png]
//
// Exit status is 0 if at least one denoiser beats "no denoising" on RMSE.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dct_denoise.hpp"
#include "toy_jets.hpp"
#include "wavelet_denoise.hpp"

struct Config : toy::Config {
    std::string out = "denoise_test.png";
};

// ---------------------------------------------------------------------------
// Denoisers: noisy image in, estimate of the clean image out.
// ---------------------------------------------------------------------------
static cv::Mat1f pedestal(const cv::Mat1f& noisy) {
    // Mean pixel value in each 8x8-pixel patch, then the median over patches:
    // robust against the few patches the jet sits in.
    cv::Mat1f patches;
    cv::resize(noisy, patches, cv::Size(noisy.cols / 8, noisy.rows / 8), 0, 0, cv::INTER_AREA);
    std::vector<float> v(patches.begin(), patches.end());
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return noisy - v[v.size() / 2];
}

static cv::Mat1f gaussian(const cv::Mat1f& noisy) {
    cv::Mat1f out;
    cv::GaussianBlur(noisy, out, cv::Size(0, 0), 1.0);
    return out;
}

static cv::Mat1f median(const cv::Mat1f& noisy) {
    cv::Mat1f out;
    cv::medianBlur(noisy, out, 3);
    return out;
}

static cv::Mat1f bilateral(const cv::Mat1f& noisy) {
    cv::Mat1f out;
    // d = 5 pixels, sigmaColor = 2 GeV, sigmaSpace = 1.5 pixels
    cv::bilateralFilter(noisy, out, 5, 2.0, 1.5);
    return out;
}

// Generic DCT denoiser (dct_denoise.hpp). It expects zero-mean noise, so the
// pedestal comes off first; sigma is then estimated from the image itself.
static cv::Mat1f dct(const cv::Mat1f& noisy) {
    cv::Mat1f flat = pedestal(noisy);
    return dn::dctDenoise(flat, dn::estimateSigma(flat));
}

static cv::Mat1f wavelet(const cv::Mat1f& noisy) { return dn::waveletDenoise(noisy).jet; }

static cv::Mat1f identity(const cv::Mat1f& noisy) { return noisy.clone(); }

struct Method {
    std::string name;
    std::function<cv::Mat1f(const cv::Mat1f&)> run;
    double sumRmse = 0.0, sumD = 0.0, sumD2 = 0.0;
};

static bool parseArgs(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) return false;
        std::string v = argv[++i];
        if (a == "--events") c.nEvents = std::atoi(v.c_str());
        else if (a == "--seed") c.seed = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--jet-pt") c.jetPt = std::atof(v.c_str());
        else if (a == "--out") c.out = v;
        else return false;
    }
    return c.nEvents > 0 && c.jetPt > 0.0;
}

int main(int argc, char** argv) {
    Config c;
    if (!parseArgs(argc, argv, c)) {
        std::fprintf(stderr, "usage: %s [--events N] [--seed S] [--jet-pt GEV] [--out panel.png]\n", argv[0]);
        return 2;
    }

    const cv::Mat1f cone = toy::coneMask(c);

    std::vector<Method> methods = {
        {"none", identity}, {"pedestal", pedestal}, {"gaussian", gaussian},
        {"median", median}, {"bilateral", bilateral}, {"dct", dct}, {"wavelet", wavelet},
    };

    std::mt19937 rng(c.seed);
    std::vector<cv::Mat> tiles;
    for (int ev = 0; ev < c.nEvents; ++ev) {
        cv::Mat1f clean = toy::makeJet(c, rng);
        cv::Mat1f noisy = clean + toy::makeBackground(c, rng);
        double truePt = clean.dot(cone);
        double vmax = 0.0;
        if (ev == 0) {
            cv::minMaxLoc(clean, nullptr, &vmax);
            tiles.push_back(toy::tile(clean, vmax, "clean"));
        }
        for (Method& m : methods) {
            cv::Mat1f out = m.run(noisy);
            double d = out.dot(cone) - truePt;
            m.sumRmse += cv::norm(out, clean, cv::NORM_L2) / c.npix;
            m.sumD += d;
            m.sumD2 += d * d;
            if (ev == 0) tiles.push_back(toy::tile(out, vmax, m.name == "none" ? "noisy" : m.name));
        }
    }

    std::printf("%d events, jet pT %.0f GeV, R = %.1f, %dx%d pixels of %.3f\n\n",
                c.nEvents, c.jetPt, c.R, c.npix, c.npix, c.pixelSize());
    std::printf("%-10s %14s %14s %14s\n", "method", "RMSE [GeV/px]", "pT bias [GeV]", "pT res [GeV]");
    double noneRmse = 0.0, bestRmse = 1e300;
    for (const Method& m : methods) {
        double rmse = m.sumRmse / c.nEvents;
        double bias = m.sumD / c.nEvents;
        double res = std::sqrt(std::max(0.0, m.sumD2 / c.nEvents - bias * bias));
        std::printf("%-10s %14.3f %14.2f %14.2f\n", m.name.c_str(), rmse, bias, res);
        if (m.name == "none") noneRmse = rmse;
        else bestRmse = std::min(bestRmse, rmse);
    }

    cv::Mat panel;
    cv::hconcat(tiles, panel);
    if (!cv::imwrite(c.out, panel)) {
        std::fprintf(stderr, "could not write %s\n", c.out.c_str());
        return 2;
    }
    std::printf("\nwrote %s (event 0: clean, noisy, then each denoiser)\n", c.out.c_str());

    if (bestRmse >= noneRmse) {
        std::printf("FAIL: no denoiser improved on the noisy image\n");
        return 1;
    }
    std::printf("PASS: best denoiser RMSE %.3f < noisy RMSE %.3f\n", bestRmse, noneRmse);
    return 0;
}
