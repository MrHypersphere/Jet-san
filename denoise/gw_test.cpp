// gw_test.cpp
//
// The same DCT denoiser as the jet test (dct_denoise.hpp), applied to a toy
// gravitational-wave signal: a 1D time series stored as a 1 x N cv::Mat.
//
//   clean = toy compact-binary chirp (frequency and amplitude rising to a
//           merger, then a damped ringdown)
//   noisy = clean + white Gaussian noise
//
// Real detector noise is coloured; whiten the strain first and this toy is
// what the denoiser then sees. The peak amplitude here is far louder relative
// to the noise than a real event, so the result is visible by eye.
//
// Build:  make gw_test
// Run:    ./gw_test [--events N] [--seed S] [--amp PEAK_OVER_SIGMA] [--out plot.png]
//
// Exit status is 0 if the denoised RMSE is below the noisy RMSE.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dct_denoise.hpp"

static const int N = 4096;          // samples
static const double FS = 4096.0;    // sample rate [Hz] -> 1 second of data
static const int PATCH = 64;        // DCT window [samples]

// Toy chirp, peak amplitude `amp`. Inspiral: f = f0 (1 - t/tc)^(-3/8),
// amplitude ~ (1 - t/tc)^(-1/4), up to the merger at tm; then a ringdown.
static cv::Mat1f makeChirp(double amp, double tm) {
    const double f0 = 30.0, tc = tm + 0.02, tau = 0.01;
    auto x = [&](double t) { return 1.0 - t / tc; };
    auto phase = [&](double t) { return 2.0 * CV_PI * f0 * tc * 1.6 * (1.0 - std::pow(x(t), 0.625)); };
    const double fm = f0 * std::pow(x(tm), -0.375);
    const double am = std::pow(x(tm), -0.25);

    cv::Mat1f h(1, N);
    for (int i = 0; i < N; ++i) {
        double t = i / FS;
        double v = t < tm
            ? std::pow(x(t), -0.25) * std::sin(phase(t))
            : am * std::exp(-(t - tm) / tau) * std::sin(phase(tm) + 2.0 * CV_PI * fm * (t - tm));
        h(0, i) = static_cast<float>(amp * v / am);
    }
    return h;
}

static double rmse(const cv::Mat& a, const cv::Mat& b) {
    return cv::norm(a, b, cv::NORM_L2) / std::sqrt(static_cast<double>(a.total()));
}

// Draw one trace into a horizontal band of the canvas.
static void drawTrace(cv::Mat& canvas, const cv::Mat1f& y, int top, int height, double scale,
                      const cv::Scalar& colour, const std::string& label) {
    std::vector<cv::Point> pts;
    for (int i = 0; i < y.cols; ++i) {
        int px = static_cast<int>(static_cast<double>(i) * (canvas.cols - 1) / (y.cols - 1));
        int py = top + height / 2 - static_cast<int>(y(0, i) / scale * height / 2);
        pts.emplace_back(px, py);
    }
    cv::polylines(canvas, pts, false, colour, 1, cv::LINE_AA);
    cv::putText(canvas, label, cv::Point(8, top + 16), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
}

int main(int argc, char** argv) {
    int nEvents = 200;
    unsigned seed = 1;
    double amp = 3.0;  // peak signal amplitude in units of the noise sigma
    std::string out = "gw_test.png";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string v = i + 1 < argc ? argv[++i] : "";
        if (a == "--events" && !v.empty()) nEvents = std::atoi(v.c_str());
        else if (a == "--seed" && !v.empty()) seed = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--amp" && !v.empty()) amp = std::atof(v.c_str());
        else if (a == "--out" && !v.empty()) out = v;
        else nEvents = 0;
    }
    if (nEvents <= 0 || amp <= 0.0) {
        std::fprintf(stderr, "usage: %s [--events N] [--seed S] [--amp PEAK_OVER_SIGMA] [--out plot.png]\n", argv[0]);
        return 2;
    }

    const double sigma = 1.0;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, static_cast<float>(sigma));
    std::uniform_real_distribution<double> mergerTime(0.6, 0.9);

    double sumNoisy = 0.0, sumDen = 0.0, sumMatch = 0.0, sumSigma = 0.0;
    cv::Mat canvas(540, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int ev = 0; ev < nEvents; ++ev) {
        cv::Mat1f clean = makeChirp(amp * sigma, mergerTime(rng));
        cv::Mat1f noisy = clean.clone();
        for (int i = 0; i < N; ++i) noisy(0, i) += gauss(rng);

        // The same two calls the jet test makes.
        double sigmaEst = dn::estimateSigma(noisy);
        cv::Mat1f den = dn::dctDenoise(noisy, sigmaEst, cv::Size(PATCH, 1));

        sumSigma += sigmaEst;
        sumNoisy += rmse(noisy, clean);
        sumDen += rmse(den, clean);
        sumMatch += den.dot(clean) / std::max(1e-12, cv::norm(den) * cv::norm(clean));

        if (ev == 0) {
            double scale = 1.5 * amp * sigma;
            drawTrace(canvas, noisy, 0, 180, scale, cv::Scalar(150, 150, 150), "noisy");
            drawTrace(canvas, clean, 180, 180, scale, cv::Scalar(40, 40, 40), "clean");
            drawTrace(canvas, den, 360, 180, scale, cv::Scalar(214, 120, 42), "DCT denoised");
        }
    }

    double rNoisy = sumNoisy / nEvents, rDen = sumDen / nEvents;
    std::printf("%d events, %d samples at %.0f Hz, peak amplitude %.1f sigma, %d-sample DCT window\n\n",
                nEvents, N, FS, amp, PATCH);
    std::printf("estimated sigma        %.3f   (true %.3f)\n", sumSigma / nEvents, sigma);
    std::printf("RMSE noisy    vs clean %.3f\n", rNoisy);
    std::printf("RMSE denoised vs clean %.3f\n", rDen);
    std::printf("match(denoised, clean) %.3f   (normalised overlap, 1 = identical shape)\n", sumMatch / nEvents);

    if (!cv::imwrite(out, canvas)) {
        std::fprintf(stderr, "could not write %s\n", out.c_str());
        return 2;
    }
    std::printf("\nwrote %s (event 0)\n", out.c_str());

    if (rDen >= rNoisy) {
        std::printf("FAIL: denoising did not reduce the RMSE\n");
        return 1;
    }
    std::printf("PASS: denoised RMSE %.3f < noisy RMSE %.3f\n", rDen, rNoisy);
    return 0;
}
