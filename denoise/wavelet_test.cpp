// wavelet_test.cpp
//
// Wavelet analysis of toy jets in a thermal background (toy_jets.hpp), using
// the starlet transform in wavelet_denoise.hpp. For every event it denoises
// the image and then quantifies both parts, scale by scale:
//
//   background  rho, the level per unit area (from the smooth plane), and the
//               fluctuation sigma_j of every wavelet scale
//   jet         how far it stands above the background on every scale, how its
//               wavelet energy is shared between scales, and its pT and girth
//               inside the cone R compared with the truth
//
// Girth = sum(pT_i r_i) / sum(pT_i) over pixels in the cone, r_i measured from
// the image centre (the true jet axis); same definition as make_jets.py.
//
// Build:  make wavelet_test
// Run:    ./wavelet_test [--events N] [--seed S] [--jet-pt GEV] [--scales J]
//                        [--k K] [--k-finest K] [--iterations N] [--out panel.png]
//
// Exit status is 0 if rho is within 5% of the truth and the denoised image is
// closer to the clean one than background subtraction alone.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "toy_jets.hpp"
#include "wavelet_denoise.hpp"

struct Stat {
    double s = 0.0, s2 = 0.0;
    int n = 0;
    void add(double x) { s += x; s2 += x * x; ++n; }
    double mean() const { return n ? s / n : 0.0; }
    double sd() const { return n ? std::sqrt(std::max(0.0, s2 / n - mean() * mean())) : 0.0; }
};

// pT and girth of an image inside the cone.
struct JetShape { double pt, girth; };
static JetShape measure(const cv::Mat1f& img, const cv::Mat1f& cone, const cv::Mat1f& radius) {
    double pt = img.dot(cone);
    return {pt, pt != 0.0 ? img.mul(cone).dot(radius) / pt : 0.0};
}

int main(int argc, char** argv) {
    toy::Config c;
    dn::WaveletOptions opt;
    std::string out = "wavelet_test.png";
    bool ok = true;
    for (int i = 1; i < argc && ok; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) { ok = false; break; }
        const char* v = argv[++i];
        if (a == "--events") c.nEvents = std::atoi(v);
        else if (a == "--seed") c.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        else if (a == "--jet-pt") c.jetPt = std::atof(v);
        else if (a == "--scales") opt.nScales = std::atoi(v);
        else if (a == "--k") opt.k = std::atof(v);
        else if (a == "--k-finest") opt.kFinest = std::atof(v);
        else if (a == "--iterations") opt.iterations = std::atoi(v);
        else if (a == "--out") out = v;
        else ok = false;
    }
    if (!ok || c.nEvents <= 0 || c.jetPt <= 0.0 || opt.nScales < 1 || opt.nScales > 4 || opt.iterations < 1) {
        std::fprintf(stderr,
                     "usage: %s [--events N] [--seed S] [--jet-pt GEV] [--scales 1-4]\n"
                     "       [--k K] [--k-finest K] [--iterations N] [--out panel.png]\n", argv[0]);
        return 2;
    }

    const int J = opt.nScales;
    const cv::Mat1f radius = toy::radiusMap(c);
    const cv::Mat1f cone = toy::coneMask(c);
    const std::vector<double> gain = dn::starletNoiseGain(J);

    // Truth for the background model: compound Poisson, pT ~ Gamma(shape 2).
    const double trueRho = c.bkgDensity * c.bkgMeanPt;
    const double trueSigma = std::sqrt(c.bkgDensity * c.pixelArea() * 1.5 * c.bkgMeanPt * c.bkgMeanPt);

    Stat rho, rmseWavelet, rmseSubtracted;
    Stat truePt, trueGirth, dPtW, dGirthW, dPtS, dGirthS;
    std::vector<Stat> sigma(J), peak(J), signif(J), nSig(J), share(J);
    std::vector<cv::Mat> tiles;

    std::mt19937 rng(c.seed);
    for (int ev = 0; ev < c.nEvents; ++ev) {
        cv::Mat1f clean = toy::makeJet(c, rng);
        cv::Mat1f noisy = clean + toy::makeBackground(c, rng);

        dn::WaveletResult w = dn::waveletDenoise(noisy, opt);
        cv::Mat1f subtracted = noisy - w.background;  // background removed, nothing denoised

        rho.add(w.rho / c.pixelArea());
        double energy = 0.0;
        for (const dn::WaveletScale& s : w.scales) energy += s.jetEnergy;
        for (int j = 0; j < J; ++j) {
            const dn::WaveletScale& s = w.scales[j];
            sigma[j].add(s.sigma);
            peak[j].add(s.peak);
            signif[j].add(s.peakSignificance);
            nSig[j].add(s.nSignificant);
            share[j].add(energy > 0.0 ? s.jetEnergy / energy : 0.0);
        }

        JetShape t = measure(clean, cone, radius);
        JetShape mw = measure(w.jet, cone, radius);
        JetShape ms = measure(subtracted, cone, radius);
        truePt.add(t.pt);
        trueGirth.add(t.girth);
        dPtW.add(mw.pt - t.pt);
        dGirthW.add(mw.girth - t.girth);
        dPtS.add(ms.pt - t.pt);
        dGirthS.add(ms.girth - t.girth);
        rmseWavelet.add(cv::norm(w.jet, clean, cv::NORM_L2) / c.npix);
        rmseSubtracted.add(cv::norm(subtracted, clean, cv::NORM_L2) / c.npix);

        if (ev == 0) {
            double vmax = 0.0;
            cv::minMaxLoc(clean, nullptr, &vmax);
            tiles.push_back(toy::tile(clean, vmax, "clean"));
            tiles.push_back(toy::tile(noisy, vmax, "noisy"));
            dn::Starlet planes = dn::starletTransform(noisy, J);
            for (int j = 0; j < J; ++j)
                tiles.push_back(toy::tile(planes.detail[j], w.scales[j].peak, "scale " + std::to_string(j + 1)));
            tiles.push_back(toy::tile(w.jet, vmax, "wavelet jet"));
        }
    }

    std::printf("%d events, jet pT %.0f GeV, R = %.1f, %dx%d pixels of %.3f\n", c.nEvents, c.jetPt, c.R,
                c.npix, c.npix, c.pixelSize());
    std::printf("starlet transform, %d scales, threshold %.1f sigma (%.1f on scale 1), %d iterations\n\n",
                J, opt.k, opt.kFinest, opt.iterations);

    std::printf("BACKGROUND\n");
    std::printf("  rho from the smooth plane  %7.1f +- %.1f GeV per unit area   (true %.1f)\n",
                rho.mean(), rho.sd(), trueRho);
    std::printf("  fluctuation per scale; 'white-equiv' is the per-pixel white noise that would\n"
                "  give that sigma_j (true per-pixel sigma %.2f GeV, flat across scales = uncorrelated)\n\n",
                trueSigma);
    std::printf("  %-5s %9s %18s %20s\n", "scale", "size", "sigma_j [GeV/px]", "white-equiv [GeV/px]");
    for (int j = 0; j < J; ++j)
        std::printf("  %-5d %9.2f %18.3f %20.2f\n", j + 1, (1 << (j + 1)) * c.pixelSize(), sigma[j].mean(),
                    sigma[j].mean() / gain[j]);

    std::printf("\nJET, scale by scale\n");
    std::printf("  %-5s %9s %14s %16s %15s %14s\n", "scale", "size", "peak [GeV/px]", "peak / sigma_j",
                "significant px", "energy share");
    for (int j = 0; j < J; ++j)
        std::printf("  %-5d %9.2f %14.2f %16.1f %15.1f %13.0f%%\n", j + 1, (1 << (j + 1)) * c.pixelSize(),
                    peak[j].mean(), signif[j].mean(), nSig[j].mean(), 100.0 * share[j].mean());

    std::printf("\nJET, inside the cone (bias +- event-to-event spread, against the truth)\n");
    std::printf("  %-10s %10s %22s %26s\n", "quantity", "truth", "wavelet denoised", "background subtracted");
    std::printf("  %-10s %10.1f %12.1f +- %-7.1f %14.1f +- %.1f\n", "pT [GeV]", truePt.mean(), dPtW.mean(),
                dPtW.sd(), dPtS.mean(), dPtS.sd());
    std::printf("  %-10s %10.3f %12.3f +- %-7.3f %14.3f +- %.3f\n", "girth", trueGirth.mean(),
                dGirthW.mean(), dGirthW.sd(), dGirthS.mean(), dGirthS.sd());
    std::printf("  %-10s %10s %12.3f %25.3f\n", "RMSE/px", "", rmseWavelet.mean(), rmseSubtracted.mean());

    cv::Mat panel;
    cv::hconcat(tiles, panel);
    if (!cv::imwrite(out, panel)) {
        std::fprintf(stderr, "could not write %s\n", out.c_str());
        return 2;
    }
    std::printf("\nwrote %s (event 0: clean, noisy, wavelet scales, denoised jet)\n", out.c_str());

    bool rhoOk = std::fabs(rho.mean() - trueRho) < 0.05 * trueRho;
    bool rmseOk = rmseWavelet.mean() < rmseSubtracted.mean();
    if (!rhoOk) std::printf("FAIL: rho %.1f is more than 5%% from the true %.1f\n", rho.mean(), trueRho);
    if (!rmseOk) std::printf("FAIL: wavelet denoising did not beat background subtraction on RMSE\n");
    if (rhoOk && rmseOk) std::printf("PASS: rho within 5%% of truth, denoised RMSE below subtraction-only RMSE\n");
    return rhoOk && rmseOk ? 0 : 1;
}
