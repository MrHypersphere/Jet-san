// dct_denoise.hpp
//
// Domain-agnostic DCT denoiser (sliding-patch DCT hard thresholding,
// Yu & Sapiro, "DCT image denoising: a simple and effective image denoising
// algorithm", IPOL 2011, Algorithm 1), written on top of OpenCV.
//
//   if the image is coloured: decorrelate the colour channels
//   for every sliding patch of each channel:
//       2D DCT  ->  zero coefficients with |c| < 3 sigma  ->  inverse 2D DCT
//   average the overlapping patches back into the channel
//   if the image is coloured: undo the colour decorrelation
//
// Nothing here knows what the data is. It assumes only that
//   (a) the noise is additive, zero-mean, white and Gaussian with std sigma, and
//   (b) the signal is sparse in a local DCT basis (smooth or oscillatory).
// A 1D signal (a time series) is a 1 x N image with a 1 x L patch.
//
// Data that breaks (a) must be brought into that form first: subtract a
// non-zero noise mean (jets: the background pedestal), whiten coloured noise
// (gravitational-wave strain: divide by the amplitude spectral density).

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

namespace dn {

// Robust noise estimate from neighbouring-sample differences:
//   sigma = 1.4826 x median(|x[i+1] - x[i]|) / sqrt(2)
// Differencing removes anything that varies slowly from sample to sample, so
// a smooth signal does not inflate the estimate; the median ignores the few
// samples where it does. Uses the first channel.
inline double estimateSigma(const cv::Mat& img) {
    cv::Mat f;
    img.convertTo(f, CV_32F);
    const int ch = f.channels();
    std::vector<float> v;
    v.reserve(f.total());
    for (int r = 0; r < f.rows; ++r) {
        const float* p = f.ptr<float>(r);
        for (int c = 0; c + 1 < f.cols; ++c) v.push_back(std::fabs(p[(c + 1) * ch] - p[c * ch]));
    }
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return 1.4826 * v[v.size() / 2] / std::sqrt(2.0);
}

// Denoise a 1- or 3-channel image (any depth; returned as CV_32F).
// Patch sides must be even or 1 (a cv::dct restriction) and are clipped to
// the image, so the default 8x8 patch becomes 1x8 on a 1 x N signal.
inline cv::Mat dctDenoise(const cv::Mat& noisy, double sigma, cv::Size patch = cv::Size(8, 8)) {
    CV_Assert(noisy.channels() == 1 || noisy.channels() == 3);
    patch.width = std::min(patch.width, noisy.cols);
    patch.height = std::min(patch.height, noisy.rows);
    CV_Assert(patch.width == 1 || patch.width % 2 == 0);
    CV_Assert(patch.height == 1 || patch.height % 2 == 0);

    cv::Mat img;
    noisy.convertTo(img, CV_32F);

    // Orthonormal 3-point DCT across the colour channels: decorrelates them
    // while leaving the noise white with the same sigma in every channel.
    const float a = 1.0f / std::sqrt(3.0f), b = 1.0f / std::sqrt(2.0f), c = 1.0f / std::sqrt(6.0f);
    const cv::Matx33f colour(a, a, a,
                             b, 0, -b,
                             c, -2 * c, c);
    const bool isColour = img.channels() == 3;
    if (isColour) cv::transform(img, img, colour);

    const float threshold = static_cast<float>(3.0 * sigma);
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (cv::Mat& ch : channels) {
        cv::Mat1f sum = cv::Mat1f::zeros(ch.size());
        cv::Mat1f count = cv::Mat1f::zeros(ch.size());
        cv::Mat1f coef, rec;
        for (int y = 0; y + patch.height <= ch.rows; ++y)
            for (int x = 0; x + patch.width <= ch.cols; ++x) {
                cv::Rect roi(x, y, patch.width, patch.height);
                cv::dct(ch(roi), coef);
                coef.setTo(0.0f, cv::abs(coef) < threshold);
                cv::idct(coef, rec);
                sum(roi) += rec;
                count(roi) += 1.0f;
            }
        ch = sum / count;
    }
    cv::merge(channels, img);

    if (isColour) cv::transform(img, img, colour.t());
    return img;
}

}  // namespace dn
