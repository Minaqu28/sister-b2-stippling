#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace stipple {

namespace {

bool normalizeDecodedMat(cv::Mat& mat, std::string& error) {
    if (mat.depth() != CV_8U) {
        double scale = 1.0;
        switch (mat.depth()) {
            case CV_16U:
                scale = 1.0 / 257.0;
                break;
            case CV_32F:
            case CV_64F:
                scale = 255.0;
                break;
            default:
                break;
        }
        cv::Mat converted;
        mat.convertTo(converted, CV_8U, scale);
        mat = converted;
    }

    switch (mat.channels()) {
        case 1:
        case 2:
            break;
        case 3:
            cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
            break;
        case 4:
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
            break;
        default:
            error = "unsupported channel count: " + std::to_string(mat.channels());
            return false;
    }

    if (!mat.isContinuous()) {
        mat = mat.clone();
    }
    return true;
}

}  // namespace

bool loadImage(const std::string& path, Image& outImage, std::string& error) {
    cv::Mat mat;
    try {
        mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    if (mat.empty()) {
        error = "failed to decode image (unsupported format, or file unreadable/corrupt)";
        return false;
    }
    if (!normalizeDecodedMat(mat, error)) {
        return false;
    }

    outImage.width = mat.cols;
    outImage.height = mat.rows;
    outImage.channels = mat.channels();
    const size_t byteCount = mat.total() * static_cast<size_t>(mat.channels());
    outImage.pixels.assign(mat.data, mat.data + byteCount);
    return true;
}

bool saveImagePng(const std::string& path, const Image& image, std::string& error) {
    std::filesystem::path fsPath(path);
    if (fsPath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(fsPath.parent_path(), ec);
        if (ec) {
            error = "failed to create output directory: " + ec.message();
            return false;
        }
    }

    if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.pixels.empty()) {
        error = "refusing to write an empty image";
        return false;
    }
    cv::Mat wrapped(image.height, image.width, CV_8UC(image.channels),
                     const_cast<unsigned char*>(image.pixels.data()));
    cv::Mat encodable;
    switch (image.channels) {
        case 3:
            cv::cvtColor(wrapped, encodable, cv::COLOR_RGB2BGR);
            break;
        case 4:
            cv::cvtColor(wrapped, encodable, cv::COLOR_RGBA2BGRA);
            break;
        default:
            encodable = wrapped;
            break;
    }

    try {
        if (!cv::imwrite(path, encodable)) {
            error = "OpenCV failed to encode the output image";
            return false;
        }
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

bool saveAnimationGif(const std::string& path, const std::vector<Image>& frames, int frameDelayMs,
                       int loopCount, std::string& error) {
    if (frames.empty()) {
        error = "no frames to write";
        return false;
    }

    const Image& first = frames.front();
    if (first.width <= 0 || first.height <= 0 || first.channels <= 0) {
        error = "first frame has invalid dimensions";
        return false;
    }

    std::filesystem::path fsPath(path);
    if (fsPath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(fsPath.parent_path(), ec);
        if (ec) {
            error = "failed to create output directory: " + ec.message();
            return false;
        }
    }
    const int delay = std::max(10, frameDelayMs);

    cv::Animation animation(loopCount);
    animation.frames.reserve(frames.size());
    animation.durations.reserve(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        const Image& f = frames[i];
        if (f.width != first.width || f.height != first.height || f.channels != first.channels) {
            error = "frame " + std::to_string(i) + " does not match the first frame's dimensions";
            return false;
        }
        const size_t expected =
            static_cast<size_t>(f.width) * static_cast<size_t>(f.height) * static_cast<size_t>(f.channels);
        if (f.pixels.size() != expected) {
            error = "frame " + std::to_string(i) + " has an inconsistent pixel buffer";
            return false;
        }

        cv::Mat wrapped(f.height, f.width, CV_8UC(f.channels),
                         const_cast<unsigned char*>(f.pixels.data()));
        cv::Mat encodable;
        switch (f.channels) {
            case 3:
                cv::cvtColor(wrapped, encodable, cv::COLOR_RGB2BGR);
                break;
            case 4:
                cv::cvtColor(wrapped, encodable, cv::COLOR_RGBA2BGRA);
                break;
            default:
                encodable = wrapped.clone();
                break;
        }
        animation.frames.push_back(std::move(encodable));
        animation.durations.push_back(delay);
    }

    try {
        if (!cv::imwriteanimation(path, animation)) {
            error = "OpenCV failed to encode the animation (is the extension a format it can write?)";
            return false;
        }
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

std::vector<float> computeWeightMap(const Image& image, bool* uniformFallbackUsed) {
    const size_t pixelCount = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    std::vector<float> weights(pixelCount, 0.0f);

    for (size_t i = 0; i < pixelCount; ++i) {
        const unsigned char* px = &image.pixels[i * static_cast<size_t>(image.channels)];
        float r, g, b;
        switch (image.channels) {
            case 1:
                r = g = b = static_cast<float>(px[0]);
                break;
            case 2: {
                const float gray = static_cast<float>(px[0]);
                const float alpha = static_cast<float>(px[1]) / 255.0f;
                const float composited = alpha * gray + (1.0f - alpha) * 255.0f;
                r = g = b = composited;
                break;
            }
            case 4: {
                const float alpha = static_cast<float>(px[3]) / 255.0f;
                r = alpha * static_cast<float>(px[0]) + (1.0f - alpha) * 255.0f;
                g = alpha * static_cast<float>(px[1]) + (1.0f - alpha) * 255.0f;
                b = alpha * static_cast<float>(px[2]) + (1.0f - alpha) * 255.0f;
                break;
            }
            case 3:
            default:
                r = static_cast<float>(px[0]);
                g = static_cast<float>(px[1]);
                b = static_cast<float>(px[2]);
                break;
        }
        const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        const float weight = std::clamp(1.0f - gray / 255.0f, 0.0f, 1.0f);
        weights[i] = weight;
    }

    float maxWeight = 0.0f;
    for (float w : weights) maxWeight = std::max(maxWeight, w);

    const bool fallback = maxWeight <= 1e-6f;
    if (uniformFallbackUsed) *uniformFallbackUsed = fallback;
    if (fallback) {
        std::fill(weights.begin(), weights.end(), 1.0f);
    }
    return weights;
}

float defaultRadius(int width, int height, int numPoints) {
    if (numPoints <= 0) return 1.0f;
    const double area = static_cast<double>(width) * static_cast<double>(height);
    const double spacing = std::sqrt(area / (static_cast<double>(numPoints) * 3.14159265358979));
    return static_cast<float>(std::clamp(spacing * 0.45, 0.4, 6.0));
}

Image renderStipple(int width, int height, const std::vector<Point>& points, float radius) {
    Image canvas;
    canvas.width = width;
    canvas.height = height;
    canvas.channels = 3;
    canvas.pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 255);

    const float r = std::max(radius, 0.1f);
    const float rSq = r * r;
    const int spread = static_cast<int>(std::ceil(r));

    for (const Point& p : points) {
        const float cx = std::clamp(p.x, 0.0f, static_cast<float>(width));
        const float cy = std::clamp(p.y, 0.0f, static_cast<float>(height));
        const int minX = std::max(0, static_cast<int>(std::floor(cx)) - spread);
        const int maxX = std::min(width - 1, static_cast<int>(std::ceil(cx)) + spread);
        const int minY = std::max(0, static_cast<int>(std::floor(cy)) - spread);
        const int maxY = std::min(height - 1, static_cast<int>(std::ceil(cy)) + spread);

        for (int y = minY; y <= maxY; ++y) {
            const float dy = (static_cast<float>(y) + 0.5f) - cy;
            for (int x = minX; x <= maxX; ++x) {
                const float dx = (static_cast<float>(x) + 0.5f) - cx;
                if (dx * dx + dy * dy <= rSq) {
                    const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
                    canvas.pixels[idx] = 0;
                    canvas.pixels[idx + 1] = 0;
                    canvas.pixels[idx + 2] = 0;
                }
            }
        }
    }
    return canvas;
}

}
