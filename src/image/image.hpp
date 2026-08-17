#pragma once

#include <string>
#include <vector>

#include "../algorithm/stippling.hpp"

namespace stipple {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

bool loadImage(const std::string& path, Image& outImage, std::string& error);
bool saveImagePng(const std::string& path, const Image& image, std::string& error);

bool saveAnimationGif(const std::string& path, const std::vector<Image>& frames, int frameDelayMs,
                       int loopCount, std::string& error);
std::vector<float> computeWeightMap(const Image& image, bool* uniformFallbackUsed = nullptr);
float defaultRadius(int width, int height, int numPoints);
Image renderStipple(int width, int height, const std::vector<Point>& points, float radius);

}
