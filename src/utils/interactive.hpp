#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "cli.hpp"

namespace stipple {

constexpr int kInteractiveMinPoints = 1;
constexpr int kInteractiveMaxPoints = 500000;
constexpr int kInteractiveMinIterations = 1;
constexpr int kInteractiveMaxIterations = 10000;
constexpr float kInteractiveMinEpsilon = 0.0f;
constexpr float kInteractiveMaxEpsilon = 10000.0f;

constexpr const char* kInteractiveInputDir = "input";
constexpr const char* kInteractiveOutputDir = "output";

bool isSupportedImageFile(const std::filesystem::path& path);
std::vector<std::filesystem::path> listSupportedImages(const std::string& dir);
enum class FolderStatus { Ready, Created, CreateFailed };
FolderStatus ensureFolderExists(const std::string& dir, std::string& errorOut);
bool parseIntInRange(const std::string& text, int min, int max, int& out, std::string& errorOut);
bool parseFloatInRange(const std::string& text, float min, float max, float& out, std::string& errorOut);
bool isQuitCommand(const std::string& text);

bool promptInt(std::istream& in, std::ostream& out, const std::string& prompt, int min, int max,
               int& result);

bool promptFloat(std::istream& in, std::ostream& out, const std::string& prompt, float min, float max,
                  float& result);

bool promptImageChoice(std::istream& in, std::ostream& out,
                        const std::vector<std::filesystem::path>& images, size_t& selectedIndex);

bool promptMode(std::istream& in, std::ostream& out, Mode& selectedMode);

// Yes/no prompt; a blank line selects `defaultYes`. Returns false if the user
// quits or the stream ends.
bool promptYesNo(std::istream& in, std::ostream& out, const std::string& prompt, bool defaultYes,
                  bool& result);
int runInteractive();

}
