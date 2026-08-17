#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "algorithm/stippling.hpp"
#include "image/image.hpp"
#include "utils/cli.hpp"
#include "utils/interactive.hpp"
#include "utils/runner.hpp"

#ifdef _MSC_VER
#pragma comment(linker, \
    "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' "  \
    "version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' " \
    "language='*'\"")
#endif

using namespace stipple;

namespace {

constexpr UINT WM_APP_PROGRESS = WM_APP + 1;
constexpr UINT WM_APP_DONE = WM_APP + 2;
constexpr UINT WM_APP_MARQUEE = WM_APP + 3;

enum ControlId : int {
    IDC_EDIT_INPUT = 1000,
    IDC_BTN_BROWSE_INPUT,
    IDC_EDIT_POINTS,
    IDC_EDIT_ITERATIONS,
    IDC_EDIT_EPSILON,
    IDC_EDIT_SEED,
    IDC_EDIT_THREADS,
    IDC_EDIT_RADIUS,
    IDC_EDIT_REPEATS,
    IDC_COMBO_MODE,
    IDC_EDIT_OUTPUT,
    IDC_BTN_BROWSE_OUTPUT,
    IDC_CHECK_ANIMATE,
    IDC_EDIT_ANIM_DELAY,
    IDC_EDIT_ANIM_LOOP,
    IDC_BTN_RUN,
    IDC_PROGRESS,
    IDC_LABEL_PROGRESS,
    IDC_EDIT_RESULTS,
    IDC_LABEL_GPU_STATUS,
};

std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int len = WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, s.data(), len, nullptr, nullptr);
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    int len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(len - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, w.data(), len);
    return w;
}

std::wstring AbsPathW(const std::string& p) {
    std::error_code ec;
    const auto abs = std::filesystem::absolute(p, ec);
    return Widen(ec ? p : abs.string());
}

std::wstring GetText(HWND h) {
    const int len = GetWindowTextLengthW(h);
    std::wstring s(static_cast<size_t>(len), L'\0');
    if (len > 0) GetWindowTextW(h, s.data(), len + 1);
    return s;
}

void SetProgressMarquee(HWND pb, bool marquee) {
    const LONG_PTR style = GetWindowLongPtrW(pb, GWL_STYLE);
    if (marquee) {
        SetWindowLongPtrW(pb, GWL_STYLE, style | PBS_MARQUEE);
        SendMessageW(pb, PBM_SETMARQUEE, TRUE, 30);
    } else {
        SendMessageW(pb, PBM_SETMARQUEE, FALSE, 0);
        SetWindowLongPtrW(pb, GWL_STYLE, style & ~static_cast<LONG_PTR>(PBS_MARQUEE));
    }
}

struct RunResult {
    bool ok = false;
    std::wstring text;
};

struct AppState {
    HWND hMain = nullptr;
    HWND edInput = nullptr, btnBrowseInput = nullptr;
    HWND edPoints = nullptr, edIterations = nullptr, edEpsilon = nullptr;
    HWND edSeed = nullptr, edThreads = nullptr, edRadius = nullptr, edRepeats = nullptr;
    HWND comboMode = nullptr;
    HWND edOutput = nullptr, btnBrowseOutput = nullptr;
    HWND checkAnimate = nullptr, edAnimDelay = nullptr, edAnimLoop = nullptr;
    HWND btnRun = nullptr, progress = nullptr, lblProgress = nullptr;
    HWND edResults = nullptr, lblGpuStatus = nullptr;

    std::vector<Mode> comboModes;
    std::thread worker;
    std::atomic<bool> running{false};
};

struct RunParams {
    std::string inputPath, outputPath;
    int numPoints = 0, maxIterations = 0, threads = 0, repeats = 1;
    int animationDelayMs = 120, animationLoopCount = 0;
    float epsilon = 0.0f, radius = -1.0f;
    uint32_t seed = 42;
    Mode mode = Mode::Serial;
    bool animate = false;
};

std::wstring ToCrlf(const std::wstring& s) {
    std::wstring out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == L'\n' && (i == 0 || s[i - 1] != L'\r')) out += L'\r';
        out += s[i];
    }
    return out;
}

void PostResult(HWND hMain, bool ok, std::wstring text) {
    auto* r = new RunResult{ok, ToCrlf(text)};
    PostMessageW(hMain, WM_APP_DONE, 0, reinterpret_cast<LPARAM>(r));
}

void RunWorker(HWND hMain, RunParams p) {
    Image inputImage;
    std::string loadErr;
    if (!loadImage(p.inputPath, inputImage, loadErr)) {
        PostResult(hMain, false, L"Error: failed to decode input image (" + Widen(loadErr) + L")");
        return;
    }

    bool uniformFallback = false;
    const std::vector<float> weights = computeWeightMap(inputImage, &uniformFallback);

    Params params;
    params.width = inputImage.width;
    params.height = inputImage.height;
    params.numPoints = p.numPoints;
    params.maxIterations = p.maxIterations;
    params.epsilon = p.epsilon;
    params.seed = p.seed;

    const float radius =
        p.radius > 0.0f ? p.radius : defaultRadius(params.width, params.height, params.numPoints);
    const std::vector<Point> initial =
        initializePoints(weights, params.width, params.height, params.numPoints, params.seed);

    std::wostringstream out;
    if (uniformFallback) {
        out << L"Note: input image has no dark pixels; falling back to a uniform point "
               L"distribution.\r\n\r\n";
    }

    if (p.mode == Mode::Benchmark) {
        PostMessageW(hMain, WM_APP_MARQUEE, TRUE, 0);
        AnimationOptions benchAnimation;
        benchAnimation.enabled = p.animate;
        benchAnimation.frameDelayMs = p.animationDelayMs;
        benchAnimation.loopCount = p.animationLoopCount;
        const BenchmarkOutcome bo = runBenchmarkAll(weights, params, initial, p.threads, p.repeats,
                                                     radius, p.outputPath, p.inputPath, benchAnimation);
        for (const std::string& w : bo.warnings) out << L"Note: " << Widen(w) << L"\r\n";
        out << L"\r\n" << Widen(bo.report) << L"\r\n";
        out << L"Benchmark report written to " << AbsPathW(bo.reportPath) << L"\r\n\r\nOutput file(s):\r\n";
        for (const std::string& path : bo.outputPaths) out << L"  - " << AbsPathW(path) << L"\r\n";
        for (const std::string& path : bo.animationPaths) out << L"  - " << AbsPathW(path) << L"\r\n";
        PostResult(hMain, bo.ok, out.str());
        return;
    }

    const int maxIter = params.maxIterations;
    IterationObserver progressObserver = [hMain, maxIter](int idx, const std::vector<Point>&) {
        PostMessageW(hMain, WM_APP_PROGRESS, static_cast<WPARAM>(idx + 1),
                     static_cast<LPARAM>(maxIter));
    };

    AnimationOptions animation;
    animation.enabled = p.animate;
    animation.frameDelayMs = p.animationDelayMs;
    animation.loopCount = p.animationLoopCount;
    animation.outputPath =
        deriveAnimationPath(p.inputPath, std::filesystem::path(p.outputPath).parent_path().string());

    const SingleRunOutcome outcome = runOnce(p.mode, weights, params, initial, p.threads, radius,
                                              p.outputPath, animation, &progressObserver);
    if (!outcome.ok) {
        out << Widen(outcome.error);
        PostResult(hMain, false, out.str());
        return;
    }

    out << Widen(outcome.summaryLine) << L"\r\n";
    if (!outcome.animationWarning.empty()) out << L"Note: " << Widen(outcome.animationWarning) << L"\r\n";
    out << L"\r\nOutput saved to: " << AbsPathW(outcome.outputPath) << L"\r\n";
    if (!outcome.animationPath.empty()) {
        out << L"Animation saved to: " << AbsPathW(outcome.animationPath) << L" ("
            << outcome.animationFrames << L" frames)\r\n";
    }
    PostResult(hMain, true, out.str());
}

HWND CreateLabel(HWND parent, int x, int y, int w, const wchar_t* text) {
    return CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE, x, y, w, 18, parent, nullptr,
                            nullptr, nullptr);
}

HWND CreateEdit(HWND parent, int id, int x, int y, int w, int h, DWORD extraStyle = 0) {
    return CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                            WS_CHILD | WS_VISIBLE | WS_TABSTOP | extraStyle, x, y, w, h, parent,
                            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
}

HWND CreateButton(HWND parent, int id, int x, int y, int w, int h, const wchar_t* text,
                   DWORD extraStyle = BS_PUSHBUTTON) {
    return CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | WS_TABSTOP | extraStyle, x, y, w,
                            h, parent, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr,
                            nullptr);
}

void ApplyDefaultFont(HWND parent) {
    HFONT font = reinterpret_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    EnumChildWindows(
        parent,
        [](HWND h, LPARAM lp) -> BOOL {
            SendMessageW(h, WM_SETFONT, static_cast<WPARAM>(lp), TRUE);
            return TRUE;
        },
        reinterpret_cast<LPARAM>(font));
}

std::wstring BrowseOpenImage(HWND owner) {
    wchar_t buf[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = L"Images (*.png;*.jpg;*.jpeg;*.bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0All files\0*.*\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrInitialDir = L"input";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameW(&ofn) ? buf : L"";
}

std::wstring BrowseSaveFile(HWND owner, const wchar_t* filter, const wchar_t* defExt) {
    wchar_t buf[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = defExt;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    return GetSaveFileNameW(&ofn) ? buf : L"";
}

void CreateControls(AppState* app) {
    HWND hwnd = app->hMain;

    CreateWindowExW(0, L"BUTTON", L"Input / output", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 10, 6, 660, 76,
                     hwnd, nullptr, nullptr, nullptr);
    CreateLabel(hwnd, 22, 28, 70, L"Image:");
    app->edInput = CreateEdit(hwnd, IDC_EDIT_INPUT, 96, 26, 440, 22);
    app->btnBrowseInput = CreateButton(hwnd, IDC_BTN_BROWSE_INPUT, 544, 25, 110, 24, L"Browse...");
    CreateLabel(hwnd, 22, 58, 70, L"Output:");
    app->edOutput = CreateEdit(hwnd, IDC_EDIT_OUTPUT, 96, 56, 440, 22);
    app->btnBrowseOutput = CreateButton(hwnd, IDC_BTN_BROWSE_OUTPUT, 544, 55, 110, 24, L"Browse...");

    CreateWindowExW(0, L"BUTTON", L"Parameters", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 10, 90, 660, 110,
                     hwnd, nullptr, nullptr, nullptr);
    CreateLabel(hwnd, 22, 112, 70, L"Points:");
    app->edPoints = CreateEdit(hwnd, IDC_EDIT_POINTS, 96, 110, 90, 22);
    CreateLabel(hwnd, 200, 112, 70, L"Iterations:");
    app->edIterations = CreateEdit(hwnd, IDC_EDIT_ITERATIONS, 274, 110, 90, 22);
    CreateLabel(hwnd, 378, 112, 60, L"Epsilon:");
    app->edEpsilon = CreateEdit(hwnd, IDC_EDIT_EPSILON, 440, 110, 90, 22);
    CreateLabel(hwnd, 544, 112, 50, L"Seed:");
    app->edSeed = CreateEdit(hwnd, IDC_EDIT_SEED, 590, 110, 64, 22);

    CreateLabel(hwnd, 22, 146, 100, L"Threads (0=auto):");
    app->edThreads = CreateEdit(hwnd, IDC_EDIT_THREADS, 130, 144, 56, 22);
    CreateLabel(hwnd, 200, 146, 110, L"Radius (blank=auto):");
    app->edRadius = CreateEdit(hwnd, IDC_EDIT_RADIUS, 318, 144, 56, 22);
    CreateLabel(hwnd, 388, 146, 130, L"Benchmark repeats:");
    app->edRepeats = CreateEdit(hwnd, IDC_EDIT_REPEATS, 520, 144, 56, 22);

    CreateLabel(hwnd, 22, 180, 60, L"Mode:");
    app->comboMode = CreateWindowExW(WS_EX_CLIENTEDGE, L"COMBOBOX", L"",
                                      WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST, 96, 176, 200,
                                      200, hwnd, reinterpret_cast<HMENU>(IDC_COMBO_MODE), nullptr,
                                      nullptr);
    app->lblGpuStatus = CreateLabel(hwnd, 310, 180, 350, L"");

    CreateWindowExW(0, L"BUTTON", L"Animation", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 10, 206, 660, 50,
                     hwnd, nullptr, nullptr, nullptr);
    app->checkAnimate = CreateWindowExW(0, L"BUTTON", L"Save animated GIF (next to output)",
                                         WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX, 22, 228,
                                         230, 22, hwnd, reinterpret_cast<HMENU>(IDC_CHECK_ANIMATE),
                                         nullptr, nullptr);
    CreateLabel(hwnd, 264, 230, 100, L"Frame delay (ms):");
    app->edAnimDelay = CreateEdit(hwnd, IDC_EDIT_ANIM_DELAY, 368, 228, 60, 22);
    CreateLabel(hwnd, 440, 230, 130, L"Loop count (0=forever):");
    app->edAnimLoop = CreateEdit(hwnd, IDC_EDIT_ANIM_LOOP, 574, 228, 60, 22);

    app->btnRun = CreateButton(hwnd, IDC_BTN_RUN, 10, 264, 140, 30, L"Run", BS_DEFPUSHBUTTON);
    app->progress = CreateWindowExW(0, PROGRESS_CLASS, nullptr, WS_CHILD | WS_VISIBLE, 158, 268, 380, 22,
                                     hwnd, reinterpret_cast<HMENU>(IDC_PROGRESS), nullptr, nullptr);
    app->lblProgress = CreateLabel(hwnd, 546, 270, 120, L"");

    app->edResults =
        CreateEdit(hwnd, IDC_EDIT_RESULTS, 10, 302, 660, 308,
                   WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_WANTRETURN);
    HFONT monospace = CreateFontW(-14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                   OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
                                   FIXED_PITCH | FF_MODERN, L"Consolas");
    SendMessageW(app->edResults, WM_SETFONT, reinterpret_cast<WPARAM>(monospace), TRUE);

    ApplyDefaultFont(hwnd);

    SetWindowTextW(app->edPoints, L"2000");
    SetWindowTextW(app->edIterations, L"30");
    SetWindowTextW(app->edEpsilon, L"0.2");
    SetWindowTextW(app->edSeed, L"42");
    SetWindowTextW(app->edThreads, L"0");
    SetWindowTextW(app->edRepeats, L"1");
    SetWindowTextW(app->edAnimDelay, L"120");
    SetWindowTextW(app->edAnimLoop, L"0");

    app->comboModes = {Mode::Serial, Mode::Cpu};
    SendMessageW(app->comboMode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Serial CPU"));
    SendMessageW(app->comboMode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Parallel CPU (OpenMP)"));

    std::wstring status;
    if (simdAvx2Available()) {
        app->comboModes.push_back(Mode::CpuSimd);
        SendMessageW(app->comboMode, CB_ADDSTRING, 0,
                     reinterpret_cast<LPARAM>(L"Parallel CPU + SIMD (AVX2)"));
        status += L"SIMD: available";
    } else {
        status += L"SIMD: no AVX2";
    }

    if (isGpuRuntimeAvailable()) {
        app->comboModes.push_back(Mode::Gpu);
        SendMessageW(app->comboMode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"GPU (CUDA)"));
        status += L" | GPU: available";
    } else if (isGpuBuildAvailable()) {
        status += L" | GPU: no CUDA device detected";
    } else {
        status += L" | GPU: not available (built without CUDA)";
    }
    SetWindowTextW(app->lblGpuStatus, status.c_str());

    app->comboModes.push_back(Mode::Benchmark);
    SendMessageW(app->comboMode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Benchmark all available"));
    SendMessageW(app->comboMode, CB_SETCURSEL, static_cast<WPARAM>(app->comboModes.size() - 1), 0);
}

Mode CurrentMode(AppState* app) {
    const int sel = static_cast<int>(SendMessageW(app->comboMode, CB_GETCURSEL, 0, 0));
    return (sel >= 0 && sel < static_cast<int>(app->comboModes.size())) ? app->comboModes[sel]
                                                                          : Mode::Benchmark;
}

void UpdateAnimationEnabled(AppState* app) {
    const bool animate = SendMessageW(app->checkAnimate, BM_GETCHECK, 0, 0) == BST_CHECKED;
    for (HWND h : {app->edAnimDelay, app->edAnimLoop}) {
        EnableWindow(h, animate);
    }
}

void SetRunningState(AppState* app, bool running) {
    for (HWND h : {app->edInput,   app->btnBrowseInput, app->edPoints, app->edIterations,
                   app->edEpsilon, app->edSeed,          app->edThreads, app->edRadius,
                   app->edRepeats, app->comboMode,       app->edOutput,  app->btnBrowseOutput,
                   app->checkAnimate, app->edAnimDelay,  app->edAnimLoop}) {
        EnableWindow(h, !running);
    }
    EnableWindow(app->btnRun, !running);
}

bool ReadInt(AppState* app, HWND edit, const char* label, int min, int max, int& out) {
    std::string err;
    if (!parseIntInRange(Narrow(GetText(edit)), min, max, out, err)) {
        MessageBoxW(app->hMain, Widen(std::string(label) + ": " + err).c_str(), L"Invalid input",
                    MB_ICONWARNING);
        SetFocus(edit);
        return false;
    }
    return true;
}

bool ReadFloat(AppState* app, HWND edit, const char* label, float min, float max, float& out) {
    std::string err;
    if (!parseFloatInRange(Narrow(GetText(edit)), min, max, out, err)) {
        MessageBoxW(app->hMain, Widen(std::string(label) + ": " + err).c_str(), L"Invalid input",
                    MB_ICONWARNING);
        SetFocus(edit);
        return false;
    }
    return true;
}

void OnRun(AppState* app) {
    if (app->running.load()) return;

    RunParams p;
    p.inputPath = Narrow(GetText(app->edInput));
    if (p.inputPath.empty() || !std::filesystem::exists(p.inputPath)) {
        MessageBoxW(app->hMain, L"Input image does not exist.", L"Invalid input", MB_ICONWARNING);
        SetFocus(app->edInput);
        return;
    }

    if (!ReadInt(app, app->edPoints, "Points", kInteractiveMinPoints, kInteractiveMaxPoints, p.numPoints))
        return;
    if (!ReadInt(app, app->edIterations, "Iterations", kInteractiveMinIterations,
                 kInteractiveMaxIterations, p.maxIterations))
        return;
    if (!ReadFloat(app, app->edEpsilon, "Epsilon", kInteractiveMinEpsilon, kInteractiveMaxEpsilon,
                    p.epsilon))
        return;

    int seedInt = 42;
    if (!ReadInt(app, app->edSeed, "Seed", 0, INT32_MAX, seedInt)) return;
    p.seed = static_cast<uint32_t>(seedInt);
    if (!ReadInt(app, app->edThreads, "Threads", 0, 1024, p.threads)) return;
    if (!ReadInt(app, app->edRepeats, "Benchmark repeats", 1, 1000, p.repeats)) return;

    const std::wstring radiusText = GetText(app->edRadius);
    if (!radiusText.empty()) {
        if (!ReadFloat(app, app->edRadius, "Radius", 0.01f, 1000.0f, p.radius)) return;
    }

    p.outputPath = Narrow(GetText(app->edOutput));
    if (p.outputPath.empty()) {
        p.outputPath = "output/" + std::filesystem::path(p.inputPath).stem().string() + "_stipple.png";
        SetWindowTextW(app->edOutput, Widen(p.outputPath).c_str());
    }

    p.mode = CurrentMode(app);
    p.animate = SendMessageW(app->checkAnimate, BM_GETCHECK, 0, 0) == BST_CHECKED;
    if (p.animate) {
        if (!ReadInt(app, app->edAnimDelay, "Frame delay", 1, 100000, p.animationDelayMs)) return;
        if (!ReadInt(app, app->edAnimLoop, "Loop count", 0, 100000, p.animationLoopCount)) return;
    }

    app->running.store(true);
    SetRunningState(app, true);
    SetWindowTextW(app->edResults, L"Running...\r\n");
    SendMessageW(app->progress, PBM_SETRANGE32, 0, 100);
    SendMessageW(app->progress, PBM_SETPOS, 0, 0);
    SetWindowTextW(app->lblProgress, L"");

    if (app->worker.joinable()) app->worker.join();
    app->worker = std::thread(RunWorker, app->hMain, p);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* app = reinterpret_cast<AppState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
        case WM_CREATE: {
            auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
            app = reinterpret_cast<AppState*>(cs->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
            app->hMain = hwnd;
            CreateControls(app);
            UpdateAnimationEnabled(app);
            return 0;
        }
        case WM_COMMAND: {
            const int id = LOWORD(wParam);
            const int code = HIWORD(wParam);
            if (id == IDC_BTN_BROWSE_INPUT && code == BN_CLICKED) {
                const std::wstring path = BrowseOpenImage(hwnd);
                if (!path.empty()) {
                    SetWindowTextW(app->edInput, path.c_str());
                    const std::string stem = std::filesystem::path(Narrow(path)).stem().string();
                    SetWindowTextW(app->edOutput, Widen("output/" + stem + "_stipple.png").c_str());
                }
            } else if (id == IDC_BTN_BROWSE_OUTPUT && code == BN_CLICKED) {
                const std::wstring path =
                    BrowseSaveFile(hwnd, L"PNG image\0*.png\0All files\0*.*\0", L"png");
                if (!path.empty()) SetWindowTextW(app->edOutput, path.c_str());
            } else if (id == IDC_CHECK_ANIMATE && code == BN_CLICKED) {
                UpdateAnimationEnabled(app);
            } else if (id == IDC_BTN_RUN && code == BN_CLICKED) {
                OnRun(app);
            }
            return 0;
        }
        case WM_APP_PROGRESS: {
            const int iter = static_cast<int>(wParam);
            const int maxIter = static_cast<int>(lParam);
            SendMessageW(app->progress, PBM_SETRANGE32, 0, maxIter > 0 ? maxIter : 1);
            SendMessageW(app->progress, PBM_SETPOS, static_cast<WPARAM>(iter), 0);
            wchar_t buf[64];
            wsprintfW(buf, L"Iteration %d / %d", iter, maxIter);
            SetWindowTextW(app->lblProgress, buf);
            return 0;
        }
        case WM_APP_MARQUEE: {
            SetProgressMarquee(app->progress, wParam != 0);
            SetWindowTextW(app->lblProgress, wParam != 0 ? L"Running..." : L"");
            return 0;
        }
        case WM_APP_DONE: {
            std::unique_ptr<RunResult> r(reinterpret_cast<RunResult*>(lParam));
            SetProgressMarquee(app->progress, false);
            SendMessageW(app->progress, PBM_SETPOS, r->ok ? 100 : 0, 0);
            SetWindowTextW(app->lblProgress, r->ok ? L"Done" : L"Failed");
            SetWindowTextW(app->edResults, r->text.c_str());
            app->running.store(false);
            SetRunningState(app, false);
            if (app->worker.joinable()) app->worker.join();
            return 0;
        }
        case WM_CLOSE: {
            if (app && app->running.load()) {
                MessageBoxW(hwnd, L"A run is in progress; please wait for it to finish.", L"Please wait",
                            MB_ICONINFORMATION);
                return 0;
            }
            DestroyWindow(hwnd);
            return 0;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

}  // namespace

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&icc);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"StippleGuiWindow";
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    RegisterClassExW(&wc);

    auto app = std::make_unique<AppState>();

    RECT r{0, 0, 690, 630};
    AdjustWindowRect(&r, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"Stipple Me This",
                                 WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                 CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                                 nullptr, nullptr, hInstance, app.get());
    if (!hwnd) return 1;

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (app->worker.joinable()) app->worker.join();
    return static_cast<int>(msg.wParam);
}
