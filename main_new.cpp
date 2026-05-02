#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pet {
namespace {

constexpr int kDefaultImageSize = 250;

struct Config {
    int image_size = kDefaultImageSize;
    int emitter_count = 0;
    int emitter_radius = 0;
    bool animate_forward_projection = false;
    bool show_windows = true;
    bool use_log_display = false;
    std::uint32_t random_seed = std::random_device{}();
    std::string output_dir = "photo-dump";
};

struct Geometry {
    int size = 0;
    double radius = 0.0;
    cv::Point2d center;
};

using ScanLines = std::vector<std::vector<cv::Point2d>>;

void print_usage(const char *program_name)
{
    std::cerr
        << "Usage: " << program_name << " <emitter_count> <emitter_radius> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --size <pixels>        Square image size. Default: " << kDefaultImageSize << "\n"
        << "  --animate <0|1>       Animate forward projection. Default: 0\n"
        << "  --show <0|1>          Show OpenCV windows. Default: 1\n"
        << "  --log-display <0|1>   Use log(1 + abs(x)) before display normalization. Default: 0\n"
        << "  --seed <uint>         Random seed. Default: random_device\n"
        << "  --out <dir>           Output directory. Default: photo-dump\n";
}

int parse_int(const std::string &value, const std::string &name)
{
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(value, &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid integer for " + name + ": " + value);
    }
}

std::uint32_t parse_u32(const std::string &value, const std::string &name)
{
    const int parsed = parse_int(value, name);
    if (parsed < 0) {
        throw std::invalid_argument(name + " must be non-negative");
    }
    return static_cast<std::uint32_t>(parsed);
}

bool parse_bool_flag(const std::string &value, const std::string &name)
{
    const int parsed = parse_int(value, name);
    if (parsed != 0 && parsed != 1) {
        throw std::invalid_argument(name + " must be 0 or 1");
    }
    return parsed == 1;
}

Config parse_config(int argc, char **argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        throw std::invalid_argument("Missing required arguments");
    }

    Config config;
    config.emitter_count = parse_int(argv[1], "emitter_count");
    config.emitter_radius = parse_int(argv[2], "emitter_radius");

    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        auto require_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + name);
            }
            return argv[++i];
        };

        if (option == "--size") {
            config.image_size = parse_int(require_value(option), option);
        } else if (option == "--animate") {
            config.animate_forward_projection = parse_bool_flag(require_value(option), option);
        } else if (option == "--show") {
            config.show_windows = parse_bool_flag(require_value(option), option);
        } else if (option == "--log-display") {
            config.use_log_display = parse_bool_flag(require_value(option), option);
        } else if (option == "--seed") {
            config.random_seed = parse_u32(require_value(option), option);
        } else if (option == "--out") {
            config.output_dir = require_value(option);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    if (config.image_size <= 0) {
        throw std::invalid_argument("image_size must be positive");
    }
    if (config.emitter_count < 0) {
        throw std::invalid_argument("emitter_count must be non-negative");
    }
    if (config.emitter_radius <= 0) {
        throw std::invalid_argument("emitter_radius must be positive");
    }
    if (2 * config.emitter_radius >= config.image_size) {
        throw std::invalid_argument("emitter_radius is too large for image_size");
    }

    return config;
}

Geometry make_geometry(int image_size)
{
    Geometry geometry;
    geometry.size = image_size;
    geometry.radius = static_cast<double>(image_size) / 2.0;
    geometry.center = cv::Point2d((image_size - 1) / 2.0, (image_size - 1) / 2.0);
    return geometry;
}

double squared_distance(const cv::Point2d &a, const cv::Point2d &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

bool inside_detector(const cv::Point2d &point, const Geometry &geometry)
{
    if (point.x < 0.0 || point.y < 0.0 ||
        point.x >= geometry.size || point.y >= geometry.size) {
        return false;
    }
    return squared_distance(point, geometry.center) <= geometry.radius * geometry.radius;
}

cv::Point2d rotate_about(const cv::Point2d &point, const cv::Point2d &origin, double angle_radians)
{
    const double dx = point.x - origin.x;
    const double dy = point.y - origin.y;
    const double c = std::cos(angle_radians);
    const double s = std::sin(angle_radians);
    return cv::Point2d(origin.x + c * dx - s * dy, origin.y + s * dx + c * dy);
}

bool rounded_detector_index(const cv::Point2d &point, const Geometry &geometry, int &x, int &y)
{
    x = static_cast<int>(std::lround(point.x));
    y = static_cast<int>(std::lround(point.y));
    return inside_detector(cv::Point2d(x, y), geometry);
}

cv::Mat make_detector_canvas(const Geometry &geometry)
{
    cv::Mat canvas(geometry.size, geometry.size, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < canvas.rows; ++y) {
        for (int x = 0; x < canvas.cols; ++x) {
            const cv::Point2d point(x, y);
            canvas.at<uchar>(y, x) = inside_detector(point, geometry) ? 0 : 128;
        }
    }
    return canvas;
}

cv::Mat make_emitter_image(const Config &config, const Geometry &geometry)
{
    cv::Mat image = make_detector_canvas(geometry);
    std::mt19937 rng(config.random_seed);
    std::uniform_int_distribution<int> pixel_dist(0, config.image_size - 1);

    const double safe_radius = geometry.radius - config.emitter_radius - 1.0;
    if (safe_radius <= 0.0) {
        throw std::invalid_argument("emitter_radius leaves no valid detector area");
    }

    for (int i = 0; i < config.emitter_count; ++i) {
        cv::Point2d center;
        do {
            center = cv::Point2d(pixel_dist(rng), pixel_dist(rng));
        } while (squared_distance(center, geometry.center) > safe_radius * safe_radius);

        cv::circle(image, center, config.emitter_radius, cv::Scalar(255), -1);
    }

    return image;
}

ScanLines make_initial_scan_lines(const Geometry &geometry)
{
    ScanLines lines(geometry.size);
    for (int x = 0; x < geometry.size; ++x) {
        for (int y = 0; y < geometry.size; ++y) {
            const cv::Point2d point(x, y);
            if (inside_detector(point, geometry)) {
                lines[x].push_back(point);
            }
        }
    }
    return lines;
}

ScanLines rotate_scan_lines(const ScanLines &initial_lines, const Geometry &geometry, double angle_radians)
{
    ScanLines rotated(initial_lines.size());
    for (std::size_t line_index = 0; line_index < initial_lines.size(); ++line_index) {
        rotated[line_index].reserve(initial_lines[line_index].size());
        for (const cv::Point2d &point : initial_lines[line_index]) {
            rotated[line_index].push_back(rotate_about(point, geometry.center, angle_radians));
        }
    }
    return rotated;
}

void draw_scan_lines(const ScanLines &lines, const Geometry &geometry, cv::Mat &canvas, int stride)
{
    canvas = make_detector_canvas(geometry);
    for (std::size_t line_index = 0; line_index < lines.size(); line_index += stride) {
        for (const cv::Point2d &point : lines[line_index]) {
            int x = 0;
            int y = 0;
            if (rounded_detector_index(point, geometry, x, y)) {
                canvas.at<uchar>(y, x) = 128;
            }
        }
    }
}

int count_emitters_on_line(const std::vector<cv::Point2d> &line,
                           const cv::Mat &emitter_image,
                           const Geometry &geometry)
{
    int count = 0;
    for (const cv::Point2d &point : line) {
        int x = 0;
        int y = 0;
        if (rounded_detector_index(point, geometry, x, y) && emitter_image.at<uchar>(y, x) == 255) {
            ++count;
        }
    }
    return count;
}

cv::Mat normalize_for_display(const cv::Mat &src, bool use_log_scale)
{
    if (src.empty()) {
        return cv::Mat();
    }

    cv::Mat f32;
    src.convertTo(f32, CV_32F);

    cv::Mat display_source = f32;
    if (use_log_scale) {
        cv::Mat magnitude;
        cv::absdiff(f32, cv::Scalar::all(0.0f), magnitude);
        cv::log(magnitude + cv::Scalar::all(1.0f), display_source);
    }

    cv::Mat normalized;
    cv::normalize(display_source, normalized, 0.0, 255.0, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    return normalized;
}

void write_image(const Config &config, const std::string &filename, const cv::Mat &image)
{
    if (image.empty()) {
        std::cerr << "Skipping empty image: " << filename << '\n';
        return;
    }

    const std::filesystem::path output_dir = config.output_dir.empty()
        ? std::filesystem::path("photo-dump")
        : std::filesystem::path(config.output_dir);

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create output directory: " + output_dir.string() + " (" + ec.message() + ")");
    }

    const std::filesystem::path full_path = output_dir / filename;
    const std::string path = full_path.string();
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to write image: " + path);
    }
    std::cout << "Wrote image: " << path << '\n';
}

cv::Mat forward_project(const Config &config,
                        const Geometry &geometry,
                        const cv::Mat &emitter_image,
                        const ScanLines &initial_lines,
                        cv::Mat &final_line_canvas)
{
    const int steps = geometry.size;
    const double angle_step = CV_PI / static_cast<double>(steps);
    cv::Mat sinogram(steps, geometry.size, CV_32F, cv::Scalar(0.0f));
    cv::Mat animated_sinogram(steps, geometry.size, CV_8UC1, cv::Scalar(0));

    for (int step = 0; step < steps; ++step) {
        const double angle = angle_step * step;
        const int output_row = steps - 1 - step;
        const ScanLines lines = rotate_scan_lines(initial_lines, geometry, angle);

        for (int detector_col = 0; detector_col < geometry.size; ++detector_col) {
            const int count = count_emitters_on_line(lines[detector_col], emitter_image, geometry);
            sinogram.at<float>(output_row, detector_col) = static_cast<float>(count);
            animated_sinogram.at<uchar>(output_row, detector_col) =
                static_cast<uchar>(std::min(count, 255));
        }

        if (config.animate_forward_projection) {
            draw_scan_lines(lines, geometry, final_line_canvas, 50);
            cv::Mat display;
            cv::hconcat(std::vector<cv::Mat>{emitter_image, final_line_canvas, animated_sinogram}, display);
            cv::imshow("Forward Projection", display);
            cv::waitKey(1);
        }
    }

    final_line_canvas = make_detector_canvas(geometry);
    return sinogram;
}

cv::Mat ramp_filter_rows(const cv::Mat &sinogram)
{
    if (sinogram.type() != CV_32F) {
        throw std::invalid_argument("ramp_filter_rows expects CV_32F input");
    }

    cv::Mat spectrum;
    cv::dft(sinogram, spectrum, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes(2);
    cv::split(spectrum, planes);

    cv::Mat ramp_row(1, spectrum.cols, CV_32F);
    const int nyquist_bin = spectrum.cols / 2;
    const float nyquist = static_cast<float>(nyquist_bin);
    for (int k = 0; k < spectrum.cols; ++k) {
        const int wrapped_frequency = (k <= nyquist_bin) ? k : spectrum.cols - k;
        ramp_row.at<float>(0, k) = nyquist > 0.0f ? static_cast<float>(wrapped_frequency) / nyquist : 0.0f;
    }

    cv::Mat ramp;
    cv::repeat(ramp_row, spectrum.rows, 1, ramp);
    cv::multiply(planes[0], ramp, planes[0]);
    cv::multiply(planes[1], ramp, planes[1]);
    cv::merge(planes, spectrum);

    cv::Mat filtered;
    cv::idft(spectrum, filtered, cv::DFT_ROWS | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    return filtered;
}

cv::Mat dft_magnitude_display(const cv::Mat &sinogram, bool use_log_display)
{
    cv::Mat spectrum;
    cv::dft(sinogram, spectrum, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes(2);
    cv::split(spectrum, planes);

    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    return normalize_for_display(magnitude, use_log_display);
}

cv::Mat backproject(const Geometry &geometry, const cv::Mat &filtered_sinogram, const ScanLines &initial_lines)
{
    if (filtered_sinogram.type() != CV_32F) {
        throw std::invalid_argument("backproject expects CV_32F filtered_sinogram");
    }

    cv::Mat reconstruction(geometry.size, geometry.size, CV_32F, cv::Scalar(0.0f));
    const int steps = geometry.size;
    const double angle_step = CV_PI / static_cast<double>(steps);

    for (int step = 0; step < steps; ++step) {
        const double angle = angle_step * step;
        const int sinogram_row = steps - 1 - step;
        const ScanLines lines = rotate_scan_lines(initial_lines, geometry, angle);

        for (int detector_col = 0; detector_col < geometry.size; ++detector_col) {
            const float measurement = filtered_sinogram.at<float>(sinogram_row, detector_col);
            for (const cv::Point2d &point : lines[detector_col]) {
                int x = 0;
                int y = 0;
                if (rounded_detector_index(point, geometry, x, y)) {
                    reconstruction.at<float>(y, x) += measurement;
                }
            }
        }
    }

    return reconstruction;
}

void show_image(const std::string &window_name, const cv::Mat &image, bool enabled, int delay_ms)
{
    if (!enabled || image.empty()) {
        return;
    }
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, image);
    cv::waitKey(delay_ms);
}

void run(const Config &config)
{
    const Geometry geometry = make_geometry(config.image_size);
    const ScanLines initial_lines = make_initial_scan_lines(geometry);
    cv::Mat line_canvas = make_detector_canvas(geometry);

    const cv::Mat emitter_image = make_emitter_image(config, geometry);

    const double start_time = static_cast<double>(cv::getTickCount());
    const cv::Mat sinogram = forward_project(config, geometry, emitter_image, initial_lines, line_canvas);
    const double elapsed_seconds =
        (static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency();

    double min_count = 0.0;
    double max_count = 0.0;
    cv::minMaxLoc(sinogram, &min_count, &max_count);
    std::cout << "Forward projection complete in " << elapsed_seconds << " seconds\n";
    std::cout << "Sinogram count range: [" << min_count << ", " << max_count << "]\n";

    const cv::Mat forward_display = normalize_for_display(sinogram, config.use_log_display);
    write_image(config, "forward_projection_normalized.png", forward_display);

    cv::Mat final_forward_display;
    cv::hconcat(std::vector<cv::Mat>{emitter_image, line_canvas, forward_display}, final_forward_display);
    show_image("Final Forward Projection", final_forward_display, config.show_windows, 2000);

    const cv::Mat spectrum_display = dft_magnitude_display(sinogram, true);
    show_image("DFT Magnitude", spectrum_display, config.show_windows, 2000);

    const cv::Mat filtered_sinogram = ramp_filter_rows(sinogram);
    const cv::Mat filtered_sinogram_display = normalize_for_display(filtered_sinogram, config.use_log_display);
    show_image("Filtered Sinogram", filtered_sinogram_display, config.show_windows, 2000);

    const cv::Mat reconstruction = backproject(geometry, filtered_sinogram, initial_lines);
    const cv::Mat reconstruction_display = normalize_for_display(reconstruction, config.use_log_display);
    write_image(config, "backprojection_unnormalized.png", reconstruction);
    write_image(config, "backprojection_normalized.png", reconstruction_display);
    show_image("Backprojection", reconstruction_display, config.show_windows, 0);

    cv::destroyAllWindows();
}

} // namespace
} // namespace pet

int main(int argc, char **argv)
{
    try {
        const pet::Config config = pet::parse_config(argc, argv);
        pet::run(config);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
