
// #include <opencv2/core/types.hpp>
// #include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/imgcodecs/imgcodecs.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
// #include <ranges>


int IMAGE_SIZE = 250; //500; //1024;
cv::Point2d CENTER = cv::Point2d(floor(IMAGE_SIZE/2)-1,floor(IMAGE_SIZE/2)-1);
int RADIUS = floor(IMAGE_SIZE / 2);
bool USE_LOG_SCALE_FOR_DISPLAY = false;
bool ANIMATING = true;


// int EMITTER_RADIUS = 40;

// CMDLINE ARGS -- ./build/PET <emitter count> <emitter radius>
//

/*

PROMPT
can you write a method to write the files to the home directory? SPecifically for the final forward projection AFTER NORMALIZATION and for the backprojection, normalized AND unnormalized

APRIL 24, 2026
- I prefer normalization to log scale

APRIL 23, 2026
- Investigated normalization regardless of max count
- Log scale for image dispay instead?

APRIL 22, 2026
- Fixed weird seg fault that occurred intermittently on runs from terminal, but not debugger (indexing issue)
- Made weird assert for image_cols == image_rows, temporary for now in in_circle (assumes image is square for the circle)


APRIL 21 2026
- Still goint through copmmenting FBP implementation, cleaning (making forward projection a function)
- Is normalizing to 0 to 255 any good if max count was less than 255?


APRIL 19&20 2026
- Back projection pretty much imlemented, it works

UPDATE APRIL 9 2026
- Started preparing for FBP integration with looking at video series by Andrew Reader (see TODO)


UPDATE APRIL 8 2026
- Changed some names
- Changed out of bounds area colour to gray (emitters outside this region still not coulored over, but not used either)
- Updated sinogram to keep track of true counts, instead of maxing out at 255, so that the final image is remapped from [0, max_count] -> [0, 255], only iff max_count > 255. 
        Not sure if best. Will have to implement FBP
- Increased IMAGE_SIZE, tested some higher counts
    - LAST TEST: ./build/PET 500 10, IMAGE_SIZE 1024
    Better IMAGE_SIZE 500 because frame skipping not implemented (waiting each IMAGE_SIZE of pixels rows drawn of sinogram, so animation is quite slow for high counts)



TODO
- Commenting / cleaning
- Is normalizing to 0 to 255 any good if max count was less than 255?
    Does not seem to make a difference
- Verify FPB
- Colour over OOB emitters DONE, BETTER SOLUTION IN populate_detector_region_with_random_emitters
- Other quality of life checks
    understanding the even circle centers (2x2 or 4x4 for example)
    arbitrary input images
    FFTSHIFt only do when ANIMATING / showing the picture
- parallel? idk think later



*/


bool in_circle(cv::Point2d &point, cv::Point2d &center, int radius)
{
    // test if a point is in / on the circle by radius
    double x_diff = point.x - center.x;
    double y_diff = point.y - center.y;
    double d_sq = x_diff * x_diff + y_diff * y_diff;
    double rad_sq = static_cast<double>(radius) * radius;
    if (d_sq > rad_sq)
        return false;
    return true;
}

// rectangle check and odd sizes (when center is even like a 2x2 or 4x4)
// the cols and rows are basically a rotation check (points can rotate out)
bool in_detector(cv::Point2d &point, int image_cols, int image_rows)
{
    // temporary 
    assert(image_cols == image_rows);
    // must be in / on the circle to be true
    // Is a more robust check than in_circle (rectangle check and odd sizes?)
    if (point.x < 0 || point.x >= image_cols || point.y < 0 || point.y >= image_rows || !in_circle(point, CENTER, floor(image_cols/2)))
    {
        return false;
    }
    return true;
}

int rotate(cv::Point2d &toRotate, cv::Point2d &about, double angle)
{

    // translate point about to origin, rotate using matrix
    double x_new = std::cos(angle) * (toRotate.x - about.x) - std::sin(angle) * (toRotate.y - about.y);
    double y_new = std::sin(angle) * (toRotate.x - about.x) + std::cos(angle) * (toRotate.y - about.y);

    // translate new point back to absolute coordinates
    x_new += about.x;
    y_new += about.y;

    // NO ROUNDING HERE , ONLY WHEN DRAWING, or else error will accumulate.
    toRotate.x = x_new;
    toRotate.y = y_new;

    // this was in case of errors
    return 0;
}

void write_image_to_project_dir(const char *filename, const cv::Mat &image)
{
    if (filename == nullptr || image.empty())
    {
        std::cerr << "write_image_to_project_dir: invalid filename or empty image" << std::endl;
        return;
    }

    const std::string project_dir = "/Users/espagrud/code/c-cpp/March 2026/PET-project/";
    const std::string full_path = project_dir + filename;
    if (!cv::imwrite(full_path, image))
    {
        std::cerr << "Failed to write image: " << full_path << std::endl;
        return;
    }
    std::cout << "Wrote image: " << full_path << std::endl;
}

cv::Mat normalize_for_display_u8(const cv::Mat &src, bool use_log_scale)
{
    if (src.empty())
        return cv::Mat();

    cv::Mat src_f32;
    src.convertTo(src_f32, CV_32F);

    cv::Mat to_normalize = src_f32;
    if (use_log_scale)
    {
        // Use magnitude before log so negative values display robustly too.
        cv::Mat abs_src;
        cv::absdiff(src_f32, cv::Scalar::all(0.0f), abs_src);
        cv::log(abs_src + cv::Scalar::all(1.0f), to_normalize);
    }

    cv::Mat out_u8;
    cv::normalize(to_normalize, out_u8, 0, 255, cv::NORM_MINMAX);
    out_u8.convertTo(out_u8, CV_8U);
    return out_u8;
}

void refresh_canvas(cv::Mat &image)
{
    for (int i = 0; i < image.rows - 1; i++)
    {
        for (int j = 0; j < image.cols - 1; j++)
        {
            cv::Point2d point = cv::Point2d(j, i);

            // COLOUR POINTS OUTSIDE DETECTOR GRAY
            if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1 || !in_circle(point, CENTER, RADIUS))
                image.row(i).col(j).setTo(cv::Scalar(128));

            // COLOUR POINTS INSIDE DETECTOR BLACK 
            else
                image.row(i).col(j).setTo(cv::Scalar(0));
        }
    }
}



void draw_line(std::vector<cv::Point2d> &line, cv::Mat &image)
{
    /*
    test function used on cv::Mat visual_line_integral_image
    helps to visualize the rotating lines that are integrated along

    TO AVOID DRAWING ERROR accumulation 
    * rounding right before drawing, everything else kept before
    */
    for (int i = 0; i < static_cast<int>(line.size()); i++)
    {
        // FINALLY ROUND BEFORE DRAWING (ACCESSING for yoru line INTEGRAL)
        long ix = std::lround(line[i].x);
        long iy = std::lround(line[i].y);
        cv::Point2d point = cv::Point2d(ix, iy);

        // Only draw in the detector
        // rotated lines can leave the image so just use the in_detector more robust check, instead of just in_circle
        if (in_detector(point, image.cols, image.rows))
        {
            image.row(iy).col(ix).setTo(cv::Scalar(128));
        }
    }
}

void backproject_sinogram_pixel(int output_col, int output_row, const std::vector<std::vector<cv::Point2d>> &scan_lines, const cv::Mat &filtered_sinogram, cv::Mat &reconstruction)
{
    if (output_row >= IMAGE_SIZE  || output_row < 0 || output_col < 0 || output_col >= IMAGE_SIZE)
    {
        std::cout << "OOB! in backproject_sinogram_pixel" << std::endl;
        return;
    }

    if (filtered_sinogram.type() != CV_32F || reconstruction.type() != CV_32F)
    {
        std::cout << "backproject_sinogram_pixel expects CV_32F mats" << std::endl;
        return;
    }

    const float measurement = filtered_sinogram.at<float>(output_row, output_col);
    const std::vector<cv::Point2d> &line = scan_lines[output_col];

    for (int i = 0; i < static_cast<int>(line.size()); i++)
    {
        int ix = static_cast<int>(std::lround(line[i].x));
        int iy = static_cast<int>(std::lround(line[i].y));
        cv::Point2d point(ix, iy);
        if (in_detector(point, reconstruction.cols, reconstruction.rows))
        {
            reconstruction.at<float>(iy, ix) += measurement;
        }
    }
}

void construct_sinogram_pixel(int &output_col, int &output_row, std::vector<std::vector<cv::Point2d>> &scan_lines, cv::Mat &true_counts_sinogram, cv::Mat &ideal_emitter_image, cv::Mat &noise_free_sinogram)
{
    // noise_free_sinogram[output_row][output_col] = dot product of all_lines[output_col] with vector of 1s
    // this is a forward projection
    // here we just count them

    if (output_row >= IMAGE_SIZE  || output_row < 0 || output_col < 0 || output_col >= IMAGE_SIZE)
    {
        std::cout << "OOB! in construct_sinogram_pixel" << std::endl;
        return;
    }

    // EQUIVALENT TO DOT PRODUCT OF THE LINE WITH VECTOR OF 1s INSIDE DETECTOR BOUNDS

    // Count along the scan line, and bucket the count to this line's cell

    int count = 0;
    for (int i = 0; i < static_cast<int>(scan_lines[output_col].size()); i++)
    {

        // DO NOT FORGET TO ROUND THESE HERE TOO
        // FINAL ROUND

        int ix = static_cast<int>(std::lround(scan_lines[output_col][i].x));
        int iy = static_cast<int>(std::lround(scan_lines[output_col][i].y));


        // If point is in detector AND point is White (an emitting pixel)
        cv::Point2d point = cv::Point2d(ix, iy);

        // if (in_circle(point, CENTER, RADIUS) && ideal_emitter_image.at<uchar>(iy, ix) == 255)
        //          uchar is compiler directive for this thing holds a byte (0 to 255)
        // rotated lines can leave the image so just use the in_detector more robust check, instead of just in_circle
        if (in_detector(point, ideal_emitter_image.cols, ideal_emitter_image.rows) && ideal_emitter_image.at<uchar>(iy, ix) == 255)
        {
            count += 1;
        }

    }



    const int curr = static_cast<int>(noise_free_sinogram.at<uchar>(output_row, output_col));

    if (curr + count >= 255)
    {
        std::cout << "A max count was reached!" << std::endl;
    }

    true_counts_sinogram.at<float>(output_row, output_col) += static_cast<float>(count);

    // this is the only place where the sinogram is updated
    // this is temporary for the scan to show line by line, the image will be normalized later
    noise_free_sinogram.at<uchar>(output_row, output_col) = static_cast<uchar>(std::min(curr + count, 255));


}

void populate_lines_with_member_points(std::vector<std::vector<cv::Point2d>> &all_lines, std::vector<std::vector<cv::Point2d>> &all_lines_initial_idx)
{
    /*
        For each column vector, only add the indeces inside the circle

        Fill all_lines and all_lines_initial_idx

        j is outer dim here
    */
    for (int j = 0; j < IMAGE_SIZE; j++)
    {
        // loop rows
        for (int i = 0; i < IMAGE_SIZE; i++)
        {
            // if the pixel matches target black, append it
            // if (visual_line_integral_image.at<uchar>(i, j) == 0)
            cv::Point2d point = cv::Point2d(j,i);
            if (in_detector(point, IMAGE_SIZE, IMAGE_SIZE))
            {
                // store separate points (not pointers to same point)
                all_lines[j].push_back(point);
                all_lines_initial_idx[j].push_back(point);
            }
        }

        // this actually deep copies 
        // all_lines_initial_idx[j] = all_lines[j];
    }
}

void populate_detector_region_with_random_emitters(cv::Mat &ideal_emitter_image, int &EMITTER_RADIUS, int &NUM_EMITTERS)
{
    // make emitters
    /*
        Build collection of lines that are in the shape of a circle

        These will be rotated to collect the sinogram and visualized
   
        There are IMAGE_SIZE # of vectors, each representing a parallel line through the detector region.   
        Each vector contains indeces of the line members. There are only as many members as teh circle is tall at that column.
        The lines start vertical.

        Having the indeces makes both visualization and sinogram easy.

        all_lines_initial_idx are to rotate by total angle (so that only one round occurs) 
    */

    refresh_canvas(ideal_emitter_image); // makes outside detector gray, inside black
    
    srand(time(NULL));
    for (int i = 0; i < NUM_EMITTERS; i++) 
    {

        // int x = rand() % IMAGE_SIZE;
        // int y = rand() % IMAGE_SIZE;
        cv::Point2d point = cv::Point2d(rand()%IMAGE_SIZE, rand()%IMAGE_SIZE);

        while (!in_detector(point, IMAGE_SIZE - 2*EMITTER_RADIUS, IMAGE_SIZE - 2*EMITTER_RADIUS))
        {
            point = cv::Point2d(rand()%IMAGE_SIZE, rand()%IMAGE_SIZE);
        }

        cv::circle(ideal_emitter_image, point, EMITTER_RADIUS, cv::Scalar(255), -1);
    }
}



int main(int argc, char** argv) 
{
   
    assert(argc >= 3);
    int NUM_EMITTERS = std::stoi(argv[1]);
    int EMITTER_RADIUS = std::stoi(argv[2]); 


    // IMAGE 1 - EMITTER IMAGE
    cv::Mat ideal_emitter_image(IMAGE_SIZE, IMAGE_SIZE, CV_8UC1, cv::Scalar(0));
    populate_detector_region_with_random_emitters(ideal_emitter_image, EMITTER_RADIUS, NUM_EMITTERS);

    // IMAGE 2 - VISUAL LINE INTEGRAL IMAGE
    cv::Mat visual_line_integral_image(IMAGE_SIZE, IMAGE_SIZE, CV_8UC1, cv::Scalar(0));

    // IMAGE 3 - NOISE-FREE SINOGRAM (and true counts for normalization)
    cv::Mat true_counts_sinogram(IMAGE_SIZE, IMAGE_SIZE, CV_32F, cv::Scalar(0.0f));
    cv::Mat noise_free_sinogram(IMAGE_SIZE, IMAGE_SIZE, CV_8UC1, cv::Scalar(0));


    // LINES -> VECTORS OF POINTS
    // PARALLEL LINES THROUGH THE DETECTOR REGION (a circle, see function)
    std::vector<std::vector<cv::Point2d>> all_lines(IMAGE_SIZE);
    std::vector<std::vector<cv::Point2d>> all_lines_initial_idx(IMAGE_SIZE);
    populate_lines_with_member_points(all_lines, all_lines_initial_idx);

 


    // FORWARD PROJECTION WITH ANIMATION SWITCH


    /*
        NOT PERFECT AND SLOW    ALSO HIGHER STEPS ARE MORE ACCURATRE. Less than 200 will seem to stop not exactly at Pi.

        Setup to rotate through total_angle radians in step steps

        draw the lines in the visualization from the lines

        cosntruct the sinogram from the lines

        was good to test with 1 line first before going to all lines
    
    */

    // double Pi = 3.1415926535897932384626433832795;
    double pi = std::acos(-1);
    double total_angle = pi;

    int steps = IMAGE_SIZE;
    double angle_step = total_angle / steps;
    double angle = 0;
    cv::Point2d midpoint = cv::Point2d(floor(IMAGE_SIZE/2),floor(IMAGE_SIZE/2));
    int err = 0;
    int output_row = IMAGE_SIZE - 1;

    // 
    // FORWARD PROJECTIOn
    //

    // auto forward_project = [](cv::Mat &ideal_emitter_image, std::vector<std::vector<cv::Point2d>> visual_line_integral_image, cv::Mat noise_free_sinogram, bool ANIMATING) -> void 
    // {

    // };

    

    

    double t1 = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Timing started." << std::endl;


    cv::Mat display;
    // cv::hconcat(std::vector<cv::Mat>{ideal_emitter_image, visual_line_integral_image, noise_free_sinogram}, display);
    // cv::namedWindow("Simulation", cv::WINDOW_AUTOSIZE);

    while(angle < total_angle)
    {

        // clear the image
        refresh_canvas(visual_line_integral_image);

        // draw the lines
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            if (j % 50 == 0 && ANIMATING)
                draw_line(all_lines[j], visual_line_integral_image);

            // noise_free_sinogram[output_row][output_col] = dot product of all_lines[j] with vector of 1s
            construct_sinogram_pixel(j, output_row, all_lines, true_counts_sinogram,ideal_emitter_image, noise_free_sinogram);
            // function signature
            // construct_sinogram_pixel(int &output_col, int &output_row, std::vector<std::vector<cv::Point2d>> &scan_lines, std::vector<std::vector<cv::Point2d>> &true_counts_sinogram, cv::Mat &ideal_emitter_image, cv::Mat &noise_free_sinogram)
        }
        output_row--;


        // display the image
        if (ANIMATING)
        {
            cv::hconcat(std::vector<cv::Mat>{ideal_emitter_image, visual_line_integral_image, noise_free_sinogram}, display);
            cv::imshow("Noise Free Sinogram FP Simulation", display);
            cv::waitKey(1);
        }



        angle += angle_step;


        // Reset the lines from last iteration
        // Rotate every point by new total angle


        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            for (int i = 0; i < static_cast<int>(all_lines[j].size()); i++)
            {
                all_lines[j][i] = all_lines_initial_idx[j][i];
                err = rotate(all_lines[j][i], midpoint, angle);
            }
        }

    }



    // Remap true_counts_sinogram (float) into display sinogram when counts exceed 8-bit range.
    double min_count = 0.0;
    double max_count = 0.0;
    cv::minMaxLoc(true_counts_sinogram, &min_count, &max_count);
    const int max_count_i = static_cast<int>(std::ceil(max_count));

    std::cout << "Max count: " << max_count_i << std::endl;


    //
    // NEEED TO CHECk if THE BOOST TO 255 is GOOD OR NOT
    //

    if (max_count_i > 255)
    {
        cv::Mat normalized_f;
        cv::normalize(true_counts_sinogram, normalized_f, 0.0, 255.0, cv::NORM_MINMAX);

        cv::Mat normalized_u8;
        normalized_f.convertTo(normalized_u8, CV_8U);
        normalized_u8.copyTo(noise_free_sinogram);

    }
    cv::Mat forward_projection_display = normalize_for_display_u8(true_counts_sinogram, USE_LOG_SCALE_FOR_DISPLAY);
    write_image_to_project_dir("forward_projection_normalized.png", forward_projection_display);







    double t2 = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Timing ended at " << t2 - t1 << " seconds." << std::endl;
    
    /*
        Dislay the final results 
    
    */


    cv::hconcat(std::vector<cv::Mat>{ideal_emitter_image, visual_line_integral_image, forward_projection_display}, display);
    cv::namedWindow("Final", cv::WINDOW_AUTOSIZE);
    cv::imshow("Final", display);
    cv::waitKey(2000);



    // basic steps from Andrew Reader, equivalent to 2D transform method
    // (take advantage of linearity, use 1D transforms )
    // for row in sinogram_rows 
    //     1D FT row
    //     row.RampFilter (multiply by abs(index), middle is zero)
    //     row.inverseFFT
    //     set_projecting(projecting, row)
    //     angle+=increment
    //     back_project(projecting, angle, image)




    // angle_step
    // projecting = vector(IMAGE_SIZE)
    // angle =0
    // angle_increment = increment

    // cv::Mat transformed_sinogram(I)






    // DFT requires CV_32F/CV_64F with 1 or 2 channels.
    cv::Mat sinogram_float;
    noise_free_sinogram.convertTo(sinogram_float, CV_32F);

    // Real-valued input -> complex-valued output (2 channels).
    cv::Mat transformed_sinogram;
    cv::dft(sinogram_float, transformed_sinogram, cv::DFT_COMPLEX_OUTPUT);

    // FFT shift: move DC from top-left to image center.
    // C++ lambda, okay!
    // Pretty much just for viewing
    auto fft_shift = [](const cv::Mat &src) -> cv::Mat
    {
        if (src.empty() || src.cols < 2 || src.rows < 2)
        {
            return src.clone();
        }

        const int cx = src.cols / 2;
        const int cy = src.rows / 2;

        cv::Mat shifted_cols;
        cv::hconcat(src.colRange(cx, src.cols), src.colRange(0, cx), shifted_cols);

        cv::Mat shifted;
        cv::vconcat(shifted_cols.rowRange(cy, src.rows), shifted_cols.rowRange(0, cy), shifted);
        return shifted;
    };
    transformed_sinogram = fft_shift(transformed_sinogram);

    // Ramp filter on each row
    // multiply frequency bins by |k - center| (high-pass in frequency domain).
    cv::Mat ramp_row = cv::Mat::zeros(1, transformed_sinogram.cols, CV_32F);
    const int center_col = transformed_sinogram.cols / 2; // after fftshift
    for (int k = 0; k < transformed_sinogram.cols; ++k) {
        ramp_row.at<float>(0, k) = static_cast<float>(std::abs(k - center_col));
    }

    cv::Mat ramp_2d;
    cv::repeat(ramp_row, transformed_sinogram.rows, 1, ramp_2d);

    // Apply ramp to both complex channels explicitly (robust for channel/type rules).
    std::vector<cv::Mat> dft_planes(2);
    cv::split(transformed_sinogram, dft_planes); // [0]=real, [1]=imag
    cv::multiply(dft_planes[0], ramp_2d, dft_planes[0]);
    cv::multiply(dft_planes[1], ramp_2d, dft_planes[1]);

    //for ifft later //                         
    cv::Mat dft_complex_ramp;
    cv::merge(dft_planes, dft_complex_ramp);

    // Get |DFT| from complex, filtered output.

    cv::Mat dft_magnitude;
    cv::magnitude(dft_planes[0], dft_planes[1], dft_magnitude);



    cv::Mat dft_magnitude_display = normalize_for_display_u8(dft_magnitude, USE_LOG_SCALE_FOR_DISPLAY);

    // display da goods
    cv::namedWindow("DFT Magnitude", cv::WINDOW_AUTOSIZE);
    cv::imshow("DFT Magnitude", dft_magnitude_display);
    cv::waitKey(20000);


    // Inverse FFT on filtered, fft-shifted spectrum.
    // must unshift, as DC is expected at [0,0]
    // and will unwind correctly to make the row we must backproject
    cv::Mat dft_complex_unshifted = fft_shift(dft_complex_ramp);

    cv::Mat filtered_sinogram_float;
    cv::idft(dft_complex_unshifted, filtered_sinogram_float, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);



    cv::Mat filtered_sinogram_display = normalize_for_display_u8(filtered_sinogram_float, USE_LOG_SCALE_FOR_DISPLAY);

    cv::namedWindow("Filtered Sinogram (iDFT)", cv::WINDOW_AUTOSIZE);
    cv::imshow("Filtered Sinogram (iDFT)", filtered_sinogram_display);
    cv::waitKey(20000);


    //
    // BACKPROJECTION
    //

    steps = IMAGE_SIZE;
    angle_step = total_angle / steps;
    angle = 0;
    output_row = IMAGE_SIZE - 1;
    cv::Mat reconstruction_float = cv::Mat::zeros(IMAGE_SIZE, IMAGE_SIZE, CV_32F);

    while(angle < total_angle)
    {

        // draw the lines
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            backproject_sinogram_pixel(j, output_row, all_lines, filtered_sinogram_float, reconstruction_float);
        }
        output_row--;


        // display the image
        // if (ANIMATING)
        // {
        //     cv::hconcat(std::vector<cv::Mat>{ideal_emitter_image, visual_line_integral_image, noise_free_sinogram}, display);
        //     cv::imshow("Noise Free Sinogram FP Simulation", display);
        //     cv::waitKey(1);
        // }



        angle += angle_step;


        // Reset the lines from last iteration
        // Rotate every point by new total angle


        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            for (int i = 0; i < static_cast<int>(all_lines[j].size()); i++)
            {
                all_lines[j][i] = all_lines_initial_idx[j][i];
                err = rotate(all_lines[j][i], midpoint, angle);
            }
        }

    }

    cv::Mat reconstruction_display = normalize_for_display_u8(reconstruction_float, USE_LOG_SCALE_FOR_DISPLAY);
    write_image_to_project_dir("backprojection_unnormalized.png", reconstruction_float);
    write_image_to_project_dir("backprojection_normalized.png", reconstruction_display);
    cv::namedWindow("Backprojection (Filtered)", cv::WINDOW_AUTOSIZE);
    cv::imshow("Backprojection (Filtered)", reconstruction_display);
    cv::waitKey(20000);












    //can use the lines from before
    /*

        Cleanup
    
    */
    cv::destroyAllWindows();




    return 0;
}



    // C++ 23 compiler support?
    // for (auto [i, row] : std::views::enumerate(true_counts))
    // {
    //     for (auto [j, count] : std::views::enumerate(row))
    //     {

    //     }
    // }