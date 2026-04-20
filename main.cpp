
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
#include <iostream>
#include <vector>
// #include <ranges>


int IMAGE_SIZE = 500; //1024;
cv::Point2d CENTER = cv::Point2d(floor(IMAGE_SIZE/2)-1,floor(IMAGE_SIZE/2)-1);
int RADIUS = floor(IMAGE_SIZE / 2);
// int EMITTER_RADIUS = 40;

// CMDLINE ARGS -- ./build/PET <emitter count> <emitter radius>
//

/*

# good backprojection animation

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
- Implement FPB to test your sinogram
    IP

    // basic steps from Andrew Reader, equivalent to 2D transform method
    // (take advantage of linearity, use 1D transforms )
    // this avoids rotating to produce the true 2D FT from the sinogram

    // projecting = vector(IMAGE_SIZE^2)
    // angle =0
    // angle_increment = increment
    // for (auto& row:img)
    //     1D FT row
    //     row.RampFilter (multiply by abs(index), middle is zero)
    //     row.inverseFFT
    //     set_projecting(projecting, row)
    //     angle+=increment
    //     back_project(projecting, angle, image)




- Colour over OOB emitters
- Other quality of life checks
    understanding the even circle centers (2x2 or 4x4 for example)
    arbitrary input images
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
bool in_detector(cv::Point2d &point, int &image_cols, int &image_rows)
{
    // must be in / on the circle to be true
    // Is a more robust check than in_circle (rectangle check and odd sizes?)
    if (point.x < 0 || point.x >= image_cols || point.y < 0 || point.y >= image_rows || !in_circle(point, CENTER, floor(IMAGE_SIZE/2)))
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

void refresh_canvas(cv::Mat &image)
{
    for (int i = 0; i < image.rows - 1; i++)
    {
        for (int j = 0; j < image.cols - 1; j++)
        {
            cv::Point2d point = cv::Point2d(j, i);

            // COLOUR POINTS OUTSIDE DETECTOR GRAY
            if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1 || !in_circle(point, CENTER, RADIUS))
                image.row(i).col(j).setTo(cv::Scalar(128,128,128));

            // COLOUR POINTS INSIDE DETECTOR BLACK 
            else
                image.row(i).col(j).setTo(cv::Scalar(0, 0, 0));
        }
    }
}



void draw_line(std::vector<cv::Point2d> &line, cv::Mat &image)
{
    /*
    test function used on cv::Mat visual_line_integral_FP_BP
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
            image.row(iy).col(ix).setTo(cv::Scalar(128, 128, 128));
        }
    }
}


// takes another argument to the final image to deposit TRUE COUNTS
// the animation will just display "about" what it is, since it is going line by line
// final version is fully corrected
//
// made it take all scan lines by reference so that the indexing makes more sense
//
void construct_sinogram_pixel(int &output_col, int &output_row, std::vector<std::vector<cv::Point2d>> &scan_lines, std::vector<std::vector<int>> &true_counts, cv::Mat &emitter_image, cv::Mat &noise_free_sinogram)
{
    // noise_free_sinogram[output_row][output_col] = dot product of all_lines[output_col] with vector of 1s
    // here we just count them

    if (output_row >= IMAGE_SIZE  || output_row < 0 || output_col < 0 || output_col >= IMAGE_SIZE)
    {
        std::cout << "OOB! in construct_sinogram_pixel" << std::endl;
        return;
    }

    // EQUIVALENT TO DOT PRODUCT OF THE LINE WITH VECTOR OF 1s INSIDE DETECTOR BOUNDS

    // Count along the scan line, and bucket the count to this line's cell

    int count = 0;
    for (int i = 0; i < static_cast<int>(scan_lines[output_col - 1].size()); i++)
    {

        // DO NOT FORGET TO ROUND THESE HERE TOO
        // FINAL ROUND

        int ix = static_cast<int>(std::lround(scan_lines[output_col - 1][i].x));
        int iy = static_cast<int>(std::lround(scan_lines[output_col - 1][i].y));


        // If point is in detector AND point is White (an emitting pixel)
        cv::Point2d point = cv::Point2d(ix, iy);

        // if (in_circle(point, CENTER, RADIUS) && emitter_image.at<cv::Vec3b>(iy, ix) == cv::Vec3b(255, 255, 255))
        // rotated lines can leave the image so just use the in_detector more robust check, instead of just in_circle
        if (in_detector(point, emitter_image.cols, emitter_image.rows) && emitter_image.at<cv::Vec3b>(iy, ix) == cv::Vec3b(255, 255, 255))
        {
            count += 1;
        }

    }



    cv::Vec3b curr = noise_free_sinogram.at<cv::Vec3b>(output_row, output_col);

    if ((curr[0] + count >= 255) || (curr[1] + count >= 255) || (curr[2] + count >= 255))
    {
        std::cout << "A max count was reached!" << std::endl;
    }



    true_counts[output_row][output_col] += count;

    // this is the only place where the sinogram is updated
    // this is temporary for the scan to show line by line, the image will be normalized later
    noise_free_sinogram.at<cv::Vec3b>(output_row, output_col) = cv::Vec3b(
        std::min(curr[0] + count, 255),
        std::min(curr[1] + count,255),
        std::min(curr[2] + count,255)
    );


}



int main(int argc, char** argv) 
{
   
    assert(argc >= 3);
    int NUM_EMITTERS = std::stoi(argv[1]);
    int EMITTER_RADIUS = std::stoi(argv[2]); 


    /* 
        Create Image of emitter, and refresh_canvas to clean / setup

        refresh_canvas marks everything outside of a IMAGE_SIZE-diametered circle white
    */
    
    cv::Mat emitter_image(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    refresh_canvas(emitter_image);


    /* 
        10 random points used to place emitters in square

        Only those in IMAGE_SIZE-diametered circle will be considered 
            (emitters are same colour as background)
    */
    srand(time(NULL));
    for (int i = 0; i < NUM_EMITTERS; i++) 
    {
        int x = rand() % IMAGE_SIZE;
        int y = rand() % IMAGE_SIZE;
        cv::circle(emitter_image, cv::Point(x, y), EMITTER_RADIUS, cv::Scalar(255, 255, 255), -1);
    }


    /*
        Setup for a Forward Projection to create a NOISE-FREE SINOGRAM

        This will be used to compare to the empirical noisy sinogram from actual simulated annihilation events

            visiual_line_integral_FP_BP  -> For visualization of parallel-line path integrals
            noise_free_sinogram -> For actually holding the sinogram

            the first is setup with the IMAGE_SIZE-diametered circle 
            the second is a square 
    
    */
    cv::Mat visual_line_integral_FP_BP(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    refresh_canvas(visual_line_integral_FP_BP);

    cv::Mat noise_free_sinogram(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    
    

    /*
        Build collection of lines that are in the shape of a circle

        These will be rotated to collect the sinogram and visualized
   
        There are IMAGE_SIZE # of vectors, each representing a parallel line through the detector region.   
        Each vector contains indeces of the line members. There are only as many members as teh circle is tall at that column.
        The lines start vertical.

        Having the indeces makes both visualization and sinogram easy.

        all_base_lines are to reset, and add the updated global / UNROUNDED sums each iteration
        this avoid error accumulating from rotating by small angles in high step counts

    */


    // we won't need all IMAGE_SIZE on inner dimension except for in the middle row
    std::vector<std::vector<cv::Point2d>> all_lines(IMAGE_SIZE);//,std::vector<cv::Point2d>(IMAGE_SIZE));
    std::vector<std::vector<cv::Point2d>> all_base_lines(IMAGE_SIZE);//,std::vector<cv::Point2d>(IMAGE_SIZE));
    std::vector<std::vector<int>> true_counts(IMAGE_SIZE, std::vector<int>(IMAGE_SIZE, 0));




    /*
        For each column vector, only add the indeces inside the circle

        Fill all_lines and all_base_lines
    */

    bool animating = true;



    // if (animating)
    // {




    // J IS OUTER DIM HERE
    // fill lines and base lines
    for (int j = 0; j < IMAGE_SIZE; j++)
    {
        // loop rows
        for (int i = 0; i < IMAGE_SIZE; i++)
        {
            // if the pixel matches target black, append it
            if (visual_line_integral_FP_BP.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0,0))
            {
                all_lines[j].push_back(cv::Point2d(j,i));
            }
        }

        // this actually deep copies 
        all_base_lines[j] = all_lines[j];
    }

    // }



    /*
        NOT PERFECT AND SLOW    ALSO HIGHER STEPS ARE MORE ACCURATRE. Less than 200 will seem to stop not exactly at Pi.

        Setup to rotate through total_angle radians in step steps

        draw the lines in the visualization from the lines

        cosntruct the sinogram from the lines

        was good to test with 1 line first before going to all lines
    
    */

    double Pi = 3.1415926535897932384626433832795;
    double total_angle = Pi;
    int steps = IMAGE_SIZE;
    double angle_step = total_angle / steps;
    double angle = 0;
    cv::Point2d midpoint = cv::Point2d(floor(IMAGE_SIZE/2),floor(IMAGE_SIZE/2));
    int err = 0;
    int output_row = IMAGE_SIZE - 1;

    double t1 = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Timing started." << std::endl;


    cv::Mat display;
    cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
    cv::namedWindow("Simulation", cv::WINDOW_AUTOSIZE);

    while(angle < total_angle)
    {

        // clear the image
        refresh_canvas(visual_line_integral_FP_BP);

        // draw the lines
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            if (j % 50 == 0 && animating)
                draw_line(all_lines[j], visual_line_integral_FP_BP);

            // noise_free_sinogram[output_row][output_col] = dot product of all_lines[j] with vector of 1s
            construct_sinogram_pixel(j, output_row, all_lines, true_counts,emitter_image, noise_free_sinogram);
            // function signature
            // construct_sinogram_pixel(int &output_col, int &output_row, std::vector<std::vector<cv::Point2d>> &scan_lines, std::vector<std::vector<cv::Point2d>> &true_counts, cv::Mat &emitter_image, cv::Mat &noise_free_sinogram)
        }
        output_row--;


        // display the image
        if (animating)
        {
            cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
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
                all_lines[j][i] = all_base_lines[j][i];
                err = rotate(all_lines[j][i], midpoint, angle);
            }
        }

    }

   

    // normalize the sinogram with true_counts -- get max and min counts, and map to 0 to 255
    int max_count = 0;
    // find max value in 2D array
    for (const auto& row : true_counts) {
        if (!row.empty()) {
            auto it = std::max_element(row.begin(), row.end());
            max_count = std::max(max_count, *it);
        }
    }

    std::cout << "Max count: " << max_count << std::endl;

    // now make everything max_count / 255 times SMALLER
    // so mat[i][j] = mat[i][j] * 255 / max_count
    // and just write it to the noise_free_sinogram



    if (max_count > 255)
    {
        int new_value =0;
        cv::Vec3b curr;
        for (int i = 0; i < IMAGE_SIZE; i++)
        {
            for (int j = 0; j < IMAGE_SIZE; j++)
            {
                curr = noise_free_sinogram.at<cv::Vec3b>(i, j);
    
                // 3 channel, but is grayscale anyways lol
                new_value = floor(true_counts[i][j] * 255 / max_count);
                if (new_value >= 254)
                    std::cout << "New value: " << new_value << std::endl;
                noise_free_sinogram.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    new_value,
                    new_value,
                   new_value 
                );
            }
        }
    }








    double t2 = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Timing ended at " << t2 - t1 << " seconds." << std::endl;
    
    /*
        Dislay the final results 
    
    */


    cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
    cv::namedWindow("Final", cv::WINDOW_AUTOSIZE);
    cv::imshow("Final", display);
    cv::waitKey(2000);



    // basic steps from Andrew Reader, equivalent to 2D transform method
    // (take advantage of linearity, use 1D transforms )




    // angle_step
    // projecting = vector(IMAGE_SIZE)
    // angle =0
    // angle_increment = increment

    // cv::Mat transformed_sinogram(I)






    // HERE
    // cv::Mat transformed_sinogram(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));

    // // assumes noise_free_sinogram is image_size tall
    // for (int i = 0; i < IMAGE_SIZE; i++)
    // {
    //     cv::dft(noise_free_sinogram.row(i), transformed_sinogram.row(i));

    // }


    // cv::namedWindow("DFT", cv::WINDOW_AUTOSIZE);
    // cv::imshow("DFT", transformed_sinogram);
    // cv::waitKey(20000);
    //     1D FT row
    //     row.RampFilter (multiply by abs(index), middle is zero)
    //     row.inverseFFT
    //     set_projecting(projecting, row)
    //     angle+=increment
    //     back_project(projecting, angle, image)






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