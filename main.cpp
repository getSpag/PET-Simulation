
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/imgcodecs/imgcodecs.hpp>
#include <cmath>
#include <vector>


int IMAGE_SIZE = 500;
// int EMITTER_RADIUS = 40;

/*
TODO
- Normalization for bucketing in busy emitters
- sinogram upside down

*/



bool in_circle(cv::Point2d point, cv::Point2d center, int radius);

void construct_sinogram(int line_number, std::vector<cv::Point2d> &scan_line, cv::Mat &emitter_image, cv::Mat &noise_free_sinogram, int &height_count)
{
    // sum along the line, and place in the appropriate bin 

    if (height_count == IMAGE_SIZE - 1)
    {
        return;
    }


    // Count along the scan line, and bucket the count to this line's cell

    int count = 0;
    for (int i = 0; i < static_cast<int>(scan_line.size()); i++)
    {

        // DO NOT FORGET TO ROUND THESE HERE TOO
        // FINAL ROUND

        const int ix = static_cast<int>(std::lround(scan_line[i].x));
        const int iy = static_cast<int>(std::lround(scan_line[i].y));
        if (ix < 0 || ix >= emitter_image.cols || iy < 0 || iy >= emitter_image.rows)
            continue;
        if (emitter_image.at<cv::Vec3b>(iy, ix) == cv::Vec3b(255, 255, 255))
            count += 1;
    }


    // Not sure if this is 100% correct, the way I am binning the counts 
    // May need to normalize for busy emitters (255 is not a lot of hits)

    cv::Vec3b curr = noise_free_sinogram.at<cv::Vec3b>(height_count, line_number);
    noise_free_sinogram.at<cv::Vec3b>(height_count, line_number) = cv::Vec3b(
        std::min(curr[0] + count, 255),
        std::min(curr[1] + count,255),
        std::min(curr[2] + count,255));



}

int rotate(cv::Point2d &toRotate, cv::Point2d about, double angle)
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

void clear_image(cv::Mat &image)
{
    for (int i = 0; i < image.rows - 1; i++)
    {
        for (int j = 0; j < image.cols - 1; j++)
        {
            // there was a shortcut taken here to also colour the outside of the circle white -- not just the edge of the whole square image
            if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1 || !in_circle(cv::Point2d(j, i), cv::Point2d(floor(IMAGE_SIZE / 2) - 1, floor(IMAGE_SIZE / 2) - 1), floor(IMAGE_SIZE / 2)))
                image.row(i).col(j).setTo(cv::Scalar(255, 255, 255));
            else
                image.row(i).col(j).setTo(cv::Scalar(0, 0, 0));
        }
    }
}


bool in_image(cv::Point2d point, int image_x, int image_y)
{
    // must be in / on the circle to be true, calls in_circle, I don't like how these functions are tied
    if (point.x < 0 || point.x >= image_x || point.y < 0 || point.y >= image_y || !in_circle(point, cv::Point2d(floor(IMAGE_SIZE/2)-1,floor(IMAGE_SIZE/2)-1), floor(IMAGE_SIZE/2)))
    {
        return false;
    }
    return true;
}

bool in_circle(cv::Point2d point, cv::Point2d center, int radius)
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
        const long ix = std::lround(line[i].x);
        const long iy = std::lround(line[i].y);
        if (in_image(cv::Point2d(ix, iy), image.cols, image.rows))
        {
            image.row(iy).col(ix).setTo(cv::Scalar(128, 128, 128));
        }
    }
}

int main(int argc, char** argv) 
{
   
    assert(argc >= 3);
    int NUM_EMITTERS = std::stoi(argv[1]);
    int EMITTER_RADIUS = std::stoi(argv[2]); 


    /* 
        Create Image of emitter, and clear_image to clean / setup

        clear_image marks everything outside of a IMAGE_SIZE-diametered circle white
    */
    cv::Mat emitter_image(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    clear_image(emitter_image);


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
    clear_image(visual_line_integral_FP_BP);

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




    /*
        For each column vector, only add the indeces inside the circle

        Fill all_lines and all_base_lines
    */


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
    int height_count = IMAGE_SIZE - 1;


    cv::Mat display;
    cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
    cv::namedWindow("Simulation", cv::WINDOW_AUTOSIZE);

    while(angle < total_angle)
    {

        // clear the image
        clear_image(visual_line_integral_FP_BP);

        // draw the lines
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            if (j % 50 == 0)
                draw_line(all_lines[j], visual_line_integral_FP_BP);

            construct_sinogram(j, all_lines[j], emitter_image, noise_free_sinogram, height_count);
        }
        height_count--;


        // display the image
        cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
        cv::imshow("Noise Free Sinogram FP Simulation", display);
        cv::waitKey(10);


        // TOTAL ANGLE FOR THIS FRAME, NOT SMALL ONES FOR ERROR-ACCUMULATING ROTATES

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

    
    /*
        Dislay the final results 
    
    */


    cv::hconcat(std::vector<cv::Mat>{emitter_image, visual_line_integral_FP_BP, noise_free_sinogram}, display);
    cv::namedWindow("Final", cv::WINDOW_AUTOSIZE);
    cv::imshow("Final", display);
    cv::waitKey(20000);




    return 0;
}