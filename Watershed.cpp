#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Read a color image
    Mat img = imread("../data/dst.jpg");

    // Convert the image to grayscale
    Mat src;
    cvtColor(img, src, COLOR_BGR2GRAY);

    // Thresholding
    Mat thresh;
    GaussianBlur(src, src, Size(5, 5), 0); // Apply Gaussian blur for smoothing
    imwrite("../data/GaussianBlur.jpg", src); // Save the image after Gaussian blur
    threshold(src, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU); // Apply thresholding using Otsu's method
    imwrite("../data/thresh.jpg", thresh);
    cv::Mat background;
    Mat ele = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(thresh, background, ele, cv::Point(-1, -1), 2); // Dilate the thresholded image
    imwrite("../data/dilate.jpg", background);
    bitwise_not(background, background); // Invert the background
    imwrite("../data/bitwise_not.jpg", background);

    // Generate a binary image representing the determined foreground
    Mat foreground;
    morphologyEx(thresh, foreground, MORPH_CLOSE, ele, cv::Point(-1, -1), 2); // Close operation on the thresholded image
    int n = connectedComponents(foreground, foreground, 8, CV_32S); // Connected component labeling
    imwrite("../data/foreground.jpg", foreground);

    // Generate a marker image
    Mat markers = foreground;
    markers.setTo(255, background); // Set determined background to 255, the rest remains unknown (0)
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10); // Scale the intensity for better visualization
    imwrite("../data/Markers_Input.jpg", markers8u);

    // Apply watershed algorithm to mark object contours
    watershed(img, markers);
    markers.convertTo(markers8u, CV_8U, 10); // Scale the intensity for better visualization
    imwrite("../data/Markers_Output.jpg", markers8u);

    // Post-processing (color filling)
    Mat mark;
    markers.convertTo(mark, CV_8U); // Convert -1 to 0
    bitwise_not(mark, mark);

    imwrite("../data/Mark.jpg", mark);

    // Assign random colors to different segments
    vector<Vec3b> colors;
    for (size_t i = 0; i < n; i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create an image with colored segments
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(n))
                dst.at<Vec3b>(i, j) = colors[index - 1];
        }
    }

    imwrite("../data/dstresult.jpg", dst);

    // Merge the segmented and colored image with the original image
    Mat result;
    addWeighted(img, 0.4, dst, 0.6, 0, result);

    imwrite("../data/final_result.jpg", result);

    return 0;
}
