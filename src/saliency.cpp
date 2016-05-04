#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img = cv::imread("test.jpg");
    cv::cvtColor(img, img, CV_BGR2GRAY);
    float ratio = 128.0 / img.cols;
    cv::resize(img, img, cv::Size(img.cols*ratio, img.rows*ratio));

    cv::Mat planes[] = {cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);
    cv::split(complexImg, planes);

    cv::Mat mag, logmag, smooth, spectralResidual;
    cv::magnitude(planes[0], planes[1], mag);
    cv::log(mag, logmag);
    cv::boxFilter(logmag, smooth, -1, cv::Size(3,3));
    cv::subtract(logmag, smooth, spectralResidual);
    cv::exp(spectralResidual, spectralResidual);

    planes[0] = planes[0].mul(spectralResidual) / mag;
    planes[1] = planes[1].mul(spectralResidual) / mag;

    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg, cv::DFT_INVERSE | cv::DFT_SCALE);
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], mag);
    cv::multiply(mag, mag, mag);
    cv::GaussianBlur(mag, mag, cv::Size(9,9), 2.5, 2.5);
    cv::normalize(mag, mag, 1.0, 0.0, NORM_MINMAX);

    cv::imshow("Saliency Map", mag);
    cv::waitKey(0);
}
