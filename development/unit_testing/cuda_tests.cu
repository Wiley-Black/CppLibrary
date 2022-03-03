
#include "wbComplete.h"
#include <iostream>

#ifdef CUDA_Support

using namespace wb;
using namespace std;
static const double M_PI = 3.14159265358979323846264338327950288;

#pragma region "gtest-like macro definitions"

/** This section provides a "gtest lite" implementation.  It provides
*   the ASSERT_...() macros in the same general format as gtest, but
*   it does not provide the test case layout format, which will have to 
*   be provided by a calling .cpp file.  It also requires the 
*   START_TEST() and FINISH_TEST() macros be at the start and end of the
*   function using the ASSERT_...() and EXPECT_...() macros.
* 
*   Implementation:
* 
*   We need to fulfill the gtest pattern:
* 
*       ASSERT_EQ(A,B) << "my message goes here.";
* 
*   The macro will need to expand to the following general format:
* 
*       if ([Test Condition Passes]) ; else trigger = messenger
* 
*   And messenger has to accept operator<<.  The operator= on "trigger"
*   will trigger an assertion or expectation.
*/
bool g_TestSucceeded = true;

class ExpectTrigger {
public:
    ExpectTrigger(const ostream& msg)
    {
        g_TestSucceeded = false;
        std::cout << ((std::stringstream&)msg).str();
        std::cout << std::endl;
    }
};

class AssertException {};

class AssertTrigger {
public:
    AssertTrigger(const ostream& msg)
    {
        g_TestSucceeded = false;
        std::cout << ((std::stringstream&)msg).str();
        std::cout << std::endl;
        throw AssertException();
    }
};

#define START_TEST()    g_TestSucceeded = true; try {
#define FINISH_TEST()   } catch (AssertException&) { return false; } return g_TestSucceeded;

#define ASSERTEXPECT_EQ(Expected, Actual, FILE, LINE, TriggerType)     \
    if ((Actual) == (Expected)) ; else TriggerType et = std::stringstream()   \
        << std::string(FILE) << "(" << std::to_string(LINE) << "): assertion failed." << std::endl \
        << "  Actual: " << (Actual) << std::endl  \
        << "Expected: " << (Expected) << std::endl

#define EXPECT_EQ(Expected, Actual)     ASSERTEXPECT_EQ(Expected, Actual, __FILE__, __LINE__, ExpectTrigger)
#define ASSERT_EQ(Expected, Actual)     ASSERTEXPECT_EQ(Expected, Actual, __FILE__, __LINE__, AssertTrigger)
#define EXPECT_TRUE(Actual)             ASSERTEXPECT_EQ(true, Actual, __FILE__, __LINE__, ExpectTrigger)
#define ASSERT_TRUE(Actual)             ASSERTEXPECT_EQ(true, Actual, __FILE__, __LINE__, AssertTrigger)

#define ASSERTEXPECT_NEAR(Expected, Actual, abs_error, FILE, LINE, TriggerType)        \
    if (abs((Actual) - (Expected)) < abs_error) ; else TriggerType et = std::stringstream()   \
        << std::string(FILE) << "(" << std::to_string(LINE) << "): assertion failed." << std::endl \
        << "   Actual: " << (Actual) << std::endl      \
        << " Expected: " << (Expected) << std::endl    \
        << "Tolerance: " << (abs_error) << std::endl

#define EXPECT_NEAR(Expected, Actual, abs_error)     ASSERTEXPECT_NEAR(Expected, Actual, abs_error, __FILE__, __LINE__, ExpectTrigger)
#define ASSERT_NEAR(Expected, Actual, abs_error)     ASSERTEXPECT_NEAR(Expected, Actual, abs_error, __FILE__, __LINE__, AssertTrigger)

#pragma endregion

/** Test Cases **/

bool CUDAImageProcessingTesting()
{
    START_TEST();

    using namespace wb::images;
     
    try
    {
        auto pGSI = std::shared_ptr<GPUSystemInfo>(new GPUSystemInfo());

        std::cout << "Selecting CUDA device [0]..." << std::endl;
        cudaThrowable(cudaSetDevice(0));
        std::cout << "CUDA device selected!" << std::endl;

        std::cout << "Retrieving GPU properties and initializing a stream..." << std::endl;
        auto gStream = GPUStream::New(pGSI, "Unit Testing Thread");
        std::cout << "GPU Max Threads/Block: " << std::to_string(gStream.GetDeviceProperties().maxThreadsPerBlock) << std::endl;

        /** Basic image functionality and host-device transfer tests **/
        {
            auto imgA = Image<float>::NewHostImage(256, 256, gStream);
            imgA.ToHost();
            imgA.Fill(1.11f);
            ASSERT_NEAR(1.11f, imgA(5, 5), 1e-4) << "Host pixel did not match expected value after host fill.\n";
            auto imgB = Image<float>::NewHostImage(256, 256, gStream);
            imgB.ToDevice();
            imgB.Fill(1.23f);
            ASSERT_NEAR(1.23f, imgB(5, 5), 1e-4) << "Host pixel did not match expected value after device fill.\n";
            imgB.ToDevice();
            imgA += imgB;
            ASSERT_NEAR(2.34f, imgA(5, 5), 1e-4) << "Host pixel did not match expected value after adding images A and B into A.\n";
            imgB.ToDevice();
            imgB += imgA;
            ASSERT_NEAR(3.57f, imgB(5, 5), 1e-4) << "Host pixel did not match expected value after adding images A and B into B.\n";
            imgB.SaturateInPlace(0.0f, 2.0f);
            ASSERT_NEAR(2.0f, imgB(5, 5), 1e-4) << "Host pixel did not match expected saturation after saturing image B in place.\n";
            imgA(5, 5) = 25.0f;
            ASSERT_NEAR(25.0f, imgA(5, 5), 1e-4) << "Host pixel did not match expected value after direct modification.\n";
            auto imgAsmall = imgA.CropTo(3, 2, 5, 5);
            // std::cout << "cropped A (3,2,5,5) now = " << imgAsmall << "\n";
            ASSERT_EQ(5, imgAsmall.Width()) << "Image did not have expected width after cropping.\n";
            ASSERT_EQ(5, imgAsmall.Height()) << "Image did not have expected height after cropping.\n";
            ASSERT_NEAR(25.0f, imgAsmall(2, 3), 1e-4) << "Specific host pixel at x=2, y=3 did not match expectation after cropping.\n";
            ASSERT_NEAR(2.34f, imgAsmall(3, 2), 1e-4) << "Host pixel at x=3, y=2 did not match expectation after cropping.\n";
            auto imgC = imgA.CopyToDevice();
        }

        /** Small image and resize test **/
        {
            auto imgA = Image<float>::NewHostImage(2, 2, gStream);
            imgA(0, 0) = 10.0f;
            imgA(1, 0) = 20.0f;
            imgA(0, 1) = 10.0f;
            imgA(1, 1) = 20.0f;
            imgA.ToDevice(false);
            auto imgLarger = imgA.ResizeTo(images::Rectangle<int>(0, 0, 3, 2), images::Rectangle<int>::Whole(), InterpolationMethods::Linear);
            imgLarger.ToHost(false);
            ASSERT_EQ(3, imgLarger.Width()) << "Image did not have expected width after resizing.\n";
            ASSERT_EQ(2, imgLarger.Height()) << "Image did not have expected height after resizing.\n";
            ASSERT_NEAR(10.0f, imgLarger(0, 0), 0.1f) << "Interpolated (resized) pixel at x=0, y=0 did not match expectation.\n";
            ASSERT_NEAR(14.1f, imgLarger(1, 0), 0.1f) << "Interpolated (resized) pixel at x=1, y=0 did not match expectation.\n";
            ASSERT_NEAR(20.0f, imgLarger(2, 0), 0.1f) << "Interpolated (resized) pixel at x=2, y=0 did not match expectation.\n";
            ASSERT_NEAR(10.0f, imgLarger(0, 1), 0.1f) << "Interpolated (resized) pixel at x=0, y=1 did not match expectation.\n";
            ASSERT_NEAR(14.1f, imgLarger(1, 1), 0.1f) << "Interpolated (resized) pixel at x=1, y=1 did not match expectation.\n";
            ASSERT_NEAR(20.0f, imgLarger(2, 1), 0.1f) << "Interpolated (resized) pixel at x=2, y=1 did not match expectation.\n";
        }

#if 0   // Disabled for now, until I revisit the order swapping question in ConvolutionKernel.
        {
            auto imgBy = Image<byte>::NewHostImage(10, 10);
            imgBy.Fill(8);
            imgBy(3, 3) = 4;
            //std::cout << "Input imgBy = " << imgBy << "\n";
            // LPF = average.  For 3x4, the average with the one substituted pixel (4) is (8*11 + 4) / 12 = 7.6 near the substituted pixel.  In a byte, 7.                        
            auto LPF3x4 = ConvolutionKernel<Int32>::New({ {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1} }, 3*4);
            auto imgFiltered = imgBy.Convolve(LPF3x4);
            //std::cout << "LPF kernel = " << LPF3x4 << "\n";
            imgFiltered.ToHost(false);
            std::cout << "LPF filtered now = " << imgFiltered << "\n";
            ASSERT_EQ(7, imgFiltered(3, 3));
        }
#endif

        /** Image file save-load test **/
        {
            std::cout << "Testing image file save-load...\n";

            auto imgA = Image<float>::NewHostImage(256, 256, gStream);
            for (int yy = 0; yy < imgA.Height(); yy++)
            {
                for (int xx = 0; xx < imgA.Width(); xx++)
                {
                    imgA(xx, yy) = (float)yy / 2.0f + (float)xx;
                }
            }

            string filename = "test.tif";
            imgA.Save(filename);

            auto imgB = Image<float>::Load(filename, gStream);
            ASSERT_EQ(imgA.Width(), imgB.Width()) << "Image width was not the same after save-load to file '" << filename << "'\n";
            ASSERT_EQ(imgA.Height(), imgB.Height()) << "Image height was not the same after save-load to file '" << filename << "'\n";
            for (int yy = 0; yy < imgA.Height(); yy++)
            {
                for (int xx = 0; xx < imgA.Width(); xx++)
                {
                    ASSERT_NEAR(imgA(xx,yy), imgB(xx,yy), 1e-4) << "Pixel values (A: " << imgA(xx,yy) << ", B: " << imgB(xx,yy) << ") at position (" << xx << "," << yy << ") was not the same after save-load to file '" << filename << "'\n";
                } 
            }

            io::File::Delete(filename);
        }

        /** Real Math test **/
        {
            auto imgA = Image<float>::NewHostImage(4, 4, gStream);
            imgA.ToHost();
            imgA.Fill(-3.5f);
            ASSERT_NEAR(-3.5f, imgA(2, 2), 1e-4) << "Host pixel did not match expected value after host fill.\n";
            auto imgB = imgA.Absolute();
            ASSERT_NEAR(3.5f, imgB(2, 2), 1e-4) << "Pixel did not match expected value after absolute operation.\n";
            ASSERT_NEAR(-3.5f, imgA(2, 2), 1e-4) << "Pixel unexpectedly affected by operation.\n";
            imgA.AbsoluteInPlace();
            ASSERT_NEAR(3.5f, imgA(2, 2), 1e-4) << "Pixel did not match expected value after in-place operation.\n";
            imgA.AbsoluteInPlace();
            ASSERT_NEAR(3.5f, imgA(2, 2), 1e-4) << "Pixel did not match expected value after in-place operation.\n";
        }

        /** Complex Math test **/
        {
            auto imgA = Image<thrust::complex<float>>::NewHostImage(4, 4, gStream);
            imgA.ToHost();
            thrust::complex<float> val(3.0f, -2.0f);
            imgA.Fill(val);
            ASSERT_NEAR(3.0f, imgA(2, 2).real(), 1e-4) << "Host pixel did not match expected value after host fill.\n";
            ASSERT_NEAR(-2.0f, imgA(2, 2).imag(), 1e-4) << "Host pixel did not match expected value after host fill.\n";
            auto imgB = imgA.Conjugate();
            ASSERT_NEAR(3.0f, imgB(2, 2).real(), 1e-4) << "Pixel did not match expected value after conjugate operation.\n";
            ASSERT_NEAR(2.0f, imgB(2, 2).imag(), 1e-4) << "Pixel did not match expected value after conjugate operation.\n";
            ASSERT_NEAR(-2.0f, imgA(2, 2).imag(), 1e-4) << "Pixel unexpectedly affected by operation.\n";
            imgA.ConjugateInPlace();
            ASSERT_NEAR(3.0f, imgA(2, 2).real(), 1e-4) << "Pixel did not match expected value after in-place conjugate operation.\n";
            ASSERT_NEAR(2.0f, imgA(2, 2).imag(), 1e-4) << "Pixel did not match expected value after in-place conjugate operation.\n";

            auto imgAbsA = imgA.Absolute();
            ASSERT_NEAR(3.6056f, imgAbsA(2, 2), 1e-4) << "Pixel did not match expected value after absolute operation.\n";
            ASSERT_NEAR(3.0f, imgA(2, 2).real(), 1e-4) << "Pixel unexpectedly affected by operation.\n";
            ASSERT_NEAR(2.0f, imgA(2, 2).imag(), 1e-4) << "Pixel unexpectedly affected by operation.\n";
        }

        /** Sum() and CUB testing **/
        {
            std::cout << "Testing sum/reduction of real and complex images...\n";
            wb::Random rng(0);
            auto sw = wb::Stopwatch::StartNew();
            int NIterations = 20;
            Int64 TotalPixels = 0;
            for (int iter = 0; iter < NIterations; iter++)
            {
                int width = rng.NextUniform(1, 8192);
                int height = rng.NextUniform(1, 2048);
                if (rng.NextUniform(0, 9) != 0) {
                    // Most of the time, choose the nearest power of 2 for the image dimensions so that most of our testing applies to power-of-2 images.
                    width = (int)pow(2, (int)log2(width));
                    height = (int)pow(2, (int)log2(height));
                }

                auto img = Image<thrust::complex<float>>::NewHostImage(width, height, gStream, HostFlags::Pinned);
                for (int yy = 0; yy < img.Height(); yy++)
                {
                    for (int xx = 0; xx < img.Width(); xx++)
                    {
                        img(xx, yy) = thrust::complex<float>(rng.NextUniform(0.0f, 10.0f), rng.NextUniform(0.0f, 10.0f));
                    }
                }

                // Sum of real part only
                auto imgReal = img.GetReal();
                auto sumHostReal = imgReal.Sum<double>();
                auto imgDeviceReal = imgReal.CopyToDevice(HostFlags::Pinned);
                imgDeviceReal.Synchronize();
                auto sumDeviceReal = imgDeviceReal.Sum<double>();
                //std::cout << "\tReal image size = (" << width << " x " << height << "), host sum = " << sumHostReal << ", device sum = " << sumDeviceReal << "\n";
                ASSERT_NEAR(sumHostReal, sumDeviceReal, width * height * 1e-4) << "Summation on host and device did not match for real component.\n";

                // Sum with ROI, real part only
                auto ROI = images::Rectangle<int>((int)(rng.NextUniform(0.0f, 1.0f) * width), (int)(rng.NextUniform(0.0f, 1.0f) * height), (int)(rng.NextUniform(0.0f, 1.0f) * width), (int)(rng.NextUniform(0.0f, 1.0f) * height));
                if (ROI.IsContainedIn(imgReal.Bounds()))
                {
                    imgReal.ToHost();
                    auto sumROIHost = imgReal.Sum<double>(ROI);
                    auto sumROIDevice = imgDeviceReal.Sum<double>(ROI);
                    ASSERT_NEAR(sumROIHost, sumROIDevice, width * height * 1e-4) << "Summation on host and device did not match for real component, ROI region of " << ROI << "\n";
                    auto ROIArea = ROI.Area();
                    auto ImgArea = imgReal.Bounds().Area();
                    auto ROIFraction = (double)ROIArea / (double)ImgArea;
                    ASSERT_NEAR(sumROIHost, sumHostReal * ROIFraction, width* height * 1e-2) << "Summation on host did not match fraction expected for ROI region of " << ROI << " (area " << ROIArea << " pixels^2) out of full image " << imgReal.Bounds() << " (area " << ImgArea << " pixels^2)\n";
                }

                // Complex
#if 0     // Temporarily disabled, see ComplexImage::Sum()
                auto sumHostComplex = img.Sum<thrust::complex<double>>();
                auto imgDeviceComplex = img.CopyToDevice(HostFlags::Pinned);
                imgDeviceComplex.Synchronize();
                auto sumDeviceComplex = imgDeviceComplex.Sum<thrust::complex<double>>();
                //std::cout << "\tComplex Image size = (" << width << " x " << height << "), host sum = " << sumHostComplex << ", device sum = " << sumDeviceComplex << "\n";
                ASSERT_NEAR(sumHostComplex, sumDeviceComplex, width * height * 1e-4) << "Summation on host and device did not match for complex image.\n";
#endif

                TotalPixels += (width * height);
            }
            sw.Stop();
            double PixelsPerSecond = (double)TotalPixels / sw.GetElapsedSeconds();
            std::cout << std::to_string(NIterations) << " iterations required " << sw.GetElapsedSeconds() << " seconds, " << PixelsPerSecond << " pixels/second, or " << (PixelsPerSecond * 8 / 1.0e9) << " GB/s.\n";
        }

        /** cuFFT test 1 **/
#if 0
        {
            auto imgA = Image<float>::NewHostImage(32, 32, gStream);
            imgA.Fill(10.0f);
            float freq = imgA.Height() / 4.0f;
            for (int yy = 0; yy < imgA.Height(); yy++)
            {
                for (int xx = 0; xx < imgA.Width(); xx++)
                {
                    imgA(xx, yy) += (float)cos(2.0 * M_PI * freq * (float)yy/imgA.Height());
                }
            }

            auto imgB = Image<thrust::complex<float>>::NewHostImage(imgA.Width(), imgA.Height(), gStream, HostFlags::Pinned);
            imgB.FillZero();
            imgB.SetReal(imgA);
            imgA.Save("imgA.tif");
            std::cout << "Image B:\n" << imgB << "\n";

            FFTPlan plan(imgB.Width(), imgB.Height(), CUFFT_C2C, gStream);
            auto imgFB = plan.Forward(imgB);
            imgFB.ToHost(false);
            imgFB.Synchronize();
            std::cout << "Image F_B:\n" << imgFB << "\n";
            for (int yy = 0; yy < imgFB.Height(); yy++)
            {
                for (int xx = 0; xx < imgFB.Width(); xx++)
                {
                    if (abs(imgFB(xx, yy)) > 1.0e-3) std::cout << "Context @ " << xx << "," << yy << ": " << imgFB(xx, yy) << "\n";
                }
            }
            auto imgFB_re = imgFB.GetReal();
            auto imgFB_im = imgFB.GetImag();
            imgFB_re.Save("imgFB_real.tif");
            imgFB_im.Save("imgFB_imag.tif");
        }
#endif

        /** Color, TIFF/PNG image load & save and cuFFT test 2 **/
        {
            std::cout << "Loading baboon test image and running FFT testing...\n";

            /** Load baboon test image and ensure it matches itself after a load-save-load cycle **/
            auto imgColor = Image<RGBPixel>::Load("data\\baboon.png", gStream);
            string test_fn = "baboon_loadsaveload.tif";
            if (io::File::Exists(test_fn)) io::File::Delete(test_fn);
            imgColor.Save(test_fn);
            auto imgColor2 = Image<RGBPixel>::Load(test_fn, gStream);
            ASSERT_EQ(imgColor.Width(), imgColor2.Width()) << "Baboon test image width did not match after load-save-load cycle.";
            ASSERT_EQ(imgColor.Height(), imgColor2.Height()) << "Baboon test image height did not match after load-save-load cycle.";
            for (int yy = 0; yy < imgColor.Height(); yy++)
            {
                for (int xx = 0; xx < imgColor.Width(); xx++)
                {
                    ASSERT_EQ((int)imgColor(xx, yy).R, (int)imgColor2(xx, yy).R) << "Pixel (" << xx << "," << yy << ") R from baboon test image did not match after load-save-load cycle.";
                    ASSERT_EQ((int)imgColor(xx, yy).G, (int)imgColor2(xx, yy).G) << "Pixel (" << xx << "," << yy << ") G from baboon test image did not match after load-save-load cycle.";
                    ASSERT_EQ((int)imgColor(xx, yy).B, (int)imgColor2(xx, yy).B) << "Pixel (" << xx << "," << yy << ") B from baboon test image did not match after load-save-load cycle.";
                }
            }
            io::File::Delete(test_fn);

            /** Convert baboon test image to grayscale and then to the real part of a complex image **/
            auto imgBaboon = imgColor.ConvertToGrayscale<float>();
            auto imgComplexBaboon = Image<thrust::complex<float>>::NewHostImage(imgBaboon.Width(), imgBaboon.Height(), gStream, HostFlags::Pinned);
            imgComplexBaboon.FillZero();
            imgComplexBaboon.SetReal(imgBaboon);
            //std::cout << "Mean of imgComplexBaboon just before FFT: " << imgComplexBaboon.Mean<thrust::complex<double>>() << "\n";

            /** Perform forward FFT **/
            FFTPlan<float> plan(imgComplexBaboon.Width(), imgComplexBaboon.Height(), FFTType::ComplexToComplex, gStream);
            auto imgFB = imgComplexBaboon.CopyToHost();
            plan.Forward(imgFB, imgComplexBaboon);
            imgFB.ToHost(false);
            imgFB.Synchronize();
            //imgFB.GetReal().Save("Freq_Baboon_Real.tif");
            //imgFB.GetImag().Save("Freq_Baboon_Imag.tif");

            /** Load reference FFT results **/
            auto imgRefBaboonFFTReal = Image<float>::Load("data\\numpy_fft_baboon_real.tif", gStream);
            auto imgRefBaboonFFTImag = Image<float>::Load("data\\numpy_fft_baboon_imag.tif", gStream);
            auto imgRefBaboonFFT = Image<thrust::complex<float>>::NewHostImage(imgRefBaboonFFTReal.Width(), imgRefBaboonFFTReal.Height(), gStream, HostFlags::None);
            imgRefBaboonFFT.SetReal(imgRefBaboonFFTReal);
            imgRefBaboonFFT.SetImag(imgRefBaboonFFTImag);

            /** Calculate error against reference and check for match **/
            auto imgError = imgRefBaboonFFT - imgFB;
            auto imgAbsError = imgError.Absolute(HostFlags::None);            
            auto sumAbsError = imgAbsError.Sum<double>();
            auto maxAbsError = imgAbsError.Max();
            std::cout << "FFT Sum of Absolute Error (this code vs. numpy reference): " << std::to_string(sumAbsError) << "\n";
            std::cout << "FFT Max of Absolute Error (this code vs. numpy reference): " << std::to_string(maxAbsError) << "\n";
            ASSERT_NEAR(0.0f, sumAbsError, imgError.Width() * imgError.Height() * 1e-4) << "Mismatch in FFT result between this code and numpy result for baboon grayscale image (sum error exceeded).\n";

            /** Perform inverse FFT **/
            auto imgBaboonPrimeComplex = Image<thrust::complex<float>>::NewHostImage(imgFB.Width(), imgFB.Height());
            plan.Inverse(imgBaboonPrimeComplex, imgFB);            
            auto imgBaboonPrime = imgBaboonPrimeComplex.Absolute();
            imgBaboonPrime /= (imgBaboonPrime.Width() * imgBaboonPrime.Height());           // cuFFT computes unnormalized FFTs such that IFFT(FFT(A))=n A, where n is # of elements.  So remove this factor.
            imgBaboonPrime.ToHost(false);
            imgBaboonPrime.Synchronize();            
            //imgBaboonPrime.Save("ABS_IFFT_FFT_Baboon.tif");
            //imgBaboon.Save("Baboon_grayscale.tif");

            /** Calculate error against original and check for match **/
            auto imgError2 = imgBaboon - imgBaboonPrime;
            auto imgAbsError2 = imgError2.Absolute(HostFlags::None);
            auto sumAbsError2 = imgAbsError2.Sum<double>();
            auto maxAbsError2 = imgAbsError2.Max();
            std::cout << "ABS_IFFT_FFT Sum of Absolute Error (this code vs. numpy reference): " << std::to_string(sumAbsError2) << "\n";
            std::cout << "ABS_IFFT_FFT Max of Absolute Error (this code vs. numpy reference): " << std::to_string(maxAbsError2) << "\n";
            ASSERT_NEAR(0.0f, sumAbsError2, imgError2.Width()* imgError2.Height() * 1e-4) << "Mismatch in Absolute value of IFFT of FFT result against original test image in grayscale.\n";
        }
    }
    catch (std::exception& ex)
    {
        ASSERT_TRUE(false) << "An exception occurred: " << std::string(ex.what());
    }

    FINISH_TEST();
}

#endif

