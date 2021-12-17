/////////
//	PixelType_Primitives.h
/////////

#ifndef __WBPixelType_Primitives_h__
#define __WBPixelType_Primitives_h__

#include "../../wbFoundation.h"

namespace wb
{
	namespace images
	{		
		/** Pixel elements **/

		#pragma pack(push,1)
		struct RGBPixel
		{
			byte	R;
			byte	G;
			byte	B;

			RGBPixel() { R = G = B = 0; }
			RGBPixel(const RGBPixel& cp) : R(cp.R), G(cp.G), B(cp.B) { }
			RGBPixel(RGBPixel&& mv) : R(mv.R), G(mv.G), B(mv.B) { }

			RGBPixel& operator=(const RGBPixel& cp) { R = cp.R; G = cp.G; B = cp.B; return *this; }
			RGBPixel& operator=(RGBPixel&& mv) { R = mv.R; G = mv.G; B = mv.B; return *this; }

			RGBPixel& operator+=(const RGBPixel& rhs)
			{
				R += rhs.R; G += rhs.G; B += rhs.B;
				return *this;
			}

			RGBPixel operator+(const RGBPixel& rhs) const
			{
				RGBPixel ret;
				ret.R = R + rhs.R;
				ret.G = G + rhs.G;
				ret.B = B + rhs.B;
				return ret;
			}

			RGBPixel& operator-=(const RGBPixel& rhs)
			{
				R -= rhs.R; G -= rhs.G; B -= rhs.B;
				return *this;
			}

			RGBPixel operator-(const RGBPixel& rhs) const
			{
				RGBPixel ret;
				ret.R = R - rhs.R;
				ret.G = G - rhs.G;
				ret.B = B - rhs.B;
				return ret;
			}

			RGBPixel& operator/=(double Scalar)
			{
				R = (byte)(R / Scalar);
				G = (byte)(G / Scalar);
				B = (byte)(B / Scalar);
				return *this;
			}

			RGBPixel operator/(double Scalar) const
			{
				RGBPixel ret;
				ret.R = (byte)(R / Scalar);
				ret.G = (byte)(G / Scalar);
				ret.B = (byte)(B / Scalar);
				return ret;
			}
		};

		struct RGBAPixel
		{
			byte	R;
			byte	G;
			byte	B;
			byte	A;
		};

		struct ABGRPixel
		{
			byte	A;
			byte	B;
			byte	G;
			byte	R;

			RGBAPixel ToRGBA() { RGBAPixel ret; ret.R = R; ret.G = G; ret.B = B; ret.A = A; return ret; }
		};
		#pragma pack(pop)	

	}// namespace images
}// namespace wb

#endif	// __WBPixelType_Primitives_h__

//	End of PixelType_Primitives.h


