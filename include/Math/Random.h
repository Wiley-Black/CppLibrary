/////////
//  Random.h
//  Copyright (C) 1998-2014 by Wiley Black
//

#ifndef __WBRandom_h__
#define __WBRandom_h__

/** Dependencies **/

#include "wbFoundation.h"
#include <math.h>
#include <time.h>
#include <assert.h>

namespace wb
{
	/**	Random Number Generators **/

	class Random
	{
		bool  m_bSeedInitialized;

		Int32 m_nRandomSeed1;
		Int32 m_nRandomSeed2;

		enum { RandomTableSize = 64 };
		Int32 iv[RandomTableSize];
		Int32 iy;

		// TODO: Should make these constants, but don't want to require a cpp file.
		#define IM1		2147483563
		#define IM1d	2147483563.0
		#define IM2		2147483399
		#define IA1		40014
		#define IA2		40692
		#define IQ1		53668
		#define IQ2		52774
		#define IR1		12211
		#define IR2		3791

		#define NDIV	((Int32)(1+((IM1-1)/RandomTableSize)))

		// We compute gaussians two-at-a-time, so we store one to be returned as the next value.
		double	m_dNextGaussian;
		bool	m_bNextGaussian;

		// For NextBit():
		UInt32	m_nNextBitSeed;

		// For NextFast():
		UInt32	m_nNextFastSeed;

	public:

		/// <summary>Initializes the random number generator with a random seed (release builds) or a zero seed (debug 
		/// builds).  Call with a seed to override or call the Seed() method.</summary>
		Random()
		{
			m_bSeedInitialized = false;
			m_bNextGaussian = false;

			iy = 0;
			Seed(0);
//SeedFromClock();
		}

		Random(UInt32 nSeed)
		{
			m_bSeedInitialized = false;
			m_bNextGaussian = false;
			iy = 0;
			Seed(nSeed);
		}

			/** Seed()
				Provides an optional seed for random number generators.  Only needed when
				intentionally trying to duplicate or synchronize random sequences.  When
				not called, the random number generators automatically seed themselves
				based on system time.

				Random Number generators derived from Numerical Recipes in C, Second Edition.
				Uniform and Seed generators from 'ran2()' method.  The "perfect" generator.
			**/
		void   Seed(UInt32 nSeed);	

		void   SeedFromClock()
		{
			unsigned long nSeconds = (unsigned long)time(nullptr);
			#ifdef Windows
			Seed( (nSeconds * 10000) + (GetTickCount() % 10000) );
			#else
			Seed(nSeconds * 10000);
			#endif
		}

			/** NextUniform()
				Returns a uniformly-distributed random number in the range of 0.0 to 
				1.0 (exclusive).

				Note: This is a very good random number generator.  The distribution
				is uniform for all practical (and most imaginable) purposes.  If cast
				to single-precision, the values will be random.  If used as 
				double-precision, the values will still be uniformly distributed, 
				and better than single-precision, but will not completely fill the 
				machine precision space.

				From Numerical Recipes.
			**/
		double NextUniform();

			/** NextUniform()
				Returns a uniformly-distributed random number in the range [Minimum,Maximum].
			**/
		double NextUniform(double Minimum, double Maximum);

		/// <summary>NextUniform() returns a uniformly-distributed random number in the range [Minimum,Maximum].</summary>
		double NextUniform(double Minimum, float Maximum);

		/// <summary>NextUniform() returns a uniformly-distributed random number in the range [Minimum,Maximum].</summary>
		double NextUniform(float Minimum, double Maximum);

		/// <summary>NextUniform() returns a uniformly-distributed random number in the range [Minimum,Maximum].</summary>
		float NextUniform(float Minimum, float Maximum);

		/// <summary>NextUniform() returns a uniformly-distributed random number in the range [Minimum,Maximum].</summary>
		int NextUniform(int Minimum, int Maximum);

			/**	RandomGaussian()
				Uses the Box-Muller method to generate a gaussian-distributed random number with a
				mean of zero and a standard deviation of one.  
			**/
		double NextGaussian();

			/** NextBit()
				Provides a fast one-bit random number.  Not suitable for constructing a large,
				supposedly random, integer or as the mantissa of a supposedly random floating-
				point number.
			**/
		bool NextBit();

			/** NextFast()
				Provides the "quick-and-dirty" generator outlined in Numerical Recipes (2nd ed)
				on page 284.  Provides a 16-bit pseudorandom number.
			**/
		UInt16 NextFast();
		
			/** NextPoisson()
			Uses the inversion by sequential search method found at:
			http://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
			**/
		UInt32 NextPoisson(UInt32 Lambda);
	};

	/** Implementation **/

	inline void Random::Seed(UInt32 nSeed)
	{
		if( !nSeed ) nSeed = 1;						// Zero cannot be used as seed.  Replace with 1.
		nSeed = nSeed % IM1;						// Maximum allowed seed value is IM1-1.

		m_nRandomSeed1 = m_nRandomSeed2 = nSeed;
		for( Int32 jj = RandomTableSize + 7; jj >= 0; jj-- ){			// Load the shuffle table (after 8 warm-ups).
			Int32 k = m_nRandomSeed1 / IQ1;
			m_nRandomSeed1 = IA1 * (m_nRandomSeed1 - k * IQ1) - k * IR1;
			if (m_nRandomSeed1 < 0) m_nRandomSeed1 += IM1;
			if (jj < RandomTableSize) iv[jj] = m_nRandomSeed1;
		}
		iy = iv[0];

		// Verify the state on exit, such that if the assertions at the start of NextUniform() trigger it shows clearly that external corruption occurred.
		assert (m_nRandomSeed1 > 0 && m_nRandomSeed1 < IM1);			// 9-8-17: I believe this is a required condition based on the %IM1 in the Seed() call.  Want to make sure no corruption of member variables happens outside NextUniform().
		assert (m_nRandomSeed2 > 0 && m_nRandomSeed2 < IM1);			//				"
		assert (iy > 0 && iy < IM1);									//				"		Note: verified passing for 100M repeated calls on 9-8-17.

		m_nNextBitSeed = nSeed;
		m_nNextFastSeed = nSeed;

		m_bSeedInitialized = true;
	}

	/** Random Number Generators **/

	inline double Random::NextUniform()
	{ 
		/** Implements ran2() in Chapter 7 (Random Numbers) on pg 282 of Numerical Recipes in C 2nd Edition **/

		const double eps = DBL_EPSILON;

		assert (m_bSeedInitialized);
		assert (m_nRandomSeed1 != 0 && m_nRandomSeed2 != 0);
		assert (m_nRandomSeed1 > 0 && m_nRandomSeed1 < IM1);			// 9-8-17: I believe this is a required condition based on the %IM1 in the Seed() call.  Want to make sure no corruption of member variables happens outside NextUniform().
		assert (m_nRandomSeed2 > 0 && m_nRandomSeed2 < IM1);			//				"
		assert (iy > 0 && iy < IM1);									//				"		Note: verified passing for 100M repeated calls on 9-8-17.
			// It is still possible for the table to end up corrupted outside of NextUniform().  Not verifying that here because it would add a good bit more time.
			// If these state variables do end up corrupted, consider that Random is not thread-safe as a first place to look.  Multiple threads asking for random numbers
			// from the same Random instance can cause failures.

		Int32 k = m_nRandomSeed1 / IQ1;
		m_nRandomSeed1 = IA1 * (m_nRandomSeed1 - k * IQ1) - k * IR1;	// Compute idum=(IA1*idum) % IM1 without overflows by Schrage's method.
		if( m_nRandomSeed1 < 0 ) m_nRandomSeed1 += IM1;
		k = m_nRandomSeed2 / IQ2;
		m_nRandomSeed2 = IA2 * (m_nRandomSeed2 - k * IQ2) - k * IR2;	// Compute idum2=(IA2*idum2) % IM2 likewise.
		if( m_nRandomSeed2 < 0 ) m_nRandomSeed2 += IM2;

		if( !m_nRandomSeed1 ) m_nRandomSeed1 ++;						// Prevent a zero from entering the sequence.  Zeros are just bad here.
		if( !m_nRandomSeed2 ) m_nRandomSeed2 ++;						// Prevent a zero from entering the sequence.  Zeros are just bad here.

		assert(iy < IM1);

		Int32 j = iy/NDIV;												// Will be in the range 0..NTAB-1.
		assert( j < RandomTableSize );
		iy = iv[j] - m_nRandomSeed2;									// Here idum is shuffled, idum and idum2 are combined to generate output.
		iv[j] = m_nRandomSeed1;
		if( iy < 1 ) iy += (IM1 - 1);
	
		double dNormalize = (1.0/IM1d) * iy;
		if( dNormalize > 1.0 - eps ) dNormalize = 1.0 - eps;	// Prevent endpoint value for symmetry (since 0.0 isn't possible) and as expected.

		// Verify the state on exit, such that if the assertions at the start of NextUniform() trigger it shows clearly that external corruption occurred.
		assert (m_nRandomSeed1 > 0 && m_nRandomSeed1 < IM1);			// 9-8-17: I believe this is a required condition based on the %IM1 in the Seed() call.  Want to make sure no corruption of member variables happens outside NextUniform().
		assert (m_nRandomSeed2 > 0 && m_nRandomSeed2 < IM1);			//				"
		assert (iy > 0 && iy < IM1);									//				"		Note: verified passing for 100M repeated calls on 9-8-17.

		assert( dNormalize > 0.0 && dNormalize < 1.0 );
		return dNormalize;
	}

	inline double Random::NextUniform(double Minimum, double Maximum)
	{
		// If the caller swapped their Minimum and Maximum, we can handle that...
		if (Maximum >= Minimum)
			return (NextUniform() * (Maximum-Minimum)) + Minimum;
		else
			return (NextUniform() * (Minimum-Maximum)) + Maximum;
	}

	inline double Random::NextUniform(double Minimum, float Maximum) { return NextUniform((double)Minimum, (double)Maximum); }
	inline double Random::NextUniform(float Minimum, double Maximum) { return NextUniform((double)Minimum, (double)Maximum); }
	inline float Random::NextUniform(float Minimum, float Maximum) { return (float)NextUniform((double)Minimum, (double)Maximum); }

	inline int Random::NextUniform(int Minimum, int Maximum)
	{
		return Round(NextUniform((double)Minimum, (double)Maximum));
	}

	inline double Random::NextGaussian()
	{
		if (m_bNextGaussian){
			m_bNextGaussian = false;
			return m_dNextGaussian;
		}

		double x1, x2, w, y1;

		do {
			x1 = 2.0 * NextUniform() - 1.0;
			x2 = 2.0 * NextUniform() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * log( w ) ) / w );
		y1 = x1 * w;
		m_dNextGaussian = x2 * w;		// We only return one of the 2 generated variables, but save it for next use.
		m_bNextGaussian = true;

		return y1;
	}

	inline bool Random::NextBit()
	{
		/** From Numerical Recipes in C, 2nd Edition, Section 7.4, Method II **/

		static const UInt32 IB1 = 1;
		static const UInt32 IB2 = 2;
		static const UInt32 IB5 = 16;
		static const UInt32 IB18 = 131072;
		static const UInt32 MASK = (IB1+IB2+IB5);

		if (m_nNextBitSeed & IB18) {
			m_nNextBitSeed = ((m_nNextBitSeed ^ MASK) << 1) | IB1;
			return true;
		}
		else {
			m_nNextBitSeed <<= 1;
			return false;
		}
	}

	inline UInt16 Random::NextFast()
	{
		/** From Numerical Recipes in C, 2nd Edition, Section 7.1, "Quick and Dirty Generators" **/

		static const UInt32 im = 714025, ia = 4096, ic = 150889;
		m_nNextFastSeed = (m_nNextFastSeed*ia + ic) % im;
		// return (UInt16)( (((UInt32)UInt16_MaxValue+1)*m_nNextFastSeed) / im );			// Proper
		return (UInt16)m_nNextFastSeed;														// Even dirtier, not verified.
	}
	
	inline UInt32 Random::NextPoisson(UInt32 Lambda)
	{
		UInt32 x = 0;
		double p = exp(-(double)Lambda);
		double s = p;
		double u = NextUniform();
		while (u > s)
		{
			x++;
			p = p * Lambda / (double)x;
			s += p;
		}
		return x;
	}

}

#endif  // __WBRandom_h__

//  End of Random.h

