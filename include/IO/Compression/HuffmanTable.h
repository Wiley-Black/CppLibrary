/////////
//	HuffmanTable.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBHuffmanTable_h__
#define __WBHuffmanTable_h__

/** Table of contents **/

namespace wb {
	namespace io {
		namespace compression {
			class HuffmanTable;
		}
	}
}

/** Dependencies **/

#include "../BitStream.h"

/** Content **/

namespace wb
{
	namespace io
	{
		namespace compression
		{			
			class HuffmanTable
			{
			public:				
				/// <summary>
				/// UnusedSymbols gives the number of symbols which have no code.  These symbols do not
				/// appear in the decompressed data and therefore do not require a code representation.
				/// </summary>
				int UnusedSymbols;

				/// <summary>
				/// UnusedCodes gives the number of codes which are not assigned to a symbol.
				/// </summary>
				int UnusedCodes;                        							

			private:
				int MinCodeLength;			   // Smallest code length, in bits.
				uint FirstCode[16];              // Given a code length (bits, index), give the first code defined at that length.  Inclusive.
				uint LastCode[16];               // Given a code length (bits, index), give the last code defined at that length.  Exclusive.
				uint* SymbolTable;            // Given a code (index), return a symbol

			public:
				HuffmanTable()
				{
					SymbolTable = nullptr;
				}
            
				HuffmanTable(uint* CodeLength, int NumCodes)
				{
					SymbolTable = nullptr;
					Init(CodeLength, NumCodes);
				}
				
				HuffmanTable(HuffmanTable&& mv)
				{
					UnusedSymbols = mv.UnusedSymbols;
					UnusedCodes = mv.UnusedCodes;
					MinCodeLength = mv.MinCodeLength;
					MoveMemory (FirstCode, mv.FirstCode, sizeof(FirstCode));
					MoveMemory (LastCode, mv.LastCode, sizeof(LastCode));
					SymbolTable = mv.SymbolTable;
					mv.SymbolTable = nullptr;
				}
				
				~HuffmanTable()
				{
					if (SymbolTable != nullptr) { delete[] SymbolTable; SymbolTable = nullptr; }
				}				

				/// <summary>
				/// Performs the transformation described in RFC 1951 Section 3.2.2.  
				/// </summary>
				/// <param name="CodeLength">The length, in bits, of each code.  A zero value indicates
				/// a symbol which will not be used.  The indices of this array correspond to the
				/// symbols being represented.</param>
				/// <returns></returns>
				void Init(uint* CodeLength, int NumCodes)
				{
					/** Example, from RFC 1951:
					 *  Consider the alphabet ABCDEFGH, with bit lengths (3, 3, 3, 3, 3, 2, 4, 4).
					 */
				
					/// The CLCount array gives the number of codes sharing a common bit-length.  For example,
					/// CLCount[3] gives the number of symbols which are represented by 3-bit codes.  CLCount[0]
					/// gives the number of symbols that have no code.					
					uint CLCount[16];
					ZeroMemory (CLCount, sizeof(CLCount));

					MinCodeLength = 1;

					// 1. Count the number of codes for each code length. Let bl_count[bl] be the number of codes of length bl, bl >= 1.					
					for (int symbol = 0; symbol < NumCodes; symbol++) 
					{
						if (CodeLength[symbol] >= 16) throw ArgumentException("Code lengths cannot exceed 15-bits.");
						CLCount[CodeLength[symbol]]++;
					}
                
					UnusedSymbols = (int)CLCount[0];
					CLCount[0] = 0;

					// Unused symbols have a CodeLength of zero, which will add to bl_count[0].  If they all
					// are unused we have a problem...
					if (CLCount[0] == (uint)NumCodes) throw Exception("No symbols used!");

					/** After step 1, we have:
					 *  N           = {   ..., 2,      3,      4, ... }
					 *  bl_count[N] = { 0, 0,  1,      5,      2, ... }     (a.k.a. CLCount)
					 */

					// 2) Find the numerical value of the smallest code for each code length:
					uint next_code[16];
					uint code = 0;
					for (int bits = 1; bits < 16; bits++)
					{
						code = (code + CLCount[bits - 1]) << 1;
						next_code[bits] = code;
					}

					/** Step 2 computes the following next_code values:
					 *  N               = { 0,  1,  2,  3,  4, ... }
					 *  next_code[N]    = { 0,  0,  0,  2, 14, ... }
					 */

					// 3) Assign numerical values to all codes, using consecutive values for all codes of the same length 
					// with the base values determined at step 2. Codes that are never used (which have a bit length of 
					// zero) must not be assigned a value.
                
					/** When decoding, our first task will be to identify to bit length.  Since codes
					 *  are all consecutive for the same bit length and a code never repeats as a
					 *  prefix in a longer code, this can be accomplished by checking the range as
					 *  we read in the sequence. **/					
					uint FinalCodeP1 = 0;
					for (int bits = 1; bits < 16; bits++)
					{
						FirstCode[bits] = (uint)next_code[bits];
						LastCode[bits] = (uint)next_code[bits] + (uint)CLCount[bits];
						if (CLCount[bits] > 0) FinalCodeP1 = LastCode[bits];
					}

					for (int bits = 15; bits > 0; bits--)
					{
						if (CLCount[bits] > 0) MinCodeLength = bits;
					}
                
					/** Step 3+ **/

					#if 0
					Debug.Write("Huffman next_code[] table:\n");
					for (int ii = 0; ii < next_code.Length; ii++)
					{
						if (CLCount[ii] > 0)
							Debug.Write(string.Format("\t[{0}] {1} (Count={2})\n", ii, next_code[ii], CLCount[ii]));
						else
							Debug.Write(string.Format("\t[{0}] {1}\n", ii, next_code[ii]));
					}
					Debug.Write("End table.\n");
					#endif
                
					/** When decoding and after having identified the bit length, the code comes
					 *  easy.  Once we have the code, we need the symbol.  The Symbol array will
					 *  give us a symbol given a code. **/
					if (SymbolTable != nullptr) delete[] SymbolTable;
					SymbolTable = new uint[FinalCodeP1];
					code = 0;
					for (uint symbol = 0; symbol < (uint)NumCodes; symbol++)
					{
						if (CodeLength[symbol] > 0)
						{
							assert (CodeLength[symbol] < 16);
							code = next_code[CodeLength[symbol]]++;
							SymbolTable[code] = symbol;
						}
					}

					#if 0
					Debug.Write("Huffman CLCount[] table:\n");
					for (int ii = 0; ii < this.CLCount.Length; ii++) Debug.Write(string.Format("\t[{0}] {1}\n", ii, this.CLCount[ii]));
					Debug.Write("End table.\n");
					#endif

					/** Step 3 produces the following code values:
						Symbol Length   Code
						------ ------   ----
						A       3        010 (2)
						B       3        011 (3)
						C       3        100 (4)
						D       3        101 (5)
						E       3        110 (6)
						F       2         00 (0)
						G       4       1110 (14)
						H       4       1111 (15)
                  
						We've also used the codes to build up the SymbolTable.
					 */
				}

				uint Decode(LSBBitStream& Src)
				{
					// TODO: Optimization!  This function is by far the most time-critical function in
					// all of DeflateStreamEx.

					uint code = 0;
					for (int bits = 0; bits < MinCodeLength; bits++)
					{
						code <<= 1;
						code |= Src.ReadBits(1);
					}
                
					for (int bits = MinCodeLength; bits < 16; bits++)
					{
						// We do not yet know if the code is complete or if we need
						// to read more bits...to find out, we check the valid range
						// of codes at this bit length...
						if (code < LastCode[bits] && code >= FirstCode[bits])
						{
							// The posited code falls within a valid range of codes at
							// this length.  Since a utilized code can never occur as a 
							// prefix to a longer code (by the rules), we have located
							// the code.  Now return the symbol.
							return SymbolTable[code];
						}

						// The code is incomplete - we need more bits!
						code <<= 1;
						code |= Src.ReadBits(1);
					}

					throw Exception("Compressed stream corrupt - an unassigned or invalid code was found within the stream.");
				}
			};			
		}
	}
}

#endif	// __WBHuffmanTable_h__

//	End of HuffmanTable.h

