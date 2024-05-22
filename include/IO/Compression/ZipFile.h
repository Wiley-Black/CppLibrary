/////////
//	ZipFile.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBZipFile_h__
#define __WBZipFile_h__

/** Table of contents **/

namespace wb {
	namespace io {
		namespace compression {
			class ZipArchiveEntry;
			class ZipArchiveDirectoryInfo;
			class ZipArchive;
		}
	}
}

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../Processing/CRC.h"
#include "DeflateStream.h"
#include "../StreamFragment.h"

/** Content **/

namespace wb
{
	namespace io
	{
		namespace compression
		{
			enum_class_start(ZipArchiveMode,int)
			{
				Create,
				Read,
				Update
			}
			enum_class_end(ZipArchiveMode);
			
			struct ZipVersion
			{ 
				UInt8 PlatformID;
				UInt8 Major;
				UInt8 Minor;

				ZipVersion(UInt16 EncodedVersion);
			};

			class ZipArchiveEntry : public FileInfo
			{
				enum_class_start(CompressionMethods,UInt16)
				{
					None = 0,				// The file is stored (no compression)
					Shrunk = 1,				// The file is Shrunk
					ReducedFactor1 = 2,		// The file is Reduced with compression factor 1
					ReducedFactor2 = 3,		// The file is Reduced with compression factor 2
					ReducedFactor3 = 4,		// The file is Reduced with compression factor 3
					ReducedFactor4 = 5,		// The file is Reduced with compression factor 4
					Imploded = 6,			// The file is Imploded
					ReservedTokenizing = 7,	// Reserved for Tokenizing compression algorithm
					Deflate = 8,			// The file is Deflated
					Deflate64 = 9,			// Enhanced Deflating using Deflate64(tm)
					IBM_TERSE_Old = 10,		// PKWARE Data Compression Library Imploding (old IBM TERSE)
					Reserved11 = 11,		// Reserved by PKWARE
					BZIP2 = 12,				// File is compressed using BZIP2 algorithm
					Reserved13 = 13,		// Reserved by PKWARE
					LZMA = 14,				// LZMA (EFS)
					Reserved15 = 15,		// Reserved by PKWARE
					Reserved16 = 16,		// Reserved by PKWARE
					Reserved17 = 17,		// Reserved by PKWARE
					IBM_TERSE_New = 18,		// File is compressed using IBM TERSE (new)
					LZ77 = 19,				// IBM LZ77 z Architecture (PFS)
					WavPack = 97,			// WavPack compressed data
					PPMdVersion1_1 = 98		// PPMd version I, Rev 1
				}
				enum_class_end(CompressionMethods);

				enum_class_start(GeneralFlags,UInt16)
				{
					Encrypted = 0x0001,
					CompressionTechnique = 0x0006,		// Meaning depends on CompressionMethod employed.
					
					/// <summary>If DataDescriptor is set, then a data descriptor record follows the data.  The CRC-32, 
					/// compressed size, and uncompressed size fields are stored in the data descriptor and are zero in
					/// the local header.</summary>
					DataDescriptor = 0x0008,
					
					PatchedData = 0x0020,
					StrongEncryption = 0x0040,
					FilenameInUTF8 = 0x0800,
					EncryptedCentralDirectory = 0x2000,

					Reserved = 0x0010 | 0x0080 | 0x0100 | 0x0200 | 0x0400 | 0x1000 | 0x4000 | 0x8000
				}
				enum_class_end(GeneralFlags);

				#pragma pack(push,1)

				/// <summary>The BaseFileRecord is the directory entry for an individual file (and directory?).  It contains
				/// an offset (LocalHeaderOffset) to the location where the file's local header entry and data reside.  The
				/// complete FileRecord also includes 3 variable fields that are not included in the BaseFileRecord but will
				/// immediately follow it and whose lengths are specified in BaseFileRecord members.</summary>
				struct BaseFileRecord
				{
					enum { SignatureValue = 0x02014b50 };

					UInt32	Signature;				// central file header signature (0x02014b50)
					UInt16	WriterVersion;			// version made by
					UInt16	RequiredReaderVersion;	// version needed to extract
					UInt16	Flags;					// general purpose bit flag
					UInt16	CompressionMethod;		// compression method              
					UInt16	LastWriteTime;			// last mod file time
					UInt16	LastWriteDate;			// last mod file date
					UInt32	CRC32;
					UInt32	CompressedSize;			// compressed size
					UInt32	UncompressedSize;	
					UInt16	FileNameLength;
					UInt16	ExtraFieldLength;
					UInt16	FileCommentLength;
					UInt16	StartingDiskNumber;		// disk number start
					UInt16	InternalAttributes;		// internal file attributes
					UInt32	ExternalAttributes;		// external file attributes
					UInt32	LocalHeaderOffset;		// relative offset of local header

					// file name (variable size)
					// extra field (variable size)
					// file comment (variable size)
				};

				/// <summary>The LocalFileHeader is found at the archive's HeaderSize offset plus the BaseFileRecord's
				/// LocalHeaderOffset.  If encrypted, it is followed by an encryption header and then the compressed [and
				/// encrypted] file data.  Finally, the file's data is followed by the data descriptor record.</summary>
				struct LocalFileHeader
				{
					enum { SignatureValue = 0x04034b50 };

					UInt32	Signature;
					UInt16	RequiredReaderVersion;
					UInt16	Flags;					// General purpose bit flags
					UInt16	CompressionMethod;
					UInt16	LastWriteTime;
					UInt16	LastWriteDate;
					UInt32	CRC32;
					UInt32	CompressedSize;
					UInt32	UncompressedSize;
					UInt16	FileNameLength;
					UInt16	ExtraFieldLength;

					// file name (variable size)
					// extra field (variable size)
				};

				#pragma pack(pop)				

				ZipArchive					*m_pArchive;
				unique_ptr<BaseFileRecord>	m_pRecord;				

				friend class ZipArchive;
				ZipArchiveEntry(ZipArchive* pArchive, unique_ptr<BaseFileRecord>&& pRecord, const string& FullName_);

			public:
				ZipArchiveEntry();
				ZipArchiveEntry(const ZipArchiveEntry&);
				~ZipArchiveEntry();
				ZipArchiveEntry& operator=(const ZipArchiveEntry&);
				
				/// <summary>
				/// Retrieves the full path of the directory or file within the archive.  For example, "Path\Examples.xml".
				/// </summary>
				Path GetFullName() const { return FullName; }

				/// <summary>
				/// Retrieves the uncompressed size of the file.
				/// </summary>
				/// <returns></returns>
				UInt64 GetLength() { return m_pRecord->UncompressedSize; }

				/// <summary>Decompresses and opens the contents of the archived file.  The caller must delete the returned Stream when
				/// complete.  Only one entry within a ZipArchive should be opened at a time.</summary>
				Stream* Open();

				ZipVersion	GetRequiredReaderVersion() { return ZipVersion(m_pRecord->RequiredReaderVersion); }
				UInt64		GetUncompressedSize() { return m_pRecord->UncompressedSize; }
				UInt64		GetCompressedSize() { return m_pRecord->CompressedSize; }
				
				/** To verify the CRC-32 value for a file, first call Open().  Wrap the returned Stream in a HashedStream<CRC32Algorithm> stream 
					and read the stream until EOF is reached.  Then, compare the HashedStream's RetrieveHash() value to the GetCRC32() value 
					of the ZipArchiveEntry for validation.  An example:

					ZipArchiveEntry* pMyEntry = ...;
					Stream* pMyFile = pMyEntry->Open();
					HashedStream<CRC32Algorithm>* pMyVerifiedFile = new HashedStream<CRC32Algorithm>(pMyFile, pMyEntry->GetCRC32Algorithm());
					// Consume pMyVerifiedFile through EOF...
					if (pMyEntry->GetCRC32() != pMyVerifiedFile->RetrieveHash()) throw ValidationException();
				**/

				UInt32 GetCRC32() { return m_pRecord->CRC32; }
				const CRC32Algorithm&	GetCRC32Algorithm();
			};

			class ZipArchiveDirectoryInfo
			{
				friend class ZipArchive;				

				Path FullName;
				DateTime LastWriteTime;

				// keys are all lowercase, as we will assume the zip archive's filename or directory name search is insensitive.
				std::unordered_map<string, ZipArchiveDirectoryInfo>		m_subdirectories;
				std::unordered_map<string, ZipArchiveEntry*>			m_files;
			public:
				ZipArchiveDirectoryInfo(Path FullName_)
				{
					this->FullName = FullName_;
				}				

				Path GetFullName() const { return FullName; }
				DateTime GetLastWriteTime() const { return LastWriteTime; }

				vector<ZipArchiveEntry>			EnumerateFiles() const;
				vector<ZipArchiveDirectoryInfo>	EnumerateDirectories() const;				
			};

			class ZipArchive
			{
				#pragma pack(push,1)				

				// End of Central Directory (EOCD) Record
				struct EOCDRecord
				{
					enum { SignatureValue = 0x06054b50 };

					UInt32		Signature;			// 0x06054b50
					UInt16		DiskNumber;			// number of this disk             
					UInt16		StartCDDiskNumber;	// number of the disk with the start of the central directory
					UInt16		DiskCDEntries;		// total number of entries in the central directory on this disk
					UInt16		TotalCDEntries;		// total number of entries in the central directory
					UInt32		CDSize;				// size of the central directory
					UInt32		CDOffset;			// offset of start of central directory with respect to the starting disk number
					UInt16		CommentLength;		// .ZIP file comment length
					// .ZIP file comment       (variable size)
				};

				// Zip64 End of Central Directory (EOCD) Record
				struct EOCD64Record
				{
					enum { SignatureValue = 0x06064b50 };

					UInt32		Signature;			// 0x06064b50
					UInt64		Size;				// size of zip64 end of central directory record
					UInt16		WriterVersion;		// version made by                 
					UInt16		RequiredReaderVersion;	// version needed to extract
					UInt32		DiskNumber;			// number of this disk
					UInt32		StartCDDiskNumber;	// number of the disk with the start of the central directory
					UInt64		DiskCDEntries;		// total number of entries in the central directory on this disk
					UInt64		TotalCDEntries;		// total number of entries in the central directory
					UInt64		CDSize;				// size of the central directory
					UInt64		CDOffset;			// offset of start of central directory with respect to the starting disk number
					// Zip64 extensible data sector
				};

				// Zip64 End of Central Directory locator
				struct EOCDL64
				{
					enum { SignatureValue = 0x07064b50 };

					UInt32		Signature;			// 0x07064b50
					UInt32		StartCDDiskNumber;	// number of the disk with the start of the central directory
					UInt64		EOCDOffset;			// relative offset of the zip64 end of central directory record
					UInt32		TotalDisks;			// total number of disks
				};

				#pragma pack(pop)

				friend class ZipArchiveEntry;
				Int64 m_HeaderSize;
				vector<ZipArchiveEntry*>			m_Files;				
				memory::r_ptr<Stream>				m_pStream;
				unique_ptr<ZipArchiveDirectoryInfo> m_pRoot;

				static CRC32Algorithm	g_CRCTable;

			public:

				// Example:
				//	auto archive = ZipArchive(r_ptr<Stream>::responsible(new io::FileStream(archive_path, io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read)), ZipArchiveMode::Read);
				ZipArchive(memory::r_ptr<Stream>&& rpStream, ZipArchiveMode Mode);
				ZipArchive(ZipArchive&& other);
				~ZipArchive();
				ZipArchive& operator=(ZipArchive&& other);

				/// <summary>Call GetEntryCount() to identify the number of files stored in the archive and ascertain the
				/// maximum Index value allowed for a call to GetEntry().</summary>
				size_t GetEntryCount();

				/// <summary>Call GetEntry() with an Index value between 0 and GetEntryCount()-1 to retrieve the matching
				/// file's ZipArchiveEntry.  The returned pointer remains the responsibility of ZipArchive and should not
				/// be deleted.</summary>
				ZipArchiveEntry* GetEntry(int Index);

				ZipArchiveDirectoryInfo& GetRoot();

				// ZipArchiveEntry* GetEntry(string entryName);							
			};			

			/** Implementation - Misc **/

			inline ZipVersion::ZipVersion(UInt16 Version)
			{				
				PlatformID = (byte)(Version >> 8);
				Major = (byte)(Version);
				Minor = (Major % 10);
				Major /= 10;
			}

			/** Implementation - ZipArchive **/

			#ifdef PrimaryModule
			/*static*/ CRC32Algorithm ZipArchive::g_CRCTable((UInt32)CRC32Algorithm::CommonPolynomials::PkZip, true, 0xFFFFFFFF);
			#endif

			inline ZipArchive::ZipArchive(ZipArchive&& other)
				: 
				m_HeaderSize(other.m_HeaderSize),
				m_pStream(std::move(other.m_pStream)),
				m_Files(std::move(other.m_Files)),
				m_pRoot(std::move(other.m_pRoot))
			{
				m_HeaderSize = 0;
			}

			inline ZipArchive& ZipArchive::operator=(ZipArchive&& other)
			{
				m_HeaderSize = other.m_HeaderSize;
				m_pStream = std::move(other.m_pStream);
				m_Files = std::move(other.m_Files);
				m_pRoot = std::move(other.m_pRoot);
				return *this;
			}

			inline ZipArchive::ZipArchive(memory::r_ptr<Stream>&& rpStream, ZipArchiveMode Mode)
				: m_pStream(std::move(rpStream))
			{
				if (Mode != ZipArchiveMode::Read) throw NotImplementedException();				

				/** Locate and read end of central directory record **/

				const UInt32 EOCDSignature = IsLittleEndian() ? EOCDRecord::SignatureValue : SwapEndian((UInt32)EOCDRecord::SignatureValue);
				const UInt32 EOCD64Signature = IsLittleEndian() ? EOCD64Record::SignatureValue : SwapEndian((UInt32)EOCD64Record::SignatureValue);
				const UInt32 EOCDL64Signature = IsLittleEndian() ? EOCDL64::SignatureValue : SwapEndian((UInt32)EOCDL64::SignatureValue);

				if (!m_pStream->CanSeek()) throw NotSupportedException(S("ZipArchive requires a seekable stream."));

				Int64 Position = m_pStream->GetLength() - 4;
				UInt32 Value;
				bool IsZip64 = false;
				for (;;)
				{
					if (Position < 0) throw FormatException(S("Not a valid zip file or unable to locate EOCD signature."));
					m_pStream->Seek(Position, SeekOrigin::Begin);
					m_pStream->Read(&Value, 4);
					if (Value == EOCDSignature) break;
					if (Value == EOCD64Signature || Value == EOCDL64Signature) { IsZip64 = true; break; }
					Position--;
				}
				m_pStream->Seek(Position, SeekOrigin::Begin);
				Int64 CDOffset;
				Int64 Entries = 0;
				if (IsZip64)
				{					
					#if 0
					EOCD64Record EOCD;
					if (Value == EOCDL64Signature)
					{
						EOCDL64 EOCDL;
						if (Stream.Read(&EOCDL, sizeof(EOCDL)) < sizeof(EOCDL)) throw FormatException(S("Invalid zip file end-of-central-record locator."));
						if (!IsLittleEndian())
						{
							EOCDL.Signature = SwapEndian(EOCDL.Signature);
							EOCDL.StartCDDiskNumber = SwapEndian(EOCDL.StartCDDiskNumber);
							EOCDL.EOCDOffset = SwapEndian(EOCDL.EOCDOffset);
							EOCDL.TotalDisks = SwapEndian(EOCDL.TotalDisks);
						}
						Position = EOCDL.EOCDOffset;
					}

					Stream.Seek(Position, SeekOrigin::Begin);
					if (Stream.Read(&EOCD, sizeof(EOCD)) < sizeof(EOCD)) throw FormatException(S("Invalid zip file end-of-central-record or locator incorrect."));
					if (!IsLittleEndian())
					{
						EOCD.Signature = SwapEndian(EOCD.Signature);
						EOCD.Size = SwapEndian(EOCD.Size);
						EOCD.WriterVersion = SwapEndian(EOCD.WriterVersion);
						EOCD.RequiredReaderVersion = SwapEndian(EOCD.RequiredReaderVersion);
						EOCD.DiskNumber = SwapEndian(EOCD.DiskNumber);
						EOCD.StartCDDiskNumber = SwapEndian(EOCD.StartCDDiskNumber);
						EOCD.DiskCDEntries = SwapEndian(EOCD.DiskCDEntries);
						EOCD.TotalCDEntries = SwapEndian(EOCD.TotalCDEntries);
						EOCD.CDSize = SwapEndian(EOCD.CDSize);
						EOCD.CDOffset = SwapEndian(EOCD.CDOffset);
					}					
					if (EOCD.Signature != EOCD64Record::SignatureValue) throw FormatException(S("Invalid zip file end-of-central-record or locator invalid."));

					m_HeaderSize = Position - (EOCD.CDOffset + EOCD.CDSize);
					CDOffset = EOCD.CDOffset;
					if (EOCD.DiskCDEntries != EOCD.TotalCDEntries) throw FormatException(S("Entries count mismatch."));
					Entries = EOCD.TotalCDEntries;
					#else
					throw NotImplementedException(S("Only partial ZIP64 implementation provided."));
					#endif
				}
				else
				{
					EOCDRecord EOCD;
					if (m_pStream->Read(&EOCD, sizeof(EOCD)) < (Int64)sizeof(EOCD)) throw FormatException(S("Invalid zip file end-of-central-record."));
					if (!IsLittleEndian())
					{
						EOCD.DiskNumber = SwapEndian(EOCD.DiskNumber);
						EOCD.StartCDDiskNumber = SwapEndian(EOCD.StartCDDiskNumber);
						EOCD.DiskCDEntries = SwapEndian(EOCD.DiskCDEntries);
						EOCD.TotalCDEntries = SwapEndian(EOCD.TotalCDEntries);
						EOCD.CDSize = SwapEndian(EOCD.CDSize);
						EOCD.CDOffset = SwapEndian(EOCD.CDOffset);
						EOCD.CommentLength = SwapEndian(EOCD.CommentLength);
					}
					m_HeaderSize = Position - (EOCD.CDOffset + EOCD.CDSize);
					CDOffset = EOCD.CDOffset;
					if (EOCD.DiskCDEntries != EOCD.TotalCDEntries) throw FormatException(S("Entries count mismatch."));
					Entries = EOCD.TotalCDEntries;
				}

				/** Read file list **/

				Int64 Offset = CDOffset + m_HeaderSize;

				for (int ii = 0; ii < Entries; ii++)
				{
					auto pFR = make_unique<ZipArchiveEntry::BaseFileRecord>();
					
					m_pStream->Seek(Offset, SeekOrigin::Begin);
					if (m_pStream->Read(pFR.get(), sizeof(ZipArchiveEntry::BaseFileRecord)) < (Int64)sizeof(ZipArchiveEntry::BaseFileRecord))
						throw FormatException(S("File record error."));
					if (!IsLittleEndian())
					{
						pFR->Signature = SwapEndian(pFR->Signature);
						pFR->WriterVersion = SwapEndian(pFR->WriterVersion);
						pFR->RequiredReaderVersion = SwapEndian(pFR->RequiredReaderVersion);
						pFR->Flags = SwapEndian(pFR->Flags);
						pFR->CompressionMethod = SwapEndian(pFR->CompressionMethod);              
						pFR->LastWriteTime = SwapEndian(pFR->LastWriteTime);
						pFR->LastWriteDate = SwapEndian(pFR->LastWriteDate);
						pFR->CRC32 = SwapEndian(pFR->CRC32);
						pFR->CompressedSize = SwapEndian(pFR->CompressedSize);
						pFR->UncompressedSize = SwapEndian(pFR->UncompressedSize);
						pFR->FileNameLength = SwapEndian(pFR->FileNameLength);
						pFR->ExtraFieldLength = SwapEndian(pFR->ExtraFieldLength);
						pFR->FileCommentLength = SwapEndian(pFR->FileCommentLength);
						pFR->StartingDiskNumber = SwapEndian(pFR->StartingDiskNumber);
						pFR->InternalAttributes = SwapEndian(pFR->InternalAttributes);
						pFR->ExternalAttributes = SwapEndian(pFR->ExternalAttributes);
						pFR->LocalHeaderOffset = SwapEndian(pFR->LocalHeaderOffset);
					}

					// Verify signature
					const UInt32 FRSignature = 0x02014b50;
					if (pFR->Signature != FRSignature) throw FormatException(S("Expected file record, mismatched or missing signature."));					

					// Read filename
					char* pszFilename = new char [pFR->FileNameLength+1];
					if (m_pStream->Read(pszFilename, pFR->FileNameLength) != pFR->FileNameLength) throw FormatException();
					pszFilename[pFR->FileNameLength] = 0;

					Offset += sizeof(ZipArchiveEntry::BaseFileRecord) + pFR->FileNameLength + pFR->ExtraFieldLength + pFR->FileCommentLength;

					ZipArchiveEntry* pEntry = new ZipArchiveEntry(this, std::move(pFR), to_string(pszFilename));					
					m_Files.push_back(pEntry);					
				}
			}			

			inline ZipArchive::~ZipArchive()
			{
				for (uint ii=0; ii < m_Files.size(); ii++) delete m_Files[ii];
				m_Files.clear();
			}

			inline size_t ZipArchive::GetEntryCount() { return m_Files.size(); }
			inline ZipArchiveEntry* ZipArchive::GetEntry(int Index) { return m_Files[Index]; }

			/** Implementation - ZipArchiveEntry **/

			inline ZipArchiveEntry::ZipArchiveEntry()
			{
				m_pArchive = nullptr;
				m_pRecord = nullptr;				
				Length = 0;
			}			

			inline ZipArchiveEntry& ZipArchiveEntry::operator=(const ZipArchiveEntry& cp)
			{
				m_pArchive = cp.m_pArchive;
				m_pRecord = make_unique<BaseFileRecord>();
				MoveMemory(m_pRecord.get(), cp.m_pRecord.get(), sizeof(BaseFileRecord));
				FullName = cp.FullName;
				CreationTime = cp.CreationTime;
				LastWriteTime = cp.LastWriteTime;
				Length = cp.Length;
				return *this;
			}

			inline ZipArchiveEntry::ZipArchiveEntry(const ZipArchiveEntry& cp)
			{
				operator=(cp);
			}

			inline ZipArchiveEntry::ZipArchiveEntry(ZipArchive* pArchive, unique_ptr<BaseFileRecord>&& pRecord, const string& FullName_)
				: m_pRecord(std::move(pRecord))
			{
				m_pArchive = pArchive;
				
				// TODO: is the creation date stored separately in the extensions / newer formats?
				CreationTime = LastWriteTime = DateTime::FromMSDOS(m_pRecord->LastWriteDate, m_pRecord->LastWriteTime);
				Length = m_pRecord->UncompressedSize;
				FullName = FullName_;
			}

			inline ZipArchiveEntry::~ZipArchiveEntry()
			{
			}

			inline Stream* ZipArchiveEntry::Open()
			{	
				auto pStream = r_ptr<Stream>::absolved(m_pArchive->m_pStream);
				if (!pStream->CanSeek()) throw NotSupportedException(S("Archive stream closed or non-seekable."));
				
				Int64 Position = m_pArchive->m_HeaderSize + m_pRecord->LocalHeaderOffset;
				pStream->Seek(Position, SeekOrigin::Begin);
				LocalFileHeader	LocalHeader;
				if (pStream->Read(&LocalHeader, sizeof(LocalHeader)) < (Int64)sizeof(LocalHeader)) throw FormatException(S("Expected complete local file header."));
				if (!IsLittleEndian())
				{
					LocalHeader.Signature = SwapEndian(LocalHeader.Signature);
					LocalHeader.RequiredReaderVersion = SwapEndian(LocalHeader.RequiredReaderVersion);
					LocalHeader.Flags = SwapEndian(LocalHeader.Flags);
					LocalHeader.CompressionMethod = SwapEndian(LocalHeader.CompressionMethod);
					LocalHeader.LastWriteTime = SwapEndian(LocalHeader.LastWriteTime);
					LocalHeader.LastWriteDate = SwapEndian(LocalHeader.LastWriteDate);
					LocalHeader.CRC32 = SwapEndian(LocalHeader.CRC32);
					LocalHeader.CompressedSize = SwapEndian(LocalHeader.CompressedSize);
					LocalHeader.UncompressedSize = SwapEndian(LocalHeader.UncompressedSize);
					LocalHeader.FileNameLength = SwapEndian(LocalHeader.FileNameLength);
					LocalHeader.ExtraFieldLength = SwapEndian(LocalHeader.ExtraFieldLength);
				}

				if (LocalHeader.Signature != LocalFileHeader::SignatureValue) throw FormatException(S("Expected local file header, mismatched signature."));

				switch ((CompressionMethods)m_pRecord->CompressionMethod)
				{					
					case CompressionMethods::Deflate: break;			
					case CompressionMethods::None: break;
					default: throw NotSupportedException(("Compression method #" + wb::to_string((int)(m_pRecord->CompressionMethod)) + " is not supported.").c_str());
				}

				if (m_pRecord->Flags & 
					~((UInt16)GeneralFlags::CompressionTechnique | (UInt16)GeneralFlags::FilenameInUTF8 | (UInt16)GeneralFlags::Reserved))
					throw NotSupportedException(("Unsupported storage flag(s): " + to_hex_string(m_pRecord->Flags)).c_str());

				Position += sizeof(LocalHeader) + LocalHeader.FileNameLength + LocalHeader.ExtraFieldLength;
				pStream->Seek(Position, SeekOrigin::Begin);				

				//if (m_pRecord->UncompressedSize != m_pRecord->CompressedSize)
				if ((CompressionMethods)m_pRecord->CompressionMethod == CompressionMethods::Deflate)
				{
					#ifdef No_DeflateStream
					throw Exception("No DEFLATE decompression is available because the No_DeflateStream option was defined during build.  Cannot decompress archive.");
					#else		
					/** File data is compressed, we will need to decompress it **/

					StreamFragment* pCompressed = new StreamFragment(*pStream, LocalHeader.CompressedSize);
					DeflateStream* pDecompressed = new DeflateStream(r_ptr<Stream>::responsible(pCompressed), CompressionMode::Decompress, 						
						wb::io::compression::DeflateStream::Flags::SkipHeaders);					
					return pDecompressed;
					#endif
				}
				else 
				{
					/** File data is not compressed, can read directly **/

					StreamFragment* pDecompressed = new StreamFragment(*pStream, m_pRecord->UncompressedSize);					
					return pDecompressed;
				}
			}

			inline const CRC32Algorithm&	ZipArchiveEntry::GetCRC32Algorithm() { return ZipArchive::g_CRCTable; }

			/** Implementation - ZipArchiveDirectoryInfo and related **/

			inline ZipArchiveDirectoryInfo& ZipArchive::GetRoot()
			{
				if (m_pRoot != nullptr) return *m_pRoot;
				m_pRoot = make_unique<ZipArchiveDirectoryInfo>(Path(""));
				for (int ii = 0; ii < GetEntryCount(); ii++)
				{
					auto* pEntry = GetEntry(ii);

					bool IsDirectoryEntry = false;
					ZipArchiveDirectoryInfo* p_parent = m_pRoot.get();					
					string fn = pEntry->GetFullName();
					string next_name;
					for (size_t jj = 0; jj < fn.length(); jj++)
					{
						if (fn[jj] == '/' || fn[jj] == '\\')
						{														
							auto next_name_search = to_lower(next_name);
							{
								auto existing_it = p_parent->m_subdirectories.find(next_name_search);
								if (existing_it == p_parent->m_subdirectories.end())
								{
									std::pair<string, ZipArchiveDirectoryInfo> pair(next_name_search, ZipArchiveDirectoryInfo(p_parent->FullName / next_name));
									p_parent->m_subdirectories.insert(std::move(pair));
								}
							}

							{
								auto existing_it = p_parent->m_subdirectories.find(next_name_search);
								if (existing_it != p_parent->m_subdirectories.end())
									p_parent = &(existing_it->second);
								else
									throw Exception("Expected key to be present in subdirectories list after insertion.");
							}

							// if the slash is trailing, then this is a directory entry.
							if (jj == fn.length() - 1)
							{
								if (pEntry->Length != 0)
									throw FormatException("Zip archive entry '" + fn + "' appears to be a directory entry but has non-zero length.");
								p_parent->LastWriteTime = pEntry->LastWriteTime;
								IsDirectoryEntry = true;
							}

							next_name.clear();
							continue;
						}
						next_name += fn[jj];
					}

					if (IsDirectoryEntry) continue;

					if (next_name.length() == 0)
						throw Exception("Reached conclusion of archive path with empty filename from original full path: " + pEntry->GetFullName().to_string());
					auto next_name_search = to_lower(next_name);
					auto existing_it = p_parent->m_files.find(next_name_search);
					if (existing_it != p_parent->m_files.end())					
						throw Exception("Multiple entries for ZipArchiveEntry '" + next_name + "' were found at the same path within the archive.");										
					std::pair<string, ZipArchiveEntry*> pair(next_name_search, pEntry);
					p_parent->m_files.insert(std::move(pair));
				}
				return *m_pRoot;
			}			

			inline vector<ZipArchiveEntry>			ZipArchiveDirectoryInfo::EnumerateFiles() const
			{
				vector<ZipArchiveEntry> result;
				for (auto& kvp : m_files)				
					result.push_back(*kvp.second);				
				return result;
			}

			inline vector<ZipArchiveDirectoryInfo>	ZipArchiveDirectoryInfo::EnumerateDirectories() const
			{
				vector<ZipArchiveDirectoryInfo> result;
				for (auto& kvp : m_subdirectories)
					result.push_back(kvp.second);
				return result;
			}
		}
	}
}

#endif	// __WBZipFile_h__

//	End of ZipFile.h

