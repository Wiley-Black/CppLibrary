/////////
//	XML Parser (Generation 4)
//	Copyright (C) 2010-2014 by Wiley Black
////
//	A simplified XML Parser which operates in a single-pass and
//	uses a minimal dependency set.  The parser utilizes linked lists
//	instead of dynamic arrays, however this feature is hidden 
//	from user code for usability.  The class hierarchy is designed
//	to be somewhat similar to the .NET heirarchy.
//
//	These features could be added to the parser, however at present
//	the following XML features are not implemented and generate an
//	error:
//		- XML Namespaces
//		- Partial/characterwise message parsing 
//
//	Some features available here but not found in the XML specification include:
//		- Multiple top-level elements permitted (automatically) if needed.
//			This will manifest as the XmlDocument object containing multiple
//			children.
//
//	Other features included:
//		- Support for comment parsing (comments are discarded)
////

#ifndef __wbXmlParser_v4_h__
#define __wbXmlParser_v4_h__

/** Table of Contents **/

namespace wb
{
	namespace xml
	{
		class XmlParser;
	}
}

/** Dependencies **/

#include "Xml.h"
#include "../../IO/Streams.h"

/** Content **/

namespace wb
{
	namespace xml
	{
		class XmlParser
		{
			XmlElement* GetCurrentElement();
			XmlNode* GetCurrentNode();			
			void StartNewChild(XmlNode* pChild);
			unique_ptr<XmlDocument> OnCloseElement(bool ClosingTag);

			bool IsWhitespace(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'; }
			void SkipWhitespace();
			bool EscapedParsing(string& EscapedChars);

			bool ParseCDATA();
			bool ParsePCDATA();
			bool ParseDOCTYPE();
			bool ParseXMLDeclaration();
			bool ParseComment();			
			bool ParseOpeningTag();
			bool ParseAttributeValueStart();
			bool ParseAttributeValue();
			bool ParseClosingTag();		

			unique_ptr<XmlDocument>	pCurrentDoc;			

			/// <summary>
			/// NodeStack contains a scaffold of the current nodes being parsed.  The root node is
			/// element [0], if the document has been started.  The latest child being parsed is 
			/// element [N].
			/// </summary>
			vector<XmlNode*> NodeStack;

			string CurrentSource;
			int CurrentLineNumber;

			string GetSource();

			io::Stream* pCurrentStream;			
			
			char Current;
			char* pNext;
			enum { MaxLoading = 64 /*bytes*/ };
			int Loaded;

			/// <summary>
			/// Need() provides a character buffer with additional lookahead of up to 10 characters for the 
			/// XmlParser class.  If Need(1) returns true, then Current is ensured valid.  If Need(2) returns 
			/// true, then Current and pNext[0] are valid.  If Need(3) is called, Current, pNext[0] and [1]
			/// are valid, and so forth.  In all cases, the stream is not advanced if the requested need is 
			/// already available.  If insufficient characters are loaded from the stream, then the stream 
			/// advances- but in the case where Current is valid, it is not shifted from Next or further.
			/// </summary>
			/// <param name="NeedChars">The minimum number of characters in the Current and pNext
			/// buffer that are needed.  The character in Current counts as 1.</param>
			/// <returns>True if the requested characters are available.  False if the stream cannot provide
			/// the needed characters at this time.</returns>
			bool Need(int NeedChars);

			/// <summary>
			/// Advance() moves the current (and lookahead) buffer forward one character.  The lookahead buffer
			/// is not guaranteed in this case, and only Current can be assumed to be valid after an Advance()
			/// call.  If characters in the pNext buffer are needed, call Need() after Advance().
			/// </summary>
			/// <returns>True if the requested character is available in Current.  Flase if the stream cannot
			/// provide the needed characters at this time.</returns>
			bool Advance();

			/// <summary>
			/// IsNextEqual() is a helper that operates like the IsEqual() comparison function for strings,
			/// but uses the "pNext" buffer as its match target.  For example, if Loaded is 4 with current
			/// being 'A' and pNext containing 'B', 'C', and 'D', then IsNextEqual("BCD") would return true.
			/// </summary>
			bool IsNextEqual(const string& match);

			enum class State
			{
				Unknown,
				Initializing,
				Idle,		
				ParsingTag,
				ParsingXMLDeclaration,
				ParsingComment,
				ParsingDOCTYPE,
				ParsingCDATA,
				ParsingPCDATA,
				ParsingOpeningTag,
				ParsingAttributeKey,
				ParsingAttributeValueStart,
				ParsingAttributeValue,
				ParsingOpenCloseTagCompletion,
				ParsingClosingTag
			};			

			State CurrentState;			

			int BWhitespace;		// Counter used within ParsePCDATA().
			char QuoteChar;			// Parsing a string that is enclosed in a single quote (apostrophe) if '\'', normal quote if '\"'.
			string CurrentKey;		// Also used for tag name being parsed when closing a node.
			string CurrentValue;

		public:

			XmlParser();
			~XmlParser();						

			/// <summary>Parses the stream, which must contain an XML document or fragment.  An exception is thrown on error.</summary>
			/// <returns>An XmlDocument parsed from the provided string.</returns>
			static unique_ptr<XmlDocument> Parse(wb::io::Stream& stream, const string& sSourceFilename = "");

			/// <summary>Parses the string.  A complete XML document or fragment must be contained in the string or an exception
			/// will be thrown.  An exception is thrown on error.</summary>
			/// <returns>An XmlDocument parsed from the provided string.</returns>
			static unique_ptr<XmlDocument> Parse(const string& str, const string& sSourceFilename = "");

			/// <summary>Parses the file, which must contain an XML document or fragment.  An exception is thrown on error.
			/// A complete XML document or fragmant must be contained in the file or an exception will be thrown.</summary>
			/// <returns>An XmlDocument parsed from the provided string.</returns>
			static unique_ptr<XmlDocument> ParseFile(const string& sSourceFilename);

			/// <summary>
			/// Error messages generated from the XmlParser are much more helpful if the parser can display the
			/// source file and line number associated with the error.  To invoke this, call StartSource() anytime
			/// a new source file (or stream or other source) is invoked for parsing.
			/// </summary>
			/// <param name="sSourceFilename">The source file that will be parsed next.</param>
			/// <param name="CurrentLineNumber">If not beginning at the start of the file, sets the line number to
			/// start tracking from.</param>
			void StartSource(const string& sSourceFilename = "", int CurrentLineNumber = 1);

			/// <summary>Parses the stream, which can contain one or more XML documents or fragments.  The parsing proceeds
			/// until a top-level node is completed, at which time the associated XmlDocument is returned.  The stream position
			/// will only be updated to the closure of the top-level node.  There is no error if the end of stream is reached
			/// with an incomplete XmlDocument or if no xml content has been detected, however a call to FinishSource() can be
			/// made when all content and streaming is completed and an incomplete XmlDocument is not expected.  An exception is 
			/// thrown on error.  CAUTION: when a node is returned, the stream may still contain unparsed data belonging to the
			/// next node.  That is, GetPosition() may be less than GetLength().</summary>
			/// <returns>An XmlDocument parsed from the provided string or nullptr if no XML content has been completed in the
			/// given stream.</returns>
			unique_ptr<XmlDocument> PartialParse(wb::io::Stream& stream);

			/// <summary>
			/// An optional call can be made to FinishSource() if the stream or string provided to Parse() is expected to have provided
			/// a complete XML document at this time.  If any XML content has been initiated and not completed- that is, an opening tag
			/// has been detected- then an exception will be thrown explaining the lack of closure.  If there is no incomplete XmlDocument
			/// being parsed, then FinishSource() has no effect.
			/// </summary>
			void FinishSource();
		};
	}
}

/** Late dependencies **/

#include "Implementation/XmlParserImpl.h"

#endif	// __wbXmlParser_h__

//	End of XmlParser.h

