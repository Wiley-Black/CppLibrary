/////////
//	XmlParserImpl.h (Generation 4)
////

#ifndef __wbXmlParserImpl_v4_h__
#define __wbXmlParserImpl_v4_h__

#ifndef __wbXmlParser_v4_h__
#error	This header should be included only via XmlParser.h.
#endif

/** Dependencies **/

#include "../XmlParser.h"						// For Intellisense's benefit only.
#include "../../../Text/StringComparison.h"
#include "../../../IO/MemoryStream.h"
#include "../../../IO/FileStream.h"

/** Content **/

namespace wb
{
	namespace xml
	{				
		/** XmlParser Implementation **/
		
		inline XmlParser::XmlParser()
		{
			Loaded = 0;
			CurrentState = State::Unknown;
			pNext = new char[MaxLoading];			// Technically could be MaxLoading-1, but allowing the 1 byte for buffer.
			if (pNext == nullptr) throw OutOfMemoryException();
			CurrentState = State::Initializing;
		}

		inline XmlParser::~XmlParser()
		{
			if (pNext != nullptr)
			{
				delete[] pNext;
				pNext = nullptr;
			}
		}

		#pragma region "Alternative entry points"

		inline void XmlParser::StartSource(const string& sSourceFilename, int CurrentLineNumber)
		{
			this->CurrentSource = sSourceFilename;
			this->CurrentLineNumber = CurrentLineNumber;
		}

		inline XmlElement* XmlParser::GetCurrentElement()
		{
			for (int ii = (int)NodeStack.size() - 1; ii >= 0; ii--)
				if (NodeStack[ii]->IsElement()) return (XmlElement*)NodeStack[ii];
			return nullptr;
		}

		inline XmlNode* XmlParser::GetCurrentNode()
		{
			if (NodeStack.size() == 0) return nullptr;
			return NodeStack[NodeStack.size() - 1];
		}

		inline void XmlParser::FinishSource()
		{
			// Clear detailed state variables...
			Loaded = 0;
			CurrentKey.clear();
			CurrentValue.clear();

			string extra;
			switch (CurrentState)
			{
			default:
			case State::Unknown:
				throw Exception("Unrecognized state.");

			case State::Initializing:
			case State::Idle:
				break;

			case State::ParsingTag: extra = " and incomplete XML tag detected."; break;
			case State::ParsingXMLDeclaration: throw FormatException("Badly formed XML (incomplete XML declaration).");
			case State::ParsingComment: extra = " and incomplete XML comment detected."; break;
			case State::ParsingCDATA: extra = " and incomplete CDATA block detected."; break;
			case State::ParsingPCDATA: extra = " and incomplete text block detected."; break;
			case State::ParsingOpeningTag: 
			case State::ParsingAttributeKey: 
			case State::ParsingAttributeValueStart:
			case State::ParsingAttributeValue: extra = " and incomplete XML opening tag detected."; break;
			case State::ParsingOpenCloseTagCompletion: extra = " and incomplete XML open-close tag detected."; break;
			case State::ParsingClosingTag: extra = " and incomplete XML closing tag detected."; break;
			}

			CurrentState = State::Initializing;

			XmlElement* pCurrent = GetCurrentElement();
			if (pCurrent != nullptr)
			{
				throw FormatException("Badly formed XML (no closing tag for '" + pCurrent->LocalName + "' from " + pCurrent->SourceLocation + extra + ")");
			}
			
			if (NodeStack.size() != 0) throw Exception();			// Shouldn't be allowed, top entry should always be an element.			
		}

		inline string XmlParser::GetSource()
		{
			if (CurrentSource.length() < 1) return "line " + ::std::to_string(CurrentLineNumber);
			return CurrentSource + ":" + ::std::to_string(CurrentLineNumber);
		}

		inline /*static*/ std::unique_ptr<XmlDocument> XmlParser::Parse(wb::io::Stream& stream, const string& sSourceFilename)
		{
			XmlParser parser;
			parser.FinishSource();
			parser.StartSource(sSourceFilename);
			auto pDoc = parser.PartialParse(stream);
			if (pDoc == nullptr)
			{
				if (parser.CurrentSource.length() < 1)
					throw ArgumentException("No XML content found.");
				else
					throw ArgumentException("No XML content found in " + parser.CurrentSource + ".");
			}
			parser.FinishSource();
			return pDoc;
		}

		inline /*static*/ std::unique_ptr<XmlDocument> XmlParser::Parse(const string& str, const string& sSourceFilename)
		{
			wb::io::MemoryStream ms;
			wb::io::StringToStream(str, ms);			
			ms.Rewind();
			return Parse(ms, sSourceFilename);
		}

		inline /*static*/ std::unique_ptr<XmlDocument> XmlParser::ParseFile(const string& sSourceFilename)
		{
			using namespace wb::io;
			FileStream fs(sSourceFilename, FileMode::Open, FileAccess::Read, FileShare::Read);
			return Parse(fs, sSourceFilename);
		}

		#pragma endregion		

		inline bool XmlParser::Need(int NeedLoaded)
		{
			if (NeedLoaded < 0 || NeedLoaded > MaxLoading) throw ArgumentException("Need() parameter must be between 0 and MaxLoading.");
			while (Loaded < NeedLoaded)
			{
				int value = pCurrentStream->ReadByte();
				if (value < 0) return false;
				if (Loaded == 0) Current = (char)value;
				else pNext[Loaded - 1] = (char)value;
				Loaded++;
			}
			return true;
		}

		inline bool XmlParser::Advance()
		{
			if (Loaded >= 1 && Current == '\n') CurrentLineNumber++;
			if (Loaded > 1)
			{
				Current = pNext[0];
				// e.g. before Loaded = 4, Current | [0] [1] [2]
				//	    after Loaded = 3, Current* | [0] [1]
				for (int ii = 0; ii < Loaded - 2; ii++) pNext[ii] = pNext[ii + 1];
				Loaded--;
				return true;
			}
			if (Loaded == 1) Loaded = 0;		// In case we are at EOS and return false, we want Need(1) to return false.
			int value = pCurrentStream->ReadByte();
			if (value < 0) return false;
			Current = (char)value;
			Loaded = 1;
			return true;
		}

		inline bool XmlParser::IsNextEqual(const string& match)
		{
			if (Loaded < match.length()) throw ArgumentException("Need(" + to_string(match.length()) + ") must be called before IsNextEqual() on a string of this length.");
			for (auto ii = 0; ii < match.length(); ii++)
				if (pNext[ii] != match[ii]) return false;
			return true;
		}

		inline void XmlParser::StartNewChild(XmlNode* pNewNode)
		{
			try
			{
				pNewNode->SourceLocation = GetSource();
				GetCurrentElement()->Children.push_back(pNewNode);
				NodeStack.push_back(pNewNode);
			}
			catch (...)
			{
				delete pNewNode;
				throw;
			}
		}

		inline std::unique_ptr<XmlDocument> XmlParser::OnCloseElement(bool ClosingTag)
		{
			// ClosingTag is false for an open-and-close tag such as <Hello /> but is true for a closing tag such as </Hello>.

			if (NodeStack.size() == 0)
				throw FormatException("Badly formed XML (illegal closing tag at top-level at " + GetSource() + ")");

			XmlElement* pElement = GetCurrentElement();
			if (ClosingTag && pElement->LocalName != CurrentKey)
				throw FormatException("Badly formed XML (mismatched closing tag '" + CurrentKey + "' at " + GetSource() + " found inside element '" + pElement->LocalName + "' from " + pElement->SourceLocation + ")");			
			CurrentKey.clear();
			
			// Close any non-elements such as XmlText before we close the element.
			while (!NodeStack[NodeStack.size() - 1]->IsElement()) NodeStack.pop_back();

			// Now close the element.
			NodeStack.pop_back();
			if (NodeStack.size() == 0) {
				CurrentState = State::Initializing;
				std::unique_ptr<XmlDocument> ret = std::move(pCurrentDoc);
				return ret;
			}
			CurrentState = State::Idle;
			return nullptr;
		}

		inline std::unique_ptr<XmlDocument> XmlParser::PartialParse(wb::io::Stream& stream)
		{
			pCurrentStream = &stream;

			try
			{
				for (;;)
				{
					switch (CurrentState)
					{
					case State::Initializing:
						if (!Need(3)) return nullptr;
						if (Current == 0xEF && pNext[0] == 0xBB && pNext[2] == 0xBF) { 
							// UTF-8 BOM.  TODO: Respond to detected encoding.
							Advance(); Advance(); Advance();
						}
						CurrentState = State::Idle;
					case State::Idle:
						while (Need(1))
						{
							if (Current == '<') {		
								CurrentState = State::ParsingTag;
								Advance();
								break;
							}

							if (IsWhitespace(Current)) {
								Advance();
								continue;
							}

							if (pCurrentDoc == nullptr)
								throw FormatException("Expected XML opening tag at top-level at " + GetSource() + ".");

							// A character other than < or whitespace has been found inside the element.  Must be text!
							StartNewChild(new XmlText());
							CurrentState = State::ParsingPCDATA;
							BWhitespace = 0;
							break;
						}
						if (CurrentState != State::Idle) continue;
						return nullptr;

					case State::ParsingXMLDeclaration:
						if (!ParseXMLDeclaration()) return nullptr;
						CurrentState = State::Idle;
						continue;

					case State::ParsingComment:
						if (!ParseComment()) return nullptr;
						CurrentState = State::Idle;
						continue;

					case State::ParsingCDATA:
						if (!ParseCDATA()) return nullptr;
						CurrentState = State::Idle;
						continue;

					case State::ParsingPCDATA:
						if (!ParsePCDATA()) return nullptr;
						CurrentState = State::Idle;
						continue;

					case State::ParsingDOCTYPE:
						if (!ParseDOCTYPE()) return nullptr;
						CurrentState = State::Idle;
						continue;

					case State::ParsingTag:
						// '<' has already been parsed.
						if (!Need(1)) return nullptr;

						if (Current == '?') {
							CurrentState = State::ParsingXMLDeclaration; 
							Advance();
						}
						else if (Current == '!') {
							if (!Need(3)) return nullptr;
							if (pNext[0] == '-' && pNext[1] == '-') {
								Advance(); Advance(); Advance();
								CurrentState = State::ParsingComment;
								continue;
							}
							else if (pNext[0] == '[') {
								const int cdata_len = (int)strlen("![CDATA[");
								if (!Need(cdata_len)) return nullptr;
								if (IsNextEqual("[CDATA[")) {
									for (int ii = 0; ii < cdata_len; ii++) Advance();
									if (pCurrentDoc == nullptr) throw FormatException("Expected top-level element at " + GetSource() + ".");									
									StartNewChild(new XmlText());
									CurrentState = State::ParsingCDATA;
									continue;
								}
								else throw FormatException("Expected [CDATA[ to follow <![ tag at " + GetSource() + ".");
							}
							else if (pNext[0] == 'D' && pNext[1] == 'O') {
								const int doctype_len = (int)strlen("!DOCTYPE");
								if (!Need(doctype_len)) return nullptr;
								if (IsNextEqual("DOCTYPE")) {
									for (int ii = 0; ii < doctype_len; ii++) Advance();
									CurrentState = State::ParsingDOCTYPE;
									continue;
								}
							}
							throw FormatException("Unrecognized tag format following <! at " + GetSource() + ".");
						}
						else {							
							if (Current == '/') {
								if (pCurrentDoc == nullptr) throw FormatException("Illegal closing tag at top-level at " + GetSource() + ".");
								CurrentState = State::ParsingClosingTag;
								Advance();
							}
							else {
								if (IsWhitespace(Current)) {
									Advance(); 
									continue;
								}

								if (pCurrentDoc == nullptr)
								{
									pCurrentDoc = std::unique_ptr<XmlDocument>(new XmlDocument());
									XmlElement* pRoot = new XmlElement();
									try {
										pRoot->SourceLocation = GetSource();
										pCurrentDoc->Children.push_back(pRoot);										
									}
									catch (...)
									{
										delete pRoot;
										throw;
									}
									NodeStack.push_back(pRoot);
								}
								else StartNewChild(new XmlElement());
								CurrentState = State::ParsingOpeningTag;
							}
						}
						continue;

					case State::ParsingOpeningTag:
						// A new XmlElement (or XmlElement) will have been added to the node stack before switching
						// into the ParseOpeningTag() state.
						if (!ParseOpeningTag()) return nullptr;
						// ParseOpeningTag() will change the CurrentState to the next state before returning true.						
						continue;

					case State::ParsingAttributeKey:
						while (Need(1))
						{
							if (Current == '=') {
								Advance();
								if (CurrentKey.length() == 0) throw FormatException("Expected an attribute name at " + GetSource() + ".");
								CurrentState = State::ParsingAttributeValueStart;								
								break;
							}
							if (IsWhitespace(Current)) {
								Advance();
								continue;
							}
							if (Current == '>') {
								Advance();
								CurrentState = State::Idle;
								break;
							}
							if (Current == '/')
							{
								Advance();
								if (CurrentKey.length() > 0) throw FormatException("Badly formed XML (attribute must be followed by = at " + GetSource() + ".)");
								if (GetCurrentElement()->LocalName.length() < 1) throw Exception();								
								CurrentState = State::ParsingOpenCloseTagCompletion;
								break;
							}
							CurrentKey += Current;
							Advance();
						}
						continue;

					case State::ParsingAttributeValueStart:
						if (!ParseAttributeValueStart()) return nullptr;
						CurrentState = State::ParsingAttributeValue;
						// Fall-through...

					case State::ParsingAttributeValue:
						// We have parsed:   <tag-name attribute-name="
						// The last thing parsed when entering this state was the opening quote sign.
						if (!ParseAttributeValue()) return nullptr;
						// ParseAttributeValue() will already have added the attribute to the current element.
						CurrentState = State::ParsingAttributeKey;
						continue;

					case State::ParsingOpenCloseTagCompletion:
						// Assumes that <.../ has been parsed but not the closing >.
						while (Need(1))
						{
							if (Current == '>')
							{
								Advance();
								auto ret = OnCloseElement(false);
								if (ret != nullptr) return ret;
								break;
							}

							if (IsWhitespace(Current)) {
								Advance();
								continue;
							}

							throw FormatException("Expected > to follow closing / of tag at " + GetSource() + ".");
						}
						continue;

					case State::ParsingClosingTag:
						if (!ParseClosingTag()) return nullptr;
						auto ret = OnCloseElement(true);
						if (ret != nullptr) return ret;
						continue;
					}
				}
			}
			catch (...)
			{
				NodeStack.clear();
				pCurrentDoc = nullptr;
				CurrentKey.clear();
				CurrentValue.clear();
				CurrentState = State::Initializing;				
				throw;
			}
		}		

		inline void XmlParser::SkipWhitespace()
		{
			if (Loaded < 1) throw Exception();
			while (IsWhitespace(Current)) Advance();
		}

		inline bool XmlParser::ParseOpeningTag()
		{
			// Assumes that the '<' character has been parsed, and that the character following it
			// has been verified as not a special indicator (such as <!-- for a comment, <? for an
			// XML declaration, etc.).

			XmlElement* pElement = GetCurrentElement();
			while (Need(1))
			{
				if (IsWhitespace(Current)) {
					Advance();
					if (pElement->LocalName.length() > 0)
					{
						CurrentState = State::ParsingAttributeKey;
						return true;
					}
					continue;
				}
				
				if (Current == '/')
				{					
					if (pElement->LocalName.length() < 1)
						throw Exception("Expected </ pattern to be detected by ParseTag processing.");
					Advance();
					CurrentState = State::ParsingOpenCloseTagCompletion;
					return true;
				}
				
				if (Current == '>')
				{					
					Advance();
					CurrentState = State::Idle;
					return true;
				}

				if (Current == '=')
					throw FormatException("Badly formed XML (equal symbol must follow an attribute key at " + GetSource() + ".)");

				pElement->LocalName += Current;
				Advance();
			}
			return false;
		}		

		inline bool XmlParser::ParseAttributeValueStart()
		{
			// We have parsed:   <tag-name attribute-name=
			// The last thing parsed was an equal sign.  We are commited to needing an attribute value
			// at this stage.
			while (Need(1))
			{
				if (Current == '\"')
				{
					Advance();
					QuoteChar = '\"';
					return true;					
				}

				if (Current == '\'')
				{
					Advance();
					QuoteChar = '\'';
					return true;
				}

				if (IsWhitespace(Current)) {
					Advance();
					continue;
				}

				throw FormatException("Badly formed XML (expected attribute value after equal for attribute " + CurrentKey + " at " + GetSource() + ".)");
			}
			return false;
		}

		inline bool XmlParser::ParseAttributeValue()
		{
			// We have parsed:   <tag-name attribute-name="
			// The last thing parsed as we transitioned into this state was the opening quote.  			
			while (Need(1))
			{
				if (Current == '&')
				{
					string EscapedChars;
					if (!EscapedParsing(EscapedChars)) return false;
					CurrentValue += EscapedChars;
					continue;
				}

				if (Current == QuoteChar)
				{
					Advance();
					XmlElement* pElement = GetCurrentElement();
					if (pElement->FindAttribute(CurrentKey.c_str()) != nullptr)
						throw FormatException("Duplicate attribute '" + CurrentKey + "' found in XML tag '" + pElement->LocalName + "' at " + GetSource() + ".");
					XmlAttribute* pNewAttr = new XmlAttribute();
					try
					{
						pNewAttr->Name = CurrentKey;
						pNewAttr->Value = CurrentValue;
						pElement->Attributes.push_back(pNewAttr);
					}
					catch (...)
					{
						delete pNewAttr;
						throw;
					}
					CurrentKey.clear();
					CurrentValue.clear();
					return true;
				}

				CurrentValue += Current;
				Advance();
			}
			return false;
		}		

		inline bool XmlParser::ParseClosingTag()
		{
			// We assume that the </ characters have already been parsed.
			// Parse the tag name into 'CurrentKey', to be verified against element being closed later.

			while (Need(1))
			{
				if (Current == '>') {
					Advance();
					return true;
				}

				if (IsWhitespace(Current)) {
					Advance();
					continue;
				}

				if (Current == '/') throw FormatException("Badly formed XML (expected no more than one / in a tag at " + GetSource() + ".)");

				CurrentKey += Current;
				Advance();
			}
			return false;
		}

		inline bool XmlParser::EscapedParsing(string& EscapedChars)
		{
			// Look for a semicolon to close off the escape sequence...

			int iSemicolon = 0;
			for (; iSemicolon < Loaded - 1; iSemicolon++)
			{
				if (pNext[iSemicolon] == ';') break;
			}
			while (iSemicolon >= Loaded - 1)
			{
				if (Loaded >= MaxLoading) throw FormatException("Unrecognized escape sequence at " + GetSource() + ".");
				if (!Need(Loaded + 1)) return false;
				for (; iSemicolon < Loaded - 1; iSemicolon++)
				{
					if (pNext[iSemicolon] == ';') break;
				}
			}

			// A complete escape sequence is loaded.  Identify which one it is, if valid.
			// Example:				Current | [0] [1] [2]	(Loaded=4)
			//						'&'		  'l' 't' ';'	(Semicolon=2)
			Advance();			// Consume the '&'
			string sequence;
			for (int ii = 0; ii < iSemicolon; ii++) { sequence += Current; Advance(); }
			// Example (cont'd):	sequence = "lt"
			Advance();			// Consume the ';'

			if (IsEqualNoCase(sequence, "quot")) EscapedChars = "\"";
			else if (IsEqualNoCase(sequence, "amp")) EscapedChars = "&";
			else if (IsEqualNoCase(sequence, "apos")) EscapedChars = "\'";
			else if (IsEqualNoCase(sequence, "lt")) EscapedChars = "<";
			else if (IsEqualNoCase(sequence, "gt")) EscapedChars = ">";
			else if (sequence.length() > 1 && sequence[0] == '#')
			{				
				if (sequence.length() > 2 && tolower(sequence[1]) == 'x')
				{
					sequence = "0x" + sequence.substr(2);		// If prefixed with 0x, then UInt32_Parse() will utilize hexadecimal.
				}
				else sequence = sequence.substr(1);				// Remove the hash at start.
				UInt32 Value = UInt32_Parse(sequence, NumberStyles::AllowBasePrefix);
				if (Value <= 127)
					EscapedChars = string(1, (char)Value);
				else
					throw NotImplementedException("Need to implement XML encoding tracking to be able to incorporate UTF-8 support.");
			}
			else throw FormatException("Unrecognized PCDATA escape sequence &" + sequence + "; at " + GetSource() + ".");
			return true;
		}

		inline bool XmlParser::ParsePCDATA()
		{
			XmlNode* pNode = GetCurrentNode();
			if (pNode->GetType() != XmlNode::Type::Text) throw Exception("ParsePCDATA() expected XmlText node as current.");
			XmlText* pText = (XmlText*)pNode;
			
			while (Need(1))
			{
				if (Current == '&')
				{
					string EscapedChars;
					if (!EscapedParsing(EscapedChars)) return false;
					pText->Text += EscapedChars;
					BWhitespace = 0;
					continue;
				}

				if (Current == '<') {
					if (BWhitespace) pText->Text = pText->Text.substr(0, pText->Text.length() - BWhitespace);
					return true;
				}

				/** Microsoft's XML Parser apparently allows a > character given no < opener, and they even use
					it in some Visual Studio/MSBuild property files (xml).  I'm not sure how I feel about it, but
					I need to parse Microsoft's XML files so guess I'm going with it.
				if (Current == '>') throw FormatException();
				**/

				if (IsWhitespace(Current)) BWhitespace++; else BWhitespace = 0;
				pText->Text += Current;
				Advance();
			}
			return false;
		}

		inline bool XmlParser::ParseDOCTYPE()
		{			
			while (Need(2))
			{
				if (Current == ']' && pNext[0] == '>') {
					Advance(); Advance();
					return true;
				}
				Advance();
			}
			return false;
		}

		inline bool XmlParser::ParseXMLDeclaration()
		{				
				// Assumes that we have parsed the <? characters.							

			/** Note: ParseAttributes() is fully setup to handle the rest of the
				XML Declaration tag, including the special ? at the end.  We just
				aren't using it at the moment. **/

			while (Need(1))
			{
				if (Current == '>')
				{
					Advance();
					return true;
				}
				Advance();
			}
			return false;
		}

		inline bool XmlParser::ParseComment()
		{
				// Assumes we have already parsed the <!-- characters.				

			while (Need(2))
			{
				if (Current == '-' && pNext[0] == '>')
				{
					Advance(); Advance();
					return true;
				}
				Advance();
			}
			return false;
		}

		inline bool XmlParser::ParseCDATA()
		{
			// Assumes we have already parsed the <![CDATA[ characters.
			// Parse until we find the CEND sequence:   ]]>			
			XmlText* pText = (XmlText*)GetCurrentNode();
			while (Need(3))
			{
				if (Current == ']' && pNext[0] == ']' && pNext[1] == '>')
				{
					Advance(); Advance(); Advance();
					return true;
				}
				pText->Text += Current;
				Advance();
			}
			return false;
		}
	}
}

#endif	// __wbXmlParserImpl_v4_h__

//	End of XmlParserImpl.h

