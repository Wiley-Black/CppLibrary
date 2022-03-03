/////////
//	JsonParserToXml.h
////

#ifndef __WBJsonParserToXml_h__
#define __WBJsonParserToXml_h__

/** Dependencies **/

#include "../IO/Streams.h"
#include "../IO/MemoryStream.h"
#include "Xml/Xml.h"

/** Content **/

namespace wb
{
	namespace json
	{
		using namespace wb::xml;

		class JsonParser
		{
			void SkipWhitespace(const char*& psz);
			bool IsWhitespace(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'; }
			string ParseString(const char *&psz);

			void ParseValue(const char *&psz, XmlNode *pParent, string& NameText, bool AsElement = false);
			void ParseObject(const char *&psz, XmlNode* pNode);

			string CurrentSource;
			int CurrentLineNumber;
			string GetSource();

		public:

			JsonParser();
			~JsonParser();

			/// <summary>Parses the string, which must contain a JSON fragment.  An exception is thrown on error.</summary>
			/// <returns>The returned XmlDocument has been allocated with new and should be delete'd when done.</returns>
			unique_ptr<XmlDocument> ParseAsXml(const char *psz, const string& sSourceFilename = "");

			/// <summary>Parses the stream, which must contain a JSON fragment.  An exception is thrown on error.</summary>
			/// <returns>The returned XmlDocument has been allocated with new and should be delete'd when done.</returns>			
			unique_ptr<XmlDocument> ParseAsXml(wb::io::Stream& stream, const string& sSourceFilename = "");
		};
	
		/** JsonParser Implementation **/
		
		inline JsonParser::JsonParser()
		{
		}

		inline JsonParser::~JsonParser()
		{
		}

		inline string JsonParser::GetSource()
		{			
			if (CurrentSource.length() < 1) return "line " + std::to_string(CurrentLineNumber);
			return CurrentSource + ":" + std::to_string(CurrentLineNumber);
		}

		inline void JsonParser::SkipWhitespace(const char*& psz)
		{
			while (*psz && IsWhitespace(*psz)) {
				if (*psz == '\n') CurrentLineNumber++;
				psz++; 
			}
		}

		inline unique_ptr<XmlDocument> JsonParser::ParseAsXml(wb::io::Stream& stream, const string& sSourceFilename)
		{
			// Optimization: Could avoid storing the whole thing in memory and parse as we go...
			wb::io::MemoryStream ms;
			wb::io::StreamToStream(stream, ms);
			ms.Seek(0, wb::io::SeekOrigin::End);
			ms.WriteByte(0);			// Add null-terminator
			ms.Rewind();
			return ParseAsXml((const char *)ms.GetDirectAccess(0), sSourceFilename);
		}

		inline unique_ptr<XmlDocument> JsonParser::ParseAsXml(const char *psz, const string& sSourceFilename)
		{
			CurrentSource = sSourceFilename;
			CurrentLineNumber = 1;

			for (;; psz++)
			{
				if (*psz == 0) {
					if (CurrentSource.length() < 1) 
						throw ArgumentException("No JSON content found.");
					else
						throw ArgumentException("No JSON content found in " + CurrentSource + ".");
				}

				if (*psz == '{') { psz++; break; }

				if (*psz == '\n') CurrentLineNumber++;

				if (!IsWhitespace(*psz))
				{
					throw FormatException("Expected JSON opening brace at top-level at " + GetSource() + ".");
				}
			}

			auto pDocument = make_unique<XmlDocument>();
			pDocument->SourceLocation = CurrentSource;
			ParseObject(psz, pDocument.get());
			return pDocument;
		}
		
		inline string JsonParser::ParseString(const char *&psz)
		{
			if (*psz != '\"') throw FormatException("Expected opening quotation at start of ParseString() operation at " + GetSource() + ".");
			psz++;
			string ret = "";		// TODO: Might be more optimal/faster to use a StringBuilder kind of thing.
			for (;; psz++)
			{
				if (*psz == '\0') throw FormatException("Unterminated string during JSON parsing at " + GetSource() + ".");
				if (*psz == '\"') { psz++; return ret; }
				
				if (*psz == '\\')
				{
					psz++;
					switch (*psz)
					{
						case 0: throw FormatException("Unterminated string during JSON parsing at " + GetSource() + ".");
						case '\"': ret += '\"'; continue;
						case '\\': ret += '\\'; continue;
						case '/': ret += '/'; continue;
						case 'b': ret += '\b'; continue;
						case 'f': ret += '\f'; continue;
						case 'n': ret += '\n'; continue;
						case 'r': ret += '\r'; continue;
						case 't': ret += '\t'; continue;
						case 'u': throw NotSupportedException("Unicode character escape sequences not currently supported in JSON parsing at " + GetSource() + ".");
						default: throw FormatException(string("Unrecognized escape sequence \\") + string(1,*psz) + " in JSON string at " + GetSource() + ".");
					}
				}

				if (*psz == '\n') CurrentLineNumber++;
				
				ret += *psz;
			}
		}

		inline void JsonParser::ParseValue(const char *&psz, XmlNode *pParent, string& NameText, bool AsElement /*= false*/)
		{
			// Precondition: name and colon have been parsed.  At start of value.
			// Postcondition: past value, but haven't parsed the comma or closing brace yet.
			// Action: Parse the value and append it to the parent node.

			// Values can be a string, number, boolean, null, array, or an object.			

			for (;;)
			{
				if (*psz == 0) throw FormatException("Badly formed JSON (missing value while parsing '" + pParent->ToString() + "' at " + GetSource() + ")");				
				if (IsWhitespace(*psz)) { 
					if (*psz == '\n') CurrentLineNumber++;
					psz++; continue; 
				}

				// Look for string, handled as an Xml attribute (unless AsElement is set).
				if (*psz == '\"') { 
					string ValueText = ParseString(psz);
					if (!pParent->IsElement()) throw FormatException("Cannot parse string value outside of an object at " + GetSource() + ".");
					XmlElement* pElement = (XmlElement*)pParent;
					if (AsElement)
						pElement->AddStringAsText(NameText.c_str(), ValueText.c_str());
					else
					{
						if (pElement->FindAttribute(NameText.c_str()) != nullptr)
							throw FormatException("JSON name '" + NameText + "' was duplicated more than once within the same object and outside of an array.");
						pElement->AddStringAsAttr(NameText.c_str(), ValueText.c_str());
					}
					return;
				}

				// Look for numerical, handled as an Xml attribute (unless AsElement is set).
				if ((*psz >= '0' && *psz <= '9') || *psz == '.' || *psz == '-')
				{
					bool FloatingPoint = false;
					string NumericText = "";
					for(;;)
					{
						if (*psz == 0) throw FormatException("Badly formed JSON (unexpected termination during numeric value while parsing '" + pParent->ToString() + "' at " + GetSource() + ")");				
						if (IsWhitespace(*psz)) break;
						if (*psz >= '0' && *psz <= '9') { NumericText += *psz; psz++; continue; }
						if (*psz == '.' || *psz == 'e' || *psz == 'E') { NumericText += *psz; psz++; FloatingPoint = true; continue; }
						break;
					}
					if (!pParent->IsElement()) throw FormatException("Cannot parse numeric value outside of an object at " + GetSource() + ".");
					XmlElement* pElement = (XmlElement*)pParent;
					if (AsElement)
					{
						if (!FloatingPoint)
							pElement->AddInt64AsAttr(NameText.c_str(), Int64_Parse(NumericText, NumberStyles::Integer));
						else
							pElement->AddDoubleAsAttr(NameText.c_str(), Double_Parse(NumericText, NumberStyles::Float));
					}
					else
					{
						if (pElement->FindAttribute(NameText.c_str()) != nullptr)
							throw FormatException("JSON name '" + NameText + "' was duplicated more than once within the same object and outside of an array.");

						if (!FloatingPoint)
							pElement->AddInt64AsText(NameText.c_str(), Int64_Parse(NumericText, NumberStyles::Integer));
						else
							pElement->AddDoubleAsText(NameText.c_str(), Double_Parse(NumericText, NumberStyles::Float));
					}
					return;
				}

				// Look for array, handled as repeated Xml elements.
				// Note: nested arrays will be flattened.
				if (*psz == '[')
				{
					psz++;
					for (;;)
					{
						SkipWhitespace(psz);
						if (*psz == ']') { psz++; break; }												
						ParseValue(psz, pParent, NameText, true);
						SkipWhitespace(psz);
						if (*psz == ',') { psz++; continue; }
						if (*psz == ']') { psz++; break; }
						throw FormatException("Expected delimiter (comma) or closing bracket during list parsing at " + GetSource() + ".");
					}
					return;
				}

				// Look for object, handled as an Xml element.
				if (*psz == '{')
				{
					psz ++;
					auto pChild = make_shared<XmlElement>(NameText.c_str());
					pChild->SourceLocation = GetSource();
					pParent->AppendChild(pChild);
					ParseObject(psz, pChild.get());
					return;
				}

				// Look for boolean, handled as an Xml attribute.
				if (StartsWithNoCase(psz, "true") || StartsWithNoCase(psz, "false"))
				{
					bool bValue = StartsWithNoCase(psz, "true");
					if (bValue) psz += 4; else psz += 5;
					if (!pParent->IsElement()) throw FormatException("Cannot parse boolean value outside of an object at " + GetSource() + ".");
					XmlElement* pElement = (XmlElement*)pParent;
					if (AsElement)
						pElement->AddBoolAsText(NameText.c_str(), bValue);
					else
					{
						if (pElement->FindAttribute(NameText.c_str()) != nullptr)
							throw FormatException("JSON name '" + NameText + "' was duplicated more than once within the same object and outside of an array.");

						pElement->AddBoolAsAttr(NameText.c_str(), bValue);
					}
					return;
				}

				// Look for null, handled as an empty, but named, Xml element.
				if (StartsWithNoCase(psz, "null"))
				{
					psz += 4;
					auto pEntry = make_shared<XmlElement>(NameText.c_str());
					pEntry->SourceLocation = GetSource();
					pParent->AppendChild(pEntry);
					return;
				}

				throw FormatException("Badly formed JSON (unrecognized value type during parsing of '" + pParent->ToString() + "' at " + GetSource() + ")");
			}
		}

		inline void JsonParser::ParseObject(const char *&psz, XmlNode* pNode)
		{
			// Precondition: Assumes that the opening brace of the object has been parsed, and that the pointer is at the beginning of object content.
			// Postcondition: ParseNode() returns after the closing brace has been parsed (assuming no errors).			
				
			for (;;)
			{
				if (*psz == 0) throw FormatException("Badly formed JSON (no closing brace for '" + pNode->ToString() + "' from " + pNode->SourceLocation + ")");
				if (IsWhitespace(*psz)) { 
					if (*psz == '\n') CurrentLineNumber++;
					psz++; continue; 
				}
				
				if (*psz == '\"')
				{
					// ParseString() expects the opening quotation and parses until past the closing quotation.
					string NameText = ParseString(psz);
					
					// Look for the colon and value.
					for (;;)
					{
						if (*psz == 0) throw FormatException("Badly formed JSON (name without value and no closing brace for '" + pNode->ToString() + "' from " + pNode->SourceLocation + ")");
						if (IsWhitespace(*psz)) { 
							if (*psz == '\n') CurrentLineNumber++;
							psz++; continue; 
						}
						
						if (*psz == ':')
						{
							psz ++;
							
							// We have reached the value side of the name:value pair.
							ParseValue(psz, pNode, NameText);
							break;
						}

						throw FormatException("Badly formed JSON (expected colon following name while parsing '" + pNode->ToString() + "' at " + GetSource() + ")");
					}

					// Look for the comma or closing brace.
					for (;;)
					{
						if (*psz == 0) throw FormatException("Badly formed JSON (no comma or closing brace for '" + pNode->ToString() + "' at " + GetSource() + ")");
						if (IsWhitespace(*psz)) { 
							if (*psz == '\n') CurrentLineNumber++;
							psz++; continue; 
						}
						if (*psz == ',') { psz++; break; }
						if (*psz == '}') { psz++; return; }
						throw FormatException("Badly formed JSON (unexpected '" + string(1, *psz) + "' character instead of comma or closing brace for '" + pNode->ToString() + "' at " + GetSource() + ")");
					}

					// If we get here, it means we found a comma.  We can parse the next name:value pair as the next addition to pNode.
					continue;
				}

				if (*psz == '}') { psz++; return; }			// Empty object.
				
				throw FormatException("Badly formed JSON (expected quote or closing brace for '" + pNode->ToString() + "' from " + pNode->SourceLocation + ")");
			}
		}
		
	}
}

#endif	// __WBJsonParserToXml_h__

//	End of JsonParserToXml.h

