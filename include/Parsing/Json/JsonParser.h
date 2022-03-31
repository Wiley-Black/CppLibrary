/////////
//	JsonParser.h
////

#ifndef __WBJsonParser_h__
#define __WBJsonParser_h__

/** Dependencies **/

#include "../../IO/Streams.h"
#include "../../IO/MemoryStream.h"
#include "../Xml/XmlParser.h"
#include "Json.h"

/** Content **/

namespace wb
{
	namespace json
	{		
		class JsonParser : public wb::xml::StreamParser<64>
		{
			typedef wb::xml::StreamParser<64> base;
			
			bool IsWhitespace(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'; }

			// Calls Advance() to pass all whitespace at the current position
			void AdvanceWhitespace();									
			
			unique_ptr<JsonNumber> ParseNumber();
			unique_ptr<JsonString> ParseJsonString();
			unique_ptr<JsonSequence> ParseArray();
			unique_ptr<JsonMapping> ParseObject();
			unique_ptr<JsonValue> ParseValue();
			
			void StartStream(wb::io::Stream& stream, const string& sSourceFilename);
			void FinishStream();

		public:

			JsonParser();
			~JsonParser();

			/// <summary>Parses the stream, which must contain JSON document(s).  An exception is thrown on error.
			/// The stream will only be advanced up to conclusion of the next JSON document.</summary>
			/// <returns>A JsonValue, or nullptr if the document was empty.</returns>
			static unique_ptr<JsonValue> Parse(wb::io::Stream& stream, const string& sSourceFilename = "");

			static unique_ptr<JsonValue> ParseString(const string& str, const string& sSourceFilename = "");
		};
	
		/** JsonParser Implementation **/
		
		inline JsonParser::JsonParser()			
		{
		}

		inline JsonParser::~JsonParser()
		{
		}						

		inline void JsonParser::AdvanceWhitespace()
		{
			if (!Need(1)) return;			
			while (IsWhitespace(Current)) {
				if (!Advance()) return;
			}
		}				

		inline unique_ptr<JsonNumber> JsonParser::ParseNumber()
		{
			if (Current != '-' && !(Current >= '0' && Current <= '9')) throw FormatException("Expected - or digit as first character in JSON number parsing at " + GetSource() + ".");
			
			auto pRetNode = make_unique<JsonNumber>(GetSource(), "");
			auto& ret = pRetNode->Content;				// For convenience and clarity, edit the string directly by reference.

			ret += Current;
			if (!Advance()) return pRetNode;

			while (Current >= '0' && Current <= '9') {
				ret += Current;
				if (!Advance()) return pRetNode;
			}

			// Parse fraction, if present
			if (Current == '.')
			{
				ret += Current;
				if (!Advance()) throw FormatException("Expected digit to follow decimal place in JSON number parsing at " + GetSource() + ".");
				while (Current >= '0' && Current <= '9') {
					ret += Current;
					if (!Advance()) return pRetNode;
				}
			}

			// Parse fraction, if present
			if (Current == '.')
			{
				ret += Current;
				if (!Advance()) throw FormatException("Expected digit to follow decimal place in JSON number parsing at " + GetSource() + ".");
				while (Current >= '0' && Current <= '9') {
					ret += Current;
					if (!Advance()) return pRetNode;
				}
			}

			// Parse exponent, if present
			if (Current == 'e' || Current == 'E')
			{
				ret += Current;
				if (!Advance()) throw FormatException("Expected -, +, or digit to follow exponent marker in JSON number parsing at " + GetSource() + ".");
				if (Current == '-' || Current == '+')
				{
					ret += Current;
					if (!Advance()) throw FormatException("Expected digit to follow signed exponent marker in JSON number parsing at " + GetSource() + ".");
				}
				while (Current >= '0' && Current <= '9') {
					ret += Current;
					if (!Advance()) return pRetNode;
				}
			}

			return pRetNode;
		}

		inline unique_ptr<JsonString> JsonParser::ParseJsonString()
		{
			if (Current != '\"') throw FormatException("Expected double-quotes as first character in JSON string parsing at " + GetSource() + ".");

			auto pRetNode = make_unique<JsonString>(GetSource(), "");
			auto& ret = pRetNode->Content;				// For convenience and clarity, edit the string directly by reference.			

			for (;;)
			{
				if (!Advance()) throw FormatException("Unterminated string beginning at " + pRetNode->Source + ".");

				if (Current == '\"')
				{
					Advance();
					return pRetNode;
				}

				if (Current == '\\')
				{
					if (!Advance()) throw FormatException("Unterminated escape sequence and string beginning at " + pRetNode->Source + ".");
					switch (Current)
					{
					case '\"': ret += '\"'; break;
					case '\\': ret += '\\'; break;
					case '/': ret += '/'; break;
					case 'b': ret += '\b'; break;
					case 'f': ret += '\f'; break;
					case 'n': ret += '\n'; break;
					case 'r': ret += '\r'; break;
					case 't': ret += '\t'; break;
					case 'u': throw NotImplementedException("This JSON parser does not implement support for escaped unicode sequences at " + GetSource() + ".");
					default: throw FormatException("Unrecognized JSON escape character at " + GetSource() + ".");
					}					
					continue;
				}

				ret += Current;
			}
		}

		inline unique_ptr<JsonSequence> JsonParser::ParseArray()
		{
			// Array in JSON is also referred to as a list or sequence.

			if (Current != '[') throw FormatException("Expected opening bracket as first character in JSON array parsing at " + GetSource() + ".");

			auto pRetNode = make_unique<JsonSequence>(GetSource());
			auto& Elements = pRetNode->Elements;				// For convenience and clarity, edit the vector directly by reference.			

			if (!Advance()) throw FormatException("Missing closing bracket for JSON array started at " + pRetNode->Source + ".");

			for (;;)
			{				
				AdvanceWhitespace();
				if (!Need(1)) throw FormatException("Missing closing bracket for JSON array started at " + pRetNode->Source + ".");

				if (Current == ']')
				{
					Advance();
					return pRetNode;
				}

				Elements.push_back(ParseValue());

				AdvanceWhitespace();

				if (Current != ',' && Current != ']') throw FormatException("Syntax error at " + GetSource() + "; missing comma or closing bracket for JSON array started at " + pRetNode->Source + ".");

				if (Current == ',')
				{
					if (!Advance()) throw FormatException("Missing closing bracket for JSON array started at " + pRetNode->Source + ".");
				}
			}
		}

		inline unique_ptr<JsonMapping> JsonParser::ParseObject()
		{
			// Object in JSON is also referred to as a dictionary or mapping.

			if (Current != '{') throw FormatException("Expected opening curly as first character in JSON object parsing at " + GetSource() + ".");

			auto pRet = make_unique<JsonMapping>(GetSource());

			if (!Advance()) throw FormatException("Missing closing curly for JSON object started at " + pRet->Source + ".");

			for (;;)
			{				
				AdvanceWhitespace();
				if (!Need(1)) throw FormatException("Missing closing curly for JSON object started at " + pRet->Source + ".");

				if (Current == '}')
				{
					Advance();
					return pRet;
				}

				auto JsonKey = ParseJsonString();
				string Key = JsonKey->Content;
				AdvanceWhitespace();
				if (!Need(1)) throw FormatException("Missing colon and closing curly for JSON object started at " + pRet->Source + ".");
				if (Current != ':') throw FormatException("Expected colon to follow string at " + GetSource() + " in JSON object at " + pRet->Source + ".");
				if (!Advance()) throw FormatException("Expected value to follow colon at " + GetSource() + " in JSON object at " + pRet->Source + ".");
				pRet->Add(Key, ParseValue());

				AdvanceWhitespace();

				if (Current != ',' && Current != '}') throw FormatException("Syntax error at " + GetSource() + "; missing comma or closing curly for JSON object started at " + pRet->Source + ".");

				if (Current == ',')
				{
					if (!Advance()) throw FormatException("Missing closing curly for JSON array started at " + pRet->Source + ".");
				}
			}
		}

		inline unique_ptr<JsonValue> JsonParser::ParseValue()
		{
			AdvanceWhitespace();
			if (!Need(1)) throw FormatException("Expected JSON value at " + GetSource() + ".");

			switch (Current)
			{
			case '[': return dynamic_pointer_movecast<JsonValue>(ParseArray());
			case '{': return dynamic_pointer_movecast<JsonValue>(ParseObject());
			case '\"': return dynamic_pointer_movecast<JsonValue>(ParseJsonString());
			}

			if (Current == 't' && Need(4) && IsNextEqual("rue")) {
				string Source = GetSource();
				Advance(4);
				return unique_ptr<JsonValue>(new JsonBoolean(Source, true));
			}

			if (Current == 'f' && Need(5) && IsNextEqual("alse")) {
				string Source = GetSource();
				Advance(5);
				return unique_ptr<JsonValue>(new JsonBoolean(Source, false));
			}

			if (Current == 'n' && Need(4) && IsNextEqual("ull")) {
				string Source = GetSource();
				Advance(4);
				return unique_ptr<JsonValue>(new JsonNull(Source));
			}

			if (Current == '-' || (Current >= '0' && Current <= '9')) return dynamic_pointer_movecast<JsonValue>(ParseNumber());

			throw FormatException("Syntax error; expected JSON value at " + GetSource() + ".");
		}

		inline void JsonParser::StartStream(wb::io::Stream& stream, const string& sSourceFilename)
		{
			CurrentSource = sSourceFilename;
			CurrentLineNumber = 1;

			pCurrentStream = &stream;
			if (Need(3) && Current == 0xEF && pNext[0] == 0xBB && pNext[2] == 0xBF) {
				// UTF-8 BOM.  TODO: Respond to detected encoding.
				Advance(); Advance(); Advance();
			}
		}

		inline void JsonParser::FinishStream()
		{
			CurrentSource = "";
			CurrentLineNumber = 1;
			pCurrentStream = nullptr;
		}

		inline /*static*/ std::unique_ptr<JsonValue> JsonParser::Parse(wb::io::Stream& stream, const string& sSourceFilename)
		{
			JsonParser parser;
			parser.StartStream(stream, sSourceFilename);
			try
			{
				auto ret = parser.ParseValue();
				parser.FinishStream();
				return ret;
			}
			catch (...)
			{
				parser.FinishStream();
				throw;
			}
		}

		inline /*static*/ unique_ptr<JsonValue> JsonParser::ParseString(const string& str, const string& sSourceFilename)
		{
			wb::io::MemoryStream ms;
			wb::io::StringToStream(str, ms);
			ms.Rewind();
			return Parse(ms, sSourceFilename);
		}
	}
}

#endif	// __WBJsonParser_h__

//	End of JsonParser.h


