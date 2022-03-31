/////////
//	Json.h
////

#ifndef __WBJson_h__
#define __WBJson_h__

/** Table of Contents **/

namespace wb
{
	namespace json
	{				
		class JsonValue;
		class JsonString;
		class JsonNumber;
		class JsonBoolean;
		class JsonNull;
		class JsonSequence;			// Also called an array, vector, list, or sequence.
		class JsonMapping;			// Also called an object, record, struct, dictionary, hash table, keyed list, or associative array.
	}
}

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../Foundation/STL/Collections/UnorderedMap.h"
#include "../../Foundation/STL/Collections/Map.h"
#include "../../IO/Streams.h"
#include "../../IO/MemoryStream.h"
#include "../Xml/Xml.h"

/** Content **/

namespace wb
{
	namespace json
	{
		using namespace std;

		/** JsonWriterOptions **/

		/// <summary>JsonWriterOptions provides a set of control options for generating a JSON string.</summary>
		class JsonWriterOptions
		{
		public:
			/// <summary>[Default=0]  Indentation level for output text.</summary>
			int		Indentation;

			JsonWriterOptions() { Indentation = 0; }
		};

		/** JsonValue **/

		class JsonValue
		{			
		protected:
			static void		Indent(const JsonWriterOptions& Options, string& OnString) 
			{
				for (int ii = 0; ii < Options.Indentation; ii++) OnString += '\t';
			}

		public:
			JsonValue(string FromSource) : Source(FromSource) { }
			virtual ~JsonValue() { }
			
			string Source;

			JsonValue& operator=(const JsonValue&) = default;
			virtual unique_ptr<JsonValue>	DeepCopy() = 0;

			virtual string	ToString(JsonWriterOptions Options = JsonWriterOptions())
			{
				throw NotImplementedException("Conversion to JSON value for node at " + Source + " was not implemented.");
			}

			virtual unique_ptr<xml::XmlElement>	ToXml()
			{
				throw NotImplementedException("Conversion to XML value for node at " + Source + " was not implemented.");
			}
		};

		/** JsonString **/

		class JsonString : public JsonValue
		{		
			typedef JsonValue base;

		public:
			JsonString(string FromSource, string Text) : base(FromSource), Content(Text) { }
			virtual ~JsonString() { }
			
			string Content;

			JsonString& operator=(const JsonString&) = default;

			unique_ptr<JsonValue>	DeepCopy() override {
				return unique_ptr<JsonValue>(new JsonString(Source, Content));
			}

			static string Escape(string Value)
			{
				string ret;
				for (auto ii = 0; ii < Value.length(); ii++)
				{
					switch (Value[ii])
					{
					case '\\': ret += "\\\\"; continue;
					case '\b': ret += "\\b"; continue;
					case '\n': ret += "\\n"; continue;
					case '\r': ret += "\\r"; continue;
					case '\f': ret += "\\f"; continue;
					case '\t': ret += "\\t"; continue;
					case '\"': ret += "\\\""; continue;
					default: ret += Value[ii];
					}
				}
				return ret;
			}

			explicit operator string() const {
				return Content;
			}

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				return "\"" + Escape(Content) + "\"";
			}

			unique_ptr<xml::XmlElement>	ToXml() override
			{
				// If ToXml() is called directly here, then this JsonString is the top-level node.
				using namespace xml;				
				auto pRet = make_unique<XmlElement>("root");
				auto pTextNode = make_shared<XmlText>();
				pTextNode->SourceLocation = Source;
				pTextNode->Text = Content;
				pRet->AppendChild(pTextNode);
				return pRet;
			}
		};

		/** JsonNumber **/

		class JsonNumber : public JsonString
		{
			typedef JsonString base;

		public:
			JsonNumber(string FromSource, string AsText) : base(FromSource, AsText) { }
			virtual ~JsonNumber() { }			

			JsonNumber& operator=(const JsonNumber&) = default;
			unique_ptr<JsonValue>	DeepCopy() override {
				return unique_ptr<JsonValue>(new JsonNumber(Source, Content));
			}

			explicit operator double() const {
				return Double_Parse(Content, wb::NumberStyles::Float);
			}

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				return Content;
			}			
		};		

		/** JsonBoolean **/

		class JsonBoolean : public JsonValue
		{
			typedef JsonValue base;

		public:
			JsonBoolean(string FromSource, bool NewValue = false) : base(FromSource), Value(NewValue) { }
			virtual ~JsonBoolean() { }

			bool Value;

			JsonBoolean& operator=(const JsonBoolean&) = default;
			unique_ptr<JsonValue>	DeepCopy() override {
				return unique_ptr<JsonValue>(new JsonBoolean(Source, Value));
			}

			explicit operator bool() const {
				return Value;
			}

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				return Value ? "true" : "false";
			}

			unique_ptr<xml::XmlElement>	ToXml() override
			{
				// If ToXml() is called directly here, then this JsonBoolean is the top-level node.
				using namespace xml;
				auto pRet = make_unique<XmlElement>("root");
				auto pTextNode = make_shared<XmlText>();
				pTextNode->SourceLocation = Source;
				pTextNode->Text = Value ? "true" : "false";
				pRet->AppendChild(pTextNode);
				return pRet;
			}
		};

		/** JsonNull **/

		class JsonNull : public JsonValue
		{
			typedef JsonValue base;

		public:
			JsonNull(string FromSource) : base(FromSource) { }
			virtual ~JsonNull() { }			

			JsonNull& operator=(const JsonNull&) = default;
			unique_ptr<JsonValue>	DeepCopy() override {
				return unique_ptr<JsonValue>(new JsonNull(Source));
			}			

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				return "null";
			}

			unique_ptr<xml::XmlElement>	ToXml() override
			{
				// If ToXml() is called directly here, then this JsonBoolean is the top-level node.
				using namespace xml;
				auto pRet = make_unique<XmlElement>("root");
				return pRet;
			}
		};

		/** JsonSequence **/

		class JsonSequence : public JsonValue
		{
			typedef JsonValue base;			

		public:			
			JsonSequence(string FromSource) : base(FromSource) { }
			
			JsonSequence(const JsonSequence&) = delete;
			JsonSequence& operator=(const JsonSequence&) = delete;

			vector<unique_ptr<JsonValue>>	Elements;

			unique_ptr<JsonValue>	DeepCopy() override {
				auto pRet = make_unique<JsonSequence>(Source);
				pRet->base::operator=(*this);		// Shallow copy the base members
				for (auto& it : Elements) pRet->Elements.push_back(it->DeepCopy());
				return dynamic_pointer_movecast<JsonValue>(std::move(pRet));
			}

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				string ret;
				ret += "[\n";
				Options.Indentation++;
				bool First = true;
				for (auto& pNode : Elements)
				{
					if (!First) ret += ",\n";
					else First = false;

					Indent(Options, ret);
					ret += pNode->ToString(Options);
				}
				ret += "\n";
				Options.Indentation--;
				Indent(Options, ret);								
				ret += "]";				
				return ret;
			}

			unique_ptr<xml::XmlElement>	ToXml() override
			{
				// If ToXml() is called directly here, then this JsonSequence is the top-level node.
				using namespace xml;
				auto pRet = make_unique<XmlElement>("root");
				AppendToXml(*pRet);
				return pRet;
			}

		protected:
			friend class JsonMapping;

			void AppendToXml(xml::XmlElement& Parent)
			{
				using namespace xml;
				for (auto& pNode : Elements)
				{
					if (is_type<JsonString>(pNode))
					{
						Parent.AddStringAsText("string", ((JsonString*)pNode.get())->Content);
					}
					else if (is_type<JsonNumber>(pNode))
					{
						Parent.AddStringAsText("number", ((JsonString*)pNode.get())->Content);
					}
					else if (is_type<JsonNull>(pNode))
					{
						Parent.AppendChild(make_shared<XmlElement>("null"));
					}
					else if (is_type<JsonBoolean>(pNode))
					{
						Parent.AddStringAsText("boolean", ((JsonBoolean*)pNode.get())->Value ? "true" : "false");
					}
					else if (is_type<JsonSequence>(pNode))
					{
						auto pChild = make_shared<XmlElement>("array");
						((JsonSequence*)pNode.get())->AppendToXml(*pChild);
					}
					else if (is_type<JsonMapping>(pNode))
					{
						((JsonSequence*)pNode.get())->AppendToXml(Parent);
					}
					else throw NotSupportedException("Unrecognized JSON type.");
				}
			}
		};

		/** JsonMapping **/

		class JsonMapping : public JsonValue
		{
			typedef JsonValue base;

		public:
			JsonMapping(string FromSource) : base(FromSource) { }
						
			map<string, unique_ptr<JsonValue>>	Map;

			void Add(const string& from, unique_ptr<JsonValue>&& pTo)
			{
				auto it = Map.find(from);
				if (it != Map.end()) throw FormatException("Duplicate key '" + from + "' found in object/mapping at " + Source + ".");
				Map.insert(make_pair(from, std::move(pTo)));
			}

			unique_ptr<JsonValue>	DeepCopy() override {
				auto pRet = make_unique<JsonMapping>(Source);
				pRet->base::operator=(*this);		// Shallow copy the base members
				for (auto& it : Map) pRet->Map.insert(make_pair(it.first, it.second->DeepCopy()));
				return dynamic_pointer_movecast<JsonValue>(std::move(pRet));
			}

			string	ToString(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				string ret;				
				ret += "{\n";
				Options.Indentation++;
				bool First = true;
				for (auto& KVP : Map)
				{
					if (!First) ret += ",\n";
					else First = false;

					Indent(Options, ret);
					ret += "\"" + JsonString::Escape(KVP.first) + "\" : ";					
					ret += KVP.second->ToString(Options);
				}
				ret += "\n";
				Options.Indentation--;
				Indent(Options, ret);
				ret += "}";
				return ret;
			}

			unique_ptr<xml::XmlElement>	ToXml() override
			{
				// If ToXml() is called directly here, then this JsonMapping is the top-level node.
				using namespace xml;
				if (Map.size() == 1)
				{
					auto pRet = make_unique<XmlElement>(Map.begin()->first);

					auto& pValue = Map.begin()->second;
					if (is_type<JsonMapping>(pValue))
					{
						((JsonMapping*)pValue.get())->AppendToXml(*pRet);
					}
					else if (is_type<JsonSequence>(pValue))
					{
						((JsonSequence*)pValue.get())->AppendToXml(*pRet);
					}
					else if (is_type<JsonString>(pValue) || is_type<JsonNumber>(pValue))
					{
						auto pTextNode = make_shared<XmlText>();
						pTextNode->SourceLocation = pValue->Source;
						pTextNode->Text = ((JsonString*)pValue.get())->Content;
						pRet->AppendChild(pTextNode);
					}
					else if (is_type<JsonNull>(pValue))
					{
					}
					else if (is_type<JsonBoolean>(pValue))
					{
						auto pTextNode = make_shared<XmlText>();
						pTextNode->SourceLocation = pValue->Source;
						pTextNode->Text = ((JsonBoolean*)pValue.get())->Value ? "true" : "false";
						pRet->AppendChild(pTextNode);
					}
					else throw NotSupportedException("Unrecognized JSON node type.");

					return pRet;
				}
				else
				{
					auto pRet = make_unique<XmlElement>("root");
					AppendToXml(*pRet);
					return pRet;
				}
			}

		protected:
			friend class JsonSequence;

			void AppendToXml(xml::XmlElement& Parent)
			{
				using namespace xml;
				for (auto& KVP : Map)
				{
					auto& key = KVP.first;
					auto& pValue = KVP.second;
					if (is_type<JsonString>(pValue) || is_type<JsonNumber>(pValue))
					{
						Parent.AddStringAsAttr(key, ((JsonString*)pValue.get())->Content);
					}					
					else if (is_type<JsonNull>(pValue))
					{
						continue;
					}
					else if (is_type<JsonBoolean>(pValue))
					{
						Parent.AddBoolAsAttr(key, ((JsonBoolean*)pValue.get())->Value);
					}
					else if (is_type<JsonSequence>(pValue))
					{
						/**	{
						*		Example: [
						*			"a string",
						*			"a node" : { }
						*		]
						*   }
						*/

						auto pChild = make_shared<XmlElement>(key);
						((JsonSequence*)pValue.get())->AppendToXml(*pChild);
					}
					else if (is_type<JsonMapping>(pValue))
					{
						auto pChild = make_shared<XmlElement>(key);
						((JsonMapping*)pValue.get())->AppendToXml(*pChild);
					}
					else throw NotSupportedException("Unrecognized JSON type.");
				}
			}
		};

		/// <summary>
		/// Compares to JsonValues, including any subnodes in sequences or mappings.
		/// </summary>		
		/// <param name="Strict">
		/// If true, then the values must be completely identical.  If false, then strings, booleans, null,
		/// and numbers can be considered equivalent if they would be identical when both were cast as
		/// strings.  For example, if not strict, then "false" (JsonString) and false (JsonBoolean) are 
		/// considered equivalent.
		/// </param>		
		/// <returns>True if the values are identical.</returns>
		inline bool IsEqual(unique_ptr<JsonValue>& pA, const unique_ptr<JsonValue>& pB, bool Strict = true)
		{
			if (Strict)
			{
				// JsonNumber is a derived class of JsonString, so we have to carefully check that the types
				// are both JsonNumber in strict mode.
				if (is_type<JsonNumber>(pA) && !is_type<JsonNumber>(pB)) return false;
				if (is_type<JsonString>(pA) && is_type<JsonNumber>(pB)) return false;
				// Now that we've verified the types, we can proceed to compare as strings.
			}
			else
			{
				// In non-strict mode, see if we can convert to strings and compare.
				string a_string, b_string;

				if (is_type<JsonBoolean>(pA)) a_string = ((JsonBoolean*)pA.get())->Value ? "true" : "false";
				else if (is_type<JsonNull>(pA)) a_string = "null";
				else if (is_type<JsonString>(pA)) a_string = ((JsonString*)pA.get())->Content;

				if (is_type<JsonBoolean>(pB)) b_string = ((JsonBoolean*)pB.get())->Value ? "true" : "false";
				else if (is_type<JsonNull>(pB)) b_string = "null";
				else if (is_type<JsonString>(pB)) b_string = ((JsonString*)pB.get())->Content;

				// If both of them are empty strings, they will fall through and do the normal 
				// string comparison that will still return true.  If only one of them was convertable to
				// a string, or already a string, then they won't be equal unless the other was also convertable
				// or an equal string.
				if (a_string.length() > 0 || b_string.length() > 0) return wb::IsEqual(a_string, b_string);
			}

			if (is_type<JsonString>(pA) || is_type<JsonNumber>(pA))
			{
				if (!is_type<JsonString>(pB)) return false;
				return wb::IsEqual(((JsonString*)pA.get())->Content, ((JsonString*)pB.get())->Content);
			}
			else if (is_type<JsonNull>(pA))
			{
				return is_type<JsonNull>(pB);
			}
			else if (is_type<JsonBoolean>(pA))
			{
				if (!is_type<JsonBoolean>(pB)) return false;
				return ((JsonBoolean*)pA.get())->Value == ((JsonBoolean*)pB.get())->Value;
			}
			else if (is_type<JsonSequence>(pA))
			{
				if (!is_type<JsonSequence>(pB)) return false;
				auto* pAs = ((JsonSequence*)pA.get());
				auto* pBs = ((JsonSequence*)pB.get());
				if (pAs->Elements.size() != pBs->Elements.size()) return false;
				for (auto ii = 0; ii < pAs->Elements.size(); ii++)
				{
					if (!IsEqual(pAs->Elements[ii], pBs->Elements[ii], Strict)) return false;
				}
				return true;
			}
			else if (is_type<JsonMapping>(pA))
			{
				if (!is_type<JsonMapping>(pB)) return false;
				auto* pAm = ((JsonMapping*)pA.get());
				auto* pBm = ((JsonMapping*)pB.get());
				if (pAm->Map.size() != pBm->Map.size()) return false;

				// Mappings are unordered.
				for (auto& a_kvp : pAm->Map)
				{
					auto b_it = pBm->Map.find(a_kvp.first);
					if (b_it == pBm->Map.end()) return false;		// Key from a was not present in b.
					if (!IsEqual(a_kvp.second, b_it->second, Strict)) return false;
				}
				return true;
			}
			else throw NotImplementedException("Illegal or unrecognized type of JsonValue node.");
		}
	}
}

#endif	// __WBJson_h__

//	End of Json.h


