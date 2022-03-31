/////////
//	Yaml.h
////

#ifndef __WBYaml_h__
#define __WBYaml_h__

/** Table of Contents **/

namespace wb
{
	namespace yaml
	{		
		class YamlNode;
		class YamlScalar;
		class YamlSequence;
		class YamlMapping;
	}
}

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../Foundation/STL/Collections/UnorderedMap.h"
#include "../../Foundation/STL/Collections/Map.h"
#include "../../IO/Streams.h"
#include "../../IO/MemoryStream.h"
#include "../BaseTypeParsing.h"

/** Content **/

namespace wb
{
	namespace yaml
	{
		using namespace std;				

		/** JsonWriterOptions **/

		/// <summary>JsonWriterOptions provides a set of control options for generating a JSON string or stream from YAML content.</summary>
		class JsonWriterOptions
		{
		public:
			/// <summary>[Default=0]  Indentation level for output text.</summary>
			int		Indentation;

			/// <summary>[Default=false] Ordinarily, all scalar content is quoted.  If true, recognize and write numeric scalar content without
			/// the quotes.
			/// </summary>
			bool	UnquoteNumbers;

			JsonWriterOptions() { Indentation = 0; UnquoteNumbers = false; }
		};

		/** YamlNode **/

		class YamlNode
		{			
		protected:
			static void		AddIndent(const JsonWriterOptions& Options, string& OnString) 
			{
				for (int ii = 0; ii < Options.Indentation; ii++) OnString += '\t';
			}

		public:
			YamlNode(string FromSource) : Source(FromSource), Tag("?") { }
			virtual ~YamlNode() { }

			string Tag;
			string Source;

			YamlNode& operator=(const YamlNode&) = default;
			virtual unique_ptr<YamlNode>	DeepCopy() = 0;

			virtual string	ToJson(JsonWriterOptions Options = JsonWriterOptions())
			{
				throw NotImplementedException("Conversion to Json value for node at " + Source + " was not implemented.");
			}
		};

		/** YamlScalar **/

		class YamlScalar : public YamlNode
		{
			typedef YamlNode base;
			friend class YamlSequence;
			friend class YamlMapping;

			string	ToJsonValue(bool UnquoteNumbers)
			{
				if (Null) return "null";
				if (!UnquoteNumbers) return "\"" + wb::json::JsonString::Escape(Content) + "\"";

				// Check if the value is entirely numeric...
				if (Content.length() == 0) return "\"\"";
				if (Content.length() > 2 && Content[0] == '0' && Content[1] == 'x')
				{
					// Hex number, write to JSON as decimal.
					UInt64 Value;
					if (UInt64_TryParse(Content, NumberStyles::Integer, Value))
					{
						char buffer[26];
						_ui64toa_s(Value, buffer, 26, 10);
						return wb::json::JsonString::Escape(buffer);
					}
				}

				// Check the first character of the number.  Here, +/- are allowed.								
				if (!(Content[0] >= '0' && Content[0] <= '9') && Content[0] != '.'
					&& Content[0] != '+' && Content[0] != '-')
					return "\"" + wb::json::JsonString::Escape(Content) + "\"";
				
				bool FractionalPart = false;
				bool ExponentialPart = false;
				bool IsFractionalPartZero = true;
				if (Content[0] == '.') FractionalPart = true;

				// Check the whole number portion
				int ii = 1;
				string WholePortion;				
				if (!FractionalPart)
				{
					WholePortion += Content[0];
					for (; ii < Content.length(); ii++)
					{
						if (Content[ii] >= '0' && Content[ii] <= '9') { WholePortion += Content[ii]; continue; }
						if (toupper(Content[ii]) == 'E') { FractionalPart = false; ExponentialPart = true; ii++; break; }
						if (Content[ii] == '.') { FractionalPart = true; ii++; break; }
						return "\"" + wb::json::JsonString::Escape(Content) + "\"";
					}
				}								

				if (FractionalPart)
				{
					for (; ii < Content.length(); ii++)
					{
						if (Content[ii] == '0') continue;
						if (Content[ii] >= '1' && Content[ii] <= '9') { IsFractionalPartZero = false; continue; }
						if (toupper(Content[ii]) == 'E') { FractionalPart = false; ExponentialPart = true; ii++; break; }
						return "\"" + wb::json::JsonString::Escape(Content) + "\"";
					}
				}

				if (ExponentialPart)
				{
					if (Content[ii] == '+' || Content[ii] == '-') ii++;
					for (; ii < Content.length(); ii++)
					{
						if (Content[ii] >= '0' && Content[ii] <= '9') continue;
						return "\"" + wb::json::JsonString::Escape(Content) + "\"";
					}
				}

				// It is entirely numeric.  Omit quotes.
				// While there's no real rule here, also convert whole numbers to integers to match the Yaml Json references
				// used in the YamlParse unit tests.
				if (IsFractionalPartZero && !ExponentialPart) return WholePortion;
				return Content;
			}

		public:			
			YamlScalar(string FromSource, string Text, bool IsNull) : base(FromSource), Content(Text), Null(IsNull) { }

			YamlScalar(const YamlScalar&) = default;
			YamlScalar& operator=(const YamlScalar&) = default;

			string	Content;
			bool	Null;
			
			unique_ptr<YamlNode>	DeepCopy() override {
				return unique_ptr<YamlNode>(new YamlScalar(*this));
			}

			string	ToJson(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				string ret;				
				ret += ToJsonValue(Options.UnquoteNumbers);
				return ret;
			}
		};

		/** YamlSequence **/

		class YamlSequence : public YamlNode
		{
			typedef YamlNode base;
		public:			
			YamlSequence(string FromSource) : base(FromSource) { }
			
			YamlSequence(const YamlSequence&) = delete;
			YamlSequence& operator=(const YamlSequence&) = delete;

			vector<unique_ptr<YamlNode>>	Entries;

			unique_ptr<YamlNode>	DeepCopy() override {
				auto pRet = make_unique<YamlSequence>(Source);
				pRet->base::operator=(*this);		// Shallow copy the base members
				for (auto& it : Entries) {
					if (!it) pRet->Entries.push_back(nullptr);
					else pRet->Entries.push_back(it->DeepCopy());
				}
				return dynamic_pointer_movecast<YamlNode>(std::move(pRet));
			}

			string	ToJson(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				string ret;				
				ret += "[\n";
				Options.Indentation++;
				bool First = true;
				for (auto& pNode : Entries)
				{
					if (!First) ret += ",\n";
					else First = false;

					AddIndent(Options, ret);
					/*
					if (is_type<YamlScalar>(pNode))
					{						
						ret += ((YamlScalar*)pNode.get())->ToJsonValue(Options.UnquoteNumbers);
					}
					else
					{
					*/
					if (pNode == nullptr) ret += "null";
					else ret += pNode->ToJson(Options);
					//}
				}
				ret += "\n";
				Options.Indentation--;
				AddIndent(Options, ret);				
				ret += "]";
				return ret;
			}
		};

		/** YamlMapping **/

		class YamlMapping : public YamlNode
		{
			typedef YamlNode base;
		public:
			YamlMapping(string FromSource) : base(FromSource) { }

			map<unique_ptr<YamlNode>, unique_ptr<YamlNode>>	Map;

			void Add(unique_ptr<YamlNode>&& pFrom, unique_ptr<YamlNode>&& pTo)
			{
				auto it = Map.find(pFrom);
				if (it != Map.end()) throw FormatException("Duplicate keys found at " + pFrom->Source + " and " + it->first->Source + " are not permitted in mapping at " + Source + ".");
				Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(std::move(pFrom), std::move(pTo)));
			}

			unique_ptr<YamlNode>	DeepCopy() override {
				auto pRet = make_unique<YamlMapping>(Source);
				pRet->base::operator=(*this);		// Shallow copy the base members				
				for (auto& it : Map) {
					if (it.first == nullptr && it.second == nullptr)
						pRet->Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(nullptr, nullptr));
					else if (it.first == nullptr)
						pRet->Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(nullptr, it.second->DeepCopy()));
					else if (it.second == nullptr)
						pRet->Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(it.first->DeepCopy(), nullptr));
					else
						pRet->Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(it.first->DeepCopy(), it.second->DeepCopy()));
				}
				return dynamic_pointer_movecast<YamlNode>(std::move(pRet));
			}

			string	ToJson(JsonWriterOptions Options = JsonWriterOptions()) override
			{
				string ret;
				ret += "{\n";
				Options.Indentation++;
				bool First = true;
				for (auto& KVP : Map)
				{
					if (!First) ret += ",\n";
					else First = false;

					/*
					if (is_type<YamlScalar>(KVP.first))
					{
						AddIndent(Options, ret);
						ret += ((YamlScalar*)KVP.first.get())->ToJsonValue(Options.UnquoteNumbers);
					}
					else
					{
					*/
					AddIndent(Options, ret);
					bool UnquoteNumbers = Options.UnquoteNumbers;
					Options.UnquoteNumbers = false;					// JSON keys are always quoted, even though values can be numbers.
					if (KVP.first == nullptr) ret += "\"\"";		// JSON does not permit a null key, so there is no perfect representation of the YAML here.
					else ret += KVP.first->ToJson(Options);
					Options.UnquoteNumbers = UnquoteNumbers;
					//}

					ret += ": ";

					/*
					if (is_type<YamlScalar>(KVP.second))
					{
						AddIndent(Options, ret);
						ret += ((YamlScalar*)KVP.second.get())->ToJsonValue(Options.UnquoteNumbers);
					}
					else
					{
					*/
					if (KVP.second == nullptr) ret += "null";
					else ret += KVP.second->ToJson(Options);
					//}
				}
				ret += "\n";
				Options.Indentation--;
				AddIndent(Options, ret);
				ret += "}";				
				return ret;
			}
		};
	}
}

#endif	// __WBYaml_h__

//	End of Yaml.h


