/////////
//	YamlParser.h
////

#ifndef __WBYamlParser_h__
#define __WBYamlParser_h__

/** Dependencies **/

#include <iostream>		// For debugging

#include "../../IO/Streams.h"
#include "../../IO/MemoryStream.h"
#include "../Xml/XmlParser.h"
#include "Yaml.h"
#include "YamlEventParser.h"

/** Content **/

namespace wb
{
	namespace yaml
	{		
		class YamlParser : public wb::yaml::YamlEventParser
		{
			typedef wb::yaml::YamlEventParser base;			

		protected:
			
			/** In order to utilize unique_ptrs instead of shared_ptrs, a certain strategy of one-copy is employed here:
			*	1) nodes are DeepCopy()'d if they need to be reserved as an anchor or dereferenced.
			*	2) +MAP or +SEQ (OnOpenEvent) adds a node to the top of the Nodes stack, but not to pKey or the parent.
			*	3) -MAP or -SEQ (OnCloseEvent) removes a node from the top of the Nodes stack and attaches it to pKey or the parent.
			*	4) Upon -DOC, there must only be a single topmost entry in Nodes, which is removed and appended to Documents.
			*/

			struct node_data
			{
				unique_ptr<YamlNode>	pNode;
				string					Anchor;

				unique_ptr<YamlNode>	pKey;			// Only applicable if pNode is a YamlMapping.

				node_data(unique_ptr<YamlNode>&& pNewNode, string AnchorName) : pNode(std::move(pNewNode)), Anchor(AnchorName) { }
			};

			stack<node_data>							Nodes;
			unordered_map<string, unique_ptr<YamlNode>>	Anchors;
			unique_ptr<YamlNode>						pNextDocument;
			vector<unique_ptr<YamlNode>>				Documents;

			void Attach(unique_ptr<YamlNode>&& pNode)
			{
				if (Nodes.empty())
				{
					if (pNextDocument)
						throw FormatException("Encountered unexpected content at " + pNode->Source + " without document closure from previous content.");
					else
						pNextDocument = std::move(pNode);
					return;
				}

				if (is_type<YamlMapping>(Nodes.top().pNode))
				{
					if (!Nodes.top().pKey) Nodes.top().pKey = std::move(pNode);
					else
					{
						auto pMapping = (YamlMapping*)Nodes.top().pNode.get();
						pMapping->Add(std::move(Nodes.top().pKey), std::move(pNode));
					}
					return;
				}
				else if (is_type<YamlSequence>(Nodes.top().pNode))
				{
					auto pSeq = (YamlSequence*)Nodes.top().pNode.get();
					pSeq->Entries.push_back(std::move(pNode));
					return;
				}
				else 
					throw NotSupportedException("Unrecognized or unsupported node type at " + Nodes.top().pNode->Source + " while attaching " + GetSource() + ".");
			}

			void OnOpenEvent(Event Event, string Anchor, string Tag, bool FlowStyleOrExplicit = false) override
			{
				switch (Event)
				{
				case Event::Stream: return;
				case Event::Document: return;
				case Event::Map:
				{
					auto pNode = unique_ptr<YamlNode>(new YamlMapping(GetSource()));
					Nodes.push(node_data(std::move(pNode), Anchor));
					return;
				}
				case Event::Sequence:
				{
					auto pNode = unique_ptr<YamlNode>(new YamlSequence(GetSource()));
					Nodes.push(node_data(std::move(pNode), Anchor));
					return;
				}
				default:
					throw NotImplementedException("Unrecognized Event type.");
				}
			}

			void OnCloseEvent(Event Event, bool Explicit) override
			{
				switch (Event)
				{
				case Event::Stream: return;
				case Event::Document: 
				{
					Anchors.clear();
					if (Nodes.size() != 0) throw FormatException("End of YAML document detected at " + GetSource() + " but nodes still open (i.e. " + Nodes.top().pNode->Source + ").");
					if (pNextDocument == nullptr) Documents.push_back(unique_ptr<YamlNode>(new YamlScalar(GetSource(), string(), true)));
					else Documents.push_back(std::move(pNextDocument));
					return;
				}
				case Event::Map:
				case Event::Sequence:
				{					
					if (Nodes.size() == 0) throw NotSupportedException("While parsing YAML, closure event received but no nodes are open at " + GetSource() + ".");
					else {
						auto Node = std::move(Nodes.top());
						Nodes.pop();
						if (Node.Anchor.length()) Anchors[Node.Anchor] = Node.pNode->DeepCopy();
						Attach(std::move(Node.pNode));
					}
					return;
				}				
				default:
					throw NotImplementedException("Unrecognized Event type.");
				}
			}

			void OnValueEvent(string Value, string Anchor, string Tag, Flags flags = NoFlags)
			{
				//wb::yaml::string_ReplaceAll(Value, "\\", "\\\\");
				//wb::yaml::string_ReplaceAll(Value, "\n", "\\n");
				//wb::yaml::string_ReplaceAll(Value, "\t", "\\t");

				auto pValue = unique_ptr<YamlNode>(new YamlScalar(GetSource(), Value, (flags & NullValueFlag) != 0));
				if (Anchor.length()) Anchors[Anchor] = pValue->DeepCopy();				
				Attach(std::move(pValue));
			}

			void OnAliasEvent(string AnchorName)
			{
				if (Anchors.find(AnchorName) == Anchors.end()) throw FormatException("Attempt to reference an unknown anchor '" + AnchorName + "' at " + GetSource() + ".");
				Attach(Anchors[AnchorName]->DeepCopy());
			}

		public:

			YamlParser();
			~YamlParser();

			/// <summary>Parses the stream, which must contain YAML document(s).  An exception is thrown on error.
			/// All YAML documents in the stream will be parsed and a vector will be returned where each
			/// entry is one document's top node.</summary>
			/// <returns>A YamlNode, or nullptr if the document was empty.</returns>
			static vector<unique_ptr<YamlNode>> Parse(wb::io::Stream& stream, const string& sSourceFilename = "");

			static vector<unique_ptr<YamlNode>> ParseString(const string& str, const string& sSourceFilename = "");
		};
	
		/** YamlParser Implementation **/				

		inline YamlParser::YamlParser()			
		{
		}

		inline YamlParser::~YamlParser()
		{
		}								

		inline /*static*/ std::vector<std::unique_ptr<YamlNode>> YamlParser::Parse(wb::io::Stream& stream, const string& sSourceFilename)
		{
			YamlParser parser;
			parser.StartStream(stream, sSourceFilename);
			try
			{
				parser.ParseTopLevel();
				parser.FinishStream();
				return std::move(parser.Documents);
			}
			catch (...)
			{
				parser.FinishStream();
				throw;
			}
		}

		inline /*static*/ vector<unique_ptr<YamlNode>> YamlParser::ParseString(const string& str, const string& sSourceFilename)
		{
			wb::io::MemoryStream ms;
			wb::io::StringToStream(str, ms);
			ms.Rewind();
			return Parse(ms, sSourceFilename);
		}
	}
}

#endif	// __WBYamlParser_h__

//	End of YamlParser.h

