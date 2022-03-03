/////////
//	XmlImpl.h (Generation 5)
//	Copyright (C) 2010-2022 by Wiley Black
////

#ifndef __WBXmlImpl_v5_h__
#define __WBXmlImpl_v5_h__

#ifndef __WBXml_v5_h__
#error	This header should be included only via Xml.h.
#endif

/** Dependencies **/

#include <string>
#include <assert.h>
#include "../../../Text/StringComparison.h"
#include "../../../Text/StringConversion.h"
#include "../../BaseTypeParsing.h"
#include "../Xml.h"					// For Intellisense, but otherwise has no effect.

/** Content **/

namespace wb
{
	namespace xml
	{
		using namespace std;

		/** XmlWriterOptions implementation **/

		inline XmlWriterOptions::XmlWriterOptions(XmlWriterOptions& cp)
		{
			IncludeContent = cp.IncludeContent;
			Indentation = cp.Indentation;
			AllowSingleTags = cp.AllowSingleTags;
			EscapeAttributeWhitespace = cp.EscapeAttributeWhitespace;
		}

		inline XmlWriterOptions::XmlWriterOptions(XmlWriterOptions&& cp) noexcept
		{
			IncludeContent = cp.IncludeContent;
			Indentation = cp.Indentation;
			AllowSingleTags = cp.AllowSingleTags;
			EscapeAttributeWhitespace = cp.EscapeAttributeWhitespace;
		}

		inline XmlWriterOptions::XmlWriterOptions(bool AndContent)
		{
			IncludeContent = AndContent;
			Indentation = 0;
			AllowSingleTags = true;
			EscapeAttributeWhitespace = false;
		}

		/** XmlNode Implementation **/

		inline XmlNode::XmlNode()
		{
		}

		inline XmlNode::~XmlNode()
		{
			Children.clear();
		}		

		inline /*static*/ string XmlNode::Escape(const XmlWriterOptions& Options, const string& str)
		{
			string ret;
			for (size_t ii=0; ii < str.length(); ii++)
			{				
				if (!Options.EscapeAttributeWhitespace)
				{
					switch (str[ii])
					{
					case '\"': ret += "&quot;"; continue;
					case '&': ret += "&amp;"; continue;
					case '\'': ret += "&apos;"; continue;
					case '<': ret += "&lt;"; continue;
					case '>': ret += "&gt;"; continue;
					default: ret += str[ii]; continue;
					}
				}
				else
				{
					switch (str[ii])
					{
					case '\"': ret += "&quot;"; continue;
					case '&': ret += "&amp;"; continue;
					case '\'': ret += "&apos;"; continue;
					case '<': ret += "&lt;"; continue;
					case '>': ret += "&gt;"; continue;
					case ' ': ret += "&#x20;"; continue;
					case '\t': ret += "&#x9;"; continue;
					case '\n': ret += "&#xA;"; continue;
					case '\r': ret += "&#xD;"; continue;
					default: ret += str[ii]; continue;
					}
				}
			}
			return ret;
		}

		inline /*static*/ string XmlNode::Escape(const JsonWriterOptions& Options, const string& str)
		{
			string ret;
			for (size_t ii=0; ii < str.length(); ii++)
			{				
				switch (str[ii])
				{
				case '\"': ret += "\\\""; continue;
				case '\\': ret += "\\\\"; continue;
				case '/': ret += "\\/"; continue;
				case '\b': ret += "\\b"; continue;
				case '\f': ret += "\\f"; continue;
				case '\n': ret += "\\n"; continue;
				case '\r': ret += "\\r"; continue;
				case '\t': ret += "\\t"; continue;
					// TODO: Can use \u to escape a Unicode character.
				default: ret += str[ii]; continue;
				}
			}
			return ret;
		}

		inline /*static*/ void XmlNode::Indent(const XmlWriterOptions& Options, string& OnString)
		{
			for (int ii=0; ii < Options.Indentation; ii++) OnString += '\t';
		}

		inline /*static*/ void XmlNode::Indent(const JsonWriterOptions& Options, string& OnString)
		{
			for (int ii=0; ii < Options.Indentation; ii++) OnString += '\t';
		}

		inline string XmlNode::ToString(XmlWriterOptions Options)
		{
			if (Options.IncludeContent)
			{
				string ret;
				//int nSinceLF = 0;
				for (size_t ii=0; ii < Children.size(); ii++)
				{
					auto pChild = Children[ii];
					string substr = pChild->ToString(Options);
					ret += substr;
					//nSinceLF += substr.length();
					//if (nSinceLF > 40) { ret += '\n'; Indent(Options, ret); nSinceLF = 0; }
					// ret += '\n';					
				}
				return ret;
			}
			else return "XmlNode";
		}

		inline string XmlNode::ToJson(JsonWriterOptions Options)
		{
			throw NotImplementedException("Conversion to Json value for this node was not implemented.");
		}

		inline string XmlNode::ToJsonValue(JsonWriterOptions Options)
		{
			throw NotImplementedException("Conversion to Json value for this node was not implemented.");
		}

		inline const vector<shared_ptr<XmlElement>> XmlNode::Elements() const
		{
			vector<shared_ptr<XmlElement>> ret;
			for (size_t ii = 0; ii < Children.size(); ii++)
			{
				if (Children[ii]->IsElement()) ret.push_back(dynamic_pointer_cast<XmlElement>(Children[ii]));
			}
			return ret;
		}

		inline shared_ptr<XmlElement> XmlNode::FindChild(const string& TagName)
		{
			for (size_t ii=0; ii < Children.size(); ii++)
			{			
				if (Children[ii]->IsElement())
				{
					auto pElement = dynamic_pointer_cast<XmlElement>(Children[ii]);
					if (IsEqual(pElement->LocalName, TagName)) return pElement;
				}
			}
			return NULL;
		}

		inline shared_ptr<XmlElement> XmlNode::FindNthChild(const string& TagName, int N)
		{
			int Matches = 0;
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				if (Children[ii]->IsElement())
				{
					auto pElement = dynamic_pointer_cast<XmlElement>(Children[ii]);
					if (IsEqual(pElement->LocalName, TagName))
					{
						if (Matches == N) return pElement;				
						Matches ++;
					}
				}
			}
			return nullptr;
		}		

		inline void XmlNode::AppendChild(shared_ptr<XmlNode> pNewChild)
		{			
			Children.push_back(pNewChild);
		}

		inline void XmlNode::RemoveChild(shared_ptr<XmlNode> pChild)
		{
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				if (Children[ii].get() == pChild.get())
				{
					Children.erase(Children.begin() + ii);

					// We assume there is only one instance of pChild in
					// the list.  Anything else is improper construction.
					return;
				}
			}
		}

		/** XmlDocument Implementation **/
		
		inline string XmlDocument::ToString(XmlWriterOptions Options)
		{		
			if (!Options.IncludeContent) return "XmlDocument";
			return XmlNode::ToString(Options);
		}

		inline string XmlDocument::ToJson(JsonWriterOptions Options)
		{			
			string ret;
			Indent(Options, ret);
			ret += "{\n";
			Options.Indentation++;			
			ret += GetDocumentElement()->ToJson(Options);
			ret += "\n";
			Options.Indentation--;
			Indent(Options, ret);
			ret += "}\n";
			return ret;
		}		

		inline shared_ptr<XmlElement> XmlDocument::GetDocumentElement() { 
			if (!Children.size()) return NULL; 
			if (!Children[0]->IsElement()) return NULL;
			return dynamic_pointer_cast<XmlElement>(Children[0]);
		}

		inline shared_ptr<XmlNode> XmlDocument::DeepCopy() 
		{ 
			auto pRet = make_shared<XmlDocument>();
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				auto pCopy = Children[ii]->DeepCopy();
				pRet->Children.push_back(pCopy);
			}
			return pRet;
		}		

		/** XmlElement Implementation **/

		inline XmlElement::XmlElement(const string& LocalName /*= ""*/)
		{			
			this->LocalName = LocalName;
		}

		inline XmlElement::~XmlElement()
		{			
			Attributes.clear();
		}

		inline shared_ptr<XmlAttribute> XmlElement::FindAttribute(const string& AttrName) const
		{
			for (size_t ii=0; ii < Attributes.size(); ii++)
			{
				if (IsEqual(Attributes[ii]->Name, AttrName)) return Attributes[ii];
			}
			return NULL;
		}

		inline void XmlElement::AddStringAsAttr(const string& AttrName, const string& Value)
		{
			auto pNewAttr = make_shared<XmlAttribute>();
			pNewAttr->Name = AttrName;
			pNewAttr->Value = Value;
			Attributes.push_back(pNewAttr);
		}		

		inline void XmlElement::AddStringAsText(const string& Name, const string& Value)
		{
			auto pNewChild = make_shared<XmlElement>();
			pNewChild->LocalName = Name;
			auto pNewText = make_shared<XmlText>();
			pNewText->Text = Value;			
			pNewChild->Children.push_back(pNewText);
			Children.push_back(pNewChild);
		}

		inline void XmlElement::AddString(const string& Value)
		{
			auto pNewText = make_shared<XmlText>();
			pNewText->Text = Value;
			Children.push_back(pNewText);
		}		

		inline string XmlElement::ToString(XmlWriterOptions Options) 
		{
			string ret;
			Indent(Options, ret);
			ret += '<';
			ret += LocalName;
			if (!Options.IncludeContent) { ret += '>'; return ret; }
			
			for (size_t ii=0; ii < Attributes.size(); ii++)
			{
				auto pAttr = Attributes[ii];			
				ret += ' ';
				ret += pAttr->Name;
				ret += '='; ret += '\"';
				ret += Escape(Options, pAttr->Value);
				ret += '\"';
			}
						
			if (Children.size() == 0 && Options.AllowSingleTags) ret += '/';
			ret += '>';
			if (Children.size() == 0 && Options.AllowSingleTags) { ret += '\n'; return ret; }
			Options.Indentation ++;
			string ChildContent = XmlNode::ToString(Options);			
			Options.Indentation --;
			if (ChildContent.length() > 40 && ChildContent[0] != '\n') { 
				ret += '\n'; 
				if (ChildContent[0] != '\t') Indent(Options,ret); 
				ret += ChildContent;
				if (ChildContent[ChildContent.length() - 1] != '\n') ret += '\n'; 
				Indent(Options,ret);
			}
			else if (ChildContent[0] == '\t') {
				ret += '\n'; 
				ret += ChildContent;
				if (ChildContent[ChildContent.length() - 1] != '\n') ret += '\n'; 
				Indent(Options,ret);
			}
			else ret += ChildContent;
			ret += '<'; ret += '/';
			ret += LocalName;
			ret += '>';
			ret += '\n';
			return ret;
		}		

		inline string XmlElement::ToJsonValue(JsonWriterOptions Options)
		{
			// First, iterate through the children and collect the result so that we can plan what to do at the higher level.
			string childstr;
			string childtext;
			Options.Indentation++;			// Indent for creation of children representation
			bool NeedsComma = false;
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				auto pChild = Children[ii];

				// Special handling for element-wrapped XmlText nodes (additional special handling for this happens after this loop):
				if (dynamic_pointer_cast<XmlText>(pChild) != nullptr) { childtext += (dynamic_pointer_cast<XmlText>(pChild))->Text; continue; }

				// Special case handling for JSON arrays:
				if (pChild->IsElement())
				{
					// First, check if this is an array that was previously handled (first element has already been iterated over):
					bool IsArrayHandled = false;					
					for (size_t jj = 0; jj < ii; jj++)
					{
						auto pOtherChild = dynamic_pointer_cast<XmlElement>(Children[jj]);
						if (pOtherChild == nullptr) continue;
						if (IsEqual(dynamic_pointer_cast<XmlElement>(pChild)->LocalName, pOtherChild->LocalName)) { IsArrayHandled = true; break; }
					}
					if (IsArrayHandled) continue;

					// We don't know yet if we have an array, but if we do, pChild is the head of it.  Next, find the length (and implicitly,
					// whether it is an array).
					Options.Indentation++;					// Indent during array (from the '['), although we may not have an array and might discard.
					vector<string> ArrayContents;					
					for (size_t jj = ii + 1; jj < Children.size(); jj++)
					{
						auto pOtherChild = dynamic_pointer_cast<XmlElement>(Children[jj]);
						if (pOtherChild == nullptr) continue;
						if (IsEqual(dynamic_pointer_cast<XmlElement>(pChild)->LocalName, pOtherChild->LocalName))
						{
							if (ArrayContents.size() == 0) ArrayContents.push_back(pChild->ToJsonValue(Options));
							ArrayContents.push_back(pOtherChild->ToJsonValue(Options));
						}
						else if (!Options.MergeArrays) break;
					}

					if (ArrayContents.size())
					{						
						if (!Options.MergeArrays)
						{
							// We have an array and have already collected it.  But we need to verify that all entries in this array appear consecutively.  If they don't,
							// we have a problem - this is a case where XML does not translate readily into JSON.
							for (size_t jj = ii + ArrayContents.size(); jj < Children.size(); jj++)
							{
								auto pOtherChild = dynamic_pointer_cast<XmlElement>(Children[jj]);
								if (pOtherChild == nullptr) continue;
								if (IsEqual(dynamic_pointer_cast<XmlElement>(pChild)->LocalName, pOtherChild->LocalName))
									throw NotSupportedException("Cannot convert this XML heirarchy to JSON because a JSON array can only be formed from repeated and consecutive elements of the same name.  In this case, we have non-consecutive occurrances of element <" + pOtherChild->LocalName + ">.");							
							}
						}
					
						if (NeedsComma) { childstr += ",\n"; Indent(Options, childstr); NeedsComma = false; }
						childstr += '\"';
						childstr += Escape(Options, dynamic_pointer_cast<XmlElement>(pChild)->LocalName);
						childstr += "\": [\n";
						for (size_t jj=0; jj < ArrayContents.size(); jj++)
						{
							if (NeedsComma) { childstr += ",\n"; NeedsComma = false; }
							Indent(Options, childstr);
							childstr += ArrayContents[jj];
							NeedsComma = true;
						}
						if (NeedsComma) childstr += "\n";
						Options.Indentation--;				// Wrapping up the array's [], so unindent back to ] level.						
						Indent(Options, childstr);
						childstr += "]";
						NeedsComma = true;
						continue;
					}
					Options.Indentation--;			// Ended up not having an array after all, so undo the indention that wasn't used in this case.
				}

				if (NeedsComma) { childstr += ",\n"; Indent(Options, childstr); NeedsComma = false; }
				string substr = pChild->ToJson(Options);
				childstr += substr;
				NeedsComma = true;
			}
			Options.Indentation--;					// Remove the indention we had while writing out the children.
			
			string ret;

			// Additional special case handling for XmlText wrappers:
			if (childtext.length() > 0)
			{
				if (Attributes.size() > 0 || childstr.length() > 0) throw NotSupportedException("Cannot convert Xml that contains a mix of text and Xml elements.  For example, <MyXml>Where I have <b>bold</b> text.</MyXml> has no clear conversion to Json.");
				return "\"" + Escape(Options, childtext) + "\"";
			}

			// At this point, we've collected the text comprising the children, and we did this because we needed to know if we were a text-wrapper element up-front.  But we're not a text wrapper, and
			// the attributes should actually be listed ahead of the children, so now we have to actually form the object from the start and insert the child representation we're saving up.

			ret = "{\n";
			Options.Indentation++;					// Bring back the indention from the children so that we can also do the attributes.  (Also affects the first child if NeedsComma goes true).
			Indent(Options, ret);

			// Add the attributes:			
			NeedsComma = false;
			for (size_t ii=0; ii < Attributes.size(); ii++)
			{
				if (NeedsComma) { ret += ",\n"; Indent(Options, ret); NeedsComma = false; }
				auto pAttr = dynamic_pointer_cast<XmlAttribute>(Attributes[ii]);
				ret += '\"';
				ret += Escape(Options, pAttr->Name);
				ret += "\": \"";
				ret += Escape(Options, pAttr->Value);
				ret += "\"";
				NeedsComma = true;
			}			
			
			// Add the child representation and close it up:
			if (NeedsComma && childstr.length() > 0) { ret += ",\n"; Indent(Options, ret); NeedsComma = false; }
			ret += childstr;

			Options.Indentation--;					// Take away the attribute (and children) indentation again.

			if (childstr.length() > 0 || Attributes.size() > 0) { ret += "\n"; Indent(Options, ret); }			
			ret += "}";
			return ret;
		}

		inline string XmlElement::ToJson(JsonWriterOptions Options) 
		{
			string ret;
			Indent(Options, ret);
			ret += '\"';
			ret += Escape(Options, LocalName);
			ret += "\": ";
			ret += ToJsonValue(Options);
			return ret;
		}

		inline shared_ptr<XmlNode> XmlElement::DeepCopy() 
		{ 
			auto pRet = make_shared<XmlElement>();
			pRet->LocalName = LocalName;
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				auto pCopy = Children[ii]->DeepCopy();
				pRet->Children.push_back(pCopy);
			}
			for (size_t ii=0; ii < Attributes.size(); ii++)
			{
				auto pCopy = make_shared<XmlAttribute>(*Attributes[ii]);
				pRet->Attributes.push_back(pCopy);
			}
			return pRet;
		}

		/** Text retrieval conveniences **/

		inline string XmlElement::GetTextAsString() const { 
			string ret;
			for (size_t ii=0; ii < Children.size(); ii++)
			{
				if (Children[ii]->GetType() != XmlNode::Type::Text) throw FormatException("Node contained non-textual content.");
				ret += dynamic_pointer_cast<XmlText>(Children[ii])->Text;
			}
			return ret;
		}

		inline int XmlElement::GetTextAsInt32() const {
			Int32 Value;
			if (Int32_TryParse(GetTextAsString().c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline unsigned int XmlElement::GetTextAsUInt32() const {			
			UInt32 Value;
			if (UInt32_TryParse(GetTextAsString().c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline Int64 XmlElement::GetTextAsInt64() const {
			Int64 Value;
			if (Int64_TryParse(GetTextAsString().c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline UInt64 XmlElement::GetTextAsUInt64() const {
			UInt64 Value;
			if (UInt64_TryParse(GetTextAsString().c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline float XmlElement::GetTextAsFloat() const {
			float Value;
			if (Float_TryParse(GetTextAsString().c_str(), NumberStyles::Float, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline double XmlElement::GetTextAsDouble() const {
			double Value;
			if (Double_TryParse(GetTextAsString().c_str(), NumberStyles::Float, Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		inline bool XmlElement::GetTextAsBool() const {			
			bool Value;
			if (Bool_TryParse(GetTextAsString().c_str(), Value)) return Value;
			throw FormatException("Xml node " + LocalName + " found but has invalid format.");
		}

		/** Attribute retrieval conveniences **/

		inline string XmlElement::GetAttribute(const string& AttrName) const
		{
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return "";
			return pAttr->Value;
		}

		inline string XmlElement::GetAttrAsString(const string& AttrName, const string& DefaultValue /*= _T("")*/) const { 
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			return pAttr->Value;
		}

		inline int XmlElement::GetAttrAsInt32(const string& AttrName, int lDefaultValue /*= 0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return lDefaultValue;
			Int32 Value;
			if (Int32_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline unsigned int XmlElement::GetAttrAsUInt32(const string& AttrName, unsigned int lDefaultValue /*= 0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return lDefaultValue;
			UInt32 Value;
			if (UInt32_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline Int64 XmlElement::GetAttrAsInt64(const string& AttrName, Int64 DefaultValue /*= 0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			Int64 Value;
			if (Int64_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline UInt64 XmlElement::GetAttrAsUInt64(const string& AttrName, UInt64 DefaultValue /*= 0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			UInt64 Value;
			if (UInt64_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline float XmlElement::GetAttrAsFloat(const string& AttrName, float DefaultValue /*= 0.0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			float Value;
			if (Float_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline double XmlElement::GetAttrAsDouble(const string& AttrName, double DefaultValue /*= 0.0*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			double Value;
			if (Double_TryParse(pAttr->Value.c_str(), NumberStyles::Integer, Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline bool XmlElement::GetAttrAsBool(const string& AttrName, bool DefaultValue /*= false*/) const {
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) return DefaultValue;
			bool Value;
			if (Bool_TryParse(pAttr->Value.c_str(), Value)) return Value;
			throw FormatException("Xml attribute " + AttrName + " found but has invalid format.");
		}

		inline void XmlElement::AddInt32AsAttr(const string& AttrName, int Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddUInt32AsAttr(const string& AttrName, unsigned int Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddInt64AsAttr(const string& AttrName, Int64 Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddUInt64AsAttr(const string& AttrName, UInt64 Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddFloatAsAttr(const string& AttrName, float Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddDoubleAsAttr(const string& AttrName, double Value) { AddStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::AddBoolAsAttr(const string& AttrName, bool Value) { AddStringAsAttr(AttrName, Convert::ToString(Value)); }

		inline bool XmlElement::IsAttrPresent(const string& AttrName) const { return FindAttribute(AttrName) != nullptr; }

		inline void	XmlElement::SetStringAsAttr(const string& AttrName, const string& sValue)
		{	
			auto pAttr = FindAttribute(AttrName);
			if (!pAttr) { AddStringAsAttr(AttrName, sValue); return; }
			pAttr->Value = sValue;
		}

		inline void XmlElement::SetInt32AsAttr(const string& AttrName, int Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }		
		inline void	XmlElement::SetUInt32AsAttr(const string& AttrName, unsigned int Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void	XmlElement::SetInt64AsAttr(const string& AttrName, Int64 Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void	XmlElement::SetUInt64AsAttr(const string& AttrName, UInt64 Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void	XmlElement::SetFloatAsAttr(const string& AttrName, float Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void	XmlElement::SetDoubleAsAttr(const string& AttrName, double Value) { SetStringAsAttr(AttrName, Convert::ToString(Value).c_str()); }
		inline void XmlElement::SetBoolAsAttr(const string& AttrName, bool Value) { SetStringAsAttr(AttrName, Convert::ToString(Value)); }

		inline void XmlElement::AddInt32AsText(const string& Name, int Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddUInt32AsText(const string& Name, unsigned int Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddInt64AsText(const string& Name, Int64 Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddUInt64AsText(const string& Name, UInt64 Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddFloatAsText(const string& Name, float Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddDoubleAsText(const string& Name, double Value) { AddStringAsText(Name, Convert::ToString(Value)); }
		inline void XmlElement::AddBoolAsText(const string& Name, bool Value) { AddStringAsText(Name, Convert::ToString(Value)); }

		inline void XmlElement::AddInt32(int Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddUInt32(unsigned int Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddInt64(Int64 Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddUInt64(UInt64 Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddFloat(float Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddDouble(double Value) { AddString(Convert::ToString(Value)); }
		inline void XmlElement::AddBool(bool Value) { AddString(Convert::ToString(Value)); }

		/** XmlText Implementation **/

		inline void String_ReplaceAll(string& input, char chFind, const char* pszReplace)
		{			
			size_t index = input.find(chFind);
			size_t len = strlen(pszReplace);
			while (index != string::npos) {
				//   0 1 2 3 4 5 6 7
				//   x & z
				//	 x & a m p ; z
				input = input.substr(0, index) + pszReplace + input.substr(index + 1);
				index = input.find(chFind, index + len);
			}
		}
		
		inline void String_ReplaceAll(string& input, const string& strFind, const string& strReplace)
		{
			size_t index = input.find(strFind);
			while (index != string::npos) {
				//   0 1 2 3 4 5 6 7
				//   x & a m p ; z
				//	 x & z
				input = input.substr(0, index) + strReplace + input.substr(index + strFind.length());
				index = input.find(strFind, index + strReplace.length());
			}
		}

		inline /*static*/ string XmlText::Escape(string RegularText)
		{
			string ret = RegularText;
			String_ReplaceAll(ret, '&', "&amp;");		// Note: This substitution must happen first.
			String_ReplaceAll(ret, '\"', "&quot;");
			String_ReplaceAll(ret, '\'', "&apos;");
			String_ReplaceAll(ret, '<', "&lt;");
			String_ReplaceAll(ret, '>', "&gt;");
			return ret;
		}

		inline /*static*/ string XmlText::Unescape(string EscapedText)
		{
			string ret = EscapedText;			
			String_ReplaceAll(ret, "&quot;", "\"");
			String_ReplaceAll(ret, "&apos;", "\'");
			String_ReplaceAll(ret, "&lt;", "<");
			String_ReplaceAll(ret, "&gt;", ">");
			String_ReplaceAll(ret, "&amp;", "&");		// Note: This substitution must happen last.
			return ret;
		}
		
		inline string XmlText::ToString(XmlWriterOptions Options) 
		{
			if (!Options.IncludeContent) return "XmlText";
			return Escape(Text);
		}

		inline string XmlText::ToJsonValue(JsonWriterOptions Options) 
		{
			throw NotSupportedException("XmlText must be intercepted by the XmlElement container before the ToJsonValue() call.  XmlText cannot be converted to Json directly, but must be converted as part of the larger Xml structure.");
		}

		inline string XmlText::ToJson(JsonWriterOptions Options) 
		{
			throw NotSupportedException("XmlText must be intercepted by the XmlElement container before the ToJson() call.  XmlText cannot be converted to Json directly, but must be converted as part of the larger Xml structure.");
		}

		inline shared_ptr<XmlNode> XmlText::DeepCopy() 
		{ 
			auto pRet = make_shared<XmlText>();
			assert (Children.size() == 0);			// XmlText nodes should not have children.			
			pRet->Text = Text;
			return pRet;
		}
	}
}

#endif	// __WBXmlImpl_v5_h__

//	End of XmlImpl.h

