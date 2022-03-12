
#include "wbCore.h"
#include "gtest/gtest.h"

using namespace std;
using namespace wb;
using namespace wb::xml;

wb::io::Path unit_testing_data_folder = "data";

#pragma region "Xml Parser Testing"

unique_ptr<XmlDocument> TestParseXml(const string& xml, const string& snippet_id)
{
	try
	{
		return XmlParser::ParseString(xml, snippet_id);
	}
	catch (...)
	{
		std::cout << "\nFailure during XML Parsing of snippet '" << snippet_id << "':\n" << xml << "\n";
		throw;
	}
}

string xml_fail_msg(wb::xml::XmlDocument& doc, string id)
{
	using namespace wb::xml;
	return "Xml parsing did not match expectation for snippet '" + id + "'.  Snippet parsed as:\n" + doc.ToString();
}

void ExpectThatXmlFailed(const string& snippet, const string& snippet_id)
{
	std::unique_ptr<XmlDocument> pDoc;
	try
	{
		wb::xml::XmlParser Parser;
		pDoc = Parser.ParseString(snippet, snippet_id);
	}
	catch (std::exception&)
	{
		// This is the normal (test passed) flow of execution.
		return;
	}
	std::cout << "Xml parsing of not-well-formed snipped '" + snippet_id + "' did not catch the error.  Snippet original:\n" + snippet + "\n\nSnippet parsed as:\n" + pDoc->ToString() + "\n\n";
	EXPECT_TRUE(false) << "Xml parsing failed to detect badly formed XML snippet " << snippet_id;
}

TEST(Library, XML)
{
	std::cout << "UnitTesting: Library.XML tests starting..." << "\n";

	unique_ptr<XmlDocument> pDoc;
	string id;

	id = "top-level-sample1";
	pDoc = TestParseXml("<JustTopLevel/>", id);
	ASSERT_TRUE(IsEqual(pDoc->GetDocumentElement()->LocalName, "JustTopLevel")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pDoc->GetDocumentElement()->Children.size(), 0) << xml_fail_msg(*pDoc, id);

	id = "top-level-sample2";
	pDoc = TestParseXml("<Just-Top-Level/>", id);
	ASSERT_TRUE(IsEqual(pDoc->GetDocumentElement()->LocalName, "Just-Top-Level")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pDoc->GetDocumentElement()->Children.size(), 0) << xml_fail_msg(*pDoc, id);

	ExpectThatXmlFailed("</Bad-Top-Level>", "top-level-sample3");

	id = "top-level-sample4";
	pDoc = TestParseXml("<Attr-Test esc=\"'please'\"/>", id);
	ASSERT_TRUE(IsEqual(pDoc->GetDocumentElement()->LocalName, "Attr-Test")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pDoc->GetDocumentElement()->Children.size(), 0) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pDoc->GetDocumentElement()->GetAttrAsString("esc"), "'please'")) << xml_fail_msg(*pDoc, id);

	id = "top-level-with-xmldecl";
	pDoc = TestParseXml("<?xml version =\"1.0\" encoding =\"UTF-8\" ?><Just-Top-Level/>", id);
	ASSERT_TRUE(IsEqual(pDoc->GetDocumentElement()->LocalName, "Just-Top-Level")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pDoc->GetDocumentElement()->Children.size(), 0) << xml_fail_msg(*pDoc, id);

	ExpectThatXmlFailed("<Top-Level></Bad-Closing></Top-Level>", "top-level-sample4");

	id = "with-openandclose";
	pDoc = TestParseXml(
		"<?xml version =\"1.0\" encoding =\"UTF-8\" ?>\n"
		"<Top-Level>\n"
		"    <Open-And-Close />\n"
		"\t  <Open-And-Close / >\n"
		"\t  <Open-And-Close/>\n"
		"</Top-Level>\n", id);
	auto pRoot = pDoc->GetDocumentElement();
	ASSERT_TRUE(IsEqual(pRoot->LocalName, "Top-Level")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pRoot->Children.size(), 3) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[0]->LocalName, "Open-And-Close")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[1]->LocalName, "Open-And-Close")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[2]->LocalName, "Open-And-Close")) << xml_fail_msg(*pDoc, id);

	id = "with-text";
	pDoc = TestParseXml(
		"<Top-Level>\n"
		"    <ContainerA>\n"
		"		And some text.\n"
		"    </ContainerA>\n"
		"\t  < ContainerB / >\n"
		"\t  < ContainerC >&lt;text&gt;</ContainerC>\n"
		"</Top-Level>\n", id);
	pRoot = pDoc->GetDocumentElement();
	ASSERT_TRUE(IsEqual(pRoot->LocalName, "Top-Level")) << xml_fail_msg(*pDoc, id);
	ASSERT_EQ(pRoot->Children.size(), 3) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[0]->LocalName, "ContainerA")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[1]->LocalName, "ContainerB")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(pRoot->Elements()[2]->LocalName, "ContainerC")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(pRoot->Elements()[0]->Children.size() == 1) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(pRoot->Elements()[1]->Children.size() == 0) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(pRoot->Elements()[2]->Children.size() == 1) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(pRoot->Elements()[0]->Children[0]->GetType() == XmlNode::Type::Text) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(pRoot->Elements()[2]->Children[0]->GetType() == XmlNode::Type::Text) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(Trim(dynamic_pointer_cast<XmlText>(pRoot->Elements()[0]->Children[0])->Text), "And some text.")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(Trim(dynamic_pointer_cast<XmlText>(pRoot->Elements()[2]->Children[0])->Text), "<text>")) << xml_fail_msg(*pDoc, id);

	std::cout << "UnitTesting: Library.XML valid and not-well-formed file tests starting..." << "\n";

	/** Run all valid test cases **/
	io::DirectoryInfo diValid(unit_testing_data_folder / "xml" / "parsing" / "valid");
	for (auto& fi : diValid.EnumerateFiles())
	{
		id = to_string(fi.GetFullName());
		io::FileStream fs(fi.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read);
		string text = io::StreamToString(fs);
		pDoc = TestParseXml(text, id);
	}

	/** Run all not-well-formed test cases **/
	io::DirectoryInfo diNWF(unit_testing_data_folder / "xml" / "parsing" / "not-well-formed");
	for (auto& fi : diNWF.EnumerateFiles())
	{
		id = to_string(fi.GetFullName());
		io::FileStream fs(fi.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read);
		string text = io::StreamToString(fs);
		ExpectThatXmlFailed(text, id);
	}
}

#pragma endregion

#pragma region "Yaml (and Json) Parser Testing"

bool CompareAttributes(shared_ptr<XmlAttribute>& a, shared_ptr<XmlAttribute>& b) {
	return a->Name < b->Name; 
}

/// <summary>
/// In order to be able to compare yaml and json test cases, we use the json->xml parsing representation and then xml->json string
/// output.  However, we need sorted attributes to generate the same sequence.
/// </summary>
/// <param name="node">Xml tree to recursively sort all attributes.</param>
void SortAttributes(XmlElement& node)
{
	for (auto& pChild : node.Elements()) SortAttributes(*pChild);

	sort(node.Attributes.begin(), node.Attributes.end(),
		CompareAttributes);
		//[](shared_ptr<XmlAttribute>& a, shared_ptr<XmlAttribute>& b) { return a->Name < b->Name; });
}

/// <summary>
/// Trim all whitespace that is outside of '...' or "..." text.  Within quotes ("), also watch for backslashes that act as escape
/// characters so as to decide properly when the quoted text ends.
/// </summary>
string RemoveUnquotedJsonWhitespace(string from)
{
	string ret;
	char quote_char = 0;
	for (auto ii = 0; ii < from.length(); )
	{
		if (from[ii] == '\'')
		{
			ret += from[ii];
			if (quote_char == '\'') quote_char = 0;
			else quote_char = '\'';
			ii++;
		}
		else if (from[ii] == '\"')
		{
			ret += from[ii];
			if (quote_char == '\"') quote_char = 0;
			else quote_char = '\"';
			ii++;
		}
		else if (from[ii] == '\\' && quote_char == '\"')
		{
			ret += from[ii];
			ii++;
			if (from[ii] == '\\' || from[ii] == '\"')
			{
				ret += from[ii];
				ii++;
				continue;
			}
			while (ii < from.length())
			{
				if ((from[ii] >= 'a' && from[ii] <= 'z') || (from[ii] >= 'A' && from[ii] <= 'Z') || (from[ii] >= '0' && from[ii] <= '9'))
				{
					ret += from[ii];
					ii++;
				}
			}
		}
		else if (quote_char == 0 && (from[ii] == ' ' || from[ii] == '\t' || from[ii] == '\n' || from[ii] == '\r')) ii++;
		else ret += from[ii++];
	}
	return ret;
}

#include <set>

TEST(Library, YAML)
{
	using namespace wb::io;

	std::cout << "UnitTesting: Library.YAML tests starting..." << "\n";

	set<string>	KnownFails;
	KnownFails.insert("4WA9");				// Explicit indentation indicators not supported.
	KnownFails.insert("4QFQ");				// Explicit indentation indicators not supported.

	/** Scan directory for test cases that we can use **/
	io::DirectoryInfo diBase(unit_testing_data_folder / "yaml" / "test-suite");
	for (auto& diCase : diBase.EnumerateDirectories())
	{
		// Locate .yaml and .json files within subfolder...
		FileInfo YamlFile, JsonFile;
		for (auto& fi : diCase.EnumerateFiles())
		{
			if (IsEqualNoCase(to_string(fi.GetName()), "in.yaml")) YamlFile = fi;
			if (IsEqualNoCase(to_string(fi.GetName()), "in.json")) JsonFile = fi;
		}

		// If both a .yaml and .json file were found, then we can compare.
		if (!YamlFile.IsEmpty() && !JsonFile.IsEmpty())
		{
			string snippet_id = to_string(diCase.GetName());
			if (KnownFails.count(snippet_id))
			{
				std::cout << "SKIP: " << snippet_id << " [known to not be supported]\n";
				continue;
			}
			std::cout << "Starting snippet_id " << snippet_id << "\n";

bool hit = false;
//if (!IsEqual(snippet_id, "4Q9F")) continue;
hit = true;
			
			string YamlParsedToJson;			
			try
			{
				io::FileStream fsYaml(YamlFile.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read);
				auto pNode = wb::yaml::YamlParser::Parse(fsYaml, to_string(diCase.GetName()) + "/" + to_string(YamlFile.GetName()));
				wb::yaml::JsonWriterOptions Options;
				Options.UnquoteNumbers = true;
				if (pNode == nullptr) pNode = unique_ptr<wb::yaml::YamlNode>(new wb::yaml::YamlScalar("empty document"));
				YamlParsedToJson = pNode->ToJson(Options);
			}
			catch (std::exception& ex)
			{
				std::cout << "\nFailure during YAML Parsing of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n";
				continue;
			}

			unique_ptr<wb::json::JsonValue> pYamlParsedToJsonParsed;
			try
			{
				// To get consistent results from the JSON parser vs YAML parser, run YAML->JSON->JSON.  That is, run the output of the
				// Yaml "ToJson()" through the Json parser to get consistent Json format.				
				pYamlParsedToJsonParsed = wb::json::JsonParser::ParseString(YamlParsedToJson.c_str(), "YAML->JSON of " + to_string(diCase.GetName()) + "/" + to_string(YamlFile.GetName()));
				//SortAttributes(*(pJsonNode->GetDocumentElement()));
				//YamlParsedToJsonToJson = pJsonNode->ToString();
			}
			catch (std::exception& ex)
			{
				std::cout << "\nFailure during JSON Re-Parsing (from YAML->JSON) of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n" << "\nYAML -> JSON:\n" << YamlParsedToJson << "\n";
				continue;
			}

			//string JsonFromFile;
			//string JsonParsed;
			unique_ptr<wb::json::JsonValue> pJsonParsed;
			try
			{
				// Parse as string...
				//io::FileStream fsJson(JsonFile.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read);
				//JsonFromFile = StreamToString(fsJson);
				
				io::FileStream fsJson(JsonFile.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read);				
				pJsonParsed = wb::json::JsonParser::Parse(fsJson, to_string(diCase.GetName()) + "/" + to_string(JsonFile.GetName()));				
				//SortAttributes(*(pNode->GetDocumentElement()));
				//JsonParsed = pNode->ToString();				
			}
			catch (std::exception& ex)
			{
				FAIL() << "\nFailure reading JSON test case target from YAML snippet '" << snippet_id << "':\n" << string(ex.what()) << "\n";
			}

			//if (!IsEqual(RemoveUnquotedJsonWhitespace(YamlParsedToJson), RemoveUnquotedJsonWhitespace(JsonFromFile))) {
			//	std::cout << "\nMismatch of YAML to target JSON in test case '" << snippet_id << "':\nYAML:\n" << YamlParsedToJson << "\nJSON:\n" << JsonFromFile << "\n";
			//}
			//if (!IsEqual(RemoveUnquotedJsonWhitespace(YamlParsedToJsonToXmlToJson), RemoveUnquotedJsonWhitespace(JsonParsedToXmlToJson)))
			//if (!IsEqual(RemoveUnquotedJsonWhitespace(YamlParsedToJsonToJson), RemoveUnquotedJsonWhitespace(JsonParsed)))
			if (!json::IsEqual(pYamlParsedToJsonParsed, pJsonParsed))
			{
				std::cout << "\nMismatch of YAML to target JSON in test case '" << snippet_id << "':"
					<< "\nYAML -> JSON:\n" << pYamlParsedToJsonParsed->ToString() << "\n"
					//<< "\nYAML -> JSON -> XML -> JSON:\n" << YamlParsedToJsonToXmlToJson << "\n" 
					<< "\nJSON:\n" << pJsonParsed->ToString() << "\n"
					//<< "\nJSON -> XML -> JSON:\n" << JsonParsedToXmlToJson << "\n"
					;
				continue;
			}
		}
	}	
}

#pragma endregion
