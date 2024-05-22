/***
*	Tests the following major subsystems/classes:
* 
*		File I/O
*		Zip Archive Reading (with no compression)
*		XML Parsing
*		JSON Parsing
*		YAML Parsing
**/

#include "wbCore.h"
#include "gtest/gtest.h"
#include <set>

using namespace std;
using namespace wb;
using namespace wb::xml;
using namespace wb::memory;
using namespace wb::io::compression;

wb::io::Path unit_testing_data_folder = "..\\test_data";

// This option instructs the YAML tests to pull their tests from the .zip file instead of from a directory tree.
// This is helpful when coming out of the git repo where LFCR can be reformatted by ensuring we have the original
// and unaltered files.
#define FromArchive

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

#include "Parsing/Yaml/YamlEventParser.h"
class YamlEventLogParser : public wb::yaml::YamlEventParser
{
	string log;

protected:

	void OnOpenEvent(Event Event, string Anchor, string Tag, bool FlowStyleOrExplicit = false) override
	{
		string entry = "+";		
		switch (Event)
		{
		case Event::Stream: entry += "STR"; break;
		case Event::Document: 
			entry += "DOC"; 			
			if (!Anchor.empty()) entry += " &" + Anchor;
			if (!Tag.empty()) entry += " " + Tag;
			if (FlowStyleOrExplicit) entry += " ---";
			break;
		case Event::Map: 
			entry += "MAP"; 			
			if (FlowStyleOrExplicit) entry += " {}";
			if (!Anchor.empty()) entry += " &" + Anchor;
			if (!Tag.empty()) entry += " " + Tag;			
			break;
		case Event::Sequence:
			entry += "SEQ";			
			if (FlowStyleOrExplicit) entry += " []";
			if (!Anchor.empty()) entry += " &" + Anchor;
			if (!Tag.empty()) entry += " " + Tag;			
			break;
		default:
			throw NotImplementedException("Unrecognized Event type.");
		}
		log += entry + "\n";
	}

	void OnCloseEvent(Event Event, bool Explicit) override
	{
		switch (Event)
		{
		case Event::Stream: log += "-STR"; break;
		case Event::Document: 
			if (Explicit)
				log += "-DOC ...";
			else
				log += "-DOC"; 
			break;
		case Event::Map: log += "-MAP"; break;
		case Event::Sequence: log += "-SEQ"; break;
		default:
			throw NotImplementedException("Unrecognized Event type.");
		}
		log += "\n";
	}

	void OnValueEvent(string Value, string Anchor, string Tag, Flags flags = NoFlags)
	{
		wb::yaml::string_ReplaceAll(Value, "\\", "\\\\");
		wb::yaml::string_ReplaceAll(Value, "\n", "\\n");
		wb::yaml::string_ReplaceAll(Value, "\t", "\\t");

		log += "=VAL";		
		if (Anchor.length()) log += " &" + Anchor;
		if (Tag.length()) log += " " + Tag;
		
		if (flags == NoFlags)
			log += " :" + Value;
		else if (flags == TextWasSingleQuoted)
			log += " \'" + Value;
		else if (flags == TextWasDoubleQuoted)
			log += " \"" + Value;
		else if (flags == TextWasLiteralBlock)
			log += " |" + Value;
		else if (flags == TextWasFoldedBlock)
			log += " >" + Value;
		else if (flags & NullValueFlag)
			log += " :";
		else throw NotSupportedException("Unrecognized flags at " + GetSource() + ".");

		log += "\n";
	}

	void OnAliasEvent(string AnchorName)
	{
		log += "=ALI *" + AnchorName + "\n";
	}

public:
	
	static string Parse(wb::io::Stream& stream, const string& sSourceFilename = "")
	{
		YamlEventLogParser parser;		
		parser.StartStream(stream, sSourceFilename);
		try
		{
			parser.ParseTopLevel();
			parser.FinishStream();
			return parser.log;
		}
		catch (...)
		{
			parser.FinishStream();
			throw;
		}
	}
};

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

string IsEqualWithNotes(string A, string B, string ALabel, string BLabel)
{
	int line_no = 1, col_no = 1;
	for (int ii = 0; ii < A.length(); ii++)
	{		
		if (ii >= B.length())
		{
			return ALabel + " is longer than " + BLabel;
		}
		if (A[ii] == '\n' && B[ii] != '\n')
		{
			return ALabel + ":" + to_string(line_no) + " is longer than " + BLabel + ":" + to_string(line_no);
		}
		if (A[ii] == '\n') {
			line_no++; col_no = 1;
		}
		if (A[ii] != B[ii])
		{
			return ALabel + ":" + to_string(line_no) + ": mismatch at column " + to_string(col_no) + ".";
		}
		col_no++;
	}
	if (B.length() > A.length())
	{
		return ALabel + " is shorter than " + BLabel;
	}
	return string();
}

#ifdef FromArchive
inline r_ptr<io::Stream> OpenTestStream(ZipArchiveEntry& fi)
{
	return r_ptr<io::Stream>::absolved(fi.Open());
}
#else
inline r_ptr<io::Stream> OpenTestStream(FileInfo fi)
{
	return r_ptr<io::Stream>::responsible(new io::FileStream(fi.GetFullName(), io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read));
}
#endif

TEST(Library, YAML)
{
	using namespace wb::io;

	// Test suite data taken from here: https://github.com/yaml/yaml-test-suite/releases/tag/data-2022-01-17

	std::cout << "UnitTesting: Library.YAML tests starting..." << "\n";

	// Cases that this YamlEventParser implementation is known not to support:
	set<string>	KnownFails;
	KnownFails.insert("4WA9");			// Explicit indentation indicators not supported.
	KnownFails.insert("4QFQ");			// Explicit indentation indicators not supported.	
	KnownFails.insert("BEC7");			// Spec calls for a warning and attempt to parse YAML 1.3, but my implementation throws an error instead.
	KnownFails.insert("D83L");			// Explicit indentation indicators not supported.	
	KnownFails.insert("F6MC");			// Explicit indentation indicators not supported.	
	KnownFails.insert("G4RS");			// Escaped unicode characters are not supported.
	KnownFails.insert("M5C3");			// Explicit indentation indicators not supported.	
	KnownFails.insert("P2AD");			// Explicit indentation indicators not supported.	
	KnownFails.insert("R4YG");			// Explicit indentation indicators not supported.	
	KnownFails.insert("Z67P");			// Explicit indentation indicators not supported.
	KnownFails.insert("MUS6\\02");		// No support for YAML 1.1 in this implementation.
	KnownFails.insert("MUS6\\03");		// No support for YAML 1.1 in this implementation.
	KnownFails.insert("MUS6\\04");		// No support for YAML 1.1 in this implementation.
	KnownFails.insert("MUS6\\06");		// No support for YAML 1.1 in this implementation.

	// Cases that should pass YAML parsing and event generation, but that the JSON parser won't support:
	set<string>	NoJSON;
	NoJSON.insert("35KP");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("5TYM");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("6XDY");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("6WLZ");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("6ZKB");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("7Z25");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("8G76");				// JSON parser requires document, empty JSON document not supported.
	NoJSON.insert("9DXL");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("98YD");				// JSON parser requires document, empty JSON document not supported.
	NoJSON.insert("9KAX");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("9WXW");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("AVM7");				// JSON parser requires document, empty JSON document not supported.
	NoJSON.insert("HWV9");				// JSON parser requires document, empty JSON document not supported.
	NoJSON.insert("JHB9");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("KSS4");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("L383");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("M7A3");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("PUW8");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("QT73");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("RZT7");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("U9NS");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("UT92");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("W4TN");				// JSON parser only reads a single document.  This isn't really valid JSON.
	NoJSON.insert("WZ62");				// JSON defines 'null', but this test case is using "".  However, null illegal on the key I believe.

	// Scan for all subdirectories of the test-suite folder
	string base_path;
#ifdef FromArchive
	auto archive_path = unit_testing_data_folder / "yaml" / "yaml-test-suite-data-2022-01-17.zip";
	auto archive = ZipArchive(r_ptr<Stream>::responsible(new io::FileStream(archive_path, io::FileMode::Open, io::FileAccess::Read, io::FileShare::Read)), ZipArchiveMode::Read);
	auto TopLevel = archive.GetRoot().EnumerateDirectories();
	if (TopLevel.size() != 1)
		throw FileNotFoundException("Expected archive (" + archive_path.to_string() + ") to contain exactly one top-level directory but found " + wb::to_string(TopLevel.size()));
	auto Subdirectories = TopLevel[0].EnumerateDirectories();
	base_path = TopLevel[0].GetFullName().to_string();
	typedef ZipArchiveEntry FileInfo_t;
#else
	io::DirectoryInfo diBase(unit_testing_data_folder / "yaml" / "test-suite");
	if (!diBase.Exists())
	{
		FAIL() << "\nYAML test data directory not found: " << diBase.GetFullName();
	}
	base_path = diBase.GetFullName();
	vector<io::DirectoryInfo> Subdirectories = diBase.EnumerateDirectories();
	for (int ii = 0; ii < Subdirectories.size(); ii++)
	{
		auto& sub = Subdirectories[ii];
		vector<io::DirectoryInfo> subsub = sub.EnumerateDirectories();
		if (subsub.size() > 0) Subdirectories.insert(Subdirectories.end(), subsub.begin(), subsub.end());
		// Since we've appended more subdirectories to the end of the vector we're enumerating over, they
		// too will get descended/walked down.
	}	
	typedef FileInfo FileInfo_t;
#endif

	/** Scan directories for test cases that we can use **/
	int success_count = 0, known_fails = 0, untested_cases = 0;
	int failure_count = 0;				// Don't actually signal failure until all test cases are handled so we can see a complete picture.	

	for (auto& diCase : Subdirectories)
	{
		// Locate .yaml and .json files within subfolder...
		bool IsErrorCase = false;
		FileInfo_t YamlFile, JsonFile, YamlEventLogFile;
		for (auto& fi : diCase.EnumerateFiles())
		{
			if (IsEqualNoCase(to_string(fi.GetName()), "in.yaml")) YamlFile = fi;
			if (IsEqualNoCase(to_string(fi.GetName()), "test.event")) YamlEventLogFile = fi;
			if (IsEqualNoCase(to_string(fi.GetName()), "in.json")) JsonFile = fi;
			if (IsEqualNoCase(to_string(fi.GetName()), "error")) IsErrorCase = true;
		}

		// If both a .yaml and .json file were found, then we can compare.
		if (!YamlFile.IsEmpty() && !JsonFile.IsEmpty() && !YamlEventLogFile.IsEmpty())
		{
			string snippet_id = to_string(Path::ToRelativePath(base_path, diCase.GetFullName()));
			if (KnownFails.count(snippet_id))
			{
				std::cout << "SKIP: " << snippet_id << " [known to not be supported]\n";
				known_fails++;
				continue;
			}
			if (IsErrorCase)
			{
				std::cout << "SKIP: " << snippet_id << " [error expected from this case]\n";
				untested_cases++;
				continue;
			}
			std::cout << "Starting snippet_id " << snippet_id << "\n";

			string source_name = wb::to_string(diCase.GetFullName().GetFileName() + L"/" + YamlFile.GetName());

			/** If debugging and you want to step through only a specific test case, uncomment the following and enter the snippet id.
			*/			
			//if (!IsEqual(snippet_id, "L24T\\01")) continue;

			string YamlParsedEventLog;
			try
			{
				auto fsYaml = OpenTestStream(YamlFile);
				YamlParsedEventLog = YamlEventLogParser::Parse(*fsYaml, source_name);
			}
			catch (std::exception& ex)
			{
				failure_count++;
				string original = io::StreamToString(*OpenTestStream(YamlFile));
				std::cout << "\nFailure during YAML Parsing of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n" 
					<< "\nOriginal YAML (" << original.length() << " characters) :\n" << original
					<< "\n"
					;
				continue;
			}

			string YamlEventLogFromFile;
			try
			{				
				auto fsLog = OpenTestStream(YamlEventLogFile);				
				YamlEventLogFromFile = StreamToString(*fsLog);
			}
			catch (std::exception& ex)
			{
				failure_count++;
				std::cout << "\nFailure while reading YAML event log of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n";
				continue;
			}			

			YamlParsedEventLog = Trim(YamlParsedEventLog);
			YamlEventLogFromFile = Trim(YamlEventLogFromFile);

			string diff = IsEqualWithNotes(YamlParsedEventLog, YamlEventLogFromFile, "Results", "expected");
			if (diff.length())
			{
				failure_count++;
				string original = io::StreamToString(*OpenTestStream(YamlFile));				
				std::cout << "\nMismatch of YAML to target YAML event log in test case '" << snippet_id << "':"
					<< "\nOriginal YAML (" << original.length() << " characters) :\n" << original
					<< "\nYAML events:\n" << YamlParsedEventLog << "\n"
					<< "\nYAML expected events:\n" << YamlEventLogFromFile << "\n"
					<< "\n" << diff << "\n"					
					;
				continue;
			}			

			if (NoJSON.count(snippet_id))
			{
				std::cout << "SKIP JSON: " << snippet_id << " [known to not be supported by JSON parser]\n";
				continue;
			}

			string YamlParsedToJson;			
			try
			{				
				auto fsYaml = OpenTestStream(YamlFile);
				auto Documents = wb::yaml::YamlParser::Parse(*fsYaml, source_name);
				wb::yaml::JsonWriterOptions Options;
				Options.UnquoteNumbers = true;
				//if (pNode == nullptr) pNode = unique_ptr<wb::yaml::YamlNode>(new wb::yaml::YamlScalar("empty document"));
				if (Documents.size() == 0) throw NotSupportedException("No document received from YamlParser.");
				if (Documents.size() > 1) {
					string original = io::StreamToString(*OpenTestStream(YamlFile));					
					std::cout << "\nMultiple documents received from YamlParser in test case '" << snippet_id << "':"
						<< "\nOriginal YAML (" << original.length() << " characters) :\n" << original
						<< "\nYAML Event Log:\n" << YamlEventLogFromFile << "\n"
						;
					throw NotSupportedException("Multiple document received from YamlParser- can't compare to JSON b/c JSON parser only supports single document.");
				}
				YamlParsedToJson = Documents[0]->ToJson(Options);
			}
			catch (std::exception& ex)
			{
				failure_count++;
				std::cout << "\nFailure during YAML Parsing of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n";
				continue;
			}

			unique_ptr<wb::json::JsonValue> pYamlParsedToJsonParsed;
			try
			{
				// To get consistent results from the JSON parser vs YAML parser, run YAML->JSON->JSON.  That is, run the output of the
				// Yaml "ToJson()" through the Json parser to get consistent Json format.				
				pYamlParsedToJsonParsed = wb::json::JsonParser::ParseString(YamlParsedToJson.c_str(), "YAML->JSON of " + source_name);
			}
			catch (std::exception& ex)
			{
				failure_count++;
				std::cout << "\nFailure during JSON Re-Parsing (from YAML->JSON) of test case '" << snippet_id << "':\n" << string(ex.what()) << "\n" << "\nYAML -> JSON:\n" << YamlParsedToJson << "\n";
				continue;
			}
			
			unique_ptr<wb::json::JsonValue> pJsonParsed;
			try
			{								
				auto fsJson = OpenTestStream(JsonFile);
				pJsonParsed = wb::json::JsonParser::Parse(*fsJson, source_name);
			}
			catch (std::exception& ex)
			{
				FAIL() << "\nFailure reading JSON test case target from YAML snippet '" << snippet_id << "':\n" << string(ex.what()) << "\n";
			}
			
			if (!json::IsEqual(pYamlParsedToJsonParsed, pJsonParsed, /*Strict=*/ false))
			{
				failure_count++;
				string original = io::StreamToString(*OpenTestStream(YamlFile));				
				std::cout << "\nMismatch of YAML to target JSON in test case '" << snippet_id << "':"
					<< "\nOriginal YAML (" << original.length() << " characters) :\n" << original
					<< "\nYAML -> JSON:\n" << pYamlParsedToJsonParsed->ToString() << "\n"					
					<< "\nJSON:\n" << pJsonParsed->ToString() << "\n"					
					<< "\nYAML Event Log:\n" << YamlEventLogFromFile << "\n"
					;				
				continue;
			}

			success_count++;
		}
	}

	if (failure_count > 0)
	{
		FAIL() << "\n" << to_string(failure_count) << " YAML test cases failed.\n";
	}
	else
	{
		std::cout << success_count << " YAML test cases succeeded.  " << known_fails << " additional are known to fail (unsupported) and " << untested_cases << " cases were not tested." << std::endl;
	}
}

#pragma endregion
