
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
		return XmlParser::Parse(xml, snippet_id);
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
		pDoc = Parser.Parse(snippet, snippet_id);
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
	ASSERT_TRUE(IsEqual(Trim(((XmlText*)pRoot->Elements()[0]->Children[0])->Text), "And some text.")) << xml_fail_msg(*pDoc, id);
	ASSERT_TRUE(IsEqual(Trim(((XmlText*)pRoot->Elements()[2]->Children[0])->Text), "<text>")) << xml_fail_msg(*pDoc, id);

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
