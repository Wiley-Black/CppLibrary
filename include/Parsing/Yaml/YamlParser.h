/////////
//	YamlParser.h
////
//	To clean this up and simplify, I would want to make the concept of a "block" much more consistent in the code.  Right now the parsing of
//	block sequences, block mappings, and block scalars (as well as block scalar literals) is all pretty much handled separately.  I think
//	a rewrite could use one routine for the parsing of the blocks and then reduce them to one node.  The rules for blocks are fairly
//  consistent, just a matter of indent, although the scalars have a lot of special rules around linebreaks.
// 
//	This simplification could remove the need for the OpenMappings stack, and instead this stack of mappings could live on the function
//  stack much as the sequences live as a variable within ParseBlockSequence().  Readahead is possible using indents in order to determine
//	when the code should stop parsing one block and complete the node.
////

#ifndef __WBYamlParser_h__
#define __WBYamlParser_h__

/** Dependencies **/

#include <iostream>		// For debugging

#include "../../IO/Streams.h"
#include "../../IO/MemoryStream.h"
#include "../Xml/XmlParser.h"
#include "Yaml.h"

/** Content **/

namespace wb
{
	namespace yaml
	{		
		class YamlParser : public wb::xml::StreamParser<8>
		{
			typedef wb::xml::StreamParser<8> base;

			enum ChompingStyle { Strip, Clip, Keep };			

			/// <summary>
			/// YamlKeyValuePair is used internally during parsing but should not be exposed in the resulting heirarchy.  All entries of YamlKeyValuePair should be transformed
			/// into YamlMapping constructs before being returned from the parser.
			/// </summary>
			class YamlKeyValuePair : public YamlNode
			{
				typedef YamlNode base;
			public:
				YamlKeyValuePair(string FromSource, int FromIndentLevel, unique_ptr<YamlNode>&& pFromKey, unique_ptr<YamlNode>&& pToValue) 
					: base(FromSource), IndentLevel(FromIndentLevel), pKey(std::move(pFromKey)), pValue(std::move(pToValue)) { }

				YamlKeyValuePair(const YamlKeyValuePair&) = delete;
				YamlKeyValuePair& operator=(const YamlKeyValuePair&) = delete;

				int						IndentLevel;
				unique_ptr<YamlNode>	pKey;
				unique_ptr<YamlNode>	pValue;

				unique_ptr<YamlNode>	DeepCopy() override {
					auto pRet = make_unique<YamlKeyValuePair>(Source, IndentLevel, pKey->DeepCopy(), pValue->DeepCopy());
					pRet->IndentLevel = IndentLevel;
					pRet->base::operator=(*this);		// Shallow copy the base members
					return dynamic_pointer_movecast<YamlNode>(std::move(pRet));
				}

				unique_ptr<YamlMapping>	MoveToMap() {					
					auto pRet = make_unique<YamlMapping>(Source);
					pRet->Map.insert(make_pair<unique_ptr<YamlNode>, unique_ptr<YamlNode>>(std::move(pKey), std::move(pValue)));
					return pRet;
					//return dynamic_pointer_movecast<YamlNode>(std::move(pRet));
				}
			};
			
			bool IsWhitespace(char ch) { return ch == ' ' || ch == '\t'; }
			bool IsLinebreak(char ch) { return ch == '\r' || ch == '\n'; }		// Note, YAML 1.2: "Line breaks inside scalar content must be normalized by the YAML processor. Each such line break must be parsed into a single line feed character."						

			// Calls Advance() to pass all whitespace at the current position and returns a count of the whitespace that was skipped over.  CurrentIndent is not updated.
			int AdvanceWhitespace();

			// Call AdvanceLine() while Current is a linebreak character to correctly step over the current and possibly a following linebreak character that should
			// be normalized (combined) into the first linebreak character, as well as indention.  If two linebreaks of the same character (i.e. "\n\n") occur, then 
			// the 2nd one is not advanced over, whereas mixed characters are (i.e. "\r\n" advance twice).  Returns false if Advance() returns false.  Indention
			// is silently counted and the current line's indention is stored in the CurrentIndent member.
			bool AdvanceLine();

			unordered_map<string, string> Tags;
			unordered_map<string, unique_ptr<YamlNode>>	Aliases;
			
			struct OpenMapping {
				unique_ptr<YamlMapping>		pMap;
				int							Indent;

				OpenMapping(unique_ptr<YamlMapping>&& pOpenMap, int WithIndent)
					: pMap(std::move(pOpenMap)), Indent(WithIndent)
				{ }
			};
			stack<OpenMapping>	OpenMappings;

			int CurrentIndent;				// Indent of the current line.  Calculated in AdvanceLine(), which is always called for a line break.
			int CurrentBlockIndent;			// Indent of current block being parsed, or Int32_MaxValue if none.  Previous values can be captured on function stacks and then restored.			
			int ParsingFlowSequences;		// Number of flow sequences currently being parsed.
			int ParsingFlowMappings;		// Number of flow mappings currently being parsed.

			static string Chomp(string input, ChompingStyle Style);

			string InSourceString() {
				if (CurrentSource.length() < 1) return ""; else return " in " + CurrentSource;
			}
			
			void SetTag(unique_ptr<YamlNode>& OnNode, string Handle)
			{
				// This is only a partial implementation of Yaml tags- it does not detect errors in resolution.
				auto it = Tags.find(Handle);
				if (it == Tags.end()) {
					OnNode->Tag = Handle;
					return;
				}
				OnNode->Tag = it->second;
				return;
			}
			
			string ParseScalarContentLiteral();
			static string Unescape(string input, string Source);
			static string ApplyFlowFolding(string input);
			string ParseScalarContent(bool& WasPlain, bool& WasLinebreak);
			unique_ptr<YamlSequence> ParseBlockSequence();
			bool FindNextContent(bool FindLinebreaks = false);
			unique_ptr<YamlNode> ParseExplicitKeyValue();
			unique_ptr<YamlSequence> ParseFlowSequence();
			unique_ptr<YamlMapping> ParseFlowMapping();
			unique_ptr<YamlNode> ParseOneNodeLevel1(bool& WasLinebreak);			
			unique_ptr<YamlNode> ParseOneNodeLevel2();
			unique_ptr<YamlNode> ParseOneNode();
			void ParseDirective();
			void StartStream(wb::io::Stream& stream, const string& sSourceFilename);
			void FinishStream();
			unique_ptr<YamlNode> ParseTopLevel();						

		public:

			YamlParser();
			~YamlParser();

			/// <summary>Parses the stream, which must contain YAML document(s).  An exception is thrown on error.
			/// The stream will only be advanced up to conclusion of the next YAML document.</summary>
			/// <returns>A YamlNode, or nullptr if the document was empty.</returns>
			static unique_ptr<YamlNode> Parse(wb::io::Stream& stream, const string& sSourceFilename = "");

			static unique_ptr<YamlNode> ParseString(const string& str, const string& sSourceFilename = "");

			/// <summary>Parses the string, which must contain a YAML fragment.  An exception is thrown on error.</summary>
			/// <returns>The returned XmlDocument has been allocated with new and should be delete'd when done.</returns>
			//unique_ptr<XmlDocument> ParseAsXml(const char *psz, const string& sSourceFilename = "");

			/// <summary>Parses the stream, which must contain a YAML fragment.  An exception is thrown on error.</summary>
			/// <returns>The returned XmlDocument has been allocated with new and should be delete'd when done.</returns>			
			//unique_ptr<XmlDocument> ParseAsXml(wb::io::Stream& stream, const string& sSourceFilename = "");
		};
	
		/** YamlParser Implementation **/
		
		// Helper functions
		inline void string_ReplaceAll(string& s, const string& search, const string& replace) 
		{
			for (size_t pos = 0; ; pos += replace.length()) {
				// Locate the substring to replace
				pos = s.find(search, pos);
				if (pos == string::npos) break;
				// Replace by erasing and inserting
				s.erase(pos, search.length());
				s.insert(pos, replace);
			}
		}

		inline void string_RemoveAll(std::string& mainStr, const std::string& toErase)
		{
			size_t pos = std::string::npos;
			while ((pos = mainStr.find(toErase)) != std::string::npos) mainStr.erase(pos, toErase.length());
		}

		inline YamlParser::YamlParser()
			: 
			CurrentIndent(0),
			ParsingFlowSequences(0),
			ParsingFlowMappings(0)
		{
		}

		inline YamlParser::~YamlParser()
		{
		}						

		inline int YamlParser::AdvanceWhitespace()
		{
			if (!Need(1)) return 0;
			int count = 0;
			while (IsWhitespace(Current)) {
				count++;
				if (!Advance()) return count;
			}
			return count;
		}		

		/*static*/ inline string YamlParser::Chomp(string input, ChompingStyle Style)
		{
			// Line breaks in 'input' are assumed to have already been normalized.  That is, only \n will appear, no \r.
			if (Style == Keep) return input;

			int last_content = (int)input.length() - 1;
			for (int ii = (int)input.length() - 1; ii >= 0; ii--)
			{
				if (input[ii] == '\n') continue;
				last_content = ii;
				break;
			}
			if (last_content <= 0) return input;

			switch (Style)
			{
			case Strip: return input.substr(0, last_content+1);
			case Clip: 
				if (last_content < input.length() - 1) return input.substr(0, last_content+1) + "\n";
				else return input;			
			default: throw NotSupportedException("Unrecognized chomping style in YAML parsing.");
			}
		}

		inline bool YamlParser::AdvanceLine()
		{
			CurrentIndent = 0;

			// First advance over one line break (which might be 2 characters if normalization is needed).

			if (Current == '\n')
			{
				if (!Advance()) return false;
				if (Current == '\r') {
					if (!Advance()) return false;
				}				
			}
			else if (Current == '\r')
			{
				if (!Advance()) return false;
				if (Current == '\n') {
					if (!Advance()) return false;
				}				
			}
			else throw Exception("Expected a linebreak character to be the current character at AdvanceLine() call.");

			// Next, count the current indent.
			// Note that there may be no indent if the line break was followed by another line break immediately.

			while (Current == ' ')
			{
				CurrentIndent++;
				if (!Advance()) return false;
			}
			return true;
		}

		inline string YamlParser::ParseScalarContentLiteral()
		{
			bool Fold = (Current == '>');
			if (!Fold && Current != '|') throw FormatException("Expected | or > to start literal block in YAML.");
			if (!Advance()) return "";
			
			//int BlockIndent = CurrentIndent;
			int BlockIndent = CurrentBlockIndent;
			ChompingStyle ChompStyle = Clip;			// Default chomping style

			string strExplicitIndentation = "";
			while (!IsLinebreak(Current))
			{
				if (Current == '+') ChompStyle = Keep;
				if (Current == '-') ChompStyle = Strip;
				if (Current >= '0' && Current <= '9') strExplicitIndentation += Current;
				if (Current == '#')
				{
					// Comment on the same line as the > or | character is valid and we ignore the rest of this line- including searching for +, -, and digits.
					if (!Advance()) return "";
					while (!IsLinebreak(Current)) { if (!Advance()) return ""; }
					break;
				}
				if (!Advance()) return "";
			}
			int ExplicitIndentation = -1;
			/**		See test case 4QFQ lines 7 and 8, which are from example 8.2:
			* 
			*		- |1
			*		  explicit
			* 
			*		The above is supposed to parse into JSON as [" explicit\n"].
			*		Example 6.2 demonstrates that the spaces, dashes, and question characters all count as indentation up to the
			*		final dash or question mark.  There is then one whitespace character following the indicator that does not
			*		count as indentation.  
			* 
			*		From 8.1.1.1: "If a block scalar has an indentation indicator, then the content indentation level of the block scalar is equal to the 
			*		indentation level of the block scalar plus the integer value of the indentation indicator character."  However, note that the content
			*		indentation level of the block scalar refers to a step above the | symbol.  That is, in test case 4QFQ above, the - is at indentation
			*		level of zero, and thus the 2nd line "explicit" has 2 spaces above a block indentation of 0 plus an explicit indentation indicator
			*		of 1, thus it has 1 surplus (retained) space.
			* 
			*		But then test case 4WA9 fails.  I'm disabling for now.
			*/			
			if (strExplicitIndentation.length() > 0) ExplicitIndentation = Int32_Parse(strExplicitIndentation, wb::NumberStyles::Integer) + BlockIndent;
			if (strExplicitIndentation.length() > 0) throw NotSupportedException("This Yaml parser does not support explicit indentation in a literal header, as found at " + GetSource() + ".");
			if (!AdvanceLine()) return "";
			
			// YAML 1.2.2 spec, section 8.1.1.1: "If no indentation indicator is given, then the content indentation level is equal to the number of leading spaces on the 
			// first non-empty line of the contents. If there is no non-empty line then the content indentation level is equal to the number of spaces on the longest line."
			
			/** Parse until we find the first non-empty text, detect indentation **/

			int Indentation = ExplicitIndentation;
			string ret;			

			// CurrentIndent is set by AdvanceLine().

			while (IsLinebreak(Current))
			{
				if (CurrentIndent > Indentation && ExplicitIndentation < 0) Indentation = CurrentIndent;
				// This is an empty line.
				//if (Fold) ret += ' '; else ret += '\n';
				ret += '\n';			// Fold does not apply to an empty line.
				if (!AdvanceLine()) return Chomp(ret, ChompStyle);
			}

			/** Parse non-empty line(s) **/
				
			// Current line contains text or content.
				
			// There may not have been any empty lines, in which case we may need to
			// set the indentation now with this line.
			if (Indentation < 0) Indentation = CurrentIndent;

			// Is tihs line part of the literal block?
			if (CurrentIndent < Indentation) {
				// We never had any content inside the literal block.  But there was at least one linebreak (from
				// the line with the folding indicator if nothing else).  From Section 6.5: "In the folded block style, the 
				// final line break and trailing empty lines are subject to chomping and are never folded."
				ret += '\n';
				return Chomp(ret, ChompStyle);		// No, we've broken out of the literal block.
			}

			// Yes, it is part of the literal block.
			// If it is further indented relative to the detected or explicit indentation, then it contains whitespace that must be
			// preserved.
			string FoldsPending = "";			// '-' for a pending fold or 'n' for an empty line.  Queued left-to-right, i.e. "-n-" came from "foo\n\nbar\n".
			bool MoreIndented = (CurrentIndent > Indentation);

			/** Parse additional lines until indentation breaks us out or until end-of-stream **/
			
			for (;;)
			{
				MoreIndented = (CurrentIndent > Indentation);
				if (MoreIndented)
				{
					// When we hit a more indented line, any \n's that might have been folded need to be retained instead.
					ret += string(FoldsPending.length(), '\n');
					FoldsPending.clear();
				}
				for (int Additional = 0; Additional < (CurrentIndent - Indentation); Additional++) ret += ' ';

				if (IsLinebreak(Current))
				{
					FoldsPending += 'n';		// Record an empty line in the pending list.
					if (!AdvanceLine()) break;
				}
				else
				{
					if (CurrentIndent < Indentation) {
						// Since it was terminal, retain the original linebreak and don't fold it.
						if (FoldsPending.length()) FoldsPending[FoldsPending.length() - 1] = 'n';						
						break;
					}

					// We've hit content at the same indentation level.  Any pending folds can now be
					// applied as planned.
					string_ReplaceAll(FoldsPending, "-n", "n");			// See test suite '4Q9F'.
					for (auto ii = 0; ii < FoldsPending.length(); ii++)
					{
						if (FoldsPending[ii] == '-') ret += ' ';
						if (FoldsPending[ii] == 'n') ret += '\n';
					}
					FoldsPending.clear();

					while (!IsLinebreak(Current)) { 
						ret += Current; 
						if (!Advance()) break;
					}
					if (Fold && !MoreIndented)
					{
						// Convert line breaks into whitespace, except if an empty line follows...
						if (!AdvanceLine()) { 
							FoldsPending += 'n';			// Example 8.12: The final line break and trailing empty lines if any, are subject to chomping and are never folded.
							break;
						}
						FoldsPending += '-';
					}
					else
					{
						ret += '\n';
						if (!AdvanceLine()) break;
					}
				}
			}

			for (auto ii = 0; ii < FoldsPending.length(); ii++)
			{
				if (FoldsPending[ii] == '-') ret += ' ';
				if (FoldsPending[ii] == 'n') ret += '\n';
			}
			return Chomp(ret, ChompStyle);
		}		

		/*static*/ inline string YamlParser::ApplyFlowFolding(string input)
		{
			// See Example 6.8: Flow Folding.  Inside quotes is a flow style and subject to folding.
			// "Folding in flow styles provides more relaxed semantics. Flow styles typically depend on explicit indicators rather than indentation 
			// to convey structure. Hence spaces preceding or following the text in a line are a presentation detail and must not be used to convey 
			// content information. Once all such spaces have been discarded, all line breaks are folded without exception.

			// The combined effect of the flow line folding rules is that each “paragraph” is interpreted as a line, empty lines are interpreted as 
			// line feedsand text can be freely more - indented without affecting the content information."

			// Precondition: when in quotes, ParseScalarContent() captures the linebreaks as a single \n character and does not apply folding until
			// close of the quotations, at which time ApplyFlowFolding is called to discard spaces and fold.
			
			string ret;
			bool FoldPending = false;			
			size_t ii = 0;
			bool FirstLine = true;
			for (;;)
			{
				auto next_linefeed = input.find('\n', ii);
				string next_line = input.substr(ii, next_linefeed - ii);

				// next_line contains all text on the next line (or is an empty string for an empty line), but excludes the \n itself.
				next_line = Trim(next_line);

				if (next_linefeed == string::npos) {					
					if (FoldPending) ret += ' ';
					if (next_line.length() > 0) ret += next_line;
					return ret;
				}
				
				if (next_line.length() == 0) {
					if (FirstLine || next_linefeed == string::npos) FoldPending = true;		// The first and last lines aren't empty- they contain the quotation characters.
					else {
						// Empty line
						ret += '\n';
						FoldPending = false;
					}
				}
				else
				{
					if (FoldPending) ret += ' ';
					ret += next_line;
					FoldPending = true;
				}
				FirstLine = false;
				
				ii = next_linefeed + 1;
			}
		}

		/*static*/ inline string YamlParser::Unescape(string input, string Source)
		{
			string ret;
			for (auto ii = 0; ii < input.length(); ii++)
			{
				if (input[ii] == '\\')
				{
					ii++;
					if (ii >= input.length())
						throw FormatException("Escaped character (\\) cannot be final character in a double-quoted string at " + Source);
					switch (input[ii])
					{
					case '0': ret += '\0'; break;
					case 'a': ret += '\a'; break;
					case 'b': ret += '\b'; break;
					case 't': ret += '\t'; break;
					case 'n': ret += '\n'; break;
					case 'v': ret += '\v'; break;
					case 'f': ret += '\f'; break;
					case 'r': ret += '\r'; break;
					case 'x': throw NotImplementedException("Escaped unicode sequences are not implemented in this YAML parser at " + Source + ".");
						// The quotation character unescape is handled while parsing and should not be handled again here.					
					case '/': ret += '/'; break;
					case '\\': ret += '\\'; break;

						// From test suite 565N, a backslash before the linebreak within a double-quoted scalar appears to have the effect of nullifying the linebreak
						// from the string without inserting whitespace.  This is similar to the C/C++ language where a backslash as the final character on a line
						// will continue the line onto the next, uninterrupted.
						// UPDATED: this is handled before Unescape(), because it needs to be handled before ApplyFlowFolding().
					// case '\n': break;

					default:
						throw NotImplementedException("Escaped \\" + string(1, input[ii]) + " sequence is not implemented in this YAML parser at " + Source + ".");
					}
				}
				else ret += input[ii];
			}
			return ret;
		}		

		/// <summary>
		/// ParseScalarContent() parses text until an indicator or other syntax is encountered.  Empty nodes are possible in
		/// Yaml, so ParseScalarContent() may return an empty string.  ParseScalarContent() is guaranteed to call Advance()
		/// at least once, as it could lead to an infinite loop if we continually parse empty nodes without advancing.  There
		/// are rules in YAML about lookahead to a colon indicator being constrained to 1024 characters and further that the
		/// colon must be found on the same line, and so the WasLinebreak boolean is returned to indicate if a linebreak split
		/// the start of the scalar content from the current position.
		/// </summary>
		inline string YamlParser::ParseScalarContent(bool& WasPlain, bool& WasLinebreak)
		{
			if (!Need(1)) throw Exception("ParseScalarContent() unable to locate scalar content as Need(1) is false, parsing at " + GetSource() + ".");

			WasLinebreak = false;
			WasPlain = true;
			bool FirstLinebreak = true;
			string ParsingQuoteStarted = "";
			char ParsingQuote = 0;
			string ret;
			bool EmptyLine = true;			
			int IndentOfBlock = CurrentIndent;
			AdvanceWhitespace();
			for (;;)
			{
				if (!Need(1)) return ret;

				if (Current == '\'' && (ret.empty() || ParsingQuote == '\''))
				{
					EmptyLine = false;					
					if (ParsingQuote == '\'') {

						// Double single-apostrophes after we've started is an escape sequence.  For example:
						//	'here''s to "quotes"'			->		here's to "quotes"'
						if (Need(2) && pNext[0] == '\'')
						{
							if (!Advance(2)) throw FormatException("Single quote starting scalar without matching closing quote at " + GetSource() + ".");
							ret += '\'';
							continue;
						}

						Advance();
						ParsingQuote = 0; 
						WasPlain = false;
						return ApplyFlowFolding(ret);
					}
					else {						
						ParsingQuote = '\'';
						ParsingQuoteStarted = GetSource();
						if (!Advance()) throw FormatException("Single quote starting scalar without matching closing quote at " + GetSource() + ".");						
					}
					continue;
				}
				if (ParsingQuote == '\"' && Current == '\\')
				{
					if (!Need(2)) throw FormatException("In double quoted scalar, backslash detected as start of escape sequence but no following character found at " + GetSource() + " from quote started at " + ParsingQuoteStarted + ".");
					// Only handle double-quote character here.  Unescape() will handle the rest later.
					if (pNext[0] == '\"') {
						ret += '\"';
						if (!Advance(2)) throw FormatException("Double quote scalar started at " + ParsingQuoteStarted + " without matching closing quote.");
					}
					else if (IsLinebreak(pNext[0])) {
						// Handle separately because we need to normalize the linebreak with AdvanceLine(), but we want to retain the escaping to avoid later folding.
						ret += '\\';
						ret += '\n';
						if (!Advance() || !AdvanceLine()) throw FormatException("Double quote scalar started at " + ParsingQuoteStarted + " without matching closing quote.");
					}
					else {
						ret += '\\';
						ret += pNext[0];
						if (!Advance(2)) throw FormatException("Double quote scalar started at " + ParsingQuoteStarted + " without matching closing quote.");
					}
					continue;
				}
				if (Current == '\"' && (ret.empty() || ParsingQuote == '\"'))
				{
					EmptyLine = false;
					if (ParsingQuote == '\"') {
						Advance();
						ParsingQuote = 0; 
						WasPlain = false;

						// From test suite 565N, a backslash before the linebreak within a double-quoted scalar appears to have the effect of nullifying the linebreak
						// from the string without inserting whitespace.  This is similar to the C/C++ language where a backslash as the final character on a line
						// will continue the line onto the next, uninterrupted.  Since the \n characters will get folded, we need to remove them as a first step.  Since
						// the \\n characters would become \n's and then get folded, we need to unescape everything else as a last step.
						string_RemoveAll(ret, "\\\n");
						ret = ApplyFlowFolding(ret);
						return Unescape(ret, ParsingQuoteStarted);
					}
					else {
						ParsingQuote = '\"';
						ParsingQuoteStarted = GetSource();
						if (!Advance()) throw FormatException("Double quote starting scalar without matching closing quote at " + GetSource() + ".");
					}
					continue;
				}				
				if (IsLinebreak(Current))
				{										
					if (ParsingQuote != 0)
					{
						// Whitespace and folding will be handled in ApplyFlowFolding() at conclusion of the quotes, so retain all whitespace and linebreaks here as \n entries.
						ret += "\n";
						if (!AdvanceLine()) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + ".");
						continue;
					}
					
					/*
					if (ParsingQuote != 0)
					{
						ret += '\n';
						if (!AdvanceLine()) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + ".");
						continue;
					}
					*/
					WasLinebreak = true;
					if (EmptyLine) {
						if (ret.length() > 0 && ret[ret.length() - 1] == ' ') ret = ret.substr(0, ret.size() - 1);		// For 36F6: clobber one whitespace if \n follows.
						ret += "\n";
					}
					else ret += " ";
					if (!AdvanceLine()) {
						if (ParsingQuote != 0) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + ".");
						return ret;
					}
					EmptyLine = true;
					if (!FindNextContent(true)) {
						if (ParsingQuote != 0) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + ".");
						return ret;
					}
					if (IsLinebreak(Current))
					{
						// If FindNextContent() stops on a linebreak having just had an AdvanceLine(), then it means we've found an empty line.						
						continue;
					}
					if (CurrentIndent < IndentOfBlock && ParsingQuote == 0) {
						//if (ParsingQuote != 0) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + " due to block indent change.");
						return Trim(ret);
					}
					if (FirstLinebreak)
					{
						IndentOfBlock = CurrentIndent;
						FirstLinebreak = false;
					}
					else
					{
						if (CurrentIndent > IndentOfBlock) ret += string(CurrentIndent - IndentOfBlock, ' ');
					}
					continue;
					//if (Trim(ret).length() > 0) { WasLinebreak = true; return ret; }
					//else continue;
				}
				if (ParsingQuote != 0)
				{
					ret += Current;
					if (!Advance()) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") started at " + ParsingQuoteStarted + ".");
					continue;
				}
				if (Current == ' ' || Current == '\t')
				{
					if (Need(2) && pNext[0] == '#')
					{
						if (!Advance(2)) return ret;
						while (!IsLinebreak(Current)) { if (!Advance()) return ret; }
						// Loop so as to hit the IsLinebreak() test.
						continue;
					}
					if (EmptyLine)
					{
						if (!Advance()) return ret;
						continue;
					}
					ret += Current;
					if (!Advance()) return ret;
					continue;
				}				
				if (Current == ':')
				{
					if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator (: ) found.");		// Should have been detected & handled at higher-level.
						return ret;		// The ": " combination is an indicator, but what precedes it constitutes the scalar content.
					}
				}
				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-')
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": end indicator (---) found.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				if (Current == '-' && (ret.empty() || ret[ret.length()-1] == ' ' || ret[ret.length()-1] == '\t' || ret[ret.length()-1] == '\n'))
				{
					if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator (- ) found.");		// Should have been detected & handled at higher-level.
						return ret;		// Detected a block sequence (8.2.1) marker.
					}
				}
				if (Current == '.')
				{
					if (Need(3) && pNext[0] == '.' && pNext[1] == '.')
					{
						if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator (...) found.");		// Should have been detected & handled at higher-level.
						return ret;			// Detected an end-of-document marker.
					}
				}
				if (Current == '|' || Current == '>')
				{
					// Check for c-b-block-header:
					bool Qualified = true;
					for (int ii = 2; ii < GetMaxLoading(); ii++)
					{
						if (!Need(ii))
						{							
							Qualified = false;
							break;
						}
						if (pNext[ii - 2] == ' ' || pNext[ii - 2] == '\t' || IsLinebreak(pNext[ii - 2])) { Qualified = true; break; }
						if (pNext[ii - 2] == '+' || pNext[ii - 2] == '-' || (pNext[ii - 2] >= '0' && pNext[ii - 2] <= '9')) continue;
						Qualified = false;
						break;
					}
					if (Qualified) 
					{
						if (!EmptyLine) throw FormatException("Illegal literal indicator following scalar content that has already initiated at " + GetSource() + ".");
						WasPlain = false;
						WasLinebreak = true;
						return ParseScalarContentLiteral();
					}
				}
				if (ParsingFlowMappings > 0 && (Current == ',' || Current == '}'))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator found in flow mapping.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				if (ParsingFlowSequences > 0 && (Current == ',' || Current == ']'))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator found in flow sequence.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				if (ret.empty() && (Current == '[' || Current == ']' || Current == '{' || Current == '}' || Current == ','))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator found.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				ret += Current;
				EmptyLine = false;
				if (!Advance()) return ret;
			}
		}		

		inline unique_ptr<YamlSequence> YamlParser::ParseBlockSequence()
		{
			if (!Need(2) || Current != '-' || (pNext[0] != ' ' && pNext[0] != '\t' && !IsLinebreak(pNext[0]))) throw Exception("Parsing block sequence without indicator at start: " + GetSource());
			
			int PreviousBlockIndent = CurrentBlockIndent;
			Advance();
			CurrentIndent++;				// Example 6.2: The - indicator counts as indentation.
			CurrentBlockIndent = CurrentIndent;

			unique_ptr<YamlSequence> pSequence = make_unique<YamlSequence>(GetSource());
			if (!FindNextContent())
			{
				// Need to add an empty node at the end of the sequence because we found a - with no content following it.
				auto pEmptyNode = unique_ptr<YamlNode>(new YamlScalar(GetSource()));
				pSequence->Entries.push_back(std::move(pEmptyNode));
				CurrentBlockIndent = PreviousBlockIndent;
				return pSequence;
			}
			auto pNextNode = ParseOneNode();
			pSequence->Entries.push_back(std::move(pNextNode));
			
			for (;;)
			{
				if (!FindNextContent()) break;				

				if (Current == '-')
				{
					if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						if (CurrentIndent + 1 < CurrentBlockIndent) break;

						// Block sequence entry identifier (- )
						Advance();
						CurrentIndent++;		// Example 6.2: The - indicator counts as indentation.
						#if 0
						// Could probably allow the following to be parsed by ParseOneNode(), but using explicit handling for now:
						bool bEOF;
						if (IsLinebreak(Current)) bEOF = !AdvanceLine(); 
						else {
							bEOF = !Advance(); CurrentIndent++;
						}						
						if (bEOF)
						#endif		
						if (!FindNextContent())
						{
							// Need to add an empty node at the end of the sequence because we found a - with no content following it.
							auto pEmptyNode = unique_ptr<YamlNode>(new YamlScalar(GetSource()));
							pSequence->Entries.push_back(std::move(pEmptyNode));
							break;
						}
						auto pNext = ParseOneNode();
						pSequence->Entries.push_back(std::move(pNext));
						continue;
					}
				}
				
				if (CurrentIndent < CurrentBlockIndent) break;

				// We should not see mapping nodes here- they should have been merged into one node by ParseOneNode().
				// A scalar without a leading dash would be an error.
				// Any other content (i.e. flow) would be an error.
				throw FormatException("Unexpected content at " + GetSource() + " following block sequence started at " + pSequence->Source + ".");
			}
			
			CurrentBlockIndent = PreviousBlockIndent;			
			return pSequence;
		}

		/// <summary>
		/// FindNextContent() advances from the current position until something other than whitespace or a comment is found.  CurrentIndent
		/// is updated during the search, in case content is found on the current line.  False is returned if Advance() or AdvanceLine()
		/// returns false.  If FindLinebreaks is true, then a linebreak is considered relevant and FindNextContent() returns at the linebreak.
		/// </summary>
		inline bool YamlParser::FindNextContent(bool FindLineBreaks /*= false*/)
		{
			// Decide what the block's indent should be.  First, check if there's content on the same line as the indicator.
			
			if (!Need(1)) return false;

			for (;;)
			{
				if (Current == ' ' || Current == '\t')
				{
					if (!Advance()) return false;
					CurrentIndent++;
					continue;
				}

				if (Current == '#')
				{
					while (!IsLinebreak(Current)) { if (!Advance()) return false; }
					if (FindLineBreaks) return true;
					AdvanceLine();
					break;
				}

				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-') return false;
				if (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.') return false;

				if (IsLinebreak(Current)) { 
					if (FindLineBreaks) return true;
					AdvanceLine(); 
					break; 
				}
				return true;
			}

			// Didn't find content on the same line, so check the next non-empty line.
						
			for (;;)
			{
				if (!Need(1))
				{
					// Indicator was the last thing in the document.  Use empty.
					return false;
				}

				// Skipping whitespace should be unnecessary here, since we have had an AdvanceLine() call.  The exception is
				// tabs, which don't count for indentation but still count as whitespace to be skipped and separating tokens.
				// And since it is possible for a space to follow a tab, we want to skip over both.
				if (Current == ' ' || Current == '\t')
				{
					if (!Advance()) return false;
					// Don't count indent at this point.
					continue;
				}

				if (Current == '#')
				{
					while (!IsLinebreak(Current)) { if (!Advance()) return false; }
					if (FindLineBreaks) return true;
					AdvanceLine();
					continue;
				}

				if (IsLinebreak(Current)) {
					if (FindLineBreaks) return true;
					AdvanceLine();
					continue;
				}

				return true;
			}
		}

		inline unique_ptr<YamlNode> YamlParser::ParseExplicitKeyValue()
		{
			if (!Need(2) || Current != '?' || (pNext[0] != ' ' && pNext[0] != '\t' && !IsLinebreak(pNext[0]))) throw Exception("Parsing explicit key without indicator at start: " + GetSource());			

			string QuestionMarkSource = GetSource();
			int PreviousBlockIndent = CurrentBlockIndent;
			int QuestionMarkIndent = CurrentIndent;

			CurrentIndent ++;		// Example 6.2: The '?' indicator counts as indentation.
			if (!Advance() || !FindNextContent())
			{
				// Question mark was the last thing in the document.  Use empty.
				return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, nullptr, nullptr));
			}

			// At this point, we've found the content following the ?.  If we hadn't, we'd have returned due to end-of-document.  We don't
			// know if it's actually a key or the : for this ? yet, depends on indent.

			unique_ptr<YamlNode> pKey;
			if (CurrentIndent >= QuestionMarkIndent + 2)
			{
				// There's key content.
				CurrentBlockIndent = CurrentIndent;
				pKey = ParseOneNode();
				// pKey might be empty still.
				CurrentBlockIndent = PreviousBlockIndent;
			}
			else
			{
				// Empty key
			}

			// Look for ? or :, or if we've left the block.

			// There might be empty lines between the end of the key (i.e. end of block) and the colon, or end of document or other content.
			if (!FindNextContent())
			{
				// No : found, so the value is empty.
				return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, std::move(pKey), nullptr));
			}

			if (CurrentIndent < QuestionMarkIndent)
			{
				// Empty value
				return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, std::move(pKey), nullptr));
			}

			if (CurrentIndent > QuestionMarkIndent) 
				throw FormatException("Unrecognized block indent; question mark indented to " + std::to_string(QuestionMarkIndent) 
					+ " at " + QuestionMarkSource + " but found block at " + std::to_string(CurrentIndent) + " at " + GetSource() + " after key.");

			// Indent matches the question mark.  Now look for ? or :.

			if (Current == '?' && Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
			{
				// Question mark at same level as original question mark.  This starts a new sibling-level key that should be merged into the open mapping that will
				// be formed by the original question mark.  The value is empty.
				return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, std::move(pKey), nullptr));
			}

			if (Current == ':' && Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
			{
				CurrentIndent += 2;
				if (!Advance() || !Advance() || !FindNextContent())
				{
					// Colon was the last thing in the document.  Use empty.
					return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, std::move(pKey), nullptr));
				}
				
				CurrentBlockIndent = CurrentIndent;
				auto pValue = ParseOneNode();
				CurrentBlockIndent = PreviousBlockIndent;

				return unique_ptr<YamlNode>(new YamlKeyValuePair(QuestionMarkSource, QuestionMarkIndent, std::move(pKey), std::move(pValue)));
			}

			throw FormatException("Expected ? or : at " + GetSource() + " or indent change following previous ? found at " + QuestionMarkSource);
		}

		inline unique_ptr<YamlSequence> YamlParser::ParseFlowSequence()
		{
			if (!Need(1) || Current != '[') throw Exception("Parsing flow sequence without indicator at start: " + GetSource());
			if (!Advance()) throw FormatException("Unterminated YAML flow sequence beginning at " + GetSource() + ".");
			CurrentIndent++;

			// Indicate to ParseOneNode() that a comma or closing ] is acceptable and should cause a return, while allowing nesting.
			ParsingFlowSequences++;

			auto pSequence = make_unique<YamlSequence>(GetSource());
			for (;;)
			{
				auto pNext = ParseOneNode();

				if (!Need(1)) throw FormatException("Unterminated YAML flow sequence beginning at " + pSequence->Source + ".");

				if (Current == ',')
				{
					if (!Advance() || !FindNextContent()) throw FormatException("Unterminated YAML flow sequence beginning at " + pSequence->Source + ".");
					if (Current != ']')			// Tolerance for extra comma: i.e. [one, two, ]
					{
						pSequence->Entries.push_back(std::move(pNext));
						continue;
					}
				}

				if (Current == ']')
				{
					Advance();
					if (pNext != nullptr) pSequence->Entries.push_back(std::move(pNext));
					ParsingFlowSequences--;
					return pSequence;
				}
			}
		}

		inline unique_ptr<YamlMapping> YamlParser::ParseFlowMapping()
		{
			if (!Need(1) || Current != '{') throw Exception("Parsing flow mapping without indicator at start: " + GetSource());
			if (!Advance()) throw FormatException("Unterminated YAML flow mapping beginning at " + GetSource() + ".");
			CurrentIndent++;

			// Indicate to ParseOneNode() that a comma or closing } is acceptable and should cause a return, while allowing nesting.
			ParsingFlowMappings++;

			// We'll use an implicit sequence to wrap the individual mappings.
			//auto pMapping = make_unique<YamlMapping>(GetSource());

			// ParseOneNode() will capture the flow mappings before returning into an OpenMappings, since the individual entries
			// for the flow mapping will be key:value pairs (YamlKeyValuePairs) before reaching here.  To account for this, we
			// explicitly add an OpenMapping with 0 indent to the top of the stack that will gather our flow mapping results.			
			OpenMappings.push(OpenMapping(make_unique<YamlMapping>(GetSource()), 0));
			YamlMapping& ThisMap = *OpenMappings.top().pMap;

			for (;;)
			{
				auto pNextNode = ParseOneNode();
				//if (pNextNode != nullptr && !is_type<YamlKeyValuePair>(pNextNode)) throw FormatException("Expected single key-value pair as entry at " + pNextNode->Source + " in flow mapping started at " + pMapping->Source + ".");
				//unique_ptr<YamlKeyValuePair> pNext = dynamic_pointer_movecast<YamlKeyValuePair>(std::move(pNextNode));

				if (!Need(1)) throw FormatException("Unterminated YAML flow mapping beginning at " + ThisMap.Source + ".");

				if (Current == ',')
				{
					if (!Advance()) throw FormatException("Unterminated YAML flow mapping beginning at " + ThisMap.Source + ".");
					//pMapping->Add(std::move(pNext->pKey), std::move(pNext->pValue));
					continue;
				}

				if (Current == '}')
				{
					Advance();
					//if (pNext != nullptr) pMapping->Add(std::move(pNext->pKey), std::move(pNext->pValue));
					ParsingFlowMappings--;
					auto pRet = std::move(OpenMappings.top().pMap);
					OpenMappings.pop();
					return std::move(pRet);
				}
			}
		}				

		/// <summary>
		/// ParseOneNodeLevel1() handles any parsing that can be done with no stack, queue, or memory 
		/// of multiple nodes, prefixes, or postfixes.  It handles the things that can be handled with
		/// no context.
		/// </summary>
		inline unique_ptr<YamlNode> YamlParser::ParseOneNodeLevel1(bool& WasLinebreak)
		{
			// 3.2.1.1 Nodes: Can have content of one of three kinds: scalar content, sequence, or mapping.						

			WasLinebreak = false;

			if (!Need(1)) return nullptr;				

			if (Current == ' ' || Current == '\t' || Current == '#' || IsLinebreak(Current)) 
				throw Exception("Unexpected character (" + string(1, Current) + ") at this level of processing at " + GetSource() + ".");

			// Detect end of Yaml document marker...
			if (Current == '.')
			{
				if (!Need(2)) throw FormatException("Dot (.) with no following data during YAML parsing at " + GetSource() + ".");
				if (pNext[1] == '.' && pNext[2] == '.') return nullptr;
			}

			// Detect block sequence (8.2.1) indicator...
			// Detect start of new Yaml document marker / end of directives, which in this context would imply the end of the current document...
			if (Current == '-')
			{
				if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					throw Exception("Unexpected character (- ) at this level of processing at " + GetSource() + ".");
				
				if (!Need(2)) throw FormatException("Dash character with no following data during YAML parsing at " + GetSource() + ".");

				if (pNext[0] == '-')
				{
					if (!Need(3)) throw FormatException("Dash pairs (--) with no following data during YAML parsing at " + GetSource() + ".");
					if (pNext[1] == '-')
					{
						// Three dashes detected.  
						Advance(); Advance(); Advance();
						return nullptr;
					}
				}
			}
			
			if (Current == '?')
			{
				if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					throw Exception("Unexpected character (? ) at this level of processing at " + GetSource() + ".");
			}
			
			if (Current == ':')
			{
				if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t'))
				{
					throw Exception("Unexpected character at this level of processing at " + GetSource() + ".");
				}
			}

			if (Current == ',' || Current == ']' || Current == '}' || Current == '!' || Current == '&')
				throw Exception("Unexpected character at this level of processing at " + GetSource() + ".");

			if (Current == '%') throw FormatException("YAML directive character (%) found outside of a directive section at " + GetSource() + ".");			

			if (Current == '*')
			{					
				if (!Advance()) throw FormatException("Alias character (*) with no token following in YAML parsing at " + GetSource() + ".");
				string anchor;
				while (Current != ' ' && Current != '\t' && Current != '[' && Current != ']' && Current != '{' && Current != '}' && Current != ',' && !IsLinebreak(Current))
				{
					anchor += Current;
					if (!Advance()) break;
				}
				auto it = Aliases.find(anchor);
				if (it == Aliases.end()) throw FormatException("Referenced node '" + anchor + "' was not found as a previous anchor at " + GetSource() + ".");
				if (it->second == nullptr) return nullptr;
				return it->second->DeepCopy();
			}

			if (Current == '[') {
				//if (pQueuedNode != nullptr) return nullptr;		// Flush the queue before starting flow style.
				return dynamic_pointer_movecast<YamlNode>(ParseFlowSequence());
			}

			if (Current == '{') {
				//if (pQueuedNode != nullptr) return nullptr;		// Flush the queue before starting flow style.
				return dynamic_pointer_movecast<YamlNode>(ParseFlowMapping());
			}
				
			string StartSource = GetSource();
			bool WasPlain = false;
			string Text = ParseScalarContent(WasPlain, WasLinebreak);
			if (WasPlain) Text = Trim(Text);
			auto pScalar = unique_ptr<YamlNode>(new YamlScalar(StartSource, Text));
			SetTag(pScalar, WasPlain ? "?" : "!");
			return pScalar;
		}

		/*
		* 
		*	A:
		*		alpha: 1
		*		beta: 2
		*	B:
		*	C: ["one", "two", "three"]
		*	D: ["one", "two", "three"] : "complicated"
		*	E: 
		*	    - first
		*	    - second
		*	F: {"one": 1, "two": 2, "three": 3} : ["fore"]
		*	G: "finisher"
		* 
		*/

		/// <summary>
		/// ParseOneNodeLevel2() parses a single node (calling ParseOneNodeLevel1()) with direct prefix and 
		/// postfix effects considered.  Prefixes include:
		///		- tags
		///		- explicit key marker
		///		- sequence item marker
		///		- anchors
		/// 
		/// The postfixes are:
		///		- colon, indicating a key : value pair contributing or constituting a mapping.		
		/// 
		/// One final postfix effect the merging of multiple block mapping nodes that are at the same level of 
		/// indentation.  This is not handled by ParseOneNodeLevel2().
		/// </summary>
		inline unique_ptr<YamlNode> YamlParser::ParseOneNodeLevel2()
		{
			string Tag = "!";
			
			string anchor;

			/** Parse for any prefixes **/

			for (;;)
			{
				if (!FindNextContent()) return nullptr;

				// Flush the queue anytime we have content at a lower indent.
				if (ParsingFlowMappings == 0 && ParsingFlowSequences == 0 && !OpenMappings.empty() && OpenMappings.top().Indent > CurrentIndent) return nullptr;				

				if (ParsingFlowMappings > 0)
				{
					if (Current == ',' || Current == '}') return nullptr;
				}

				if (ParsingFlowSequences > 0)
				{
					if (Current == ',' || Current == ']') return nullptr;
				}				

				// Detect block sequence (8.2.1) indicator...				
				if (Current == '-')
				{
					if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						/**
						*	Are we continuing a sequence or are we starting a new sequence?
						*
						*	- A:
						*		- alpha
						*		- beta
						*	- B: done
						*
						*	We need to make sure that the initial value of CurrentBlockIndent is such that the - A line starts a block sequence.
						*	The reduced indent on - B should be detected and cause us to return, closing the current node.
						*
						*   We'll return an empty node (nullptr) between beta and - B.  That's ambiguous with:
						*
						*	- A:
						*		- alpha
						*		- beta
						*		-
						*	- B: done
						*
						*	The nullptr should (eventually) propagate all the way up through ParseNode() to the calling ParseBlockSequence()
						*	without advancing beyond the dash character.  Then ParseBlockSequence() can distinguish whether it is the top
						*	case or the ambiguous bottom case.  The bottom case should not present a nullptr but instead an empty YamlScalar.
						*/

						// Because we are not advancing, we are not incrementing CurrentIndent to count the '-' character itself, however this does
						// count against the indent.  To predict whether we need to continue or start a new block, we need to include the +1 here.
						// 
						// Also see Example 8.22 in 8.2.3. Block Nodes, where a special exception/adjustment to the rule regarding block node indent is described.
						if (CurrentIndent + 1 < CurrentBlockIndent) return nullptr;		// Note: we do not advance in this case, caller must reduce CurrentBlockIndent to proceed.
						else {
							return ParseBlockSequence();
						}
					}
				}
				else if (!ParsingFlowMappings && !ParsingFlowSequences && CurrentIndent < CurrentBlockIndent) return nullptr;		// Note: we do not advance in this case, caller must reduce CurrentBlockIndent to proceed.

				if (Current == '&')
				{
					// Question: would it apply at this level or one level up?  For example, if you have a key:value is that one node?  
					// A. No, I think, only if there were a
					//		&anchor ? key : value 
					// would it count as one node.  Otherwise, it must be taken that key:value breaks out as &anchor key : value with the anchor referencing the key
					// only.

					int IndentAtStart = CurrentIndent;
					if (!Advance()) throw FormatException("Anchor character with no token following in YAML parsing at " + GetSource() + ".");
					if (anchor.length() > 0) throw FormatException("Repeated anchors not permitted at " + GetSource() + ".");
					while (Current != ' ' && Current != '\t' && Current != '[' && Current != ']' && Current != '{' && Current != '}' && Current != ',' && !IsLinebreak(Current))
					{
						anchor += Current;
						if (!Advance()) throw FormatException("Anchor name specified without node content at " + GetSource() + ".");
					}
					while (Current == ' ' || Current == '\t')
					{
						if (!Advance()) throw FormatException("Anchor name specified without node content at " + GetSource() + ".");
					}

					// Don't count the &anchor or whitespace after it as indent when we call FindNextContent() later.
					AdvanceWhitespace();
					// Remove the anchor as affecting the indent.  Although really the key to this is parsing the whitespace above, since we didn't increment CurrentIndent.					
					CurrentIndent = IndentAtStart;			
					// Default the alias to nullptr, in case we return a nullptr before we end up parsing a node.
					Aliases[anchor] = nullptr;
					continue;
				}

				if (Current == '!')
				{
					if (!Advance()) throw FormatException("Tag (!) character with no token following in YAML parsing at " + GetSource() + ".");
					string handle;
					while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
					{
						handle += Current;
						if (!Advance()) throw FormatException("Tag (!) handle specified without node at " + GetSource() + ".");
					}
					Tag = handle;

					// Don't count the !tag or whitespace after it as indent when we call FindNextContent() later.
					AdvanceWhitespace();
					continue;
				}				

				// Detect explicit key indicator...				
				if (Current == '?')
				{
					if (Need(2) && (pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						// Because we are not advancing, we are not incrementing CurrentIndent to count the '-' character itself, however this does
						// count against the indent.  To predict whether we need to continue or start a new block, we need to include the +1 here.
						if (CurrentIndent + 1 < CurrentBlockIndent) return nullptr;		// Note: we do not advance in this case, caller must reduce CurrentBlockIndent to proceed.
						else {
							return ParseExplicitKeyValue();
						}

						/*
						if (!Advance()) {
							auto Seq = make_unique<YamlSequence>(GetSource());
							Seq->Entries.push_back(nullptr);
							return dynamic_pointer_movecast<YamlNode>(std::move(Seq));
						}
						*/
					}
				}

				break;
			}

			/** Advance to the node itself **/

			// Since the above clauses all lead to continue's and the loop proceeds through whitespace and linebreaks,
			// we are guaranteed to be on a relevant character at this point.

			bool WasLinebreak = false;
			int IndentAtStart = CurrentIndent;
			auto pLeft = ParseOneNodeLevel1(WasLinebreak);
			if (pLeft == nullptr)
			{
				// In case an anchor has been specified, it defaults to nullptr and has already been set as such.
				return nullptr;
			}
			SetTag(pLeft, Tag);

			if (anchor.length() > 0) Aliases[anchor] = pLeft->DeepCopy();			// Override anchor if it already exists.

			if (WasLinebreak && !ParsingFlowMappings)
			{
				// If there was a linebreak, then a colon is not allowed.  We have already applied all prefixes except possibly the explicit key marker.
				return pLeft;
			}

			/** Parse for any postfixes **/

			for (;;)
			{
				if (!Need(1)) break;

				if (Current == ' ' || Current == '\t')
				{
					if (!Advance()) break;
					continue;
				}

				if (Current == '#')
				{					
					while (!IsLinebreak(Current)) { if (!Advance()) break; }					
					if (!AdvanceLine()) break;
					continue;
				}

				// A linebreak here precludes the possibility of a colon, except in flow mappings as some random exception to the rule.  Yaml has so many exceptions...
				if (IsLinebreak(Current)) {
					if (!AdvanceLine()) break;
					if (ParsingFlowMappings) continue;
					break;
				}

				if (Current == ':')
				{
					if (!Advance())
					{
						return unique_ptr<YamlNode>(make_unique<YamlKeyValuePair>(GetSource(), IndentAtStart, std::move(pLeft), nullptr));
					}
					
					/** Simplified test case 6KGN:
					* 
					*	---
					*	a: 
					*	b: 
					*	---
					* 
					*	This test case should parse as { "a" : null, "b" : null }, because these are presented as sibling nodes.  This can
					*   only be recognized by indent.  So when we hit a colon, we need to latch the indentation temporarily.
					*/

					int PrevBlockIndent = CurrentBlockIndent;
					// I'm not sure where this is in the spec (i.e. that the colon counts as an indent here), but it makes sense for value parsing and 6KGN above.
					if (CurrentIndent + 1 > CurrentBlockIndent) CurrentBlockIndent = CurrentIndent + 1;		

					// std::cout << "Parsing : at " << GetSource() << "\n";
					auto ColonSource = GetSource();					// Since ParseOneNode() has side-effects, make sure we record location of the colon.
					auto pRet = unique_ptr<YamlNode>(make_unique<YamlKeyValuePair>(ColonSource, IndentAtStart, std::move(pLeft), ParseOneNode()));
					
					CurrentBlockIndent = PrevBlockIndent;

					return pRet;
				}

				break;
			}

			return pLeft;
		}

		/// <summary>
		/// ParseOneNode() handles the last postfix effect not address by ParseOneNodeLevel1() and ParseOneNodeLevel2(),
		/// which is the merging of multiple block mapping nodes that are at the same level of indention.  ParseOneNode()
		/// maintains a single-item buffer or queue that is retained between calls to accomplish this merging process.
		/// </summary>
		inline unique_ptr<YamlNode> YamlParser::ParseOneNode()
		{
			// Note: see also ParseFlowMappings(), which uses the structures here.

			/**
			*	A:
			*		alpha:		1
			*		beta:		2
			*		charlie:	3
			*	B:
			* 
			*   Two mappings should be formed here.  Written in JSON:
			*		{ A : { alpha:1, beta:2, charlie:3 }, B : "" }
			* 
			*	Recursion is necessary if the indent increases.  This is accomplished inside ParseOneNodeLevel2(), however,
			*   as the colon in the A will induce a recursive call to ParseOneNode().  Our job here is only to merge
			*	alpha, beta, and charlie into one node, which will then become the value for A.  Upon return, the new
			*   picture will look like:
			* 
			*	A: value
			*	B:
			* 
			*	Thus, we must parse alpha, beta, and charlie.  Upon parsing B, we will notice the change in indent (equivalently,
			*	any non-key-value-pair content here) and break as we have concluded a single node.
			*/
			
			for (;;)
			{
				auto pSecond = ParseOneNodeLevel2();

				// A nullptr value from ParseOneNodeLevel2() indicates that we need to close the current mapping or block, 
				// or that we've reach EOF (which would also close the mapping).  To know if it's a mapping that needs closure,
				// let's check the indent level.
				if (pSecond == nullptr)
				{
					while (!OpenMappings.empty() && CurrentIndent < OpenMappings.top().Indent)
					{
						auto& mapping_info = OpenMappings.top();
						auto pRet = std::move(mapping_info.pMap);
						OpenMappings.pop();
						return std::move(pRet);
					}

					// If we've gotten here, then either we received a nullptr because of a block and not a mapping, or
					// we've reached EOF.  Blocks are handled higher up, as are EOFs.
					return nullptr;
				}				

				if (pSecond == nullptr || !is_type<YamlKeyValuePair>(pSecond)) return pSecond;				

				// Is there an open mapping to attach this to?  Do we need to make this the new
				// open mapping?

				auto pSecondKVP = dynamic_pointer_movecast<YamlKeyValuePair>(std::move(pSecond));

				// Note: YamlMappings can come up from ParseOneNodeLevel2() as well, but they will be flow style
				// and should not merge with OpenMappings.  However, that only applies to completed flow style
				// entries and not to individual entries within the flow style.  ParseFlowMapping() accounts for
				// this by placing the flow mapping on top of the stack with 0 indent until completion.  We
				// do, however, need to tweak the indent here.
				if (ParsingFlowMappings > 0) pSecondKVP->IndentLevel = 0;

				// Lastly, check if we need to open a new mapping or merge with the top open one.

				if (OpenMappings.empty() || pSecondKVP->IndentLevel > OpenMappings.top().Indent)
				{
					int AtIndent = pSecondKVP->IndentLevel;			// Grab this first as MoveToMap() has side effects.
					OpenMappings.push(OpenMapping(pSecondKVP->MoveToMap(), AtIndent));
					continue;
				}

				// At this point, the new KVP must be at the same level as the open one, and they can be
				// merged.
													
				OpenMappings.top().pMap->Add(std::move(pSecondKVP->pKey), std::move(pSecondKVP->pValue));
			}
		}

		inline void YamlParser::ParseDirective()
		{
			if (Current != '%') throw ArgumentException("Expected % as YAML current character.");

			if (Need(5) && IsNextEqual("YAML"))
			{
				if (!Advance(5)) throw FormatException("Encountered YAML directive without value at " + GetSource() + ".");
				if (Current != ' ' && Current != '\t') throw FormatException("Unrecognized directive at " + GetSource() + ".");
				if (!Advance()) throw FormatException("Encountered YAML directive without value at " + GetSource() + ".");
				AdvanceWhitespace();
				string strVersion = "";
				while ((Current >= '0' && Current <= '9') || Current == '.') {
					strVersion += Current;
					if (!Advance()) break;
				}
				if (strVersion.length() < 3 || strVersion.substr(0, 3) != "1.2") throw FormatException("Only YAML Version 1.2 is supported.");
			}
			else if (Need(4) && IsNextEqual("TAG"))
			{
				if (!Advance(4)) throw FormatException("Encountered YAML directive without value at " + GetSource() + ".");
				if (Current != ' ' && Current != '\t') throw FormatException("Unrecognized directive at " + GetSource() + ".");
				if (!Advance()) throw FormatException("Encountered YAML tag directive without handle at " + GetSource() + ".");
				AdvanceWhitespace();
				string handle = "";
				while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
				{
					handle += Current;					
					if (!Advance()) throw FormatException("Encountered YAML tag directive without value at " + GetSource() + ".");
				}
				AdvanceWhitespace();
				string tag = "";
				while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
				{
					tag += Current;
					if (!Advance()) break;
				}
				if (Tags.count(handle) > 0) throw FormatException("YAML tag directive must only be given at most once per handle in the same document.");
				Tags[handle] = tag;
			}
			else
			{
				// YAML spec indicates an unrecognized directive should be ignored with a warning.  We are not producing any warnings here, so we
				// will just ignore it.
				// throw FormatException("Encountered unrecognized YAML directive at " + GetSource() + ".");
			}

			while (!IsLinebreak(Current)) if (!Advance()) return;
			AdvanceLine();
		}

		inline unique_ptr<YamlNode> YamlParser::ParseTopLevel()
		{
			Tags.clear();
			Aliases.clear();
			while (!OpenMappings.empty()) OpenMappings.pop();		// Clear OpenMappings.
			CurrentBlockIndent = 0;
			ParsingFlowSequences = 0;
			ParsingFlowMappings = 0;

			/** Parse directives, and if we encounter anything then automatically switch to content **/
			
			for (;;)
			{
				if (!Need(1)) return nullptr;

				if (Current == ' ' || Current == '\t')
				{
					if (!Advance()) return make_unique<YamlScalar>(GetSource(), "");
					continue;
				}

				if (Current == '#')
				{
					while (!IsLinebreak(Current)) { if (!Advance()) return make_unique<YamlScalar>(GetSource(), ""); }
					if (!AdvanceLine()) return make_unique<YamlScalar>(GetSource(), "");
					continue;
				}

				if (IsLinebreak(Current)) {
					if (!AdvanceLine()) return make_unique<YamlScalar>(GetSource(), "");
					continue;
				}

				if (Current == '%')
				{
					ParseDirective();
					continue;
				}

				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-')
				{										
					// Three dashes.  Either separates the first document from directives, or signals the start of an additional document
					// (and the end of the current document).
					Advance(3);
					break;
				}

				// Anything else, switch to content.
				break;
			}

			/** Parse content **/

			auto ret = ParseOneNode();

			if (ret == nullptr)
			{
				if (OpenMappings.size() > 1) throw Exception("Unclosed mappings at end of parsing at " + GetSource() + ".");
				if (OpenMappings.size() == 1)
				{
					auto& mapping_info = OpenMappings.top();
					ret = dynamic_pointer_movecast<YamlNode>(std::move(mapping_info.pMap));
					OpenMappings.pop();
				}
			}

			if (Need(3) && Current == '-' && pNext[0] == '-' && pNext[1] == '-')
			{
				// Three dashes.  Either separates the first document from directives, or signals the start of an additional document
				// (and the end of the current document).							
				Advance(3);

				// We've either already had a --- on this document or we have found nodes already.  Thus, this additional --- marker
				// must end the current document.  It wasn't clear to me whether new directives were allowed after the --- marker
				// if there are two documents, but since we can detect node content it seems safest to allow a new directive section
				// for the new document.  Unclear to me if tags and aliases also reset.
				return ret;
			}
			else if (Need(3) && Current == '.' && pNext[0] == '.' && pNext[1] == '.')
			{
				Advance(3);

				// Example 6.18 demonstrates that new directives can follow the terminator sequence, so we need to start a new
				// directive section and then return the content from the now-ended document.  Unclear to me if tags and aliases
				// also reset.
				return ret;
			}
			else if (!Need(1)) return ret;
			else if (!FindNextContent()) return ret;		// If the only content left is whitespace, it's fine to call this terminated.
			else throw Exception("Unknown reason for termination of node parsing at " + GetSource() + ".");
		}

		inline void YamlParser::StartStream(wb::io::Stream& stream, const string& sSourceFilename)
		{
			CurrentSource = sSourceFilename;
			CurrentLineNumber = 1;
			CurrentIndent = 0;

			pCurrentStream = &stream;
			if (Need(3) && Current == 0xEF && pNext[0] == 0xBB && pNext[2] == 0xBF) {
				// UTF-8 BOM.  TODO: Respond to detected encoding.
				Advance(); Advance(); Advance();
			}
		}

		inline void YamlParser::FinishStream()
		{
			CurrentSource = "";
			CurrentLineNumber = 1;
			pCurrentStream = nullptr;
		}

		inline /*static*/ std::unique_ptr<YamlNode> YamlParser::Parse(wb::io::Stream& stream, const string& sSourceFilename)
		{
			YamlParser parser;			
			parser.StartStream(stream, sSourceFilename);
			try
			{
				auto ret = parser.ParseTopLevel();
				parser.FinishStream();
				return ret;
			}
			catch (...)
			{
				parser.FinishStream();
				throw;
			}
		}

		inline /*static*/ unique_ptr<YamlNode> YamlParser::ParseString(const string& str, const string& sSourceFilename)
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

