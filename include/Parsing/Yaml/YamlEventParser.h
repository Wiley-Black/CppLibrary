/////////
//	YamlEventParser.h
////
//	Provides the bottom layer of YAML parsing by converting from a serialization stream into a sequence of events.
////

#ifndef __WBYamlEventParser_h__
#define __WBYamlEventParser_h__

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
		class YamlEventParser : public wb::xml::StreamParser<1024>
		{
			typedef wb::xml::StreamParser<1024> base;

		protected:

			enum class Event
			{
				Stream,
				Document,
				Map,
				Sequence
			};

			enum Flags
			{
				NoFlags = 0x00,
				TextWasSingleQuoted = 0x01,
				TextWasDoubleQuoted = 0x02,
				TextWasLiteralBlock = 0x04,
				TextWasFoldedBlock = 0x08,
				NullValueFlag = 0x10
			};

		private:

			enum ChompingStyle { Strip, Clip, Keep };			

			unordered_map<string, string> Tags;
			unordered_map<string, unique_ptr<YamlNode>>	Aliases;

			int CurrentIndent;				// Indent of the current line.  Calculated in AdvanceLine(), which is always called for a line break.
			int CurrentBlockIndent;			// Indent of current block being parsed, or Int32_MaxValue if none.  Previous values can be captured on function stacks and then restored.						

			enum class Styles
			{
				UnknownBlock,
				BlockSequence,
				BlockMapping,
				FlowSequence,
				FlowMapping
			};
			Styles CurrentStyle;

			/** Helper classes and routines **/
						
			bool IsLinebreak(char ch) { return ch == '\r' || ch == '\n'; }		// Note, YAML 1.2: "Line breaks inside scalar content must be normalized by the YAML processor. Each such line break must be parsed into a single line feed character."						

			// Calls Advance() to pass all whitespace at the current position and returns a count of the whitespace that was skipped over.  CurrentIndent is not updated.
			int AdvanceWhitespace();

			// Call AdvanceLine() while Current is a linebreak character to correctly step over the current and possibly a following linebreak character that should
			// be normalized (combined) into the first linebreak character, as well as indention.  If two linebreaks of the same character (i.e. "\n\n") occur, then 
			// the 2nd one is not advanced over, whereas mixed characters are (i.e. "\r\n" advance twice).  Returns false if Advance() returns false.  Indention
			// is silently counted and the current line's indention is stored in the CurrentIndent member.
			bool AdvanceLine();												
			
			/// <summary>
			/// FindNextContent() is used throughout the YamlParser.  It advances the current position until something
			/// other than whitespace, linebreaks, and comments is found while tracking indent.  Termination of the
			/// document is monitored including end-of-stream, triple dashes, and triple dots.  It is also responsible
			/// for detecting and consuming & and ! entries into the anchor and tag members, respectively.
			/// </summary>
			/// <param name="FindLinebreaks">[Default false] If true, then linebreaks are considered content for 
			/// this search.</param>
			/// <returns>True if the current position was advanced to content.  False if the end of the document was
			/// reached before finding additional content.</returns>
			bool FindNextContent(bool InitiallyCountingIndent, bool FindLinebreaks = false);			

			/** Scalar parsing **/
			static string Chomp(string input, ChompingStyle Style);
			static string ChompFoldList(string FoldList, ChompingStyle Style);
			string ParseScalarContentLiteral();			
			static string Unescape(string input, string Source);
			static string ApplyFlowFolding(string input);
			string ParseScalarContent(int AtIndent, Flags& TextFlags, bool& WasLinebreak, bool IgnoreColon = false);

			string DecodeTag(string tag);
			string DereferenceTag(string shorthand);

			/** Node-Level parsing **/
			string anchor;
			string tag;
			string block_anchor;
			string block_tag;
			bool TryParseAlias();
			void ParseAliasOrScalar(bool IgnoreColon = false);
			bool ColonLookahead();			
			void ParseExplicitBlockMapping();			
			void ParseNode(Styles InStyle, int AtIndent);
			bool ParseUnknownBlock(int AtIndent, bool create_new = false);

			/** Document-level parsing **/
			void ParseDirective();		

			/** Event Helpers **/
			void OnOpenEvent(Event Event, bool FlowStyleOrExplicit = false)
			{				
				if (Event == Event::Document)
				{
					OnOpenEvent(Event, string(), string(), FlowStyleOrExplicit);
				}
				else
				{
					OnOpenEvent(Event, block_anchor, block_tag, FlowStyleOrExplicit);
					block_anchor = block_tag = string();
				}
			}

			void OnValueEvent(string Value, Flags flags = NoFlags)
			{
				OnValueEvent(Value, anchor, tag, flags);
				anchor = tag = string();
			}

			void OnNullValueEvent() {
				OnValueEvent(string(), NullValueFlag);
			}

		protected:
			
			void StartStream(wb::io::Stream& stream, const string& sSourceFilename);
			void ParseTopLevel();
			void FinishStream();						

			virtual void OnOpenEvent(Event Event, string Anchor, string Tag, bool FlowStyleOrExplicit = false) = 0;
			virtual void OnCloseEvent(Event Event, bool Explicit = false) = 0;
			virtual void OnValueEvent(string Value, string Anchor, string Tag, Flags flags = NoFlags) = 0;
			virtual void OnAliasEvent(string AnchorName) = 0;			

		public:

			YamlEventParser();
			~YamlEventParser();
		};
	
		/** YamlEventParser Implementation **/
		
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

		inline YamlEventParser::YamlEventParser()
			: 
			CurrentIndent(0),
			CurrentBlockIndent(0),
			CurrentStyle(Styles::UnknownBlock)
		{
		}

		inline YamlEventParser::~YamlEventParser()
		{
		}						

		inline int YamlEventParser::AdvanceWhitespace()
		{
			if (!Need(1)) return 0;
			int count = 0;
			while (IsWhitespace(Current)) {
				count++;
				if (!Advance()) return count;
			}
			return count;
		}		

		/*static*/ inline string YamlEventParser::Chomp(string input, ChompingStyle Style)
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

		/*static*/ inline string YamlEventParser::ChompFoldList(string FoldList, ChompingStyle Style)
		{
			// Line breaks in 'input' are assumed to have already been normalized.  That is, only \n will appear, no \r.
			// The FoldList can contain '-' for fold candidates and 'n' for empty lines or the final linebreak.
			// The FoldList contains only entries found at the end of content- those subject to chomping.
			if (Style == Keep) return FoldList;						

			switch (Style)
			{
			case Strip: return "";
			case Clip:
				if (FoldList.length() == 0) return "";
				return "n";
			default: throw NotSupportedException("Unrecognized chomping style in YAML parsing.");
			}
		}

		inline bool YamlEventParser::AdvanceLine()
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

		inline string YamlEventParser::ParseScalarContentLiteral()
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
			
			/** Parse until we find the first non-empty text, track longest indentation **/

			// Note: from test case 7T8X, comments are to be excluded from the content.
			// CurrentIndent is set by AdvanceLine(), which moves past ' ' but stops at tabs or '#'.

			int Indentation = ExplicitIndentation;
			string ret;						

			int LongestEmptyIndent = 0;
			while (IsLinebreak(Current))
			{
				//if (CurrentIndent > Indentation && ExplicitIndentation < 0) Indentation = CurrentIndent;
				if (CurrentIndent > LongestEmptyIndent) LongestEmptyIndent = CurrentIndent;
				// This is an empty line.
				ret += '\n';			// Fold does not apply to an empty line.
				if (!AdvanceLine()) return Chomp(ret, ChompStyle);
			}

			/** Parse non-empty line(s) **/
				
			// Current line contains text or content.
				
			// There may not have been any empty lines, in which case we may need to
			// set the indentation now with this line.
			if (Indentation < 0) Indentation = CurrentIndent;

			// Is this line part of the literal block?
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
						FoldsPending += 'n';				// Although in this style a '\n' is appended, we use FoldsPending to manage chomping on the end.
						if (!AdvanceLine()) break;
					}
				}
			}

			FoldsPending = ChompFoldList(FoldsPending, ChompStyle);
			for (auto ii = 0; ii < FoldsPending.length(); ii++)
			{
				if (FoldsPending[ii] == '-') ret += ' ';
				if (FoldsPending[ii] == 'n') ret += '\n';
			}
			return ret;
			//return Chomp(ret, ChompStyle);
		}		

		/*static*/ inline string YamlEventParser::ApplyFlowFolding(string input)
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
				// Note: the first and last line are never empty- they contain the quotation marks.  Therefore the trimming must be adjusted.
				if (FirstLine) next_line = TrimEnd(next_line);
				else if (next_linefeed == string::npos) next_line = TrimStart(next_line);
				else next_line = Trim(next_line);

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

		/*static*/ inline string YamlEventParser::Unescape(string input, string Source)
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
		inline string YamlEventParser::ParseScalarContent(int AtIndent, Flags& TextFlags, bool& WasLinebreak, bool IgnoreColon)
		{
			if (!Need(1)) throw Exception("ParseScalarContent() unable to locate scalar content as Need(1) is false, parsing at " + GetSource() + ".");

			WasLinebreak = false;
			TextFlags = NoFlags;
			bool FirstLinebreak = true;
			string ParsingQuoteStarted = "";
			char ParsingQuote = 0;
			string ret;
			bool EmptyLine = true;			
			//int IndentOfBlock = CurrentIndent;
			int IndentOfBlock = AtIndent;
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
						TextFlags = TextWasSingleQuoted;
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
						TextFlags = TextWasDoubleQuoted;

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

					//if (Current == '#' || IsLinebreak(Current)) continue;						

					/*
					if (!FindNextContent(true, true)) {
						if (ParsingQuote != 0) throw FormatException("Unterminated YAML quotation (" + string(1, ParsingQuote) + ") at " + GetSource() + ".");
						return ret;
					}
					*/

					// Handle specifically the case where the first character is a #.  The case of whitespace-comment is handled
					// separately below.
					if (!ParsingQuote && Current == '#')
					{
						// We're in plain, so skip over the comment and proceed.
						while (!IsLinebreak(Current))
						{
							if (!Advance()) return ret;
						}
					}

					if (IsLinebreak(Current))
					{
						// If FindNextContent() stops on a linebreak having just had an AdvanceLine(), then it means we've found an empty line.						
						continue;
					}

					switch (CurrentStyle)
					{
					case Styles::UnknownBlock:
					case Styles::BlockSequence:
					case Styles::BlockMapping:
						if (CurrentIndent < IndentOfBlock && ParsingQuote == 0) {
							return Trim(ret);
						}
						break;
					case Styles::FlowMapping:					
						if (!ParsingQuote && (Current == ',' || Current == '}')) return Trim(ret);
						break;
					case Styles::FlowSequence:
						if (!ParsingQuote && (Current == ',' || Current == ']')) return Trim(ret);
						break;
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
				if (!IgnoreColon && Current == ':' 
				 && (CurrentStyle == Styles::FlowMapping
					 || !Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": indicator (: ) found.");		// Should have been detected & handled at higher-level.
					return ret;		// The ": " combination is an indicator, but what precedes it constitutes the scalar content.
				}
				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
				 && (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2])))
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
						TextFlags = Current == '|' ? TextWasLiteralBlock : TextWasFoldedBlock;
						WasLinebreak = true;
						return ParseScalarContentLiteral();
					}
				}
				if (CurrentStyle == Styles::FlowMapping && (Current == ',' || Current == '}'))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": unexpected indicator (" + string(1,Current) + ") found in flow mapping.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				if (CurrentStyle == Styles::FlowSequence && (Current == ',' || Current == ']'))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": unexpected indicator (" + string(1, Current) + ") found in flow sequence.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				if (ret.empty() && (Current == '[' || Current == ']' || Current == '{' || Current == '}' || Current == ','))
				{
					if (ret.empty()) throw Exception("Unable to parse scalar content at " + GetSource() + ": unexpected indicator (" + string(1, Current) + ") found.");		// Should have been detected & handled at higher-level.
					return ret;
				}
				ret += Current;
				EmptyLine = false;
				if (!Advance()) return ret;
			}
		}
		
		inline string YamlEventParser::DecodeTag(string tag)
		{
			// I'm not exactly sure if DecodeTag() belongs here or in the receiver of the events.

			string ret;
			for (auto ii = 0; ii < tag.length(); )
			{
				if (tag[ii] == '%')
				{					
					ii++;
					int hexcode = 0;
					if (ii < tag.length())
					{
						char ch = toupper(tag[ii]);
						if (ch >= '0' && ch <= '9') hexcode += ch - '0';
						else if (ch >= 'A' && ch <= 'F') hexcode += ch - 'A' + 10;
						else throw FormatException("Illegal character in tag " + tag + " following escape % character at " + GetSource() + ".");
						ii++;
					}
					if (ii < tag.length())
					{
						hexcode *= 0x10;
						char ch = toupper(tag[ii]);
						if (ch >= '0' && ch <= '9') hexcode += (ch - '0');
						else if (ch >= 'A' && ch <= 'F') hexcode += (ch - 'A' + 10);
						else throw FormatException("Illegal character in tag " + tag + " following escape % character at " + GetSource() + ".");						
						ii++;
					}
					ret += (char)hexcode;
				}
				else {
					ret += tag[ii];
					ii++;
				}
			}
			return ret;
		}

		inline string YamlEventParser::DereferenceTag(string shorthand)
		{
			// From test case 7FWL and example 6.24: Verbatim Tags.  Surrounded by <> and
			// not subject to tag resolution.
			// !<tag:yaml.org,2002:str> translates to a tag of <tag:yaml.org,2002:str>.
			if (shorthand[0] == '<' && shorthand[shorthand.length() - 1] == '>')
				return DecodeTag(shorthand);

			// shorthand excludes the initial !, but Tags (the map we're comparing against) contains it.
			// Also, only part of the shorthand is going to match to the tag listing and then we append the rest.
			// I believe that the tags listing has to terminate with a !, though I'm not sure that's a strict rule.
			auto close = shorthand.find('!');
			string match_prefix = "";
			if (close != string::npos && close > 0)
			{
				match_prefix = shorthand.substr(0, close + 1);
				auto it = Tags.find("!" + match_prefix);
				if (it != Tags.end()) {
					string prefix = it->second;
					string after = shorthand.substr(match_prefix.length());
					return DecodeTag("<" + prefix + after + ">");
				}
			}			

			// Handle the case such as "!!foo", which here shows up as just one !:
			if (shorthand.length() > 1 && shorthand[0] == '!')
			{
				return DecodeTag("<tag:yaml.org,2002:" + shorthand.substr(1) + ">");
			}

			auto it_default = Tags.find("!");
			if (it_default != Tags.end())
			{
				// The default (!) tag has been redefined, so use the redefined version.				
				string prefix = it_default->second;
				return DecodeTag("<" + prefix + shorthand + ">");
			}

			// Go with the default.
			return DecodeTag("<!" + shorthand + ">");
		}

		/// <summary>
		/// FindNextContent() advances from the current position until something other than whitespace or a comment is found.  CurrentIndent
		/// is updated during the search, in case content is found on the current line.  False is returned if Advance() or AdvanceLine()
		/// returns false.  If FindLinebreaks is true, then a linebreak is considered relevant and FindNextContent() returns at the linebreak.
		/// </summary>
		inline bool YamlEventParser::FindNextContent(bool InitiallyCountingIndent, bool FindLineBreaks /*= false*/)
		{
			if (!Need(1)) return false;
			bool CountingIndent = InitiallyCountingIndent;

			for (;;)
			{
				if (Current == ' ' || Current == '\t')
				{
					if (!Advance()) return false;
					if (CountingIndent) CurrentIndent++;
					continue;
				}

				if (Current == '#')
				{
					if (tag.length()) block_tag = tag;
					if (anchor.length()) block_anchor = anchor;
					tag = anchor = string();
					while (!IsLinebreak(Current)) { if (!Advance()) return false; }
					if (FindLineBreaks) return true;
					if (!AdvanceLine()) return false;
					CountingIndent = true;
					continue;
				}

				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
				 && (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2]))) return false;
				if (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.') return false;

				if (IsLinebreak(Current)) {
					if (tag.length()) block_tag = tag;
					if (anchor.length()) block_anchor = anchor;
					tag = anchor = string();
					if (FindLineBreaks) return true;
					if (!AdvanceLine()) return false;
					CountingIndent = true;
					continue;
				}

				if (Current == '!')
				{
					if (!Advance()) return false;
					if (tag.length()) throw FormatException("Tag already detected at " + GetSource() + ".");
					string handle;
					while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
					{
						handle += Current;
						if (!Advance())
						{
							tag = DereferenceTag(handle);
							return false;
						}
					}
					tag = DereferenceTag(handle);

					// Don't count the !tag or whitespace after it as indent.  But a newline would restart the counter.
					CountingIndent = false;
					continue;
				}

				if (Current == '&')
				{
					if (!Advance()) return false;
					if (anchor.length()) throw FormatException("Anchor already detected at " + GetSource() + ".");
					anchor = string();
					while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
					{
						anchor += Current;
						if (!Advance()) return false;						
					}

					// Don't count the &anchor or whitespace after it as indent.  But a newline would restart the counter.
					CountingIndent = false;
					continue;
				}

				return true;
			}
		}
		
		inline bool YamlEventParser::TryParseAlias()
		{
			if (Need(1) && Current == '*')
			{
				if (!Advance()) throw FormatException("Alias marker (*) without handle specified at " + GetSource() + ".");
				string handle;
				while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
				{
					handle += Current;
					if (!Advance())
					{
						OnAliasEvent(handle);
						return true;
					}
				}
				OnAliasEvent(handle);
				while (Current != ' ' && Current != '\t' && !IsLinebreak(Current))
				{
					if (!Advance()) return true;
				}
				return true;
			}

			return false;
		}

		inline bool YamlEventParser::ColonLookahead()
		{
			// ColonLookahead() is only called at the start of content on a line, and the content is not a - or ? symbol.
			// Therefore, we can assume that the character preceding Current qualifies as whitespace.

			char ParsingQuote = 0;		
			bool ParsingAnchor = false;
			bool ParsingTag = false;
			if (!Need(1)) return false;
			if (Current == ':' && (CurrentStyle == Styles::FlowMapping || !Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) return true;
			else if (CurrentStyle == Styles::FlowSequence && (Current == ',' || Current == ']')) return false;
			else if (Current == '\'') ParsingQuote = '\'';
			else if (Current == '\"') ParsingQuote = '\"';
			else if (Current == '|') return false;
			else if (Current == '>') return false;
			else if (Current == '#') return false;
			else if (Current == '!') ParsingTag = true;
			else if (Current == '&' || Current == '*') ParsingAnchor = true;
			else if (IsLinebreak(Current)) return false;
			else if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
				&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2]))) return false;
			else if (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.') return false;			
			
			char Prev3 = 0;
			char Prev2 = 0;
			char Prev = 0;
			char Next = Current;
			bool Escaped = false;
			for (int ii = 1; ii < GetMaxLoading() - 2; ii++)
			{
				if (!Need(ii + 1)) return false;
								
				Prev3 = Prev2;
				Prev2 = Prev;
				Prev = Next;
				Next = pNext[ii - 1];
				if (ParsingQuote == '\'')
				{
					if (Escaped)
					{
						Escaped = false;
						if (Next == '\'')
						{
							// Two apostrophes in a row is an escaped sequence here.
							continue;
						}
						// The previous character was a potential "escape" character,
						// which is the apostrophe.  But, it was not followed by a second
						// apostrophe, so it was not actually an escape sequence when we 
						// saw it as Next on the last loop.  We end the quotation and also
						// still need to process our current Next character, since that
						// turned out to not be escaped/part of the quote.						
						ParsingQuote = 0;
						// Do not stop here because we've acted on Prev and not Next.
					}
					else if (Next == '\'') 
					{
						Escaped = true;
						continue;
					}
					else continue;
				}

				if (ParsingQuote == '\"')
				{
					if (Escaped)
					{
						Escaped = false;
						continue;
					}
					if (Next == '\\')
					{
						Escaped = true;
						continue;
					}					
					if (Next == '\"') {						
						ParsingQuote = 0;
						continue;
					}
					continue;
				}

				// 2SXE: A colon is a valid character within an anchor name, so ColonLookahead needs to ignore the anchor name until whitespace.
				if (ParsingAnchor || ParsingTag)
				{
					if (Next == ' ' || Next == '\t' || IsLinebreak(Next)) ParsingAnchor = ParsingTag = false;
					else continue;
				}

				if (Prev == ' ' || Prev == '\t')
				{
					// Test all cases that require whitespace preceding it...
					if (Next == '\'') ParsingQuote = '\'';					
					else if (Next == '\"') ParsingQuote = '\"';
					else if (Next == '#') return false;
					else if (Next == '!') ParsingTag = true;
					else if (Next == '&' || Next == '*') ParsingAnchor = true;
					else if (Next == '|') return false;
					else if (Next == '>') return false;
				}
				
				if (Next == ':' && (CurrentStyle == Styles::FlowMapping || !Need(ii + 2) || pNext[ii] == ' ' || pNext[ii] == '\t' || IsLinebreak(pNext[ii]))) return true;
				// If we were only parsing one token here, quoting would be disallowed as we've already seen that the first character
				// is not a quote.  But, whitespace is possible, and multiple tokens are possible.				
				else if (IsLinebreak(Next)) return false;
				else if (Prev == '-' && Prev2 == '-' && Prev3 == '-'
					&& (Next == ' ' || Next == '\t' || IsLinebreak(Next))) return false;
				else if (Next == '.' && Prev == '.' && Prev2 == '.') return false;
				else if ((CurrentStyle == Styles::FlowMapping || CurrentStyle == Styles::FlowSequence) && Next == ',') return false;
			}
			return false;
		}		

		inline void YamlEventParser::ParseAliasOrScalar(bool IgnoreColon)
		{
			if (TryParseAlias()) return;
			Flags Flags;
			bool WasLinebreak;
			string Text = ParseScalarContent(CurrentBlockIndent, Flags, WasLinebreak, IgnoreColon);
			if (Flags == NoFlags) Text = Trim(Text);		// NoFlags = "plain" in this case.

bool hit = false;
if (Text == "top6")
	hit = true;

			OnValueEvent(Text, Flags);
		}

		inline void YamlEventParser::ParseExplicitBlockMapping()
		{
			if (Current != '?' || !(!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
				throw Exception("Expected ParseExplicitBlockMapping() to be initiated on the ? indicator.");

			/** The 5WE3 test case is described in the spec as Example 8.17 and it is pointed out a rule:
				If the “ ? ” indicator is specified, the optional value node must be specified on a separate line, denoted
				by the “ : ” indicator.  Note that YAML allows here the same compact in-line notation described above for block sequence entries.
				Also related is the 36F6 test case, which would otherwise flow down the same logic if not for explicit handling of the ? case.
				This could be handled by defining a style specifically for explicit ? syntax (ExplicitBlockMapping), but the change in rules is
				small enough to be handled by the following.
			**/

			int QuestionMarkIndent = CurrentIndent;						
			Advance(); CurrentIndent++;
			// A couple of ways to go here: set the ParentIndent equal to the ? plus one.  Or advance the whitespace
			// and then define the parent indent.  Or maybe even it's correct to go with the ? minus one, since it
			// is the parent indent it wants...  ParseUnknownBlock() will return when it hits a sibling level.
			//CurrentIndent += AdvanceWhitespace()			
			for (;;)
			{
				if (!ParseUnknownBlock(QuestionMarkIndent + 1))
				{
					OnNullValueEvent();						// Null key.
					OnNullValueEvent();						// Null value.
					break;
				}
				if (!Need(1))
				{
					OnNullValueEvent();						// Null value.
					break;
				}
				if (CurrentIndent < QuestionMarkIndent)
				{
					OnNullValueEvent();						// Null value.
					break;
				}				
				if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
				{
					OnNullValueEvent();						// Null value.
					Advance(); CurrentIndent++;
					continue;
				}
				else if (Current == ':' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
				{
					Advance(); CurrentIndent++;
					if (!ParseUnknownBlock(QuestionMarkIndent + 1))
					{
						OnNullValueEvent();						// Null value.
						break;
					}

					if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						// Continue the explicit BlockMapping, same indent, no need to open or close.
						Advance(); CurrentIndent++;
						continue;
					}
					else break;
				}
				else {
					/** Proceed as if we are in an ordinary block mapping, which allows additional key:value pairs
					*	to become attached to this mapping.  Test case 7W2P:
					*
					*	? a
					*	? b
					*	c:
					*/
					OnNullValueEvent();						// Null value.
					ParseNode(Styles::BlockMapping, QuestionMarkIndent);
					return;
				}
			}
		}

		inline void YamlEventParser::ParseNode(Styles InStyle, int AtIndent)
		{
			if (InStyle == Styles::UnknownBlock) throw Exception("Expected UnknownBlock style to be handled exclusively by ParseUnknownBlock().");

			/** Most useful examples:
			* 
			*	key:
			*   - value_a
			*   - value_b
			* 
			*   This above example highlights how the sequence is considered indented relative to the key: by 1.  The colon causes an indent +1, and
			*   the dashes are perceived by readers as being indented, so YAML specifies that this is indented.
			* 
			*	This one presents the challenge that the dash character implies special indenting.  After parsing "key: ", ParseNode(Mapping) will call
			*	ParseUnknownBlock() with an indent of 1, due to the colon/value of a map parsing.  The dashes need to be recognized or handled as an
			*   indent level of 1 or 2.  
			*	
			*	ParseUnknownBlock() begins and finds the colon lookahead, calls ParseNode(BlockMapping, AtIndent=0).
			*		In ParseNode(BlockMapping, 0) the scalar "key" is parsed then the colon.  After the colon, ParseUnknownBlock(AtIndent=1).
			*			ParseUnknownBlock recognizes a sequence at indent 1.
			* 
			* 
			*	Test case '6BCT':
			*
			*	- foo:	 bar
			*	- - baz1
			*	  - baz2
			* 
			*	ParseUnknownBlock() begins and finds the dash, +SEQ then calls ParseNode(BlockSequence, AtIndent=1).  
			*		In ParseNode(BlockSequence, 1), a ParseUnknownBlock(AtIndent=2) call is made.
			*			ParseUnknownBlock will find the colon and initiate ParseNode(BlockMapping, AtIndent=2).
			*				ParseNode() will parse the scalar key "foo" and colon, then initiate ParseUnknownBlock(AtIndent=3).
			*					ParseUnknownBlock() will recognize bar as a scalar and parse it, then return.
			*				After the key: value, ParseNode(BlockMapping, AtIndent=2) loops.  
			*				Hits the 2nd line dash with CurrentIndent=0, effective 1, returns.
			*			returns.
			*		After the "foo:bar" block, ParseNode() loops.  
			*		ParseNode() is at the 2nd line 1st dash, a ParseUnknownBlock(AtIndent=1) call is made.
			*			ParseUnknownBlock finds the 2nd line 2nd dash, +SEQ, and initiates ParseNode(BlockSequence, AtIndent=3).
			*				ParseNode() sees current indent of 2 effective 3 and a ParseUnknownBlock(AtIndent=3) call is made.
			*					ParseUnknownBlock returns the "baz1" scalar.
			*				After the baz1 block has been parsed, ParseNode() loops.
			*				ParseNode() sees the 3rd line dash at a current indent of 2 effective 3 and a ParseUnknownBlock(AtIndent=3) call is made.
			*					ParseUnknownBlock returns the "baz2" scalar.						
			*/

			/** AtIndent should correspond to where the block's indent actually is.  Some examples:
			* 
			*	- hello
			* 
			*	ParseUnknownBlock() hits the "- " and calls ParseNode() with an indent of 0.  ParseNode() here will then advance over the "- "
			*   and in BlockSequence processing will ParseUnknownBlock() with an indent of 2, which will find "hello" as a scalar.
			* 
			*	- !!map
			*	  key: value
			* 
			*	ParseUnknownBlock() this the "- " and calls ParseNode() with an indent of 0.  ParseNode() here will then advance over the "- "
			*	and in BlockSequence processing will ParseUnknownBlock() with an indent of 2.  That will FindNextContent() of key: value, a mapping,
			*	and call ParseNode() in the BlockMapping style with an indent of 2.  The ParseNode() will be in the BlockMapping style and so it will
			*	parse the key as a scalar and then ParseUnknownBlock() on the value with an indent of 3, which will also be a scalar.
			* 
			*	- !!map
			*	  key: 
			*	   - a
			*      - b
			* 
			*   Similar to the above, ParseUnknownBlock() will be called after the colon with an indent of 3.  Then a sequence will be found at indent
			*   of 3.  ParseUnknownBlock will be called on "a" at an indent of 5 but because the "- " on the next line is at indent 3 it will return
			*	and then ParseUnknownBlock will get called again on "b" similarly.					
			* 
			* 	-
			*	  "flow in block"
			*	- >
			*	 Block scalar
			*	- !!map # Block collection
			*	  foo : bar
			* 
			*	The above (Example 8.22) highlights how the indent after the dash must be at least one so that "flow in block" and foo:bar 
			*	are recognized as blocks within the sequence.  An indent of 2 would also work in this example.
			* 
			*	-
			*	  name: Mark McGwire
			*	  hr:   65
			*	  avg:  0.278
			*	-
			*	  name: Sammy Sosa
			*	  hr:   63
			*	  avg:  0.288
			* 
			*	The above test case (229Q) shows that when parsing the value of the sequence entries the indent must be +1 or +2, otherwise the 2nd dash
			*	character would not be sufficient to break the mapping out.  An alternative approach might be to always break out of a mapping when a dash
			*	is encountered at a sibling-or-lower level, but that seems not the intention here and will probably fail other test cases.					
			* 
			*	Summarizing the rules:
			*	 -	AtIndent shall always refer to the effective column position (0-indexed) of the character starting the block.  That is, AtIndent 
			*		shall match the start position of a "key: value", but would be +1 from the actual column of a dash or ? character.
			*	 -  Blocks within a sequence will begin with an indent +1 from the dash.
			*	 -	Blocks after a colon will begin with an indent +1 relative to the key: block's indent.
			*	 -  When testing for indent level on top of a dash or ? character, add plus one to current indent.
			* 
			*	 -  ParseUnknownBlock() calls ParseNode(BlockSequence) with the AtIndent being assigned as the effective indent not current, even
			*		though the "-" has not been advanced over yet.  ParseNode() must use the effective to match, so it also +1's before advancing.
			*	 -  Same indentation rules for ? as for - should apply I believe, as the ? will be perceived as indent the same way - is.
			*/			

			for (;;)
			{
				// TODO: would be better to pass these down toward ParseScalarContent() and such, but not doing that atm.
				CurrentStyle = InStyle;
				CurrentBlockIndent = AtIndent;

				if (!FindNextContent(true)) return;

				// Detect end of Yaml document marker...
				if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
					&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2]))) break;

				// Detect block sequence (8.2.1) indicator...
				// Detect start of new Yaml document marker / end of directives, which in this context would imply the end of the current document...
				if (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.') break;								
				
				// Check indent levels if we're in block mode...
				if (InStyle == Styles::BlockSequence || InStyle == Styles::BlockMapping)
				{					
					// Count the whitespace that follows the current character.
					int WhitespaceAfter = 0;
					if (Current == '-' || Current == '?')
					{
						for (int ii = 0; ii < GetMaxLoading() - 2; ii++)
						{
							if (!Need(ii + 2)) break;
							if (pNext[ii] == ' ' || pNext[ii] == '\t') WhitespaceAfter++; 
							//else if (IsLinebreak(pNext[ii])) { WhitespaceAfter++; break;}
							else break;
						}
					}					

					if (Current == '-' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						// See indent discussion at start of this function.

						if (CurrentIndent + 1 < AtIndent)						
						{
							// Note: we do not advance in this case, caller must reduce CurrentBlockIndent to proceed.							
							return;
						}
						else if (CurrentIndent + 1 > AtIndent)						
						{
							Advance(); CurrentIndent++;
							int BlockAtIndent = CurrentIndent + 1;

							OnOpenEvent(Event::Sequence, false);
							ParseNode(Styles::BlockSequence, BlockAtIndent);
							OnCloseEvent(Event::Sequence);
							continue;
						}
						else
						{
							// Block mappings have a different indentation metric than block sequences do, so if we hit a - with an equal
							// indent to the mapping *after* the dash itself has been accounted for (as this calculation has done), then the dash 
							// is actually to the left of the mapping.  Consider test case '6BCT':
							/*
							*	- foo:	 bar
							*	- - baz
							*	  - baz
							*/
							//if (InStyle == Styles::BlockMapping) return;

							// We're at a sibling node/level and should continue/append to the block sequence as-is.
							Advance(); CurrentIndent++;
							FindNextContent(true, true);
							//continue;							
						}
					}
					else if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						if (CurrentIndent + 1 < AtIndent)
						{
							// Note: we do not advance in this case, caller must reduce CurrentBlockIndent to proceed.							
							return;
						}
						else if (CurrentIndent + 1 > AtIndent)
						{							
							OnOpenEvent(Event::Map, false);
							ParseExplicitBlockMapping();
							OnCloseEvent(Event::Map);
							continue;
						}
						else
						{
							// We're at a sibling node/level and should continue/append to the block mapping as-is.
							Advance(); CurrentIndent++;
							FindNextContent(true, true);
							//continue;
						}
					}
					else
					{
						if (CurrentIndent < AtIndent)
						{
							// Note: we do not advance in this case, caller must reduce AtIndent to proceed.							
							return;
						}
					}
				}

				if (Current == '{')
				{
					Advance();
					OnOpenEvent(Event::Map, true);
					ParseNode(Styles::FlowMapping, CurrentIndent + 1);
					OnCloseEvent(Event::Map);

					if (InStyle == Styles::FlowMapping || InStyle == Styles::FlowSequence)
					{
						if (Current == ',') { Advance(); continue; }
					}
					continue;
				}

				if (Current == '[')
				{
					Advance();
					OnOpenEvent(Event::Sequence, true);
					ParseNode(Styles::FlowSequence, CurrentIndent + 1);
					OnCloseEvent(Event::Sequence);

					if (InStyle == Styles::FlowMapping || InStyle == Styles::FlowSequence)
					{
						if (Current == ',') { Advance(); continue; }
					}
					continue;
				}

				// Handle as a prefix- which indicates a null key.
				if (Current == ':' && (CurrentStyle == Styles::FlowMapping || !Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
				{
					Advance();
					OnNullValueEvent();						// Null key.
					continue;
				}
				
#if 0
				if (InStyle != Styles::BlockMapping && InStyle != Styles::FlowMapping)				
				{
					// Check if what we're about to parse is part of a key:value pair.  If yes and we're not currently 
					// parsing a mapping, then we have detected a mapping implicitly and need to initiate a mapping 
					// before we parse the key.
					if (ColonLookahead())
					{
						OnOpenEvent(Event::Map, false);												
						// I'm a little unsure what the indent should be here.  We are perhaps starting a new block here,
						// and it's a transition from whatever style we're in to a BlockMapping style.  Though in a flow
						// style the indent isn't supposed to matter at all.  I may need a special case for that as there
						// is an allowance for a flow within a sequence, but you wouldn't have a closing } to end it.
						//if (InStyle == Styles::FlowSequence)
							//ParseNode(Styles::BlockMapping, CurrentIndent);
						//else
						ParseNode(Styles::BlockMapping, CurrentIndent);
						OnCloseEvent(Event::Map);
						continue;
					}
				}
#endif

				//string StartSource = GetSource();
				//bool WasPlain = false;
				//bool WasLinebreak = false;
				//string Text;
				switch (InStyle)
				{				
				case Styles::BlockSequence:				
					if (IsLinebreak(Current))
					{
						if (!AdvanceLine())
						{
							OnNullValueEvent();						// Null value.
							return;
						}
					
						if (!ParseUnknownBlock(AtIndent, true))
						{
							OnNullValueEvent();						// Null value.
							break;
						}
						continue;
					}

					// If the lack of a +1 or +2 on AtIndent seems sus, consider that ParseNode(BlockMapping) gets
					// called with the effective AtIndent to begin with.  So if the document begins with "- ",
					// it goes ParseTopLevel() -> ParseUnknownBlock(0) -> ParseNode(BlockMapping, 1) -> advance() over '-' -> here.
					if (!ParseUnknownBlock(AtIndent, true))
					{
						OnNullValueEvent();						// Null value.
						break;
					}
					continue;

				case Styles::BlockMapping:
				{
					ParseAliasOrScalar();

					// After the key but before the colon, find the colon if there is one.  Pay attention to whether
					// we hit a linefeed, which would start a block.
					for (;;)
					{
						if (Current == ' ' || Current == '\t')
						{
							if (!Advance())
							{
								OnNullValueEvent();						// Null value.
								return;
							}
							continue;
						}

						if (IsLinebreak(Current))
						{
							if (!AdvanceLine())
							{
								OnNullValueEvent();						// Null value.
								return;
							}

							if (!FindNextContent(true))
							{
								OnNullValueEvent();						// Null value.
								return;
							}

							if (CurrentIndent + 1 < AtIndent)
							{
								OnNullValueEvent();						// Null value.
								// We have already advanced, at least over the key.
								// Note: we do not advance further in this case, caller must reduce CurrentBlockIndent to proceed.							
								return;
							}
							else if (CurrentIndent + 1 > AtIndent)
							{
								throw FormatException("Indented content before expected colon illegal at " + GetSource());
							}
							// else at sibling node/level.
						}

						// We're either on a newline at a sibling node/level and should continue/append to the block sequence as-is,
						// or we're still on the same line.
						// We're still searching for the colon.  There might be one if we did a ?: block.  Otherwise, this looks 
						// like an error.
						if (Current == ':' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) break;
						if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) break;
						if (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
							&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2]))) break;
						if (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.') break;
						throw FormatException("Expected colon at " + GetSource());
					}

					// If we found a post-fix colon...
					if (Current == ':' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						Advance();
						CurrentIndent++;			// I'm not sure this is in the spec, but treating the : as an indent seems to work.

						if (!ParseUnknownBlock(AtIndent + 1))
						{
							OnNullValueEvent();						// Null value.
							break;
						}

						continue;
					}
					else 
					{
						// Do not advance, but need a null value entry.
						OnNullValueEvent();						// Null value.
					}
					continue;
				}

				case Styles::FlowMapping:
					if (Current == '}')
					{
						Advance();
						return;
					}
									
					ParseAliasOrScalar();

					if (!FindNextContent(false)) throw FormatException("Unterminated flow mapping at " + GetSource() + ".");

					// Handle as a post-fix
					if (Current == ':')		// Note: the usual requirement for a : to be followed by a space is N/A to flow mappings, see test case 5MUD.
					{
						Advance();
						
						if (!FindNextContent(false)) throw FormatException("Unterminated flow mapping at " + GetSource() + ".");
						ParseAliasOrScalar(true);
					}
					else //if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
					{
						// Do not advance, but need a null value entry.
						OnNullValueEvent();						// Null value.
					}

					if (Current == ',')
					{
						Advance();
						continue;
					}

					continue;

				case Styles::FlowSequence:
					if (Current == ']')
					{						
						Advance();
						return;
					}					
					
					// See spec section 7.4.1 / Examples 7.14, 7.19, 7.20, 7.21, and 7.22.
					if (ColonLookahead())
					{
						OnOpenEvent(Event::Map, true);
						// I may need to define a new style of "SinglePairMapping" here, since FlowMapping would search
						// for a closing }.  I probably also have to handle the explicit case of a ?.												
						// ParseNode(Styles::BlockMapping, CurrentIndent);

						ParseAliasOrScalar();						

						if (!FindNextContent(false)) throw FormatException("Unterminated flow sequence at " + GetSource() + ".");

						// Handle as a post-fix
						if (Current == ':')		// Note: the usual requirement for a : to be followed by a space is N/A to flow mappings, see test case 5MUD.
						{
							Advance();

							if (!FindNextContent(false)) throw FormatException("Unterminated flow sequence at " + GetSource() + ".");
							ParseAliasOrScalar(true);
						}
						else 
						{
							// Do not advance, but need a null value entry.
							OnNullValueEvent();						// Null value.
						}

						OnCloseEvent(Event::Map);						
					}
					else ParseAliasOrScalar();

					if (Current == ',')
					{
						Advance();
						continue;
					}
					continue;

				default:
					throw Exception("Unrecognized style at " + GetSource());
				}
			}
		}

		inline bool YamlEventParser::ParseUnknownBlock(int AtIndent, bool create_new)
		{
			/** If I understand correctly, I believe these examples explain tagging.  See also 74H7 test case and 57H4 below.
			*	To tag a block:
			* 
			*	example: !!seq
			*	- a
			*	- b
			* 
			*	example: !!map
			*	 foo: bar
			* 
			*	To tag scalars:
			* 
			*	!!str a: b
			*	!!str 23: !!bool false
			*/

			/*	Test case '57H4', tagging blocks.
			*
			*	sequence: !!seq
			*	- entry
			*	- !!seq
			*    - nested
			*	mapping: !!map
			*	 foo: bar
			*
			*	ParseUnknownBlock() begins and finds colon lookahead, +MAP and calls ParseNode(BlockMapping, AtIndent=0).
			*		In ParseNode(BlockMapping, 0), parses the key, colon, then calls ParseUnknownBlock(AtIndent=1).
			*			ParseUnknownBlock detects a sequence, +SEQ, calls ParseNode(BlockSequence, AtIndent=1).
			*				ParseNode() parse the value via ParseUnknownBlock(AtIndent=1).
			*					ParseUnknownBlock() parses "entry" as scalar.
			*				ParseNode() loops and is on the 2nd dash (3rd line).
			*				ParseNode() has a dash at current indent 0, effective 1=AtIdent, so parses the value with ParseUnknownBlock(AtIndent=1).
			*					ParseUnknownBlock() parses "!!seq" and detects the indented (nested) dash, calls ParseNode(BlockSequence, AtIndent=2).
			*						ParseNode(2) has current indent 1, effective 2=AtIndent, so parses the value with ParseUnknownBlock(AtIndent=2).
			*							ParseUnknownBlock() returns "nested" as scalar.
			*						ParseNode(2) encounters "mapping" at an indent of 0 < 2 and returns.
			*					returns.
			*				ParseNode(1) loops and encounters "mapping" at an indent of 0 < 1 and returns.
			*			returns.
			*		ParseNode(0) parses the key and after the colon calls ParseUnknownBlock(AtIndent=1).
			*			ParseUnknownBlock detects "foo:", +MAP, and calls ParseNode(BlockMapping, AtIndent=1).
			*				ParseNode() parses "foo" key and colon, calls ParseUnknownBlock(AtIndent=2).
			*					ParseUnknownBlock() return scalar "bar".
			*/									

			/** To accomodate these tag rules, I've setup FindNextContent() to track regular and block-level tags and anchors.  Detection of a linebreak
			*	will shift from the regular to the block-level.  Block-level will only be consumed by OnOpenEvent() for a DOC, SEQ, or MAP.
			*/

			// Detect end of stream (now allowing linefeeds), end of Yaml document marker, or document marker...
			if (!FindNextContent(true)
				|| (Current == '-' && Need(3) && pNext[0] == '-' && pNext[1] == '-'
					&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2])))
				|| (Current == '.' && Need(3) && pNext[0] == '.' && pNext[1] == '.'))
			{
				return false;
			}

			if (Current == '{')
			{
				Advance();
				OnOpenEvent(Event::Map, block_anchor, block_tag, true);
				ParseNode(Styles::FlowMapping, AtIndent + 1);
				OnCloseEvent(Event::Map);
				return true;
			}

			if (Current == '[')
			{
				Advance();
				OnOpenEvent(Event::Sequence, block_anchor, block_tag, true);
				ParseNode(Styles::FlowSequence, AtIndent + 1);
				OnCloseEvent(Event::Sequence);
				return true;
			}

			// Because we are not advancing, we are not incrementing CurrentIndent to count the '-' character itself, however this does
			// count against the indent.  To predict whether we need to continue or start a new block, we need to include the +1 here.
			// 
			// Also see Example 8.22 in 8.2.3. Block Nodes, where a special exception/adjustment to the rule regarding block node indent is described.

			int WhitespaceAfter = 0;
			for (int ii = 0; ii < GetMaxLoading() - 2; ii++)
			{
				if (!Need(ii + 2)) break;
				if (pNext[ii] == ' ' || pNext[ii] == '\t') WhitespaceAfter++; else break;
			}

			int IndentOffset = 0;			
			// While I sometimes check for FlowMapping style when hitting a colon, we are parsing an unknown *block* so it is N/A here.
			if (Current == ':' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) IndentOffset = 1; //IndentOffset++;
			else if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) IndentOffset = 1; //IndentOffset += WhitespaceAfter + 1;
			else if (Current == '-' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0]))) IndentOffset = 1; //IndentOffset += WhitespaceAfter + 1;
			//IndentOffset = 0;

			// Because we are not advancing, we are not incrementing CurrentIndent to count the '-' character itself, however this does
			// count against the indent.  To predict whether we need to continue or start a new block, we need to include the +1 here.
			// 
			// Also see Example 8.22 in 8.2.3. Block Nodes, where a special exception/adjustment to the rule regarding block node indent is described.
			if (CurrentIndent + IndentOffset < AtIndent)
			{
				// Note: we do not advance in this case, caller must reduce AtIndent to proceed.							
				// Also, tag/anchor would apply to the null node that follows since there is no following block.
				if (anchor.length() == 0) anchor = block_anchor;
				if (tag.length() == 0) tag = block_tag;
				block_anchor = block_tag = string();
				return false;
			}			
			else if (CurrentIndent + IndentOffset > AtIndent)
			{
				// Without advancing, we allow the indent to be redefined to be further to the right for this new block.
				//return ParseUnknownBlock(CurrentIndent + IndentOffset, create_new);
				//AtIndent = CurrentIndent + IndentOffset;
			}
						
			// Handle as a prefix- which indicates a null key.
			if (Current == ':' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
			{
				// Do not advance: BlockMapping parsing will need to see the : to know there's a null key.
				//OnOpenEvent(Event::Map, block_anchor, block_tag, false);
				OnOpenEvent(Event::Map, false);
				//ParseNode(Styles::BlockMapping, AtIndent);
				ParseNode(Styles::BlockMapping, CurrentIndent);
				OnCloseEvent(Event::Map);
				return true;
			}
				
			if (Current == '-' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
			{				
				//OnOpenEvent(Event::Sequence, block_anchor, block_tag, false);
				OnOpenEvent(Event::Sequence, false);
				//Advance(); CurrentIndent++; IndentOffset--;
				//Advance(); CurrentIndent++;
				//ParseNode(Styles::BlockSequence, AtIndent);
				ParseNode(Styles::BlockSequence, CurrentIndent + 1);
				OnCloseEvent(Event::Sequence);
				return true;
			}

			if (Current == '?' && (!Need(2) || pNext[0] == ' ' || pNext[0] == '\t' || IsLinebreak(pNext[0])))
			{				
				/*
				OnOpenEvent(Event::Map, block_anchor, block_tag, false);
				Advance(); CurrentIndent++; IndentOffset--;
				ParseNode(Styles::BlockMapping, CurrentIndent + IndentOffset);
				OnCloseEvent(Event::Map);
				*/
				//anchor = block_anchor;
				//tag = block_tag;
				//block_anchor = block_tag = string();
				OnOpenEvent(Event::Map, false);
				ParseExplicitBlockMapping();
				OnCloseEvent(Event::Map);
				return true;
			}				

			if (ColonLookahead())
			{
				// What we're about to parse is part of a key:value pair.  We have detected a mapping implicitly 
				// and need to initiate a mapping before we parse the key.
				//OnOpenEvent(Event::Map, block_anchor, block_tag, false);
				OnOpenEvent(Event::Map, false);
				//ParseNode(Styles::BlockMapping, AtIndent);
				ParseNode(Styles::BlockMapping, CurrentIndent);
				OnCloseEvent(Event::Map);
				return true;
			}
				
			// Since we didn't see a - or a lookahead :, this node must be a single scalar value or an alias.

			/** This could come about two ways:
			* 
			*	--- !!str
			*	text
			* 
			*	Or:
			* 
			*	---
			*	!!str text
			* 
			*	And both should be equivalent.
			*/

			if (tag.length() && block_tag.length()) throw FormatException("Conflicting tags '" + block_tag + "' and '" + tag + "' at " + GetSource() + ".");
			if (anchor.length() && block_anchor.length()) throw FormatException("Conflicting anchors '" + block_anchor + "' and '" + anchor + "' at " + GetSource() + ".");
			if (tag.length() == 0) tag = block_tag;
			if (anchor.length() == 0) anchor = block_anchor;
			block_tag = block_anchor = string();

			CurrentBlockIndent = AtIndent;
			ParseAliasOrScalar();
			return true;
		}

		inline void YamlEventParser::ParseDirective()
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

		inline void YamlEventParser::ParseTopLevel()
		{
			bool FirstDocument = true;
			for (;;)
			{
				Tags.clear();
				Aliases.clear();
				CurrentBlockIndent = 0;
				CurrentIndent = 0;
				CurrentStyle = Styles::UnknownBlock;
				block_anchor = block_tag = string();
				anchor = tag = string();

				/** Parse directives, and if we encounter anything then automatically switch to content **/

				bool EOS = false;
				for (;;)
				{
					if (!Need(1)) { EOS = true; break; }

					if (Current == ' ' || Current == '\t')
					{
						if (!Advance()) { 
							//EOS = true; break; 
							// Test case 8G76: no document generated here.
							OnCloseEvent(Event::Stream);
							return;
						}
						CurrentIndent++;
						continue;
					}

					if (Current == '#')
					{
						while (!IsLinebreak(Current)) { 
							if (!Advance()) { 
								// Test case 98YD: no document generated here.
								OnCloseEvent(Event::Stream);
								return;
							} 
						}
						if (!AdvanceLine()) { 
							// Test case 98YD: no document generated here.
							OnCloseEvent(Event::Stream);
							return;
						}
						continue;
					}

					if (IsLinebreak(Current)) {
						if (!AdvanceLine()) { 
							//EOS = true; break; 
							// Test case 8G76: no document generated here.							
							OnCloseEvent(Event::Stream);
							return;
						}
						continue;
					}

					if (Current == '%')
					{
						ParseDirective();
						continue;
					}

					if (Current == '!')
					{
						// Since there was no ---, the tag must apply to the first content item and not +DOC.
						OnOpenEvent(Event::Document, /*explicitly marked=*/ false);
						break;
					}

					FindNextContent(true);

					if (Need(3) && Current == '-' && pNext[0] == '-' && pNext[1] == '-'
						&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2])))
					{
						// Three dashes.  Either separates the first document from directives, or signals the start of an additional document
						// (and the end of the current document).
						Advance(3);						
						OnOpenEvent(Event::Document, /*explicitly marked=*/ true);						
						break;
					}

					// Anything else, switch to content.
					OnOpenEvent(Event::Document, /*explicitly marked=*/ false);
					break;
				}

				if (EOS)
				{
					if (FirstDocument)
					{
						OnOpenEvent(Event::Document, false);
						OnCloseEvent(Event::Document);
					}
					break;
				}
				FirstDocument = false;

				/** Parse content **/

				if (!ParseUnknownBlock(0, true))
				{
					// From test case 6ZKB: an empty document should contain a null entry.
					OnNullValueEvent();						// Null value.
				}

				FindNextContent(true, false);

				if (Need(3) && Current == '-' && pNext[0] == '-' && pNext[1] == '-'
					&& (!Need(4) || pNext[2] == ' ' || pNext[2] == '\t' || IsLinebreak(pNext[2])))
				{
					// Either separates the first document from directives, or signals the start of an additional document
					// (and the end of the current document).												
					// We don't advance here because a pattern like:
					//	--- !!seq
					// Is how a document can start.  So we let the "start of document" parsing advance the three dashes.					

					// We've either already had a --- on this document or we have found nodes already.  Thus, this additional --- marker
					// must end the current document.  It wasn't clear to me whether new directives were allowed after the --- marker
					// if there are two documents, but since we can detect node content it seems safest to allow a new directive section
					// for the new document.  Unclear to me if tags and aliases also reset.
					OnCloseEvent(Event::Document);
					continue;
				}
				else if (Need(3) && Current == '.' && pNext[0] == '.' && pNext[1] == '.')
				{
					Advance(3);

					// Example 6.18 demonstrates that new directives can follow the terminator sequence, so we need to start a new
					// directive section and then return the content from the now-ended document.  Unclear to me if tags and aliases
					// also reset, but based on test case 5TYM it appears that they do in the case of "..." at least.
					OnCloseEvent(Event::Document, true);
					continue;
				}
				else if (!Need(1))		// If the only content left is whitespace, it's fine to call this terminated.
				{
					OnCloseEvent(Event::Document);					
					break;
				}
				else throw Exception("Unknown reason for termination of document parsing at " + GetSource() + ".");
			}
			OnCloseEvent(Event::Stream);
		}

		inline void YamlEventParser::StartStream(wb::io::Stream& stream, const string& sSourceFilename)
		{
			CurrentSource = sSourceFilename;
			CurrentLineNumber = 1;
			CurrentIndent = 0;

			pCurrentStream = &stream;
			if (Need(3) && Current == 0xEF && pNext[0] == 0xBB && pNext[2] == 0xBF) {
				// UTF-8 BOM.  TODO: Respond to detected encoding.
				Advance(); Advance(); Advance();
			}

			block_anchor = block_tag = string();
			anchor = tag = string();
			OnOpenEvent(Event::Stream, false);
		}

		inline void YamlEventParser::FinishStream()
		{
			//OnEvent(EventType::Close, Event::Stream);

			CurrentSource = "";
			CurrentLineNumber = 1;
			pCurrentStream = nullptr;
		}		
	}
}

#endif	// __WBYamlEventParser_h__

//	End of YamlEventParser.h

