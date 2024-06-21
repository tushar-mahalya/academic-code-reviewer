from dataclasses import dataclass


@dataclass
class Prompt:
    prompt: str


_map_summary_prompt_template = """Write a concise descriptive technical summary of the following content:

{text}

Summary:
"""

_reduce_summary_prompt_template = """The following is set of summaries:

{doc_summaries}

Summarize the above descriptive summaries with all technical and key details
Summary:
"""

_code_comparison_prompt_template = '''Compare the following Python codes and provide sufficiently descriptive, constructive and relevant feedback. Highlight any inefficiencies, errors, and areas for improvement in the first code with respect to the second code. Do not write any actual code or give obvious hints. Use a second-person tone to make the feedback feel more personal and instructional. It should also provide important points of improvement in first code compared to second code.

Your Code:
{student_code}

Solution Code:
{solution_code}

The format of the output should be:

Feedback:
(Feedback Paragraph)

Points to Improve:
- (bullet point 1)
- (bullet point 2)
- (bullet point 3)
- (add more bullet points if necessary to cover all improvements)

The output should strictly follow this format only.
'''
map_summary_prompt = Prompt(prompt=_map_summary_prompt_template)
reduce_summary_prompt = Prompt(prompt=_reduce_summary_prompt_template)
code_comparison_prompt = Prompt(prompt=_code_comparison_prompt_template)
