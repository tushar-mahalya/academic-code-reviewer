from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Internal Functions
from src.preprocessing import cleanTranscript
from src.acquisition import extractVideoTranscript
from src.prompts import map_summary_prompt, reduce_summary_prompt, code_comparison_prompt


class GenerativeFeatures:
    def __init__(self):
        # LLM Model
        self.llm = OpenAI(temperature=0.1)

        # Intrinsically used Prompts
        self.map_summary_prompt = map_summary_prompt
        self.reduce_summary_prompt = reduce_summary_prompt
        self.code_comparison_prompt = code_comparison_prompt

    def generate_lecture_summary(self, yt_video_url):
        transcript = str(extractVideoTranscript(yt_video_url))
        cleaned_transcript = cleanTranscript(transcript)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        transcript_chunks = text_splitter.create_documents([cleaned_transcript])
        map_reduce_chain = self.map_reduce_chain()
        summary = map_reduce_chain.run(transcript_chunks)

        return summary

    def generate_code_feedback(self, student_code, solution_code):
        prompt = PromptTemplate.from_template(self.code_comparison_prompt.prompt)
        code_comparison_chain = LLMChain(
            prompt=prompt,
            llm=self.llm
        )
        feedback = code_comparison_chain.run(student_code=student_code, solution_code=solution_code)

        return feedback

    def map_reduce_chain(self):
        # Map Chain
        map_template = self.map_summary_prompt.prompt
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(
            prompt=map_prompt,
            llm=self.llm
        )

        # Reduce Chain
        reduce_template = self.reduce_summary_prompt.prompt
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(
            llm=self.llm,
            prompt=reduce_prompt
        )
        stuff_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )
        reduce_chain = ReduceDocumentsChain(
            combine_documents_chain=stuff_chain
        )

        # Map-Reduce Chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            document_variable_name="text",
            reduce_documents_chain=reduce_chain
        )

        return map_reduce_chain
