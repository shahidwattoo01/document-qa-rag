"""Prompt templates for answering, LLM re-ranking, and evaluation judging."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# Fixed abstention sentence: makes "did the model refuse?" checkable without a judge.
NO_ANSWER = "I don't know based on the provided documents."

ANSWER_SYSTEM = f"""You are a precise assistant that answers questions about a company's internal documents.

Rules:
1. Use ONLY the information in the numbered context passages below. Do not use prior knowledge.
2. Cite the passages that support each statement using their numbers in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply with exactly: "{NO_ANSWER}"
   Do not guess, and do not answer a different question.
4. If the context only partially answers the question, answer the part that is supported and say what is missing.
5. Be concise: one to four sentences unless the question asks for a list.

Context passages:
{{context}}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_SYSTEM), ("human", "Question: {question}")]
)

RERANK_SYSTEM = """You are a search relevance judge. For each numbered passage, rate how useful it is for answering the user's question:
3 = directly contains the answer
2 = closely related, partially answers
1 = same topic but does not help answer
0 = irrelevant
Return a score for every passage."""

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RERANK_SYSTEM), ("human", "Question: {question}\n\nPassages:\n{passages}")]
)

FAITHFULNESS_SYSTEM = """You verify whether an answer is grounded in the given context.
Break the ANSWER into its individual factual claims (ignore citation markers like [1]).
For each claim decide whether it is fully supported by the CONTEXT. A claim that adds details
not in the context, or contradicts it, is NOT supported. If the answer is a refusal / says the
information is not available, return an empty claim list."""

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", FAITHFULNESS_SYSTEM), ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}")]
)

CORRECTNESS_SYSTEM = """You grade an answer against a reference answer written by a human expert.
Score 1.0 if the answer contains all key facts of the reference and nothing contradicting it,
0.5 if it is partially correct or misses an important detail, and 0.0 if it is wrong, contradicts
the reference, or fails to answer. Extra correct details do not reduce the score."""

CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CORRECTNESS_SYSTEM),
        ("human", "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}"),
    ]
)
