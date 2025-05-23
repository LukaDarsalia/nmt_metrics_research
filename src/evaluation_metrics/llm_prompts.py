"""
LLM evaluation prompts for machine translation quality assessment.

This module contains prompt templates for both reference-based and reference-free
evaluation of machine translation quality using Large Language Models with examples.
"""

from typing import Dict, Any


class LLMPrompts:
    """Container for LLM evaluation prompts and scoring instructions."""

    @staticmethod
    def get_examples_section() -> str:
        """
        Get examples section for few-shot learning.

        Returns:
            str: Examples section with sample evaluations
        """
        return "<examples>\n<example>\n<source>\nHe was greeted by Singapore's Deputy Prime Minister Wong Kan Seng and discussed trade and terrorism issues with the Singapore Prime Minister Lee Hsien Loong.\n</source>\n<hypothesis>\nმას სინგაპურის ვიცე-პრემიერმა ვონგ კან სენგმა უმასპინძლა და სინგაპურის პრემიერ-მინისტრ ლის ჰსიენ ლონგთან ერთად განიხილა ვაჭრობისა და ტერორიზმის საკითხები.\n</hypothesis>\n<ideal_output>\n90\n</ideal_output>\n</example>\n<example>\n<source>\nEfforts to search for the crash site are being met by bad weather and harsh terrain.\n</source>\n<hypothesis>\nავარიის ადგილის ძიების მცდელობები ცუდ ამინდსა და მკაცრ რელიეფს ხვდება.\n</hypothesis>\n<ideal_output>\n76\n</ideal_output>\n</example>\n<example>\n<source>\nIt had been scheduled to be cut down on Tuesday, but was saved after an emergency court ruling.\n</source>\n<hypothesis>\nმისი შემცირება სამშაბათს იყო დაგეგმილი, მაგრამ გადაარჩინეს სასწრაფო სასამართლოს გადაწყვეტილების შემდეგ.\n</hypothesis>\n<ideal_output>\n66\n</ideal_output>\n</example>\n<example>\n<source>\nThe announcement was made after Trump had a phone conversation with Turkish President Recep Tayyip Erdoğan.\n</source>\n<hypothesis>\nგამოგზავნილია: 16 May 2018, 14:04 #517 · პროფილი · პირადი მიმოწერა · ჩატი · ელფოსტა\n</hypothesis>\n<ideal_output>\n0\n</ideal_output>\n</example>\n</examples>\n\n"

    @staticmethod
    def get_reference_based_examples_section() -> str:
        """
        Get examples section for reference-based evaluation.

        Returns:
            str: Examples section with reference-based sample evaluations (exact from user's paste.txt)
        """
        return "<examples>\n<example>\n<source>\nHe was greeted by Singapore's Deputy Prime Minister Wong Kan Seng and discussed trade and terrorism issues with the Singapore Prime Minister Lee Hsien Loong.\n</source>\n<reference>\nმას ვონგ დახვდა კან სენგი, სინგაპურის ვიცე პრემიერ მინისტრი და სინგაპურის პრემიერ მინისტრთან ლი სიან ლუნთან ვაჭრობის და ტერორიზმის საკითხები განიხილა.\n</reference>\n<hypothesis>\nმას სინგაპურის ვიცე-პრემიერმა ვონგ კან სენგმა უმასპინძლა და სინგაპურის პრემიერ-მინისტრ ლის ჰსიენ ლონგთან ერთად განიხილა ვაჭრობისა და ტერორიზმის საკითხები.\n</hypothesis>\n<ideal_output>\n90\n</ideal_output>\n</example>\n<example>\n<source>\nEfforts to search for the crash site are being met by bad weather and harsh terrain.\n</source>\n<reference>\nცუდი ამინდი და რთული რელიეფი ხელს უშლიდა ავარიის ადგილის აღმოჩენის მცდელობას.\n</reference>\n<hypothesis>\nავარიის ადგილის ძიების მცდელობები ცუდ ამინდსა და მკაცრ რელიეფს ხვდება.\n</hypothesis>\n<ideal_output>\n76\n</ideal_output>\n</example>\n<example>\n<source>\nIt had been scheduled to be cut down on Tuesday, but was saved after an emergency court ruling.\n</source>\n<reference>\nდაგეგმილი იყო მისი მოჭრა სამშაბათს, თუმცა სასამართლოს დაჩქარებული განჩინების საფუძველზე მოხერხდა მისი გადარჩენა.\n</reference>\n<hypothesis>\nმისი შემცირება სამშაბათს იყო დაგეგმილი, მაგრამ გადაარჩინეს სასწრაფო სასამართლოს გადაწყვეტილების შემდეგ.\n</hypothesis>\n<ideal_output>\n66\n</ideal_output>\n</example>\n<example>\n<source>\nThe announcement was made after Trump had a phone conversation with Turkish President Recep Tayyip Erdoğan.\n</source>\n<reference>\nამის შესახებ განცხადება გაკეთდა მას შემდეგ, რაც ტრამპი თურქეთის პრეზიდენტ რეჯეფ თაიფ ერდოღანს ტელეფონით ესაუბრა.\n</reference>\n<hypothesis>\nგამოგზავნილია: 16 May 2018, 14:04 #517 · პროფილი · პირადი მიმოწერა · ჩატი · ელფოსტა\n</hypothesis>\n<ideal_output>\n0\n</ideal_output>\n</example>\n</examples>\n\n"

    @staticmethod
    def get_reference_based_prompt() -> str:
        """
        Get prompt template for reference-based MT evaluation.

        Returns:
            str: Prompt template with placeholders for source, hypothesis, and reference
        """
        return """You are an expert machine translation evaluator. Your task is to evaluate the quality of a machine translation by comparing it to a reference translation.

**Evaluation Criteria:**
- **Adequacy**: How well does the translation convey the meaning of the reference?
- **Fluency**: How natural and grammatically correct is the translation?
- **Precision**: How accurately are specific terms and concepts translated?

**Instructions:**
1. Read the source text, machine translation, and reference translation carefully
2. Compare the machine translation to the reference translation
3. Provide a score from 0 to 100 where:
   - 90-100: Excellent - Near perfect translation, minor differences from reference
   - 80-89: Good - High quality with some minor issues
   - 70-79: Acceptable - Generally correct but with noticeable errors
   - 60-69: Poor - Significant errors affecting meaning or fluency
   - 50-59: Bad - Major errors, difficult to understand
   - 0-49: Very Bad - Mostly incorrect or incomprehensible

**Input:**
Source Text: {source}
Reference Translation: {reference}
Machine Translation: {hypothesis}

**Output Format:**
Provide only a numerical score between 0 and 100. Do not include any explanation or additional text.

Score:"""

    @staticmethod
    def get_reference_free_prompt() -> str:
        """
        Get prompt template for reference-free MT evaluation.

        Returns:
            str: Prompt template with placeholders for source and hypothesis
        """
        return """You are an expert machine translation evaluator. Your task is to evaluate the quality of a machine translation by assessing how well it translates the source text, without seeing any reference translation.

**Evaluation Criteria:**
- **Adequacy**: How completely does the translation convey the source meaning?
- **Fluency**: How natural and grammatically correct is the translation in the target language?
- **Fidelity**: How accurately are source concepts, terms, and nuances preserved?
- **Completeness**: Are all important elements from the source included?

**Instructions:**
1. Read the source text and machine translation carefully
2. Assess the translation quality based on the criteria above
3. Provide a score from 0 to 100 where:
   - 90-100: Excellent - Accurate, fluent, complete translation
   - 80-89: Good - High quality with minor issues or omissions
   - 70-79: Acceptable - Generally correct but with some errors
   - 60-69: Poor - Noticeable errors affecting meaning or readability
   - 50-59: Bad - Significant errors, meaning partially preserved
   - 0-49: Very Bad - Major errors, meaning lost or distorted

**Input:**
Source Text: {source}
Machine Translation: {hypothesis}

**Output Format:**
Provide only a numerical score between 0 and 100. Do not include any explanation or additional text.

Score:"""

    @staticmethod
    def get_system_prompt() -> str:
        """
        Get system prompt for the LLM to set evaluation context.

        Returns:
            str: System prompt for consistent evaluation behavior
        """
        return """You are a professional machine translation quality assessment expert. You evaluate translations objectively and consistently, providing numerical scores based on established criteria. Always respond with only the numerical score as requested."""

    @staticmethod
    def get_prompt_config() -> Dict[str, Any]:
        """
        Get configuration for prompt parameters matching the provided example.

        Returns:
            Dict[str, Any]: Configuration dictionary with prompt settings
        """
        return {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 20000,
            "temperature": 0,
            "stop_sequences": ["\n", "Explanation:", "Reasoning:", "Note:"]
        }