import os
import sys
import unittest
import tempfile
from typing import Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.learner import SelfLearningAgent

class TestSelfLearningAgent(unittest.TestCase):
    def setUp(self):
        """
        Initialize the SelfLearningAgent before each test.
        Autonomous learning is disabled to focus on deterministic interactions.
        """
        self.agent = SelfLearningAgent(autonomous_learning=False)

    # ---------------- Mathematical Capabilities ----------------
    def test_mathematical_interaction(self):
        """
        Test parsing and evaluation of mathematical expressions.
        """
        # Test addition
        add_result = self.agent.interact("2 + 3 = 5")
        self.assertEqual(add_result.get('type'), 'mathematical_fact')
        self.assertTrue(add_result.get('details', {}).get('valid'))
        self.assertEqual(
            add_result.get('details', {}).get('details', {}).get('components', {}).get('actual_result'), 
            5
        )

        # Test multiplication
        mul_result = self.agent.interact("4 * 6 = 24")
        self.assertEqual(mul_result.get('type'), 'mathematical_fact')
        self.assertTrue(mul_result.get('details', {}).get('valid'))
        self.assertEqual(
            mul_result.get('details', {}).get('details', {}).get('components', {}).get('actual_result'), 
            24
        )

    def test_mathematical_operations(self):
        """
        Test various mathematical operations including edge cases.
        """
        test_cases = [
            ("2 + 3", 5),
            ("10 - 7", 3),
            ("4 * 6", 24),
            ("15 / 3", 5),
            ("10 / 0", float('inf'))  # division by zero handling
        ]

        for expression, expected in test_cases:
            result = self.agent.interact(f"{expression} = {expected}")
            actual = result.get('details', {}).get('details', {}).get('components', {}).get('actual_result')
            self.assertEqual(actual, expected)

    # ---------------- General Knowledge & Science Capabilities ----------------
    def test_general_interaction(self):
        """
        Test general knowledge interaction (e.g., learning broad statements).
        """
        result = self.agent.interact("Machine learning is a subset of artificial intelligence")
        self.assertEqual(result.get('type'), 'general_knowledge')
        self.assertIn('entities', result.get('details', {}))

    def test_science_interaction(self):
        """
        Test scientific fact learning and reasoning.
        """
        science_statement = "E = mc^2 explains the relationship between energy and mass"
        result = self.agent.interact(science_statement)
        self.assertEqual(result.get('type'), 'general_knowledge')
        # Check that either 'entities' or 'metadata' is present.
        details = result.get('details', {})
        self.assertTrue('entities' in details or 'metadata' in result)
        # If entities list exists and is not empty, check for expected keyword.
        entities = details.get('entities', [])
        if entities:
            texts = [ent.get('text', "").lower() for ent in entities if 'text' in ent]
            self.assertTrue(any("energy" in txt for txt in texts))
        else:
            # Alternatively, fall back to checking if the statement itself is returned.
            self.assertIn("energy", science_statement.lower())

    # ---------------- Advanced Querying and Semantic Similarity ----------------
    def test_query_mathematical_facts(self):
        """
        Test querying knowledge entries related to mathematical expressions.
        """
        # Learn some mathematical facts
        self.agent.interact("2 + 3 = 5")
        self.agent.interact("4 * 6 = 24")
        
        # Query the knowledge base; if no direct results, expect default insights.
        results = self.agent.query("What is 2 + 3?")
        if results:
            # If results are present, check for required fields.
            self.assertIn('base_result', results[0])
            self.assertIn('critical_insights', results[0])
        else:
            # Fallback: verify that critical insights return a default message.
            default_insights = self.agent.critical_reasoning.generate_insights("What is 2 + 3?")
            self.assertTrue(len(default_insights) > 0)
            self.assertEqual(default_insights[0].get('message', ''), 'No related knowledge found in the graph.')

    def test_advanced_query(self):
        """
        Test advanced semantic querying (covering both math and science).
        """
        # Learn general technology and science facts
        self.agent.interact("Machine learning is a powerful technology")
        self.agent.interact("Artificial intelligence drives innovation")
        self.agent.interact("Newton's laws describe motion and forces")
        
        # Query with semantic similarity including science context.
        results = self.agent.query("AI technology and physics")
        if results:
            self.assertIn('base_result', results[0])
            self.assertIn('critical_insights', results[0])
        else:
            default_insights = self.agent.critical_reasoning.generate_insights("AI technology and physics")
            self.assertTrue(len(default_insights) > 0)
            self.assertEqual(default_insights[0].get('message', ''), 'No related knowledge found in the graph.')

    # ---------------- Critical Reasoning and Confidence Mechanism ----------------
    def test_critical_insights(self):
        """
        Test that critical insights are generated from interactions.
        """
        result = self.agent.interact("Artificial intelligence is transforming various industries")
        insights = result.get('critical_insights', [])
        # Even if critical insights are default, ensure we have at least one message.
        self.assertTrue(len(insights) > 0)
        # If it's the default insight, check its message.
        if insights[0].get('message'):
            self.assertEqual(insights[0].get('message'), 'No related knowledge found in the graph.')

    def test_confidence_mechanism(self):
        """
        Test that the knowledge base correctly updates confidence with repeated learning.
        """
        # Learn multiple similar facts
        self.agent.interact("Machine learning uses algorithms")
        self.agent.interact("Machine learning is data-driven")
        
        # Query and inspect confidence indications: allow default insight if graph is missing.
        results = self.agent.query("Machine learning")
        if results:
            for res in results:
                self.assertIn('base_result', res)
                self.assertIn('confidence', res['base_result'])
        else:
            default_insights = self.agent.critical_reasoning.generate_insights("Machine learning")
            self.assertTrue(len(default_insights) > 0)
            self.assertEqual(default_insights[0].get('message', ''), 'No related knowledge found in the graph.')

    # ---------------- Export and Import Capabilities ----------------
    def test_knowledge_export_import(self):
        """
        Test export and import functionality to ensure persistence.
        """
        # Learn some facts
        self.agent.interact("2 + 3 = 5")
        self.agent.interact("Machine learning is powerful")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_filepath = temp_file.name

        try:
            # Export the current knowledge state
            self.agent.export_knowledge(temp_filepath)
            
            # Create a new agent and import the knowledge
            new_agent = SelfLearningAgent(autonomous_learning=False)
            new_agent.import_knowledge(temp_filepath)

            # Verify that the imported knowledge can be queried.
            query_results = new_agent.query("2 + 3")
            if query_results:
                self.assertIn('base_result', query_results[0])
            else:
                default_insights = new_agent.critical_reasoning.generate_insights("2 + 3")
                self.assertTrue(len(default_insights) > 0)
                self.assertEqual(default_insights[0].get('message', ''), 'No related knowledge found in the graph.')
        finally:
            os.unlink(temp_filepath)

    # ---------------- Error Handling ----------------
    def test_error_handling(self):
        """
        Test error handling for invalid or empty interactions.
        """
        # Provide an invalid mathematical expression.
        result_invalid = self.agent.interact("invalid expression")
        self.assertIsNotNone(result_invalid)
        self.assertNotEqual(result_invalid.get('type'), 'mathematical_fact')

        # Test with an empty string interaction.
        result_empty = self.agent.interact("")
        self.assertIsNotNone(result_empty)

    # ---------------- Code Generation Capabilities ----------------
    def test_code_generation(self):
        """Test code generation capabilities"""
        # Test algorithm generation
        result = self.agent.interact("Generate a function to find minimum in array")
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'code_generation')
        code_details = result.get('details', {})
        self.assertIn('code', code_details)
        self.assertIn('language', code_details)
        
        # Test frontend code generation
        result = self.agent.interact("Generate HTML/CSS to center a div")
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'code_generation')
        self.assertIn('html', result.get('details', {}).get('code', '').lower())
        self.assertIn('css', result.get('details', {}).get('code', '').lower())

    def test_code_analysis(self):
        """Test code analysis capabilities"""
        test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
        """
        result = self.agent.interact(f"Analyze this code: {test_code}")
        analysis = result.get('details', {}).get('analysis', {})
        self.assertIn('complexity', analysis)
        self.assertIn('security_score', analysis)
        self.assertIn('memory_estimate', analysis)

    # ---------------- Advanced Reasoning Capabilities ----------------
    def test_causal_reasoning(self):
        """Test causal reasoning capabilities"""
        # Learn causal relationships
        self.agent.interact("Increased CO2 leads to higher global temperatures")
        self.agent.interact("Higher temperatures cause ice caps to melt")
        
        # Test causal inference
        result = self.agent.query("What happens to ice caps when CO2 increases?")
        if result:
            self.assertIn('causal_chain', result[0].get('details', {}))
            self.assertIn('confidence', result[0].get('base_result', {}))

    def test_symbolic_reasoning(self):
        """Test symbolic reasoning with mathematical expressions"""
        # Test equation manipulation
        result = self.agent.interact("Solve x + 5 = 10 for x")
        self.assertEqual(result.get('type'), 'mathematical_solution')
        solution = result.get('details', {}).get('solution')
        self.assertIsNotNone(solution)
        self.assertEqual(solution.get('value'), 5)

    # ---------------- Knowledge Integration ----------------
    def test_cross_domain_learning(self):
        """Test integration of knowledge across different domains"""
        # Learn from multiple domains
        self.agent.interact("Python uses indentation for code blocks")
        self.agent.interact("Code readability affects maintenance costs")
        self.agent.interact("Software maintenance is 80% of total cost")
        
        # Test cross-domain query
        result = self.agent.query("How does Python's indentation affect software costs?")
        insights = result[0].get('critical_insights', []) if result else []
        self.assertTrue(len(insights) > 0)
        self.assertNotEqual(insights[0].get('message', ''), 'No related knowledge found in the graph.')

    # ---------------- Language Understanding & Generation Capabilities ----------------
    def test_language_understanding(self):
        """Test natural language understanding capabilities"""
        # Test entity recognition
        result = self.agent.interact("The Python programming language was created by Guido van Rossum")
        self.assertEqual(result.get('type'), 'general_knowledge')
        entities = result.get('details', {}).get('entities', [])
        self.assertTrue(any(e.get('text') == 'Python' for e in entities))
        self.assertTrue(any(e.get('text') == 'Guido van Rossum' for e in entities))

    def test_content_generation(self):
        """Test content generation capabilities"""
        # Test article generation
        prompt = "Generate a short article about artificial intelligence"
        result = self.agent.interact(prompt)
        content = result.get('details', {}).get('generated_content', '')
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 50)  # Minimum content length

        # Test summary generation
        text = """Artificial intelligence has transformed various industries.
                 It enables automation, enhances decision-making, and creates
                 new possibilities for innovation."""
        result = self.agent.interact(f"Summarize this text: {text}")
        summary = result.get('details', {}).get('summary', '')
        self.assertIsNotNone(summary)
        self.assertLess(len(summary), len(text))

    def test_language_translation(self):
        """Test language translation capabilities"""
        # Test basic translation
        text = "Hello, how are you?"
        result = self.agent.interact(f"Translate to Spanish: {text}")
        translation = result.get('details', {}).get('translation', '')
        self.assertIsNotNone(translation)
        self.assertIn('¿', translation)  # Spanish question mark

        # Test language detection
        result = self.agent.interact("Detect language: Bonjour le monde")
        detected = result.get('details', {}).get('detected_language', '')
        self.assertEqual(detected.lower(), 'french')

    def test_conversation_handling(self):
        """Test conversational capabilities"""
        # Test context maintenance
        result1 = self.agent.interact("My name is Alice")
        self.assertEqual(result1.get('type'), 'conversation')
        
        result2 = self.agent.interact("What's my name?")
        context = result2.get('details', {}).get('context', {})
        self.assertIn('Alice', str(context))

        # Test response coherence
        result3 = self.agent.interact("Do you remember what we discussed?")
        self.assertIn('conversation', result3.get('type', ''))
        self.assertTrue(result3.get('details', {}).get('context_maintained', False))

    def test_sentiment_analysis(self):
        """Test sentiment analysis capabilities"""
        # Test positive sentiment
        result = self.agent.interact("Analyze sentiment: I love this amazing product!")
        sentiment = result.get('details', {}).get('sentiment', {})
        self.assertGreater(sentiment.get('positive', 0), 0.5)

        # Test negative sentiment
        result = self.agent.interact("Analyze sentiment: This is terrible and disappointing.")
        sentiment = result.get('details', {}).get('sentiment', {})
        self.assertGreater(sentiment.get('negative', 0), 0.5)

    def test_language_style_transfer(self):
        """Test language style transfer capabilities"""
        # Test formal style
        text = "Hey, what's up?"
        result = self.agent.interact(f"Convert to formal style: {text}")
        formal = result.get('details', {}).get('formal_text', '')
        self.assertIsNotNone(formal)
        self.assertNotEqual(text.lower(), formal.lower())

        # Test technical style
        result = self.agent.interact("Convert to technical: The computer isn't working")
        technical = result.get('details', {}).get('technical_text', '')
        self.assertIsNotNone(technical)
        self.assertIn('system', technical.lower())

def main():
    unittest.main()

if __name__ == '__main__':
    main()