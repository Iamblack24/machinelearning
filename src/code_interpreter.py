import ast
import big_o
from typing import Dict, Any, List, Optional
import ast
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
import re
import json

class SecurityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class CodeAnalysisResult:
    complexity: str
    security_score: int
    potential_issues: List[str]
    ast_analysis: Dict[str, Any]
    memory_estimate: str
    safe_to_execute: bool

class CodeGenerator:
    def __init__(self):
        self.patterns = {
            'function': self._generate_function,
            'class': self._generate_class,
            'test': self._generate_test,
            'algorithm': self._generate_algorithm,
            'frontend': self._generate_frontend,
            'style': self._generate_style
        }
        
        self.language_patterns = {
            r'find.*minimum|least.*value|smallest.*number': ('algorithm', 'find_minimum'),
            r'center.*div|align.*middle': ('frontend', 'center_div'),
            r'sort.*array|order.*list': ('algorithm', 'sort_array'),
            r'create.*class|make.*class': ('class', 'custom_class'),
            r'test.*function|unit test': ('test', 'test_function')
        }

    def parse_request(self, request: str) -> Dict[str, Any]:
        """Parse natural language request into code specification"""
        request = request.lower()
        
        for pattern, (type_, name) in self.language_patterns.items():
            if re.search(pattern, request):
                return {
                    'type': type_,
                    'name': name,
                    'description': request
                }
        
        return {'type': 'function', 'name': 'custom_function', 'description': request}

    def _generate_algorithm(self, spec: Dict[str, Any]) -> str:
        """Generate algorithm implementations"""
        algorithms = {
            'find_minimum': '''def find_minimum(array: List[float]) -> float:
    """Find the minimum value in an array."""
    if not array:
        raise ValueError("Array cannot be empty")
    return min(array)  # Using Python's built-in min for efficiency
''',
            'sort_array': '''def sort_array(array: List[float]) -> List[float]:
    """Sort an array in ascending order."""
    return sorted(array)  # Using Python's built-in sorted for efficiency
'''
        }
        return algorithms.get(spec['name'], 'pass')

    def _generate_frontend(self, spec: Dict[str, Any]) -> str:
        """Generate frontend-related code"""
        frontend_patterns = {
            'center_div': '''<!-- HTML -->
<div class="centered-container">
    <div class="centered-content">
        Your content here
    </div>
</div>

/* CSS */
.centered-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;  /* Full viewport height */
}

.centered-content {
    /* Optional: Add styling for the content */
    padding: 20px;
    border: 1px solid #ccc;
}'''
        }
        return frontend_patterns.get(spec['name'], '')

    def _generate_function(self, spec: Dict[str, Any]) -> str:
        """Generate function based on specification"""
        params = spec.get('params', [])
        param_str = ', '.join([f"{p['name']}: {p['type']}" for p in params])
        return_type = spec.get('return_type', 'Any')
        
        template = f"""def {spec['name']}({param_str}) -> {return_type}:
    \"\"\"
    {spec.get('description', 'Generated function')}
    \"\"\"
    # Implementation
    {spec.get('body', 'pass')}
"""
        return template

    def _generate_class(self, spec: Dict[str, Any]) -> str:
        """Generate class based on specification"""
        methods = spec.get('methods', [])
        class_body = []
        
        for method in methods:
            class_body.append(self._generate_function(method))
        
        template = f"""class {spec['name']}:
    \"\"\"
    {spec.get('description', 'Generated class')}
    \"\"\"
    def __init__(self):
        {spec.get('init_body', 'pass')}
        
    {'    '.join(class_body)}
"""
        return template

    def _generate_test(self, spec: Dict[str, Any]) -> str:
        """Generate test case based on specification"""
        template = f"""def test_{spec['name']}():
    \"\"\"
    {spec.get('description', 'Test case')}
    \"\"\"
    # Setup
    {spec.get('setup', '')}
    
    # Execute
    {spec.get('execute', '')}
    
    # Assert
    {spec.get('assert', 'assert True')}
"""
        return template

class CodeInterpreter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_patterns = {
            'eval': SecurityLevel.HIGH,
            'exec': SecurityLevel.HIGH,
            'os.system': SecurityLevel.HIGH,
            'subprocess': SecurityLevel.HIGH,
            'open': SecurityLevel.MEDIUM,
            'input': SecurityLevel.MEDIUM
        }
        self.generator = CodeGenerator()

    def _analyze_complexity(self, code: str) -> str:
        """Enhanced complexity analysis using big-O notation"""
        try:
            complexity = big_o.big_o(
                lambda _: self._safe_execute(code),
                lambda n: big_o.datagen.n_(n)
            )
            return str(complexity)
        except Exception as e:
            self.logger.warning(f"Complexity analysis failed: {e}")
            return "Unknown"

    def _analyze_ast(self, code: str) -> Dict[str, Any]:
        """Analyze code structure using AST"""
        try:
            tree = ast.parse(code)
            return {
                'functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                'loops': len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
                'imports': [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
            }
        except Exception as e:
            self.logger.error(f"AST analysis failed: {e}")
            return {}

    def _check_security(self, code: str) -> tuple[int, List[str]]:
        """Analyze code for security concerns"""
        issues = []
        score = 0
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.security_patterns:
                        level = self.security_patterns[func_name]
                        issues.append(f"Potentially unsafe operation: {func_name}")
                        score += level.value

        return score, issues

    def _estimate_memory(self, code: str) -> str:
        """Estimate memory usage of the code"""
        try:
            tree = ast.parse(code)
            # Basic heuristics for memory estimation
            variables = len([node for node in ast.walk(tree) if isinstance(node, ast.Name)])
            lists = len([node for node in ast.walk(tree) if isinstance(node, ast.List)])
            dicts = len([node for node in ast.walk(tree) if isinstance(node, ast.Dict)])
            
            estimate = variables * 128 + lists * 512 + dicts * 1024  # Rough estimates in bytes
            return f"~{estimate // 1024}KB"
        except:
            return "Unknown"

    def _safe_execute(self, code: str) -> Any:
        """Safely execute code in a restricted environment"""
        restricted_globals = {}
        try:
            tree = ast.parse(code)
            # Check for unsafe operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        raise SecurityError("Unsafe operation detected")
            
            exec(compile(tree, '<string>', 'exec'), restricted_globals)
            return restricted_globals
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return None

    def interpret(self, code: str) -> CodeAnalysisResult:
        """
        Comprehensive code interpretation and analysis
        
        :param code: Source code to analyze
        :return: CodeAnalysisResult object with analysis details
        """
        try:
            # Perform various analyses
            complexity = self._analyze_complexity(code)
            security_score, issues = self._check_security(code)
            ast_analysis = self._analyze_ast(code)
            memory_estimate = self._estimate_memory(code)
            
            # Determine if code is safe to execute
            safe_to_execute = security_score < SecurityLevel.HIGH.value
            
            return CodeAnalysisResult(
                complexity=complexity,
                security_score=security_score,
                potential_issues=issues,
                ast_analysis=ast_analysis,
                memory_estimate=memory_estimate,
                safe_to_execute=safe_to_execute
            )
        except Exception as e:
            self.logger.error(f"Code interpretation failed: {e}")
            return CodeAnalysisResult(
                complexity="Error",
                security_score=SecurityLevel.HIGH.value,
                potential_issues=[str(e)],
                ast_analysis={},
                memory_estimate="Unknown",
                safe_to_execute=False
            )

    def generate_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code based on specification
        
        :param spec: Dictionary containing code generation specifications
        :return: Dictionary containing generated code and analysis
        """
        try:
            code_type = spec.get('type', 'function')
            generator_func = self.generator.patterns.get(code_type)
            
            if not generator_func:
                raise ValueError(f"Unsupported code type: {code_type}")
                
            generated_code = generator_func(spec)
            
            # Analyze generated code
            analysis = self.interpret(generated_code)
            
            return {
                'code': generated_code,
                'analysis': analysis,
                'safe_to_execute': analysis.safe_to_execute
            }
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {
                'error': str(e),
                'code': None,
                'analysis': None
            }

    def suggest_improvements(self, code: str) -> List[str]:
        """Suggest code improvements based on analysis"""
        suggestions = []
        analysis = self.interpret(code)
        
        # Complexity-based suggestions
        if 'O(n^2)' in analysis.complexity:
            suggestions.append("Consider using more efficient algorithms to reduce complexity")
            
        # Memory-based suggestions
        if 'KB' in analysis.memory_estimate:
            mem_kb = int(analysis.memory_estimate.split('~')[1].split('KB')[0])
            if mem_kb > 1000:
                suggestions.append("Consider optimizing memory usage")
                
        # Security-based suggestions
        if analysis.security_score > SecurityLevel.LOW.value:
            suggestions.append("Consider implementing additional security measures")
            
        return suggestions

    def generate_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Generate and analyze code from natural language prompt"""
        try:
            # Generate code from prompt
            generated = self.generator.generate_from_text(prompt)
            
            if generated['code']:
                # Analyze generated code
                analysis = self.interpret(generated['code'])
                
                return {
                    'code': generated['code'],
                    'language': generated['language'],
                    'analysis': analysis,
                    'safe_to_execute': analysis.safe_to_execute,
                    'suggestions': self.suggest_improvements(generated['code'])
                }
            
        except Exception as e:
            self.logger.error(f"Code generation from prompt failed: {e}")
            return {
                'error': str(e),
                'code': None,
                'analysis': None
            }

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass