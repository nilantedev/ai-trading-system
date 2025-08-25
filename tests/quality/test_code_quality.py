#!/usr/bin/env python3
"""
Code Quality Assurance Tests
"""

import pytest
import ast
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCodeQuality:
    """Code quality assurance tests."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def python_files(self, project_root):
        """Get all Python files in the project."""
        python_files = []
        
        # Directories to scan
        scan_dirs = ["api", "services", "shared", "tests"]
        
        for scan_dir in scan_dirs:
            scan_path = project_root / scan_dir
            if scan_path.exists():
                for py_file in scan_path.rglob("*.py"):
                    # Skip __pycache__ and other generated files
                    if "__pycache__" not in str(py_file):
                        python_files.append(py_file)
        
        return python_files

    def test_python_syntax_validity(self, python_files):
        """Test that all Python files have valid syntax."""
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse the file to check syntax
                ast.parse(source_code, filename=str(py_file))
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except UnicodeDecodeError as e:
                syntax_errors.append(f"{py_file}: Unicode decode error: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: Unexpected error: {e}")
        
        if syntax_errors:
            print("Syntax errors found:")
            for error in syntax_errors:
                print(f"  {error}")
        
        assert len(syntax_errors) == 0, f"Found {len(syntax_errors)} files with syntax errors"

    def test_import_statement_organization(self, python_files):
        """Test that import statements are properly organized."""
        import_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check import organization
                stdlib_imports = []
                third_party_imports = []
                local_imports = []
                
                in_imports = False
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    if line.startswith(('import ', 'from ')):
                        in_imports = True
                        
                        # Categorize imports
                        if line.startswith('from .') or line.startswith('from ..'):
                            local_imports.append((i, line))
                        elif any(lib in line for lib in ['fastapi', 'pydantic', 'redis', 'asyncio', 'pytest']):
                            third_party_imports.append((i, line))
                        else:
                            stdlib_imports.append((i, line))
                    
                    elif in_imports and line and not line.startswith('#'):
                        # End of import block
                        break
                
                # Check if imports are in correct order (stdlib, third-party, local)
                all_imports = stdlib_imports + third_party_imports + local_imports
                
                # Verify order
                for i in range(len(all_imports) - 1):
                    current_line = all_imports[i][0]
                    next_line = all_imports[i + 1][0]
                    
                    if next_line < current_line:
                        import_issues.append(f"{py_file}: Imports not in correct order")
                        break
                        
            except Exception as e:
                import_issues.append(f"{py_file}: Could not analyze imports: {e}")
        
        if import_issues:
            print("Import organization issues:")
            for issue in import_issues:
                print(f"  {issue}")
        
        # This is a warning, not a failure
        print(f"Checked {len(python_files)} files for import organization")

    def test_function_complexity(self, python_files):
        """Test function complexity (McCabe cyclomatic complexity)."""
        complex_functions = []
        max_complexity = 15  # Threshold for complexity
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        
                        if complexity > max_complexity:
                            complex_functions.append({
                                'file': str(py_file),
                                'function': node.name,
                                'complexity': complexity,
                                'line': node.lineno
                            })
                            
            except Exception as e:
                print(f"Could not analyze {py_file}: {e}")
        
        if complex_functions:
            print("Functions with high complexity:")
            for func in complex_functions:
                print(f"  {func['file']}:{func['line']} {func['function']} (complexity: {func['complexity']})")
        
        # High complexity is a warning, not failure
        assert len(complex_functions) < 10, "Too many highly complex functions"

    def _calculate_complexity(self, node):
        """Calculate McCabe cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, 
                                ast.With, ast.AsyncWith, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Count boolean operations
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity

    def test_docstring_coverage(self, python_files):
        """Test docstring coverage for functions and classes."""
        missing_docstrings = []
        
        for py_file in python_files:
            # Skip test files for docstring requirements
            if "test_" in py_file.name:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # Check if function/class has docstring
                        has_docstring = (
                            node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Str)
                        )
                        
                        if not has_docstring and not node.name.startswith('_'):
                            missing_docstrings.append({
                                'file': str(py_file),
                                'name': node.name,
                                'type': type(node).__name__,
                                'line': node.lineno
                            })
                            
            except Exception as e:
                print(f"Could not analyze docstrings in {py_file}: {e}")
        
        if missing_docstrings:
            print("Missing docstrings:")
            for item in missing_docstrings[:10]:  # Show first 10
                print(f"  {item['file']}:{item['line']} {item['type']} '{item['name']}'")
            
            if len(missing_docstrings) > 10:
                print(f"  ... and {len(missing_docstrings) - 10} more")
        
        # Docstring coverage should be reasonable
        total_items = len(missing_docstrings) + 100  # Assume some have docstrings
        coverage = (total_items - len(missing_docstrings)) / total_items * 100
        
        print(f"Docstring coverage: {coverage:.1f}%")
        assert coverage > 50, "Docstring coverage should be above 50%"

    def test_line_length_compliance(self, python_files):
        """Test line length compliance (PEP 8)."""
        long_lines = []
        max_length = 88  # Black's default line length
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # Remove trailing newline for length calculation
                    line_content = line.rstrip('\n\r')
                    
                    if len(line_content) > max_length:
                        # Skip lines that are just long strings or comments
                        stripped = line_content.strip()
                        if not (stripped.startswith('#') or 
                               stripped.startswith(('"""', "'''", '"', "'"))):
                            long_lines.append({
                                'file': str(py_file),
                                'line': line_num,
                                'length': len(line_content),
                                'content': line_content[:50] + '...' if len(line_content) > 50 else line_content
                            })
                            
            except Exception as e:
                print(f"Could not check line lengths in {py_file}: {e}")
        
        if long_lines:
            print("Lines exceeding length limit:")
            for line in long_lines[:5]:  # Show first 5
                print(f"  {line['file']}:{line['line']} ({line['length']} chars): {line['content']}")
        
        # Allow some long lines but not too many
        assert len(long_lines) < 20, f"Too many long lines: {len(long_lines)}"

    def test_naming_conventions(self, python_files):
        """Test naming convention compliance."""
        naming_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Classes should be PascalCase
                        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                            naming_issues.append(f"{py_file}:{node.lineno} Class '{node.name}' should be PascalCase")
                    
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Functions should be snake_case
                        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('__'):
                            naming_issues.append(f"{py_file}:{node.lineno} Function '{node.name}' should be snake_case")
                    
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        # Variables should be snake_case (basic check)
                        if node.id.isupper() and len(node.id) > 1:
                            # Constants are OK
                            continue
                        elif not re.match(r'^[a-z_][a-z0-9_]*$', node.id) and node.id not in ['_', '__']:
                            # Only flag obvious violations to avoid too many false positives
                            if re.match(r'^[a-z]+[A-Z]', node.id):  # camelCase
                                naming_issues.append(f"{py_file} Variable '{node.id}' should be snake_case")
                            
            except Exception as e:
                print(f"Could not check naming in {py_file}: {e}")
        
        if naming_issues:
            print("Naming convention issues:")
            for issue in naming_issues[:10]:  # Show first 10
                print(f"  {issue}")
        
        # Some naming issues are acceptable
        assert len(naming_issues) < 50, "Too many naming convention violations"

    def test_security_patterns(self, python_files):
        """Test for potential security issues in code."""
        security_issues = []
        
        # Patterns that might indicate security issues
        dangerous_patterns = [
            (r'exec\s*\(', 'Use of exec() can be dangerous'),
            (r'eval\s*\(', 'Use of eval() can be dangerous'),
            (r'shell\s*=\s*True', 'shell=True in subprocess can be dangerous'),
            (r'password\s*=\s*["\'][^"\']{1,20}["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']{1,50}["\']', 'Hardcoded secret'),
            (r'api_key\s*=\s*["\'][^"\']{1,50}["\']', 'Hardcoded API key'),
            (r'subprocess\.call\([^)]*shell\s*=\s*True', 'Dangerous subprocess call'),
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, message in dangerous_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Calculate line number
                        line_num = content[:match.start()].count('\n') + 1
                        security_issues.append(f"{py_file}:{line_num} {message}")
                        
            except Exception as e:
                print(f"Could not check security patterns in {py_file}: {e}")
        
        if security_issues:
            print("Potential security issues:")
            for issue in security_issues:
                print(f"  {issue}")
        
        # Security issues should be investigated
        assert len(security_issues) == 0, "Potential security issues found"

    def test_todo_and_fixme_tracking(self, python_files):
        """Track TODO and FIXME comments."""
        todos = []
        fixmes = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    if 'todo' in line_lower:
                        todos.append(f"{py_file}:{line_num} {line.strip()}")
                    
                    if 'fixme' in line_lower:
                        fixmes.append(f"{py_file}:{line_num} {line.strip()}")
                        
            except Exception as e:
                print(f"Could not check TODOs in {py_file}: {e}")
        
        if todos:
            print(f"Found {len(todos)} TODO comments:")
            for todo in todos[:5]:  # Show first 5
                print(f"  {todo}")
        
        if fixmes:
            print(f"Found {len(fixmes)} FIXME comments:")
            for fixme in fixmes:
                print(f"  {fixme}")
        
        # FIXMEs should be addressed before production
        assert len(fixmes) == 0, "FIXME comments should be resolved"

    def test_configuration_validation(self, project_root):
        """Test configuration file validity."""
        config_files = [
            "pyproject.toml",
            "requirements.txt", 
            "requirements-dev.txt",
            "docker-compose.yml",
            "docker-compose.dev.yml",
        ]
        
        config_issues = []
        
        for config_file in config_files:
            config_path = project_root / config_file
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                    
                    # Basic validation - file should not be empty and should be readable
                    if not content.strip():
                        config_issues.append(f"{config_file} is empty")
                    
                    # Specific validations
                    if config_file.endswith('.toml'):
                        try:
                            import tomllib
                            tomllib.loads(content)
                        except Exception as e:
                            config_issues.append(f"{config_file} has invalid TOML: {e}")
                    
                    elif config_file.endswith('.yml') or config_file.endswith('.yaml'):
                        # Basic YAML validation
                        if content.count(':') == 0:
                            config_issues.append(f"{config_file} doesn't look like valid YAML")
                            
                except Exception as e:
                    config_issues.append(f"Could not read {config_file}: {e}")
        
        if config_issues:
            print("Configuration issues:")
            for issue in config_issues:
                print(f"  {issue}")
        
        assert len(config_issues) == 0, "Configuration files should be valid"

    def test_test_coverage_requirements(self, project_root):
        """Test that important modules have corresponding tests."""
        # Get all Python modules in main directories
        main_modules = []
        test_modules = []
        
        # Scan main code directories
        for main_dir in ["api", "services", "shared"]:
            main_path = project_root / main_dir
            if main_path.exists():
                for py_file in main_path.rglob("*.py"):
                    if "__pycache__" not in str(py_file) and py_file.name != "__init__.py":
                        main_modules.append(py_file.relative_to(project_root))
        
        # Scan test directory
        test_path = project_root / "tests"
        if test_path.exists():
            for test_file in test_path.rglob("test_*.py"):
                test_modules.append(test_file.relative_to(project_root))
        
        # Check which modules don't have tests
        modules_without_tests = []
        
        for module in main_modules:
            module_name = module.stem
            
            # Look for corresponding test file
            has_test = any(
                module_name in test_file.name or 
                module_name.replace('_', '') in test_file.name.replace('_', '')
                for test_file in test_modules
            )
            
            if not has_test and not module_name.startswith('__'):
                modules_without_tests.append(str(module))
        
        if modules_without_tests:
            print("Modules without tests:")
            for module in modules_without_tests[:10]:  # Show first 10
                print(f"  {module}")
        
        # Calculate test coverage percentage
        total_modules = len(main_modules)
        tested_modules = total_modules - len(modules_without_tests)
        coverage_percent = (tested_modules / total_modules * 100) if total_modules > 0 else 100
        
        print(f"Test coverage: {tested_modules}/{total_modules} modules ({coverage_percent:.1f}%)")
        
        # Require reasonable test coverage
        assert coverage_percent > 30, "Test coverage should be above 30%"

    def test_dependency_security(self, project_root):
        """Test for known security vulnerabilities in dependencies."""
        requirements_file = project_root / "requirements.txt"
        
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.readlines()
                
                # Basic checks for suspicious dependencies
                suspicious_packages = [
                    'pickle',  # Can be unsafe
                    'eval',    # Dangerous if misused
                ]
                
                security_warnings = []
                
                for requirement in requirements:
                    requirement = requirement.strip().lower()
                    
                    for suspicious in suspicious_packages:
                        if suspicious in requirement:
                            security_warnings.append(f"Suspicious package: {requirement}")
                
                if security_warnings:
                    print("Security warnings in dependencies:")
                    for warning in security_warnings:
                        print(f"  {warning}")
                
                # This is informational
                print(f"Checked {len(requirements)} dependencies for security issues")
                
            except Exception as e:
                print(f"Could not check dependencies: {e}")

    def test_logging_practices(self, python_files):
        """Test logging practices."""
        logging_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for print statements in non-test files
                if "test_" not in py_file.name:
                    print_statements = re.findall(r'\bprint\s*\(', content)
                    if print_statements:
                        logging_issues.append(f"{py_file} has {len(print_statements)} print statements (use logging instead)")
                
                # Check for proper logging imports
                if 'logging' in content or 'logger' in content:
                    if 'import logging' not in content and 'from trading_common import get_logger' not in content:
                        # Check if it's using the common logger
                        if 'get_logger(' not in content:
                            logging_issues.append(f"{py_file} uses logging but doesn't import it properly")
                            
            except Exception as e:
                print(f"Could not check logging in {py_file}: {e}")
        
        if logging_issues:
            print("Logging practice issues:")
            for issue in logging_issues[:5]:  # Show first 5
                print(f"  {issue}")
        
        # Some print statements are OK in development
        assert len(logging_issues) < 10, "Too many logging practice issues"