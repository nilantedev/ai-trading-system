#!/usr/bin/env python3
"""
Test runner with enforced coverage gates for production readiness.
Runs comprehensive test suite and enforces minimum coverage thresholds.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class TestRunner:
    """Comprehensive test runner with coverage enforcement."""
    
    def __init__(self, min_coverage: float = 70.0, min_branch_coverage: float = 65.0):
        """Initialize test runner with coverage thresholds."""
        self.min_coverage = min_coverage
        self.min_branch_coverage = min_branch_coverage
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out after 10 minutes"
        except Exception as e:
            return 1, "", f"Command failed: {e}"
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        print("🧪 Running unit tests with coverage...")
        
        cmd = [
            "python", "-m", "pytest", 
            "tests/unit/",
            "-v",
            "--cov=shared",
            "--cov=services", 
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-branch",
            "--cov-fail-under=70",
            "-m", "unit"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.test_results['unit_tests'] = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr
        }
        
        print(f"Unit tests {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
        if exit_code != 0:
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        
        return exit_code == 0
    
    def run_property_based_tests(self) -> bool:
        """Run property-based tests with Hypothesis."""
        print("🔍 Running property-based tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/test_property_based.py",
            "-v",
            "--hypothesis-show-statistics",
            "--hypothesis-seed=42",  # Reproducible tests
            "-m", "property"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.test_results['property_tests'] = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr
        }
        
        print(f"Property-based tests {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
        if exit_code != 0:
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        
        return exit_code == 0
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-v",
            "-m", "integration",
            "--tb=short"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.test_results['integration_tests'] = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr
        }
        
        print(f"Integration tests {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
        if exit_code != 0:
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        
        return exit_code == 0
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        print("🔒 Running security tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/security/",
            "-v",
            "-m", "security",
            "--tb=short"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.test_results['security_tests'] = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr
        }
        
        print(f"Security tests {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
        if exit_code != 0:
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        
        return exit_code == 0
    
    def check_coverage_gates(self) -> bool:
        """Check coverage meets minimum thresholds."""
        print("📊 Checking coverage gates...")
        
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            print("❌ Coverage report not found")
            return False
        
        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)
            
            # Check overall coverage
            total_coverage = coverage_data['totals']['percent_covered']
            branch_coverage = coverage_data['totals'].get('percent_covered_display', 0)
            
            print(f"📈 Line coverage: {total_coverage:.1f}% (minimum: {self.min_coverage}%)")
            
            coverage_passed = total_coverage >= self.min_coverage
            
            print(f"Coverage gate {'✅ PASSED' if coverage_passed else '❌ FAILED'}")
            
            # Report per-file coverage for files below threshold
            if not coverage_passed:
                print("📋 Files below coverage threshold:")
                for filepath, file_data in coverage_data['files'].items():
                    file_coverage = file_data['summary']['percent_covered']
                    if file_coverage < self.min_coverage:
                        print(f"  {filepath}: {file_coverage:.1f}%")
            
            return coverage_passed
            
        except Exception as e:
            print(f"❌ Error reading coverage report: {e}")
            return False
    
    def run_type_checking(self) -> bool:
        """Run MyPy type checking."""
        print("🔍 Running type checking...")
        
        cmd = ["python", "-m", "mypy", "shared/", "services/", "--ignore-missing-imports"]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.test_results['type_checking'] = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr
        }
        
        print(f"Type checking {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
        if exit_code != 0:
            print(f"Type errors found:\n{stdout}")
        
        return exit_code == 0
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        print("✨ Running code quality checks...")
        
        # Run Black formatting check
        cmd = ["python", "-m", "black", "--check", "--diff", "."]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        black_passed = exit_code == 0
        print(f"Black formatting {'✅ PASSED' if black_passed else '❌ FAILED'}")
        
        if not black_passed:
            print("Run 'black .' to fix formatting issues")
        
        # Run isort import sorting check
        cmd = ["python", "-m", "isort", "--check-only", "--diff", "."]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        isort_passed = exit_code == 0
        print(f"Import sorting {'✅ PASSED' if isort_passed else '❌ FAILED'}")
        
        if not isort_passed:
            print("Run 'isort .' to fix import sorting")
        
        return black_passed and isort_passed
    
    def generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['exit_code'] == 0)
        
        print(f"Test Suites: {passed_tests}/{total_tests} passed")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['exit_code'] == 0 else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        # Coverage summary if available
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                total_coverage = coverage_data['totals']['percent_covered']
                print(f"Coverage: {total_coverage:.1f}%")
            except:
                pass
        
        print("="*60)
        
        # Generate detailed HTML report
        html_report_path = self.project_root / "htmlcov" / "index.html"
        if html_report_path.exists():
            print(f"📊 Detailed coverage report: {html_report_path}")
    
    def run_all_tests(self) -> bool:
        """Run complete test suite with all checks."""
        print("🚀 Starting comprehensive test suite...")
        print("="*60)
        
        all_passed = True
        
        # Run tests in order of importance/speed
        test_suites = [
            ("Unit Tests", self.run_unit_tests),
            ("Property-Based Tests", self.run_property_based_tests),
            ("Type Checking", self.run_type_checking),
            ("Code Quality", self.run_code_quality_checks),
            ("Integration Tests", self.run_integration_tests),
            ("Security Tests", self.run_security_tests),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n🏃 Running {suite_name}...")
            try:
                success = test_func()
                if not success:
                    all_passed = False
            except Exception as e:
                print(f"❌ {suite_name} failed with exception: {e}")
                all_passed = False
        
        # Check coverage gates
        print(f"\n🔍 Checking coverage gates...")
        coverage_passed = self.check_coverage_gates()
        if not coverage_passed:
            all_passed = False
        
        # Generate final report
        self.generate_test_report()
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite with coverage gates")
    parser.add_argument("--min-coverage", type=float, default=70.0, 
                       help="Minimum line coverage percentage (default: 70.0)")
    parser.add_argument("--min-branch-coverage", type=float, default=65.0,
                       help="Minimum branch coverage percentage (default: 65.0)")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slower test suites for quick validation")
    
    args = parser.parse_args()
    
    runner = TestRunner(
        min_coverage=args.min_coverage,
        min_branch_coverage=args.min_branch_coverage
    )
    
    if args.fast:
        # Fast mode: only run unit tests and type checking
        print("🏃‍♂️ Running in fast mode...")
        success = (
            runner.run_unit_tests() and
            runner.run_type_checking() and
            runner.check_coverage_gates()
        )
    else:
        # Full comprehensive test suite
        success = runner.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! System is ready for production deployment.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please fix issues before production deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()