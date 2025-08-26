#!/usr/bin/env python3
"""
Local security vulnerability scanner for AI Trading System.
Runs various security tools to identify vulnerabilities before deployment.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityScanner:
    """Local security vulnerability scanner."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "security-reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.scan_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "scans": {},
            "summary": {}
        }
    
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans."""
        logger.info("🔒 Starting comprehensive security scan...")
        
        # Dependency vulnerability scans
        self.run_safety_scan()
        self.run_pip_audit_scan()
        
        # Code security analysis
        self.run_bandit_scan()
        self.run_semgrep_scan()
        
        # Secrets detection
        self.run_secrets_scan()
        
        # Configuration analysis
        self.analyze_configuration()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.scan_results
    
    def run_safety_scan(self):
        """Run Safety vulnerability scan."""
        logger.info("Running Safety dependency vulnerability scan...")
        
        try:
            # Install safety if not present
            subprocess.run([sys.executable, "-m", "pip", "install", "safety"], 
                         capture_output=True, check=True)
            
            # Run safety check
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", 
                "--json", "--full-report"
            ], capture_output=True, text=True)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                self.scan_results["scans"]["safety"] = {
                    "status": "completed",
                    "vulnerabilities": safety_data,
                    "vulnerability_count": len(safety_data.get("vulnerabilities", []))
                }
            else:
                self.scan_results["scans"]["safety"] = {
                    "status": "no_vulnerabilities",
                    "vulnerability_count": 0
                }
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Safety scan failed: {e}")
            self.scan_results["scans"]["safety"] = {
                "status": "failed",
                "error": str(e)
            }
        except Exception as e:
            logger.warning(f"Safety scan error: {e}")
            self.scan_results["scans"]["safety"] = {
                "status": "error",
                "error": str(e)
            }
    
    def run_pip_audit_scan(self):
        """Run pip-audit vulnerability scan."""
        logger.info("Running pip-audit vulnerability scan...")
        
        try:
            # Install pip-audit if not present
            subprocess.run([sys.executable, "-m", "pip", "install", "pip-audit"], 
                         capture_output=True, check=True)
            
            # Run pip-audit
            result = subprocess.run([
                sys.executable, "-m", "pip_audit", "--format", "json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                self.scan_results["scans"]["pip_audit"] = {
                    "status": "completed",
                    "vulnerabilities": audit_data,
                    "vulnerability_count": len(audit_data.get("vulnerabilities", []))
                }
            else:
                self.scan_results["scans"]["pip_audit"] = {
                    "status": "no_vulnerabilities",
                    "vulnerability_count": 0
                }
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"pip-audit scan completed with issues: {e}")
            # pip-audit exits with non-zero when vulnerabilities found
            if e.stdout:
                try:
                    audit_data = json.loads(e.stdout)
                    self.scan_results["scans"]["pip_audit"] = {
                        "status": "completed",
                        "vulnerabilities": audit_data,
                        "vulnerability_count": len(audit_data.get("vulnerabilities", []))
                    }
                except json.JSONDecodeError:
                    self.scan_results["scans"]["pip_audit"] = {
                        "status": "failed",
                        "error": "Could not parse pip-audit output"
                    }
        except Exception as e:
            logger.warning(f"pip-audit scan error: {e}")
            self.scan_results["scans"]["pip_audit"] = {
                "status": "error", 
                "error": str(e)
            }
    
    def run_bandit_scan(self):
        """Run Bandit code security scan."""
        logger.info("Running Bandit code security scan...")
        
        try:
            # Install bandit if not present
            subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], 
                         capture_output=True, check=True)
            
            # Run bandit
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", ".", 
                "-f", "json", "-x", "/.venv/,/tests/,/scripts/"
            ], capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                issues = bandit_data.get("results", [])
                
                # Categorize by severity
                severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for issue in issues:
                    severity = issue.get("issue_severity", "LOW")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                self.scan_results["scans"]["bandit"] = {
                    "status": "completed",
                    "issues": issues,
                    "issue_count": len(issues),
                    "severity_breakdown": severity_counts,
                    "metrics": bandit_data.get("metrics", {})
                }
            else:
                self.scan_results["scans"]["bandit"] = {
                    "status": "no_issues",
                    "issue_count": 0
                }
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Bandit scan completed with issues found")
            if e.stdout:
                try:
                    bandit_data = json.loads(e.stdout)
                    issues = bandit_data.get("results", [])
                    
                    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                    for issue in issues:
                        severity = issue.get("issue_severity", "LOW")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    self.scan_results["scans"]["bandit"] = {
                        "status": "completed",
                        "issues": issues,
                        "issue_count": len(issues),
                        "severity_breakdown": severity_counts,
                        "metrics": bandit_data.get("metrics", {})
                    }
                except json.JSONDecodeError:
                    self.scan_results["scans"]["bandit"] = {
                        "status": "failed",
                        "error": "Could not parse bandit output"
                    }
        except Exception as e:
            logger.warning(f"Bandit scan error: {e}")
            self.scan_results["scans"]["bandit"] = {
                "status": "error",
                "error": str(e)
            }
    
    def run_semgrep_scan(self):
        """Run Semgrep static analysis."""
        logger.info("Running Semgrep static analysis...")
        
        try:
            # Check if semgrep is available
            result = subprocess.run(["which", "semgrep"], capture_output=True)
            if result.returncode != 0:
                # Try to install semgrep
                subprocess.run([sys.executable, "-m", "pip", "install", "semgrep"], 
                             capture_output=True, check=True)
            
            # Run semgrep with auto config
            result = subprocess.run([
                "semgrep", "--config=auto", "--json", "."
            ], capture_output=True, text=True)
            
            if result.stdout:
                semgrep_data = json.loads(result.stdout)
                findings = semgrep_data.get("results", [])
                
                # Categorize by severity
                severity_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
                for finding in findings:
                    severity = finding.get("extra", {}).get("severity", "INFO")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                self.scan_results["scans"]["semgrep"] = {
                    "status": "completed",
                    "findings": findings,
                    "finding_count": len(findings),
                    "severity_breakdown": severity_counts
                }
            else:
                self.scan_results["scans"]["semgrep"] = {
                    "status": "no_findings",
                    "finding_count": 0
                }
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Semgrep scan issues: {e}")
            self.scan_results["scans"]["semgrep"] = {
                "status": "failed",
                "error": str(e)
            }
        except Exception as e:
            logger.warning(f"Semgrep scan error: {e}")
            self.scan_results["scans"]["semgrep"] = {
                "status": "error",
                "error": str(e)
            }
    
    def run_secrets_scan(self):
        """Run secrets detection scan."""
        logger.info("Running secrets detection scan...")
        
        try:
            # Simple regex-based secrets scan
            secrets_patterns = [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', "Password in code"),
                (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', "API key in code"),
                (r'secret[_-]?key\s*=\s*["\'][^"\']{16,}["\']', "Secret key in code"),
                (r'token\s*=\s*["\'][^"\']{20,}["\']', "Token in code"),
                (r'aws[_-]?access[_-]?key[_-]?id', "AWS Access Key ID"),
                (r'aws[_-]?secret[_-]?access[_-]?key', "AWS Secret Access Key"),
                (r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----', "Private key"),
            ]
            
            secrets_found = []
            
            # Scan Python files
            for py_file in self.project_root.rglob("*.py"):
                if "/.venv/" in str(py_file) or "/venv/" in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern, description in secrets_patterns:
                        import re
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_number = content[:match.start()].count('\n') + 1
                            secrets_found.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_number,
                                "type": description,
                                "matched_text": match.group()[:50] + "..." if len(match.group()) > 50 else match.group()
                            })
                            
                except Exception as e:
                    logger.debug(f"Error scanning {py_file}: {e}")
            
            # Also scan configuration files
            for config_file in [".env", ".env.local", ".env.production"]:
                config_path = self.project_root / config_file
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            lines = f.readlines()
                            
                        for i, line in enumerate(lines, 1):
                            if "=" in line and not line.strip().startswith("#"):
                                key, value = line.split("=", 1)
                                if len(value.strip()) > 20 and not value.strip().startswith("${"):
                                    secrets_found.append({
                                        "file": config_file,
                                        "line": i,
                                        "type": "Configuration value",
                                        "matched_text": f"{key.strip()}=***"
                                    })
                                    
                    except Exception as e:
                        logger.debug(f"Error scanning {config_file}: {e}")
            
            self.scan_results["scans"]["secrets"] = {
                "status": "completed",
                "secrets_found": secrets_found,
                "secret_count": len(secrets_found)
            }
            
        except Exception as e:
            logger.warning(f"Secrets scan error: {e}")
            self.scan_results["scans"]["secrets"] = {
                "status": "error",
                "error": str(e)
            }
    
    def analyze_configuration(self):
        """Analyze configuration security."""
        logger.info("Analyzing configuration security...")
        
        config_issues = []
        
        # Check for default/weak configurations
        env_files = [".env", ".env.local", ".env.production"]
        
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        content = f.read()
                    
                    # Check for default passwords
                    default_patterns = [
                        "password123",
                        "admin123", 
                        "changeme",
                        "default",
                        "secret",
                        "password"
                    ]
                    
                    for pattern in default_patterns:
                        if pattern.lower() in content.lower():
                            config_issues.append({
                                "file": env_file,
                                "issue": f"Potential default/weak value: {pattern}",
                                "severity": "HIGH"
                            })
                    
                    # Check for debug mode in production
                    if "production" in env_file and "DEBUG=True" in content:
                        config_issues.append({
                            "file": env_file,
                            "issue": "Debug mode enabled in production config",
                            "severity": "HIGH"
                        })
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {env_file}: {e}")
        
        # Check Docker configurations
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            try:
                with open(dockerfile_path, 'r') as f:
                    content = f.read()
                
                if "USER root" in content and "USER " not in content[content.find("USER root")+10:]:
                    config_issues.append({
                        "file": "Dockerfile",
                        "issue": "Container running as root user",
                        "severity": "MEDIUM"
                    })
                    
            except Exception as e:
                logger.debug(f"Error analyzing Dockerfile: {e}")
        
        self.scan_results["scans"]["configuration"] = {
            "status": "completed",
            "issues": config_issues,
            "issue_count": len(config_issues)
        }
    
    def generate_summary(self):
        """Generate scan summary."""
        logger.info("Generating security scan summary...")
        
        total_vulnerabilities = 0
        total_code_issues = 0
        total_secrets = 0
        total_config_issues = 0
        
        critical_issues = []
        
        # Count vulnerabilities
        if "safety" in self.scan_results["scans"]:
            total_vulnerabilities += self.scan_results["scans"]["safety"].get("vulnerability_count", 0)
            
        if "pip_audit" in self.scan_results["scans"]:
            total_vulnerabilities += self.scan_results["scans"]["pip_audit"].get("vulnerability_count", 0)
        
        # Count code issues
        if "bandit" in self.scan_results["scans"]:
            bandit_data = self.scan_results["scans"]["bandit"]
            total_code_issues += bandit_data.get("issue_count", 0)
            
            # Check for critical Bandit issues
            severity_breakdown = bandit_data.get("severity_breakdown", {})
            if severity_breakdown.get("HIGH", 0) > 0:
                critical_issues.append(f"Bandit found {severity_breakdown['HIGH']} high-severity code issues")
        
        if "semgrep" in self.scan_results["scans"]:
            semgrep_data = self.scan_results["scans"]["semgrep"]
            total_code_issues += semgrep_data.get("finding_count", 0)
        
        # Count secrets
        if "secrets" in self.scan_results["scans"]:
            secrets_data = self.scan_results["scans"]["secrets"]
            total_secrets = secrets_data.get("secret_count", 0)
            if total_secrets > 0:
                critical_issues.append(f"Found {total_secrets} potential secrets in code")
        
        # Count configuration issues
        if "configuration" in self.scan_results["scans"]:
            config_data = self.scan_results["scans"]["configuration"]
            total_config_issues = config_data.get("issue_count", 0)
            
            # Check for high-severity config issues
            high_severity_config = [
                issue for issue in config_data.get("issues", []) 
                if issue.get("severity") == "HIGH"
            ]
            if high_severity_config:
                critical_issues.append(f"Found {len(high_severity_config)} high-severity configuration issues")
        
        # Overall risk assessment
        if total_vulnerabilities > 10 or total_secrets > 0 or len(critical_issues) > 0:
            risk_level = "HIGH"
        elif total_vulnerabilities > 5 or total_code_issues > 20:
            risk_level = "MEDIUM" 
        else:
            risk_level = "LOW"
        
        self.scan_results["summary"] = {
            "risk_level": risk_level,
            "total_vulnerabilities": total_vulnerabilities,
            "total_code_issues": total_code_issues,
            "total_secrets": total_secrets,
            "total_config_issues": total_config_issues,
            "critical_issues": critical_issues,
            "recommendations": self.generate_recommendations(risk_level, critical_issues)
        }
    
    def generate_recommendations(self, risk_level: str, critical_issues: List[str]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Address critical security issues before deployment")
            
        if critical_issues:
            recommendations.append("🔥 Fix critical issues first:")
            recommendations.extend([f"  - {issue}" for issue in critical_issues])
        
        recommendations.extend([
            "🔒 Update all vulnerable dependencies to secure versions",
            "🧹 Review and fix code security issues flagged by static analysis",
            "🔑 Ensure no secrets or credentials are committed to the repository",
            "⚙️  Review configuration files for security best practices",
            "🧪 Run security scans regularly in CI/CD pipeline",
            "📝 Document security review process and findings"
        ])
        
        return recommendations
    
    def save_results(self):
        """Save scan results to files."""
        # Save JSON report
        json_report = self.reports_dir / f"security-scan-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        with open(json_report, 'w') as f:
            json.dump(self.scan_results, f, indent=2)
        
        # Save text summary
        text_report = self.reports_dir / f"security-summary-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.txt"
        with open(text_report, 'w') as f:
            f.write("🔒 SECURITY SCAN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Scan Date: {self.scan_results['timestamp']}\n")
            f.write(f"Risk Level: {self.scan_results['summary']['risk_level']}\n\n")
            
            f.write("FINDINGS:\n")
            f.write(f"  - Dependency Vulnerabilities: {self.scan_results['summary']['total_vulnerabilities']}\n")
            f.write(f"  - Code Security Issues: {self.scan_results['summary']['total_code_issues']}\n")
            f.write(f"  - Potential Secrets: {self.scan_results['summary']['total_secrets']}\n")
            f.write(f"  - Configuration Issues: {self.scan_results['summary']['total_config_issues']}\n\n")
            
            if self.scan_results['summary']['critical_issues']:
                f.write("CRITICAL ISSUES:\n")
                for issue in self.scan_results['summary']['critical_issues']:
                    f.write(f"  🚨 {issue}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            for rec in self.scan_results['summary']['recommendations']:
                f.write(f"  {rec}\n")
        
        logger.info(f"📊 Security scan results saved:")
        logger.info(f"  JSON: {json_report}")
        logger.info(f"  Summary: {text_report}")
        
        return json_report, text_report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading System Security Scanner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = SecurityScanner(args.project_root)
    
    if args.output_dir:
        scanner.reports_dir = Path(args.output_dir)
        scanner.reports_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run all scans
        results = scanner.run_all_scans()
        
        # Print summary
        summary = results["summary"]
        print("\n" + "=" * 60)
        print("🔒 SECURITY SCAN RESULTS")
        print("=" * 60)
        print(f"Risk Level: {summary['risk_level']}")
        print(f"Vulnerabilities: {summary['total_vulnerabilities']}")
        print(f"Code Issues: {summary['total_code_issues']}")
        print(f"Secrets: {summary['total_secrets']}")
        print(f"Config Issues: {summary['total_config_issues']}")
        
        if summary['critical_issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"  - {issue}")
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        
        print("=" * 60)
        
        # Exit with error code if high risk
        if summary['risk_level'] == 'HIGH':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()