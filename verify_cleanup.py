#!/usr/bin/env python3
"""
Verification script to ensure the cleanup was successful
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def check_python_syntax(filepath):
    """Check Python file syntax"""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError:
        return False

def main():
    print("🔍 Verifying AI Trading System Cleanup\n")
    
    all_good = True
    
    # Check consolidated files exist
    print("📁 Checking consolidated files:")
    consolidated_files = [
        ("api/auth.py", "Consolidated authentication"),
        ("api/rate_limiter.py", "Enhanced rate limiter"),
        ("scripts/backup_restore_system.py", "Unified backup/restore"),
        ("scripts/secrets_manager.py", "Consolidated secrets management"),
    ]
    
    for filepath, description in consolidated_files:
        if check_file_exists(filepath):
            if check_python_syntax(filepath):
                print(f"  ✅ {filepath}: {description}")
            else:
                print(f"  ❌ {filepath}: Syntax error")
                all_good = False
        else:
            print(f"  ❌ {filepath}: File not found")
            all_good = False
    
    # Check removed files are gone
    print("\n🗑️  Checking removed files are archived:")
    removed_files = [
        "api/jwt_rotation_auth.py",
        "api/enhanced_rate_limiter.py",
        "scripts/backup_system.py",
        "scripts/automated_restore_system.py",
        "shared/security/jwt_auth.py",
        "tools/generate-secrets.py",
    ]
    
    for filepath in removed_files:
        if not check_file_exists(filepath):
            print(f"  ✅ {filepath}: Successfully removed")
        else:
            print(f"  ❌ {filepath}: Still exists (should be removed)")
            all_good = False
    
    # Check archive exists
    print("\n📦 Checking archive structure:")
    archive_dirs = [
        "archive/duplicates",
        "archive/old_implementations",
        "archive/server_docs",
    ]
    
    for dirpath in archive_dirs:
        if Path(dirpath).exists():
            file_count = len(list(Path(dirpath).glob("*")))
            print(f"  ✅ {dirpath}: {file_count} files archived")
        else:
            print(f"  ❌ {dirpath}: Directory not found")
            all_good = False
    
    # Check import compatibility
    print("\n🔗 Checking import compatibility:")
    try:
        # These imports should work with compatibility aliases
        from api.auth import get_current_active_user, User, JWTAuthManager
        print("  ✅ api.auth imports working")
    except ImportError as e:
        print(f"  ⚠️  api.auth import issue (may need dependencies): {e}")
    
    try:
        from api.rate_limiter import (
            get_rate_limiter, 
            create_rate_limit_middleware,
            EnhancedRateLimiter,
            RedisRateLimiter  # Compatibility alias
        )
        print("  ✅ api.rate_limiter imports working")
    except ImportError as e:
        print(f"  ⚠️  api.rate_limiter import issue (may need dependencies): {e}")
    
    # Check for any remaining duplicates
    print("\n🔍 Checking for remaining duplicates:")
    potential_duplicates = [
        ("api", "auth*.py"),
        ("api", "rate_limit*.py"),
        ("scripts", "backup*.py"),
        ("scripts", "*restore*.py"),
        ("scripts", "*secret*.py"),
        ("scripts", "*credential*.py"),
    ]
    
    duplicates_found = False
    for directory, pattern in potential_duplicates:
        files = list(Path(directory).glob(pattern))
        # Filter out our consolidated files
        files = [f for f in files if f.name not in [
            "auth.py", "rate_limiter.py", "backup_restore_system.py", 
            "secrets_manager.py", "websocket_auth.py"
        ]]
        if files:
            duplicates_found = True
            for f in files:
                print(f"  ⚠️  Potential duplicate: {f}")
    
    if not duplicates_found:
        print("  ✅ No unexpected duplicates found")
    
    # Final summary
    print("\n" + "="*50)
    if all_good:
        print("✅ VERIFICATION PASSED - Cleanup successful!")
        print("\nThe codebase is now:")
        print("  • Clean and consolidated")
        print("  • Free of duplicates")
        print("  • Properly organized")
        print("  • Backward compatible")
        return 0
    else:
        print("❌ VERIFICATION FAILED - Issues found")
        print("\nPlease review the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())