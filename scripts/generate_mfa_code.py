#!/usr/bin/env python3
"""Generate current MFA TOTP code from secret"""
import pyotp
import sys

if len(sys.argv) > 1:
    secret = sys.argv[1]
else:
    secret = "I23JMI2MTIR4LNNOAUCJOCYPNINIGL4M"

totp = pyotp.TOTP(secret)
current_code = totp.now()
print(current_code)
