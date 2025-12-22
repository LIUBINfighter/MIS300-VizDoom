"""Simple script to assert installed package versions roughly match requirements.txt
Usage: python scripts/check_deps.py
"""
import pkg_resources

REQUIREMENTS_FILE = "requirements.txt"

if __name__ == "__main__":
    reqs = []
    with open(REQUIREMENTS_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)

    for req in reqs:
        try:
            pkg_resources.require(req)
            print(f"OK: {req}")
        except Exception as e:
            print(f"MISMATCH: {req} -> {e}")
