import os
import symbolic_pofk
import subprocess

def run_tests():
    # Get the path to the symbolic_pofk module
    symbolic_pofk_path = os.path.dirname(symbolic_pofk.__file__)
    
    # Construct the pytest command with the correct coverage path
    pytest_command = [
        'pytest',
        '--cov-report=xml',
        f'--cov={symbolic_pofk_path}',
        'tests/test_syren.py'
    ]
    
    # Run pytest
    subprocess.run(pytest_command)

if __name__ == "__main__":
    run_tests()
