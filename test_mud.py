#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/tim/projects/2024/0621_chalk_line_follow/chalktracks')

from chalk.model.convert_model.convert_model import generate_mud_file_info, write_mud_file
from pathlib import Path
import tempfile

def test_mud_generation():
    print("Testing mud file generation...")
    
    # Test the mud file generation
    mud_info = generate_mud_file_info('test-model.cvimodel')
    print('Generated mud info:')
    for section, data in mud_info.items():
        print(f'[{section}]')
        for key, value in data.items():
            print(f'{key} = {value}')
        print()

    # Test writing to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mud', delete=False) as f:
        mud_file_path = Path(f.name)

    write_mud_file(mud_file_path, mud_info)
    print(f'Mud file written to: {mud_file_path}')
    print('Contents:')
    with open(mud_file_path) as f:
        print(f.read())

    # Cleanup
    mud_file_path.unlink()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_mud_generation()
