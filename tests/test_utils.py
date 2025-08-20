import pytest
import tempfile
import shutil
from pathlib import Path
from chalk.util.utils import put_files_into_dir


class TestUtils:
    """Test utility functions in chalk.util.utils module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.test_dir / "source"
        self.dest_dir = self.test_dir / "dest"
        
        # Create source directory with test files
        self.source_dir.mkdir()
        self.dest_dir.mkdir()
        
        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = self.source_dir / f"test_file_{i}.txt"
            test_file.write_text(f"Test content {i}")
            self.test_files.append(test_file)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_put_files_into_dir_symlink(self):
        """Test putting files into directory with symlinks."""
        put_files_into_dir(self.test_files, self.dest_dir, symlink=True)
        
        # Check that symlinks were created
        for test_file in self.test_files:
            symlink_path = self.dest_dir / test_file.name
            assert symlink_path.exists(), f"Symlink {symlink_path} should exist"
            assert symlink_path.is_symlink(), f"{symlink_path} should be a symlink"
            
            # Check that symlink points to correct file
            assert symlink_path.resolve() == test_file.resolve()
            
            # Check content is accessible through symlink
            assert symlink_path.read_text() == test_file.read_text()

    def test_put_files_into_dir_copy(self):
        """Test putting files into directory with copying."""
        put_files_into_dir(self.test_files, self.dest_dir, symlink=False)
        
        # Check that copies were created
        for test_file in self.test_files:
            copy_path = self.dest_dir / test_file.name
            assert copy_path.exists(), f"Copy {copy_path} should exist"
            assert not copy_path.is_symlink(), f"{copy_path} should not be a symlink"
            
            # Check content was copied correctly
            assert copy_path.read_text() == test_file.read_text()

    def test_put_files_into_dir_empty_list(self):
        """Test putting empty list of files."""
        put_files_into_dir([], self.dest_dir, symlink=True)
        
        # Destination directory should be empty
        dest_contents = list(self.dest_dir.iterdir())
        assert len(dest_contents) == 0, "Destination directory should be empty"

    def test_put_files_into_dir_nonexistent_dest(self):
        """Test behavior with non-existent destination directory."""
        nonexistent_dest = self.test_dir / "nonexistent"
        
        # Should raise an exception when trying to create symlinks to non-existent dir
        with pytest.raises(FileNotFoundError):
            put_files_into_dir(self.test_files, nonexistent_dest, symlink=True)
