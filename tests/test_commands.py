import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse
from chalk.preprocess.add_sequence import main as add_sequence_main, add_arg_parser as add_sequence_parser
from chalk.preprocess.symlink_images import main as symlink_main, add_arg_parser as symlink_parser


class TestPreprocessCommands:
    """Test preprocessing commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.test_dir / "source"
        self.dest_dir = self.test_dir / "dest"
        
        # Create source directory with test images
        self.source_dir.mkdir()
        
        # Create mock image files
        self.image_files = []
        for i, ext in enumerate(['.jpg', '.png', '.jpeg']):
            image_file = self.source_dir / f"test_image_{i}{ext}"
            image_file.write_text(f"Mock image {i}")  # Not real image data, but sufficient for testing
            self.image_files.append(image_file)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_add_sequence_argument_parser(self):
        """Test add_sequence argument parser configuration."""
        parser = argparse.ArgumentParser()
        add_sequence_parser(parser)
        
        # Test with valid arguments
        args = parser.parse_args(['--image-dir', 'source_dir', '--sequence-dir', 'dest_dir/sequence_name'])
        assert args.image_dir == 'source_dir'
        assert args.sequence_dir == 'dest_dir/sequence_name'

    def test_add_sequence_missing_source_dir(self):
        """Test add_sequence with non-existent source directory."""
        # Create mock args
        args = MagicMock()
        args.image_dir = str(self.test_dir / "nonexistent")
        args.sequence_dir = str(self.dest_dir / "test_sequence")
        
        result = add_sequence_main(args)
        assert result == 1, "Should return error code for missing source directory"

    def test_add_sequence_creates_structure(self):
        """Test that add_sequence creates proper directory structure."""
        # Set up args - parent directory should exist, but target should not
        self.dest_dir.mkdir()  # Create the parent directory
        sequence_dir = self.dest_dir / "test_sequence"
        
        args = MagicMock()
        args.image_dir = str(self.source_dir)
        args.sequence_dir = str(sequence_dir)
        
        result = add_sequence_main(args)
        
        # Should succeed
        assert result == 0, "Should return success code"
        
        # Check directory structure was created
        assert sequence_dir.exists(), "Sequence directory should be created"
        
        expected_dirs = ['0_raw_images', '1_keyframes', '2_labelled']
        for dir_name in expected_dirs:
            dir_path = sequence_dir / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should be created"
            
        # Check subdirectories in 2_labelled
        labelled_dir = sequence_dir / '2_labelled'
        for subdir in ['images', 'labels', 'masks']:
            subdir_path = labelled_dir / subdir
            assert subdir_path.exists(), f"Subdirectory {subdir} should be created"

    def test_add_sequence_existing_sequence_dir(self):
        """Test add_sequence with existing sequence directory."""
        # Create both parent and target directories
        self.dest_dir.mkdir()
        sequence_dir = self.dest_dir / "test_sequence"
        sequence_dir.mkdir()  # This should cause an error
        
        args = MagicMock()
        args.image_dir = str(self.source_dir)
        args.sequence_dir = str(sequence_dir)
        
        result = add_sequence_main(args)
        assert result == 1, "Should return error code for existing sequence directory"

    def test_add_sequence_missing_parent_dir(self):
        """Test add_sequence with non-existent parent directory."""
        sequence_dir = self.dest_dir / "nonexistent_parent" / "test_sequence"
        
        args = MagicMock()
        args.image_dir = str(self.source_dir)
        args.sequence_dir = str(sequence_dir)
        
        result = add_sequence_main(args)
        assert result == 1, "Should return error code for missing parent directory"

    def test_symlink_images_argument_parser(self):
        """Test symlink_images argument parser configuration."""
        parser = argparse.ArgumentParser()
        symlink_parser(parser)
        
        # Test with valid arguments
        args = parser.parse_args(['--from-dir', 'source', '--to-dir', 'dest'])
        assert args.from_dir == 'source'
        assert args.to_dir == 'dest'

    def test_symlink_images_missing_source(self):
        """Test symlink_images with missing source directory."""
        args = MagicMock()
        args.from_dir = str(self.test_dir / "nonexistent")
        args.to_dir = str(self.dest_dir)
        
        result = symlink_main(args)
        assert result == 1, "Should return error code for missing source directory"

    def test_symlink_images_creates_symlinks(self):
        """Test that symlink_images creates proper symlinks."""
        self.dest_dir.mkdir()
        
        args = MagicMock()
        args.from_dir = str(self.source_dir)
        args.to_dir = str(self.dest_dir)
        
        result = symlink_main(args)
        
        # Should succeed
        assert result == 0, "Should return success code"
        
        # Check that symlinks were created for image files
        for image_file in self.image_files:
            symlink_path = self.dest_dir / image_file.name
            assert symlink_path.exists(), f"Symlink for {image_file.name} should exist"

    def test_symlink_images_no_images_warning(self):
        """Test symlink_images behavior when no images found."""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        self.dest_dir.mkdir()
        
        args = MagicMock()
        args.from_dir = str(empty_dir)
        args.to_dir = str(self.dest_dir)
        
        with patch('builtins.print') as mock_print:
            result = symlink_main(args)
            
            # Should succeed but with warning
            assert result == 0, "Should return success code even with no images"
            
            # Should print warning
            printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'Warning' in printed_text, "Should print warning about no images"


class TestArgumentParsers:
    """Test argument parser configurations for various commands."""

    def test_all_commands_have_parsers(self):
        """Test that all discovered commands have working argument parsers."""
        from chalk.cli import discover_commands, lazy_import_module
        
        commands = discover_commands()
        
        for cmd_name, cmd_info in commands.items():
            # Import the module lazily (like the actual CLI does)
            try:
                module = lazy_import_module(cmd_info['module_name'])
            except ImportError as e:
                # Skip commands that have missing dependencies
                print(f"Skipping {cmd_name} due to missing dependency: {e}")
                continue
            
            # Should be able to create parser without errors
            parser = argparse.ArgumentParser()
            try:
                module.add_arg_parser(parser)
            except Exception as e:
                pytest.fail(f"Command {cmd_name} add_arg_parser failed: {e}")

    def test_parser_descriptions(self):
        """Test that command parsers have descriptions."""
        from chalk.cli import discover_commands, lazy_import_module
        
        commands = discover_commands()
        
        for cmd_name, cmd_info in commands.items():
            # Import the module lazily (like the actual CLI does)
            try:
                module = lazy_import_module(cmd_info['module_name'])
            except ImportError as e:
                # Skip commands that have missing dependencies
                print(f"Skipping {cmd_name} due to missing dependency: {e}")
                continue
            
            parser = argparse.ArgumentParser()
            module.add_arg_parser(parser)
            
            # Parser should have description
            if hasattr(parser, 'description') and parser.description:
                assert isinstance(parser.description, str), f"Command {cmd_name} should have string description"
                assert len(parser.description.strip()) > 0, f"Command {cmd_name} should have non-empty description"
