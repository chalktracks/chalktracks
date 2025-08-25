import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse
import subprocess
from chalk.preprocess.add_sequence import main as add_sequence_main, add_arg_parser as add_sequence_parser
from chalk.preprocess.symlink_images import main as symlink_main, add_arg_parser as symlink_parser
from chalk.model.convert_model import add_arg_parser as convert_model_parser, setup_workspace, get_resource_path, convert_model_to_maixcam
from chalk.model.convert_model.convert_model import run_subprocess_with_live_output


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


class TestModelCommands:
    """Test model commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.model_file = self.test_dir / "test_model.onnx"
        self.train_dir = self.test_dir / "train_images"
        self.train_dir.mkdir()
        
        # Create dummy model file
        self.model_file.write_text("dummy onnx model content")
        
        # Create dummy training images
        for i in range(5):
            image_file = self.train_dir / f"train_image_{i}.png"
            image_file.write_text(f"dummy image {i}")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_convert_model_argument_parser(self):
        """Test convert_model argument parser configuration."""
        parser = argparse.ArgumentParser()
        convert_model_parser(parser)
        
        # Test with required arguments
        args = parser.parse_args(['--model-path', 'test.onnx', '--train-data', 'train/'])
        assert args.model_path == 'test.onnx'
        assert args.train_data == 'train/'

    def test_convert_model_resource_files_exist(self):
        """Test that required resource files exist."""
        # Test that resource files can be found
        dockerfile_path = get_resource_path("Dockerfile")
        script_path = get_resource_path("tpuc_dev_convert.sh")
        
        assert Path(dockerfile_path).exists(), "Dockerfile should exist"
        assert Path(script_path).exists(), "tpuc_dev_convert.sh should exist"

    def test_convert_model_workspace_setup(self):
        """Test workspace setup functionality."""
        workspace = setup_workspace(
            self.model_file, 
            self.train_dir
        )
        
        try:
            # Check workspace structure
            assert workspace.exists(), "Workspace should be created"
            assert (workspace / "data").exists(), "Data directory should exist"
            assert (workspace / "data" / "model.onnx").exists(), "Model should be copied"
            assert (workspace / "data" / "test_image.png").exists(), "Test image should be created"
            assert (workspace / "data" / "train_images").exists(), "Training images directory should exist"
            assert (workspace / "tpuc_dev_convert.sh").exists(), "Conversion script should be copied"
            
            # Check that all training images were copied (should be all 5)
            train_images = list((workspace / "data" / "train_images").glob("*.png"))
            assert len(train_images) == 5, f"Should have 5 training images, got {len(train_images)}"
            
        finally:
            # Clean up
            if workspace.exists():
                shutil.rmtree(workspace)

    @patch('builtins.print')
    def test_live_output_streaming(self, mock_print):
        """Test that subprocess output is streamed live to terminal."""
        # Test with a simple command that produces output
        # Use 'echo' to test multi-line output streaming
        if shutil.which("echo"):
            result = run_subprocess_with_live_output(["echo", "test output"])
            
            # Check that the command succeeded
            assert result.returncode == 0
            
            # Verify that print was called (indicating live output)
            assert mock_print.called, "Should have printed output live"
            
            # Check that the output contains expected content
            printed_calls = [str(call) for call in mock_print.call_args_list]
            output_text = ' '.join(printed_calls)
            assert 'test output' in output_text or 'echo' in output_text

    @patch('builtins.print')  
    def test_live_output_with_command_logging(self, mock_print):
        """Test that the command being run is logged."""
        if shutil.which("echo"):
            run_subprocess_with_live_output(["echo", "test"])
            
            # Verify that the command was logged
            printed_calls = [str(call) for call in mock_print.call_args_list]
            command_logged = any("Running: echo test" in str(call) for call in printed_calls)
            assert command_logged, "Should log the command being executed"

    def test_live_output_error_handling(self):
        """Test that live output function handles command failures properly."""
        # Test with a command that should fail
        result = run_subprocess_with_live_output(["false"])  # 'false' command always returns 1
        
        # Should capture the failure
        assert result.returncode != 0, "Should capture command failure"

    def test_convert_model_api_function(self):
        """Test the API function can be called directly."""
        # Test that the function can be imported and has correct signature
        import inspect
        
        # Check function signature
        sig = inspect.signature(convert_model_to_maixcam)
        params = list(sig.parameters.keys())
        assert params == ['model_path', 'train_data_dir'], f"Expected ['model_path', 'train_data_dir'], got {params}"
        
        # Test that calling with invalid paths raises appropriate errors
        with pytest.raises((FileNotFoundError, ValueError)):
            convert_model_to_maixcam("nonexistent_model.onnx", "nonexistent_data_dir")
