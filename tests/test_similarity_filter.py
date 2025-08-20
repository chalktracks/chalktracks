import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import cv2
from chalk.preprocess.similarity_filter import main as similarity_main, add_arg_parser as similarity_parser


class TestSimilarityFilter:
    """Test similarity filtering functionality."""

    def setup_method(self):
        """Set up test fixtures with mock images."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.image_dir = self.test_dir / "images"
        self.image_dir.mkdir()
        
        # Create mock image files (we'll create simple test images)
        self.image_files = []
        for i in range(5):
            image_path = self.image_dir / f"test_image_{i:03d}.png"
            
            # Create simple test image (just different colored squares)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img.fill(i * 50)  # Different brightness for each image
            cv2.imwrite(str(image_path), img)
            
            self.image_files.append(image_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_similarity_filter_argument_parser(self):
        """Test similarity filter argument parser configuration."""
        import argparse
        parser = argparse.ArgumentParser()
        similarity_parser(parser)
        
        # Test with valid arguments
        args = parser.parse_args([
            '--ssim-threshold', '0.8',
            '--image-dir', 'test_dir'
        ])
        
        assert args.ssim_threshold == 0.8
        assert args.image_dir == 'test_dir'
        assert not args.visualise
        assert not args.dry_run

    def test_similarity_filter_with_flags(self):
        """Test similarity filter with optional flags."""
        import argparse
        parser = argparse.ArgumentParser()
        similarity_parser(parser)
        
        args = parser.parse_args([
            '--ssim-threshold', '0.9',
            '--image-dir', 'test_dir',
            '--visualise',
            '--dry-run'
        ])
        
        assert args.visualise
        assert args.dry_run

    def test_similarity_filter_missing_directory(self):
        """Test similarity filter with missing directory."""
        args = MagicMock()
        args.image_dir = str(self.test_dir / "nonexistent")
        args.ssim_threshold = 0.8
        args.visualise = False
        args.dry_run = False
        
        result = similarity_main(args)
        assert result == 1, "Should return error code for missing directory"

    @patch('matplotlib.pyplot.ion')
    @patch('matplotlib.pyplot.show')  
    @patch('matplotlib.pyplot.pause')
    def test_similarity_filter_dry_run(self, mock_pause, mock_show, mock_ion):
        """Test similarity filter in dry run mode."""
        args = MagicMock()
        args.image_dir = str(self.image_dir)
        args.ssim_threshold = 0.5  # Low threshold to trigger filtering
        args.visualise = False
        args.dry_run = True
        
        # Count files before
        files_before = len(list(self.image_dir.glob("*.png")))
        
        result = similarity_main(args)
        
        # Should succeed
        assert result == 0, "Should return success code"
        
        # Files should not be deleted in dry run mode
        files_after = len(list(self.image_dir.glob("*.png")))
        assert files_before == files_after, "Files should not be deleted in dry run mode"

    def test_similarity_filter_no_images(self):
        """Test similarity filter with directory containing no images."""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        args = MagicMock()
        args.image_dir = str(empty_dir)
        args.ssim_threshold = 0.8
        args.visualise = False
        args.dry_run = False
        
        result = similarity_main(args)
        assert result == 0, "Should return success code even with no images"

    def test_similarity_filter_very_high_threshold(self):
        """Test similarity filter with very high threshold (should keep all images)."""
        args = MagicMock()
        args.image_dir = str(self.image_dir)
        args.ssim_threshold = 0.99  # Very high threshold
        args.visualise = False
        args.dry_run = False
        
        files_before = len(list(self.image_dir.glob("*.png")))
        
        result = similarity_main(args)
        
        # Should succeed
        assert result == 0, "Should return success code"
        
        # Most/all files should be kept with very high threshold
        files_after = len(list(self.image_dir.glob("*.png")))
        assert files_after >= files_before - 1, "Most files should be kept with high threshold"

    def test_similarity_filter_very_low_threshold(self):
        """Test similarity filter with very low threshold (should remove most images)."""
        args = MagicMock()
        args.image_dir = str(self.image_dir)
        args.ssim_threshold = 0.1  # Very low threshold
        args.visualise = False
        args.dry_run = False
        
        files_before = len(list(self.image_dir.glob("*.png")))
        
        result = similarity_main(args)
        
        # Should succeed
        assert result == 0, "Should return success code"
        
        # Should have fewer files after filtering
        files_after = len(list(self.image_dir.glob("*.png")))
        assert files_after <= files_before, "Should have same or fewer files after filtering"
        assert files_after > 0, "Should keep at least one keyframe"


class TestImageExtensions:
    """Test handling of different image file extensions."""

    def setup_method(self):
        """Set up test fixtures with various image extensions."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.image_dir = self.test_dir / "images"
        self.image_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_various_image_extensions_recognized(self):
        """Test that various image extensions are recognized."""
        # Create files with different extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        created_files = []
        
        for i, ext in enumerate(extensions):
            image_path = self.image_dir / f"test_image_{i}{ext}"
            
            # Create simple test image
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img.fill(100)
            cv2.imwrite(str(image_path), img)
            created_files.append(image_path)
        
        # Run similarity filter
        args = MagicMock()
        args.image_dir = str(self.image_dir)
        args.ssim_threshold = 0.8
        args.visualise = False
        args.dry_run = True  # Don't actually delete
        
        result = similarity_main(args)
        
        # Should succeed and process all extensions
        assert result == 0, "Should successfully process various image extensions"

    def test_case_insensitive_extensions(self):
        """Test that uppercase and lowercase extensions work."""
        # Create files with mixed case extensions
        extensions = ['.PNG', '.jpg', '.JPEG', '.Bmp']
        
        for i, ext in enumerate(extensions):
            image_path = self.image_dir / f"test_image_{i}{ext}"
            
            # Create simple test image
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img.fill(100)
            cv2.imwrite(str(image_path), img)
        
        # Run similarity filter
        args = MagicMock()
        args.image_dir = str(self.image_dir)
        args.ssim_threshold = 0.8
        args.visualise = False
        args.dry_run = True
        
        result = similarity_main(args)
        
        # Should succeed with mixed case extensions
        assert result == 0, "Should handle mixed case extensions"
