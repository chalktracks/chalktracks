import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
from chalk.preprocess.label_tool.label_tool import DirectoryConfig, save_mask_file, rbg_to_int


class TestLabelTool:
    """Test label tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directory structure
        for dir_path in [self.input_dir, self.images_dir, self.masks_dir, self.labels_dir]:
            dir_path.mkdir(parents=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_directory_config_creation(self):
        """Test DirectoryConfig dataclass creation."""
        config = DirectoryConfig(
            input_images_dir=self.input_dir,
            output_images_dir=self.images_dir,
            output_masks_dir=self.masks_dir,
            output_labels_dir=self.labels_dir
        )
        
        assert config.input_images_dir == self.input_dir
        assert config.output_images_dir == self.images_dir
        assert config.output_masks_dir == self.masks_dir
        assert config.output_labels_dir == self.labels_dir

    def test_rbg_to_int_function(self):
        """Test RGB to integer conversion function."""
        # Test with known color mappings
        # These values depend on the segmentation_classes configuration
        
        # Test background (should map to 0)
        background_rgb = np.array([0, 0, 0])  # Assuming background is black
        result = rbg_to_int(background_rgb)
        assert isinstance(result, (int, np.integer)), "Should return integer"
        
        # Test with RGB array
        test_rgb = np.array([255, 0, 0])  # Red
        result = rbg_to_int(test_rgb)
        assert isinstance(result, (int, np.integer)), "Should return integer"

    def test_save_mask_file(self):
        """Test saving mask file functionality."""
        # Create a simple test mask (RGB image)
        test_mask = np.zeros((10, 10, 3), dtype=np.uint8)
        test_mask[0:5, 0:5] = [255, 0, 0]  # Red region
        test_mask[5:10, 5:10] = [0, 255, 0]  # Green region
        
        mask_file = self.masks_dir / "test_mask.png"
        
        # Save the mask
        save_mask_file(test_mask, mask_file)
        
        # Check that file was created
        assert mask_file.exists(), "Mask file should be created"
        
        # Check that saved file can be read back
        saved_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        assert saved_mask is not None, "Saved mask should be readable"
        assert saved_mask.shape == (10, 10), "Saved mask should have correct dimensions"

    def test_label_tool_missing_directories(self):
        """Test label tool behavior with missing required directories."""
        from chalk.preprocess.label_tool import main as label_tool_main
        
        # Test with missing output directories
        args = MagicMock()
        args.source_image_dir = str(self.input_dir)
        args.output_dir = str(self.test_dir / "missing_output")
        
        result = label_tool_main(args)
        
        # Should fail
        assert result == 1, "Should return error code with missing directories"

    def test_label_tool_missing_source(self):
        """Test label tool behavior with missing source directory."""
        from chalk.preprocess.label_tool import main as label_tool_main
        
        args = MagicMock()
        args.source_image_dir = str(self.test_dir / "missing_source")
        args.output_dir = str(self.output_dir)
        
        result = label_tool_main(args)
        
        # Should fail
        assert result == 1, "Should return error code with missing source directory"


class TestImageProcessing:
    """Test image processing related functions."""

    def test_mask_rgb_conversion_consistency(self):
        """Test that RGB to int conversion is consistent."""
        # Test that same RGB values produce same integer outputs
        rgb1 = np.array([255, 0, 0])
        rgb2 = np.array([255, 0, 0])
        
        result1 = rbg_to_int(rgb1)
        result2 = rbg_to_int(rgb2)
        
        assert result1 == result2, "Same RGB values should produce same integer"

    def test_mask_processing_with_different_colors(self):
        """Test mask processing with various colors."""
        # Create test image with different colored regions
        test_image = np.zeros((20, 20, 3), dtype=np.uint8)
        
        # Add different colored regions
        test_image[0:10, 0:10] = [255, 0, 0]    # Red
        test_image[0:10, 10:20] = [0, 255, 0]   # Green  
        test_image[10:20, 0:10] = [0, 0, 255]   # Blue
        test_image[10:20, 10:20] = [0, 0, 0]    # Black
        
        # Process each pixel
        for y in range(20):
            for x in range(20):
                pixel_rgb = test_image[y, x]
                result = rbg_to_int(pixel_rgb)
                assert isinstance(result, (int, np.integer)), f"Pixel at ({x},{y}) should produce integer result"
