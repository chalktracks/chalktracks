import pytest
import tempfile
import shutil
from pathlib import Path
from chalk.util.view_images import detect_dataset_structure, main
from unittest.mock import patch, MagicMock
import argparse


class TestViewImages:
    """Test view_images functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_detect_dataset_structure_simple(self):
        """Test detection of simple image directory structure."""
        # Create simple directory with images
        images_dir = self.test_dir / "simple_images"
        images_dir.mkdir()
        
        # Create test images
        (images_dir / "image1.jpg").write_text("fake image")
        (images_dir / "image2.png").write_text("fake image")
        
        structure = detect_dataset_structure(images_dir)
        
        assert structure["type"] == "simple"
        assert structure["images_path"] == images_dir
        assert structure["has_labels"] == False
        assert structure["has_masks"] == False

    def test_detect_dataset_structure_structured_with_masks(self):
        """Test detection of structured directory with images and masks."""
        # Create structured directory
        structured_dir = self.test_dir / "structured"
        images_dir = structured_dir / "images"
        masks_dir = structured_dir / "masks"
        
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Create test files
        (images_dir / "image1.jpg").write_text("fake image")
        (masks_dir / "mask1.png").write_text("fake mask")
        
        structure = detect_dataset_structure(structured_dir)
        
        assert structure["type"] == "structured"
        assert structure["images_path"] == images_dir
        assert structure["has_labels"] == False
        assert structure["has_masks"] == True
        assert structure["masks_path"] == masks_dir

    def test_detect_dataset_structure_structured_with_labels(self):
        """Test detection of structured directory with images and labels."""
        # Create structured directory
        structured_dir = self.test_dir / "structured"
        images_dir = structured_dir / "images"
        labels_dir = structured_dir / "labels"
        
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create test files
        (images_dir / "image1.jpg").write_text("fake image")
        (labels_dir / "label1.txt").write_text("fake label")
        
        structure = detect_dataset_structure(structured_dir)
        
        assert structure["type"] == "structured"
        assert structure["images_path"] == images_dir
        assert structure["has_labels"] == True
        assert structure["has_masks"] == False
        assert structure["labels_path"] == labels_dir

    def test_detect_dataset_structure_structured_with_both(self):
        """Test detection of structured directory with images, labels, and masks."""
        # Create structured directory
        structured_dir = self.test_dir / "structured"
        images_dir = structured_dir / "images"
        labels_dir = structured_dir / "labels"
        masks_dir = structured_dir / "masks"
        
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Create test files
        (images_dir / "image1.jpg").write_text("fake image")
        (labels_dir / "label1.txt").write_text("fake label")
        (masks_dir / "mask1.png").write_text("fake mask")
        
        structure = detect_dataset_structure(structured_dir)
        
        assert structure["type"] == "structured"
        assert structure["images_path"] == images_dir
        assert structure["has_labels"] == True
        assert structure["has_masks"] == True
        assert structure["labels_path"] == labels_dir
        assert structure["masks_path"] == masks_dir

    def test_detect_dataset_structure_empty(self):
        """Test detection of empty directory."""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        structure = detect_dataset_structure(empty_dir)
        
        assert structure["type"] == "empty"
        assert structure["images_path"] == empty_dir
        assert structure["has_labels"] == False
        assert structure["has_masks"] == False

    def test_detect_dataset_structure_structured_but_empty_images(self):
        """Test structured directory but with empty images folder."""
        # Create structured directory but leave images empty
        structured_dir = self.test_dir / "structured"
        images_dir = structured_dir / "images"
        masks_dir = structured_dir / "masks"
        
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Only create mask, no images
        (masks_dir / "mask1.png").write_text("fake mask")
        
        structure = detect_dataset_structure(structured_dir)
        
        assert structure["type"] == "structured"
        assert structure["images_path"] == images_dir
        assert structure["has_masks"] == True

    @patch('chalk.util.view_images.FIFTYONE_AVAILABLE', False)
    def test_main_without_fiftyone(self):
        """Test main function when fiftyone is not available."""
        # Create simple directory with images
        images_dir = self.test_dir / "simple_images"
        images_dir.mkdir()
        (images_dir / "image1.jpg").write_text("fake image")
        
        # Mock args
        args = argparse.Namespace(
            image_dir=str(images_dir),
            name=None
        )
        
        # Test that it fails gracefully when fiftyone is not available
        # This will print an error message but should not raise an exception
        main(args)  # Should exit gracefully with error message

    @patch('chalk.util.view_images.FIFTYONE_AVAILABLE', True)
    @patch('chalk.util.view_images.fo.launch_app')
    @patch('chalk.util.view_images.fo.list_datasets')
    def test_main_directory_does_not_exist(self, mock_list_datasets, mock_launch_app):
        """Test main function with non-existent directory."""
        mock_list_datasets.return_value = []
        
        nonexistent_dir = self.test_dir / "does_not_exist"
        
        args = argparse.Namespace(
            image_dir=str(nonexistent_dir),
            name=None
        )
        
        with pytest.raises(FileNotFoundError):
            main(args)

    @patch('chalk.util.view_images.FIFTYONE_AVAILABLE', True)
    @patch('chalk.util.view_images.fo.launch_app')
    @patch('chalk.util.view_images.fo.list_datasets')
    def test_main_path_is_not_directory(self, mock_list_datasets, mock_launch_app):
        """Test main function when path is not a directory."""
        mock_list_datasets.return_value = []
        
        # Create a file instead of directory
        test_file = self.test_dir / "not_a_directory.txt"
        test_file.write_text("not a directory")
        
        args = argparse.Namespace(
            image_dir=str(test_file),
            name=None
        )
        
        with pytest.raises(ValueError, match="Path is not a directory"):
            main(args)

    def test_case_insensitive_image_extensions(self):
        """Test that image detection is case-insensitive."""
        # Create directory with mixed case extensions
        images_dir = self.test_dir / "mixed_case"
        images_dir.mkdir()
        
        # Create files with different case extensions
        (images_dir / "image1.JPG").write_text("fake image")
        (images_dir / "image2.PNG").write_text("fake image")
        (images_dir / "image3.jpeg").write_text("fake image")
        (images_dir / "image4.TIFF").write_text("fake image")
        
        structure = detect_dataset_structure(images_dir)
        
        assert structure["type"] == "simple"
        assert structure["images_path"] == images_dir

    def test_ignores_non_image_files(self):
        """Test that non-image files are ignored in simple directory detection."""
        # Create directory with mixed file types
        mixed_dir = self.test_dir / "mixed_files"
        mixed_dir.mkdir()
        
        # Create some image files and some non-image files
        (mixed_dir / "image1.jpg").write_text("fake image")
        (mixed_dir / "document.txt").write_text("text file")
        (mixed_dir / "script.py").write_text("python file")
        (mixed_dir / "image2.png").write_text("fake image")
        
        structure = detect_dataset_structure(mixed_dir)
        
        # Should still detect as simple since there are image files
        assert structure["type"] == "simple"
        assert structure["images_path"] == mixed_dir

    @patch('chalk.util.view_images.FIFTYONE_AVAILABLE', True)
    @patch('chalk.util.view_images.fo.launch_app')
    @patch('chalk.util.view_images.fo.list_datasets')
    @patch('chalk.util.view_images.fo.Dataset.from_images_dir')
    def test_main_with_simple_dataset(self, mock_from_images_dir, mock_list_datasets, mock_launch_app):
        """Test main function with a simple image dataset (without opening browser)."""
        # Setup mocks
        mock_list_datasets.return_value = []
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.app_config = MagicMock()
        mock_from_images_dir.return_value = mock_dataset
        
        mock_session = MagicMock()
        mock_launch_app.return_value = mock_session
        
        # Create simple directory with images
        images_dir = self.test_dir / "simple_images"
        images_dir.mkdir()
        (images_dir / "image1.jpg").write_text("fake image")
        
        # Mock args
        args = argparse.Namespace(
            image_dir=str(images_dir),
            name="test_dataset"
        )
        
        # Run main function - should not open browser due to mocking
        main(args)
        
        # Verify calls
        mock_from_images_dir.assert_called_once()
        mock_launch_app.assert_called_once_with(mock_dataset)
        mock_session.wait.assert_called_once()
