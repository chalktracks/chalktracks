import pytest
from unittest.mock import patch, MagicMock, call
import importlib
import sys
from pathlib import Path
import argparse
from chalk.cli import discover_commands, print_custom_help, main


class TestCLI:
    """Test CLI command discovery and help functionality."""

    def test_discover_commands_finds_valid_commands(self):
        """Test that command discovery finds modules with required functions."""
        commands = discover_commands()
        
        # Should find some commands
        assert len(commands) > 0, "Should discover at least one command"
        
        # Check that discovered commands have required structure
        for cmd_name, cmd_info in commands.items():
            assert 'module_name' in cmd_info, f"Command {cmd_name} should have module_name info"
            assert 'module_path' in cmd_info, f"Command {cmd_name} should have module_path"
            assert 'category' in cmd_info, f"Command {cmd_name} should have category"
            assert 'docstring' in cmd_info, f"Command {cmd_name} should have docstring"
            
            # Verify category is valid
            assert cmd_info['category'] in ['preprocess', 'model', 'util'], f"Command {cmd_name} should have valid category"

    def test_discover_commands_categories(self):
        """Test that commands are properly categorized."""
        commands = discover_commands()
        
        categories = set(cmd_info['category'] for cmd_info in commands.values())
        
        # Should have expected categories (at least some of them)
        expected_categories = {'preprocess', 'model', 'util'}
        found_categories = categories.intersection(expected_categories)
        assert len(found_categories) > 0, "Should find commands in expected categories"

    def test_discover_commands_excludes_special_modules(self):
        """Test that special modules are excluded from discovery."""
        commands = discover_commands()
        
        # Check that no discovered command comes from excluded modules
        for cmd_name, cmd_info in commands.items():
            module_name = cmd_info['module_name']
            assert '.__init__' not in module_name, "Should not discover __init__ modules"
            assert '.cli' not in module_name, "Should not discover cli module"
            assert '.__main__' not in module_name, "Should not discover __main__ modules"

    @patch('builtins.print')
    def test_print_custom_help(self, mock_print):
        """Test that custom help prints correctly."""
        # Create mock commands for testing
        mock_commands = {
            'test-cmd1': {
                'category': 'preprocess',
                'docstring': 'Test command 1 description'
            },
            'test-cmd2': {
                'category': 'util',
                'docstring': 'Test command 2 description'
            }
        }
        
        print_custom_help(mock_commands)
        
        # Verify print was called
        assert mock_print.called, "print_custom_help should call print"
        
        # Check that help contains expected elements
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        assert 'usage:' in printed_text
        assert 'chalk' in printed_text
        assert 'preprocess' in printed_text
        assert 'util' in printed_text

    def test_command_names_are_valid(self):
        """Test that discovered command names are valid."""
        commands = discover_commands()
        
        for cmd_name in commands.keys():
            # Command names should be valid identifiers (with hyphens converted from underscores)
            identifier_name = cmd_name.replace('-', '_')
            assert identifier_name.isidentifier(), f"Command name {cmd_name} should be a valid identifier"
            
            # Should not be empty
            assert len(cmd_name) > 0, "Command name should not be empty"


class TestCLIFunctional:
    """Functional tests for CLI operations."""

    def test_specific_commands_exist(self):
        """Test that specific expected commands exist."""
        commands = discover_commands()
        
        # These commands should exist based on the codebase
        expected_commands = [
            'add_sequence',
            'similarity_filter', 
            'symlink_images',
            'label_tool'
        ]
        
        found_commands = set(commands.keys())
        
        for expected_cmd in expected_commands:
            assert expected_cmd in found_commands, f"Expected command {expected_cmd} should be discovered"

    def test_commands_have_docstrings(self):
        """Test that commands have proper documentation."""
        commands = discover_commands()
        
        for cmd_name, cmd_info in commands.items():
            # Should have docstring field
            assert 'docstring' in cmd_info, f"Command {cmd_name} should have docstring field"
            
            docstring = cmd_info['docstring']
            if docstring and docstring != f"{cmd_name} command":
                assert len(docstring.strip()) > 0, f"Command {cmd_name} should have non-empty docstring"


class TestCLIHelp:
    """Test CLI help functionality."""

    @patch('sys.argv', ['chalk', '--help'])
    @patch('chalk.cli.print_custom_help')
    @patch('chalk.cli.discover_commands')
    def test_general_help_display(self, mock_discover, mock_print_help):
        """Test that general help is displayed correctly."""
        # Mock commands
        mock_discover.return_value = {
            'test_cmd': {
                'category': 'preprocess',
                'docstring': 'Test command'
            }
        }
        
        result = main()
        
        assert result == 0
        mock_print_help.assert_called_once()

    @patch('sys.argv', ['chalk', 'convert_model', '--help'])
    @patch('chalk.cli.lazy_import_module')
    @patch('chalk.cli.discover_commands')
    def test_command_specific_help_display(self, mock_discover, mock_import):
        """Test that command-specific help is displayed correctly."""
        # Mock commands with convert_model
        mock_discover.return_value = {
            'convert_model': {
                'module_name': 'chalk.model.convert_model',
                'category': 'model',
                'docstring': 'Convert model'
            }
        }
        
        # Mock the imported module
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        
        # Mock argparse.ArgumentParser and its print_help method
        with patch('argparse.ArgumentParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            
            result = main()
            
            assert result == 0
            mock_import.assert_called_once_with('chalk.model.convert_model')
            mock_module.add_arg_parser.assert_called_once()
            mock_parser.print_help.assert_called_once()

    @patch('sys.argv', ['chalk', 'nonexistent_command', '--help'])
    @patch('chalk.cli.discover_commands')
    def test_help_for_nonexistent_command(self, mock_discover):
        """Test behavior when help is requested for non-existent command."""
        # Mock no commands
        mock_discover.return_value = {}
        
        result = main()
        
        # Should not crash and should return 1 for "no commands found"
        assert result == 1

    @patch('sys.argv', ['chalk', 'convert_model', '--help'])
    @patch('chalk.cli.lazy_import_module')
    @patch('chalk.cli.discover_commands')
    def test_command_help_import_error(self, mock_discover, mock_import):
        """Test behavior when module import fails during help display."""
        # Mock commands
        mock_discover.return_value = {
            'convert_model': {
                'module_name': 'chalk.model.convert_model',
                'category': 'model',
                'docstring': 'Convert model'
            }
        }
        
        # Mock import failure
        mock_import.side_effect = ImportError("Module not found")
        
        with patch('builtins.print') as mock_print:
            result = main()
            
            assert result == 1
            # Should print error message
            error_calls = [call for call in mock_print.call_args_list 
                          if len(call[0]) > 0 and 'Error showing help' in str(call[0][0])]
            assert len(error_calls) > 0

    @patch('sys.argv', ['chalk'])
    @patch('chalk.cli.print_custom_help')
    @patch('chalk.cli.discover_commands')
    def test_no_args_shows_help(self, mock_discover, mock_print_help):
        """Test that calling with no args shows general help."""
        mock_discover.return_value = {'test': {'category': 'test', 'docstring': 'test'}}
        
        result = main()
        
        assert result == 0
        mock_print_help.assert_called_once()

    def test_command_execution_integration(self):
        """Integration test that command execution still works after help fix."""
        # This is a simple integration test that verifies the CLI structure
        # without complex mocking that could break internal argparse behavior
        
        # Test that we can import and discover commands
        commands = discover_commands()
        assert 'convert_model' in commands
        
        # Test that the convert_model command has the expected structure  
        cmd_info = commands['convert_model']
        # The module name could be either the package or the module within it
        assert cmd_info['module_name'] in ['chalk.model.convert_model', 'chalk.model.convert_model.convert_model']
        assert cmd_info['category'] == 'model'
        assert 'docstring' in cmd_info
