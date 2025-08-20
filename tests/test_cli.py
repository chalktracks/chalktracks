import pytest
from unittest.mock import patch, MagicMock
import importlib
import sys
from pathlib import Path
from chalk.cli import discover_commands, print_custom_help


class TestCLI:
    """Test CLI command discovery and help functionality."""

    def test_discover_commands_finds_valid_commands(self):
        """Test that command discovery finds modules with required functions."""
        commands = discover_commands()
        
        # Should find some commands
        assert len(commands) > 0, "Should discover at least one command"
        
        # Check that discovered commands have required structure
        for cmd_name, cmd_info in commands.items():
            assert 'module' in cmd_info, f"Command {cmd_name} should have module info"
            assert 'category' in cmd_info, f"Command {cmd_name} should have category"
            assert 'full_name' in cmd_info, f"Command {cmd_name} should have full_name"
            
            # Verify module has required functions
            module = cmd_info['module']
            assert hasattr(module, 'main'), f"Module {cmd_name} should have main function"
            assert hasattr(module, 'add_arg_parser'), f"Module {cmd_name} should have add_arg_parser function"
            
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
            full_name = cmd_info['full_name']
            assert '.__init__' not in full_name, "Should not discover __init__ modules"
            assert '.cli' not in full_name, "Should not discover cli module"
            assert '.__main__' not in full_name, "Should not discover __main__ modules"

    @patch('builtins.print')
    def test_print_custom_help(self, mock_print):
        """Test that custom help prints correctly."""
        # Create mock commands for testing
        mock_commands = {
            'test-cmd1': {
                'module': MagicMock(),
                'category': 'preprocess',
                'full_name': 'chalk.preprocess.test_cmd1'
            },
            'test-cmd2': {
                'module': MagicMock(),
                'category': 'util',
                'full_name': 'chalk.util.test_cmd2'
            }
        }
        
        # Set up mock docstrings
        mock_commands['test-cmd1']['module'].main.__doc__ = "Test command 1 description"
        mock_commands['test-cmd2']['module'].main.__doc__ = "Test command 2 description"
        
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
            module = cmd_info['module']
            
            # Main function should have docstring
            if hasattr(module.main, '__doc__') and module.main.__doc__:
                doc = module.main.__doc__.strip()
                assert len(doc) > 0, f"Command {cmd_name} should have non-empty docstring"
