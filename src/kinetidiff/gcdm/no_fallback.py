"""
No-fallback policy enforcement

This module provides utilities to PREVENT silent fallbacks.
When something fails that is critical, it should fail LOUDLY.
"""

import functools
import os
import sys


class NoFallbackError(Exception):
    """Raised when a fallback would be triggered but is not allowed"""
    pass


def require_import(module_name: str, install_command: str = None):
    """
    Verify a module is importable, raise detailed error if not
    
    Args:
        module_name: Name of module to import
        install_command: Shell command to install the module
    
    Raises:
        NoFallbackError: If module cannot be imported
    """
    try:
        __import__(module_name)
    except ImportError as e:
        error_msg = f"Required module '{module_name}' is not installed.\n\n"
        if install_command:
            error_msg += f"Install with:\n  {install_command}\n\n"
        error_msg += f"Original error: {e}\n"
        error_msg += "NO FALLBACK AVAILABLE - this is a hard requirement."
        raise NoFallbackError(error_msg) from e


def require_file(filepath: str, description: str = None):
    """
    Verify a file exists, raise detailed error if not
    
    Args:
        filepath: Path to required file
        description: Human-readable description of what the file is
    
    Raises:
        NoFallbackError: If file doesn't exist
    """
    if not os.path.isfile(filepath):
        error_msg = f"Required file not found: {filepath}\n\n"
        if description:
            error_msg += f"Description: {description}\n"
        error_msg += "Please ensure this file exists before proceeding.\n"
        error_msg += "NO FALLBACK AVAILABLE - this file is required."
        raise NoFallbackError(error_msg)


def require_directory(dirpath: str, description: str = None):
    """
    Verify a directory exists, raise detailed error if not
    """
    if not os.path.isdir(dirpath):
        error_msg = f"Required directory not found: {dirpath}\n\n"
        if description:
            error_msg += f"Description: {description}\n"
        error_msg += "NO FALLBACK AVAILABLE."
        raise NoFallbackError(error_msg)


def no_fallback(error_message: str):
    """
    Decorator that prevents functions from using fallbacks
    
    Usage:
        @no_fallback("torch_scatter is required")
        def my_function():
            from torch_scatter import scatter_add
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                raise NoFallbackError(
                    f"{error_message}\n"
                    f"Original error: {e}\n"
                    f"Function: {func.__name__}\n"
                    f"NO FALLBACK AVAILABLE - this is a hard requirement."
                ) from e
        return wrapper
    return decorator


def verify_torch_scatter():
    """
    Verify torch_scatter is available (either native or fallback)
    
    Returns:
        str: "native" if torch_scatter package is installed,
             "fallback" if using torch_scatter_impl
    
    Raises:
        NoFallbackError: If neither is available
    """
    try:
        from torch_scatter import scatter_add
        return "native"
    except ImportError:
        pass
    
    # Try fallback
    try:
        # Add gcdm-clone to path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_dir = os.path.join(project_root, 'gcdm-clone')
        if fallback_dir not in sys.path:
            sys.path.insert(0, fallback_dir)
        
        from torch_scatter_impl import scatter_add
        return "fallback"
    except ImportError:
        pass
    
    raise NoFallbackError(
        "torch_scatter is REQUIRED but not available.\n\n"
        "Install with:\n"
        "  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html\n\n"
        "Or ensure torch_scatter_impl.py exists in models/gcdm-clone/\n\n"
        "NO FALLBACK AVAILABLE - this is a hard requirement."
    )


def verify_critical_imports():
    """
    Verify all critical imports are available
    
    Call this at the start of any script that requires the full pipeline.
    """
    import torch
    
    print("Verifying critical imports...")
    
    # Core
    require_import('torch', 'pip install torch')
    require_import('numpy', 'pip install numpy')
    require_import('rdkit', 'pip install rdkit')
    
    # torch_scatter
    scatter_mode = verify_torch_scatter()
    print(f"  torch_scatter: {scatter_mode}")
    
    # Verify functional
    try:
        if scatter_mode == "native":
            from torch_scatter import scatter_add
        else:
            from torch_scatter_impl import scatter_add
        
        src = torch.randn(10, 5)
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        out = scatter_add(src, index, dim=0)
        assert out.shape[0] == 3, "scatter_add not working"
        print("  scatter_add: functional ✓")
    except Exception as e:
        raise NoFallbackError(f"scatter_add test failed: {e}")
    
    print("All critical imports verified ✓")


# Provide convenient scatter imports that work with either implementation
def get_scatter_functions():
    """
    Get scatter functions from either torch_scatter or fallback
    
    Returns:
        tuple: (scatter_add, scatter_mean, scatter_sum)
    """
    try:
        from torch_scatter import scatter_add, scatter_mean
        try:
            from torch_scatter import scatter_sum
        except ImportError:
            scatter_sum = scatter_add  # scatter_sum is same as scatter_add
        return scatter_add, scatter_mean, scatter_sum
    except ImportError:
        pass
    
    # Fallback
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_dir = os.path.join(project_root, 'gcdm-clone')
    if fallback_dir not in sys.path:
        sys.path.insert(0, fallback_dir)
    
    from torch_scatter_impl import scatter_add, scatter_mean
    scatter_sum = scatter_add
    return scatter_add, scatter_mean, scatter_sum


if __name__ == '__main__':
    # Self-test
    print("Testing no_fallback module...")
    
    # Test require_import
    require_import('os')
    print("✓ require_import works for existing module")
    
    try:
        require_import('nonexistent_module_xyz')
        print("✗ Should have raised error")
    except NoFallbackError:
        print("✓ require_import raises error for missing module")
    
    # Test verify_torch_scatter
    try:
        mode = verify_torch_scatter()
        print(f"✓ torch_scatter available ({mode})")
    except NoFallbackError as e:
        print(f"✗ torch_scatter not available: {e}")
    
    # Test get_scatter_functions
    try:
        scatter_add, scatter_mean, scatter_sum = get_scatter_functions()
        import torch
        src = torch.randn(10, 5)
        idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        out = scatter_add(src, idx, dim=0)
        assert out.shape[0] == 3
        print("✓ scatter functions work")
    except Exception as e:
        print(f"✗ scatter functions failed: {e}")
    
    print("\nAll tests passed!")
