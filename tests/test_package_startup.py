import subprocess
import sys


def test_package_import_does_not_preload_morning_runner():
    code = "import sys, src; assert 'src.morning_runner' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_morning_runner_module_execution_has_no_circular_runtime_warning():
    code = "import runpy; runpy.run_module('src.morning_runner', run_name='not_main')"
    result = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "found in sys.modules after import of package 'src'" not in result.stderr
