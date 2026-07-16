"""
IntelliMoE Root Entrypoint Wrapper.
Redirects execution to ui/app.py while preserving parent package references and context.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("ui/app.py", run_name="__main__")
