@echo off
REM ---- IJIES revision runner (Windows) -------------------------------------
REM Run this from anywhere; it `cd`s to the project root before invoking python.

setlocal
set "REPO=%~dp0..\.."
pushd "%REPO%"

python "recent-review-comments\revision\code\run_all.py" %*

popd
endlocal
