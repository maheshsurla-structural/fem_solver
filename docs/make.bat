@ECHO OFF
REM Windows build helper for Sphinx documentation.

pushd %~dp0

set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
goto end

:end
popd
