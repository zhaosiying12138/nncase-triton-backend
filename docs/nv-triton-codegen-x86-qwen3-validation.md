# nv-triton-codegen x86 build and Qwen3 validation

This report records the local build and validation performed from the
`origin/dev/3.0` baseline for the `nv-triton-codegen` branch.

## Baseline

- Branch created from detached `origin/dev/3.0`.
- Commit: `dad56f4e97943e65647544366a0cd17a96f24035`
  (`Feature/reshard opt (#1466)`).
- No source code changes were required to make the x86 build or Qwen3 test pass.
- Build artifacts, model cache, test outputs, and generated local dependency
  packages were left untracked.

## Environment

- Python virtual environment: `zsy-nncase`
- Python version: `3.10.7`
- .NET SDK: `8.0.420`
- .NET root: `$HOME/.dotnet/zsy-nncase-dotnet8`
- Native compiler used for this checkout: `gcc-13` / `g++-13`
- Conan package folder: `.nuget/packages`
- Target scope: x86 CPU backend only

The public Conan/NuGet dependencies were restored from the public remotes.
Private `sunnycase` packages were unavailable during this run, so the local
environment supplied only the missing runtime/bootstrap dependencies:

- `nethost/8.0.8` was exported to the local Conan cache from the installed
  .NET SDK host pack.
- `OrtKISharp/0.0.2` and `libortki/0.0.2` were supplied through a local NuGet
  source generated from the public `nncase==2.11.0` Python wheel. These were
  used only to restore and publish the current source tree; the compiler DLL
  used in the test was rebuilt from this checkout.

## Build Method

The Python environment was created as:

```bash
/home/zhaosiying/anaconda3/envs/py312/bin/virtualenv \
  -p /usr/bin/python3.10 zsy-nncase

source zsy-nncase/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install conan==2.6.0 gitpython cmake==3.30.3 pytest \
  numpy toml pillow opencv-python transformers==4.52.4 \
  'accelerate>=0.26.0' safetensors ml-dtypes
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1+cpu
```

Native x86 build:

```bash
export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export CC=gcc-13
export CXX=g++-13

conan remote disable sunnycase
conan install . --build=missing \
  -s build_type=Release \
  -pr:a=toolchains/x86_64-linux.profile.jinja \
  -s:a compiler.version=13 \
  -o "&:runtime=False" \
  -o "&:python=True" \
  -o "&:tests=False" \
  -o "&:python_root=$PWD/zsy-nncase"

cmake --preset conan-release -DBUILD_BENCHMARK=OFF
cmake --build build/Release --config Release --parallel 4
cmake --install build/Release --prefix install
```

C# compiler publish:

```bash
export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$DOTNET_ROOT:$PATH"
export NUGET_PACKAGES="$PWD/.nuget/packages"
export NUGET_CERT_REVOCATION_MODE=offline
```

When the `sunnycase` NuGet source is reachable, the repository `NuGet.Config`
can be used directly. In this run it returned Cloudflare 521, so a generated
local config under `build/zsy-nuget/NuGet.local.Config` was used instead. Its
source mapping was limited to the missing private packages:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local-private" value="build/zsy-nuget/packages" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="local-private">
      <package pattern="OrtKISharp" />
      <package pattern="libortki" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
```

Restore and publish commands:

```bash

dotnet restore src/Nncase.Compiler/Nncase.Compiler.csproj \
  -r linux-x64 \
  --configfile build/zsy-nuget/NuGet.local.Config \
  -p:RestoreLockedMode=false \
  -v:minimal

dotnet publish src/Nncase.Compiler/Nncase.Compiler.csproj \
  -c Release --no-restore --sc false -r linux-x64 \
  -o install -v:minimal

cp -f install/lib/*.so install/
cp -f "$DOTNET_ROOT/packs/Microsoft.NETCore.App.Host.linux-x64/8.0.26/runtimes/linux-x64/native/libnethost.so" \
  install/lib/
cp -f install/lib/libnethost.so install/
```

Installed artifacts used by the test included:

- `install/Nncase.Compiler.dll`
- `install/lib/_nncase.cpython-310-x86_64-linux-gnu.so`
- `install/lib/libNncase.Runtime.Native.so`
- `install/lib/libnethost.so`
- `install/libortki.so`
- `install/libortools.so.9`

The runtime target check after build was:

```text
python 3.10.7
cpu True
k230 False
k510 False
vulkan False
compiler_exists True
```

## Test Method

The test was run with a clean native library search path. This is required
because inheriting the conda library path can make OrTools load an incompatible
`libgcc_s.so.1`.

```bash
export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests"
export LD_LIBRARY_PATH="$PWD/install:$PWD/install/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_PLUGIN_PATH="$PWD/install"
export NNCASE_TILING_MAX_SOLUTIONS=1

python -m pytest -vv -s tests/importer/huggingface_/disabled_test_qwen3.py
```

The test downloaded and saved `Qwen/Qwen3-0.6B` under:

```text
tests/llm/Qwen/Qwen3-0.6B
```

## Test Results

Pytest result:

```text
1 passed in 360.87s (0:06:00)
```

Output comparison summary:

```text
Pass [ eval cpu noptq 0 ] Output 0: cosine = 0.9999884963, threshold = 0.999
Pass [ eval cpu noptq 0 ] Output 1: cosine = 1.0000007153, threshold = 0.999
Pass [ eval cpu noptq 1 ] Output 0: cosine = 0.9999936223, threshold = 0.999
Pass [ eval cpu noptq 1 ] Output 1: cosine = 0.9999973774, threshold = 0.999
Pass [ eval cpu noptq 2 ] Output 0: cosine = 0.9999965429, threshold = 0.999
Pass [ eval cpu noptq 2 ] Output 1: cosine = 0.9999986887, threshold = 0.999
Pass [ infer cpu noptq 0 ] Output 0: cosine = 0.9990783334, threshold = 0.999
Pass [ infer cpu noptq 0 ] Output 1: cosine = 0.9999955297, threshold = 0.999
Pass [ infer cpu noptq 1 ] Output 0: cosine = 0.9997196794, threshold = 0.999
Pass [ infer cpu noptq 1 ] Output 1: cosine = 0.9998880029, threshold = 0.999
Pass [ infer cpu noptq 2 ] Output 0: cosine = 0.9998898506, threshold = 0.999
Pass [ infer cpu noptq 2 ] Output 1: cosine = 0.9999316335, threshold = 0.999
```

Token comparison:

```text
cpu eval token match ratio: 1.0000
cpu infer token match ratio: 1.0000
HuggingFace token ids: 151667, 198, 32313
nncase token ids:      151667, 198, 32313
generated tokens:      ['<think>', '\n', 'Okay']
```
