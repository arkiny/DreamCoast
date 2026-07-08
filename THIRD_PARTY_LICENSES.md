# Third-Party Licenses

DreamCoast itself is licensed under the [MIT License](LICENSE). It builds on the
third-party components below, each under its own license. This file is a summary;
the authoritative license for each crate ships in that crate's source (and is
reported by `cargo metadata` / `cargo license`).

All bundled components are permissively licensed (MIT / Apache-2.0 / BSD-family),
compatible with redistribution under MIT.

## Rust dependencies (linked into the binary)

| Crate | Version | License |
|-------|---------|---------|
| [`ash`](https://github.com/ash-rs/ash) | 0.38 | MIT OR Apache-2.0 |
| [`windows`](https://github.com/microsoft/windows-rs) / `windows-core` | 0.62 | MIT OR Apache-2.0 |
| [`objc2`](https://github.com/madsmtm/objc2) / `objc2-foundation` / `objc2-app-kit` / `objc2-metal` / `objc2-quartz-core` (macOS) | 0.6 / 0.3 | MIT OR Apache-2.0 |
| [`glam`](https://github.com/bitshifter/glam-rs) | 0.30 | MIT OR Apache-2.0 |
| [`imgui`](https://github.com/imgui-rs/imgui-rs) / `imgui-sys` | 0.12 | MIT OR Apache-2.0 |
| [`gltf`](https://github.com/gltf-rs/gltf) | 1.4 | MIT OR Apache-2.0 |
| [`image`](https://github.com/image-rs/image) | 0.25 | MIT OR Apache-2.0 |
| [`tracing`](https://github.com/tokio-rs/tracing) | 0.1 | MIT |
| [`tracing-subscriber`](https://github.com/tokio-rs/tracing) | 0.3 | MIT |
| [`anyhow`](https://github.com/dtolnay/anyhow) | 1 | MIT OR Apache-2.0 |
| [`thiserror`](https://github.com/dtolnay/thiserror) | 2 | MIT OR Apache-2.0 |

These pull in further transitive dependencies, virtually all under MIT and/or
Apache-2.0. A complete, exact list is reproducible with:

```bash
cargo metadata --format-version 1   # license field per package
# or: cargo install cargo-license && cargo license
```

## Bundled native code

- **[Dear ImGui](https://github.com/ocornut/imgui)** — MIT License, © Omar Cornut.
  Vendored and built through the `imgui-sys` crate.

## Bundled shader code

### Academy Color Encoding System components

`crates/shader/shaders/aces.slang` is a Slang port (a derivative work) of the ACES 1.3
reference transforms — the RRT and the sRGB (100 nits, dim surround) ODT — from the
A.M.P.A.S. reference implementation, [`aces-dev` v1.3](https://github.com/ampas/aces-dev)
(`transforms/ctl/rrt`, `transforms/ctl/odt/sRGB`, `transforms/ctl/lib`). Per the license
below, the copyright notice, the list of conditions, and the Disclaimer of Warranty are
retained here in full.

> #### License Terms for Academy Color Encoding System Components
>
> Academy Color Encoding System (ACES) software and tools are provided by the Academy
> under the following terms and conditions: A worldwide, royalty-free, non-exclusive
> right to copy, modify, create derivatives, and use, in source and binary forms, is
> hereby granted, subject to acceptance of this license.
>
> Copyright © 2015 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.). Portions
> contributed by others as indicated. All rights reserved.
>
> Performance of any of the aforementioned acts indicates acceptance to be bound by the
> following terms and conditions:
>
> * Copies of source code, in whole or in part, must retain the above copyright notice,
>   this list of conditions and the Disclaimer of Warranty.
> * Use in binary form must retain the above copyright notice, this list of conditions
>   and the Disclaimer of Warranty in the documentation and/or other materials provided
>   with the distribution.
> * Nothing in this license shall be deemed to grant any rights to trademarks,
>   copyrights, patents, trade secrets or any other intellectual property of A.M.P.A.S.
>   or any contributors, except as expressly stated herein.
> * Neither the name "A.M.P.A.S." nor the name of any other contributors to this
>   software may be used to endorse or promote products derivative of or based on this
>   software without express prior written permission of A.M.P.A.S. or the contributors,
>   as appropriate.
>
> This license shall be construed pursuant to the laws of the State of California, and
> any disputes related thereto shall be subject to the jurisdiction of the courts
> therein.
>
> Disclaimer of Warranty: THIS SOFTWARE IS PROVIDED BY A.M.P.A.S. AND CONTRIBUTORS "AS
> IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT
> ARE DISCLAIMED. IN NO EVENT SHALL A.M.P.A.S., OR ANY CONTRIBUTORS OR DISTRIBUTORS, BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, RESITUTIONARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
> OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
> AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, THE ACADEMY SPECIFICALLY DISCLAIMS
> ANY REPRESENTATIONS OR WARRANTIES WHATSOEVER RELATED TO PATENT OR OTHER INTELLECTUAL
> PROPERTY RIGHTS IN THE ACADEMY COLOR ENCODING SYSTEM, OR APPLICATIONS THEREOF, HELD BY
> PARTIES OTHER THAN A.M.P.A.S., WHETHER DISCLOSED OR UNDISCLOSED.

## Build-time tooling (not linked into the binary)

These produce build artifacts (shader bytecode) but are not part of the shipped
binary. They live in the gitignored `tools/` directory and are obtained
separately.

- **[Slang](https://github.com/shader-slang/slang)** (`slangc`) — Apache-2.0.
  Compiles `.slang` shaders to SPIR-V and DXIL (Windows) and `metallib` (macOS).
- **DXC / DirectXShaderCompiler** (`dxcompiler`, bundled with Slang for DXIL
  output) — Apache-2.0 WITH LLVM-exception.
- **Apple Metal toolchain** (`xcrun metal`, from Xcode) — Apple SDK license.
  Invoked by Slang on macOS to produce `metallib`; not part of this repo and not
  shipped in the binary.

## Development-only (never shipped)

- **[Vulkan Validation Layers](https://github.com/KhronosGroup/Vulkan-ValidationLayers)**
  (Khronos) — Apache-2.0. Fetched into the gitignored `tools/vulkan-layers/`
  (`tools/fetch-vulkan-layers.py`) and loaded only in development builds;
  validation is compiled out of release builds (`cfg!(debug_assertions)`), so the
  layer is never part of a shipped artifact.
- **[renderdoc-mcp](https://github.com/Linkingooo/renderdoc-mcp)** — MIT (declared
  in its README). Cloned into the gitignored `tools/renderdoc-mcp/` and run as a
  separate MCP process for graphics debugging; never linked into the engine.
- **[RenderDoc](https://github.com/baldurk/renderdoc)** (Baldur Karlsson) — MIT.
  Built from source (gitignored `tools/renderdoc-src/`) only to produce a
  Python-3.12 `renderdoc.pyd` for the MCP above. RenderDoc-bundled **SWIG** (GPL)
  is invoked solely as a local build tool — its generated output is unrestricted by
  SWIG's GPL exception. No RenderDoc/SWIG sources or binaries are committed.

The RenderDoc MCP setup is documented in [`docs/renderdoc-mcp.md`](docs/renderdoc-mcp.md).

## Sample assets

CC0 / public-domain sample models may be fetched at runtime into the gitignored
`assets/` directory; none are committed to this repository.
