RPCS3 Metal
===========

[![macOS build](https://img.shields.io/github/actions/workflow/status/sirlocomotiva/rpcs3-metal/metal-mac.yml?branch=metal-renderer&logo=github&label=macOS%20build)](https://github.com/sirlocomotiva/rpcs3-metal/actions/workflows/metal-mac.yml)
[![Latest release](https://img.shields.io/github/v/release/sirlocomotiva/rpcs3-metal?label=download)](https://github.com/sirlocomotiva/rpcs3-metal/releases/latest)

An unofficial fork of [RPCS3](https://github.com/RPCS3/rpcs3), the open-source PlayStation 3 emulator, that adds a native Metal renderer for macOS.

On macOS, RPCS3 renders with Vulkan through MoltenVK, which translates Vulkan calls to Metal. This fork adds a third renderer next to Vulkan and OpenGL. It drives Metal directly and uses Apple features such as MetalFX upscaling, pipeline binary archives, framebuffer fetch and unified memory. The goal is to help people who only have a Mac get better performance out of their games.

All credit for the emulator goes to the RPCS3 team and its contributors. This fork only adds a renderer on top of their work, and it is not affiliated with or endorsed by them. **Please don't report problems with this build to the RPCS3 team.** Open an issue [in this repository](https://github.com/sirlocomotiva/rpcs3-metal/issues) instead, and mention that you use the Metal renderer. To support RPCS3 itself, see their [Patreon](https://rpcs3.net/patreon).

## Download

Get the latest build from [Releases](https://github.com/sirlocomotiva/rpcs3-metal/releases/latest). It needs an Apple Silicon Mac with macOS 15 or newer.

1. Open the DMG and drag **RPCS3 Metal** to Applications. Its name differs from the official app, so both can be installed side by side. They share settings, firmware, games and caches.
2. The app is not notarized, so macOS blocks the first launch. Go to System Settings → Privacy & Security and click **Open Anyway**, or run:
   ```bash
   xattr -dr com.apple.quarantine "/Applications/RPCS3 Metal.app"
   ```
3. Each time it starts, RPCS3 shows an *Experimental Build Warning*, because this is not an official release. Click **Yes** to continue.
4. Select the renderer in *Configuration → GPU → Renderer → Metal*, globally or per game.

Set up firmware and games as described in the official [quickstart](https://rpcs3.net/quickstart). Don't use *Check for Updates* in this build: it would replace it with the official RPCS3, which has no Metal renderer.

Every push to the [`metal-renderer`](https://github.com/sirlocomotiva/rpcs3-metal/tree/metal-renderer) branch is built by GitHub Actions ([metal-mac.yml](.github/workflows/metal-mac.yml)) and published as a release.

## Status

The Metal renderer is new, and so far it has mostly been tested with Grand Theft Auto IV. Other games may render incorrectly or not at all. If one does, switch that game back to Vulkan in its custom configuration, and please open an issue.

In progress:
* Shader interpreter. The interpreter modes currently fall back to asynchronous shader compilation.
* MetalFX temporal upscaling.

Design notes, build instructions, debugging switches and known limitations are in [rpcs3/Emu/RSX/Metal/README.md](rpcs3/Emu/RSX/Metal/README.md).

## How it was made

The Metal renderer was developed with the help of an AI coding assistant (Claude, by Anthropic) and tested in games on real hardware. Nothing from this fork has been submitted to RPCS3. Any future submission would follow their [AI use policy](#ai-use).

---

The rest of this page is the upstream RPCS3 README.

RPCS3
=====

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/RPCS3/rpcs3/rpcs3.yml?branch=master&logo=github&label=Actions)](https://github.com/RPCS3/rpcs3/actions/workflows/rpcs3.yml)
[![RPCS3 Discord Server](https://img.shields.io/discord/272035812277878785?color=5865F2&label=RPCS3%20Discord&logo=discord&logoColor=white)](https://discord.gg/rpcs3)

The world's first free and open-source PlayStation 3 emulator/debugger, written in C++ for Windows, Linux, macOS and FreeBSD.

You can find some basic information on our [**website**](https://rpcs3.net/). Game info is being populated on the [**Wiki**](https://wiki.rpcs3.net/).
For discussion about this emulator, PS3 emulation, and game compatibility reports, please visit our [**forums**](https://forums.rpcs3.net) and our [**Discord server**](https://discord.gg/RPCS3).

[**Support the Lead Developers on Patreon**](https://rpcs3.net/patreon)

## Contributing

If you want to help the project but do not code, the best way to help out is to test games and make bug reports. See:
* [Quickstart](https://rpcs3.net/quickstart)

If you want to contribute as a developer, please take a look at the following pages:

* [Coding Style](https://github.com/RPCS3/rpcs3/wiki/Coding-Style)
* [Developer Information](https://github.com/RPCS3/rpcs3/wiki/Developer-Information)

You should also contact any of the developers in the forums or in the Discord server to learn more about the current state of the emulator.

### AI Use

Use of AI tools for research and reverse engineering purposes is permitted. However, contributors are expected to fully own and understand all code they submit. Any communication with the team — including code, code comments, and GitHub comments — must come from the human contributor, not an AI agent acting autonomously.

We have unfortunately seen a rise in untested and unverified AI-generated slop being submitted to this project. This wastes maintainer time and, in worse cases, such changes get merged and break functionality for all users. Repeated violations will result in a ban from the repository. Please be respectful of everyone's time.

**Pull requests opened by AI agents or automated tools must include a disclosure in the PR description** stating the scope of AI involvement — which parts were AI-generated and what human testing or review was performed prior to submission. PRs that omit this disclosure may be closed without review.

If you are unsure about your work, open a discussion issue to talk it through with the team, or reach out to a maintainer on [Discord](https://discord.gg/RPCS3).

## Building

See [BUILDING.md](BUILDING.md) for more information about how to setup an environment to build RPCS3.

## Running

Check our friendly [quickstart](https://rpcs3.net/quickstart) guide to make sure your computer meets the minimum system requirements to run RPCS3.

Don't forget to have your graphics driver up to date and to install the [Visual C++ Redistributable Packages for Visual Studio 2022](https://aka.ms/vs/17/release/VC_redist.x64.exe) if you are a Windows user.

## License

Most files are licensed under the terms of GNU GPL-2.0-only License; see LICENSE file for details. Some files may be licensed differently; check appropriate file headers for details.
