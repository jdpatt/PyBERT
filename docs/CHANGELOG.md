# Changelog

All notable changes to this project will be documented in this file.  Notes for a future PR.

## [4.0.0 - Unreleased]

### Added

- Pybert now generates a `pybert.log` file in the working directory.
- Added a command line interface to pybert.
- Added test suite to start unit testing pybert.
- Added menubar to gui.
- Added "shortcuts" to common actions like "Ctrl + R" to run a simulation.
- Added ability to save configuration as yaml or pickle without hard coding the choice.
- Added ability to clear a loaded waveform. (`View -> Clear Reference Waveform(s)`)

### Fixed

- Fixed sweeping simulations.
- Made the gui's console log immutable.
- Fixed loading results which failed because rx_in was being saved but had no plot.

### Changed

- Old documentation around testing with pip/tox to conda based.
- Reorganized package structure.
- Moved to python's builtin logging infrastructure over custom system.
- Jitter is now one attribute that is a dictionary instead of attributes for every individual
  jitter element.
- AMI and IBIS models should handle their own paths not pybert.
- Reduced what different parts of pybert modify internal state of pybert.
- Moved help and about to menubar like a traditional application.
- Removed all of the conda tied tooling so where both pypi or conda env. could be used/created.
