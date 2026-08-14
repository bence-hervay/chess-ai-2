//! Minimal CPU-first self-play game AI research lab.
//!
//! Subsystems (see `ARCHITECTURE.md`):
//! - [`game`]: the core game interface;
//! - [`games`]: concrete game rule implementations;
//! - [`evaluation`]: paired-match arenas and statistics;
//! - [`experiment`]: run directories, manifests, resource probes;
//! - [`search`]: exact solver (oracle) and production alpha–beta search.

pub mod evaluation;
pub mod experiment;
pub mod game;
pub mod games;
pub mod search;
